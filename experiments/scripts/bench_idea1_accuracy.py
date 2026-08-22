"""Idea-1 (head-aware mamba-checkpoint prefix cache) ACCURACY probe.

Adapted from ``dataset/bench_unified.py`` but self-contained for THIS box:
  - reads the extracted datasets under the external extracted dataset directory directly
    with pandas (correct on-disk schemas), no hardcoded dolphinfs paths;
  - fixes the reasoning-helper import path to ``benchmark/reasoning_benchmark``
    (bench_unified pointed one level too high) so ``answer_extraction`` +
    ``eval_utils.math_equal`` load;
  - scores ``gpqa_diamond`` (which on disk is boxed short-answers, columns
    problem/solution/domain — NOT the 4-choice MCQ bench_unified assumes) with
    the MATH scorer, pulling the ground truth out of the solution's \\boxed{...}.

WHY this exists: needle / RULER prove Idea-1 is accuracy-NEUTRAL only at a
ceiling (dense==1.000, no room to drop). These hard reasoning tasks (AIME /
HMMT / IMO / GPQA / MMLU-Pro) have real headroom, so an accuracy DROP under the
most aggressive head-aware setting (W_max=16 -> most channels classified local
-> dropped, NORECON) would actually show. The A/B is two SERVER CONFIGS on the
identical single script:
    DENSE  : head-aware OFF (lossless mamba prefix reuse)  -> baseline accuracy
    IDEA1  : head-aware route A, W_max=16, NORECON=1        -> lossy prefix reuse
Same dataset / sampling / num_tries in both arms; ``num_tries`` repeats of an
identical prompt are what create the shared-prefix radix hit that makes the
head-aware pool reconstruct (and, under W_max=16, drop) that prompt's mamba
state. The accuracy delta isolates the cost of head-aware channel dropping.

This talks to an ALREADY-RUNNING SGLang server (``--port``); it does not launch
one. The RULER suite launchers in ``experiments/scripts/`` invoke this client.

Example:
    python3 experiments/scripts/bench_idea1_accuracy.py --task aime2025 \
        --num-tries 4 --temperature 0.6 --parallel 32 --port 31007 \
        --result-file idea1_acc.jsonl
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

import sglang as sgl
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)

# RULER generation is shipped with this experiment bundle. The reasoning helpers
# and extracted reasoning datasets remain in the separate SGLang checkout.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_RULER_DIR = os.path.join(_REPO_ROOT, "datasets", "ruler")
sys.path.insert(0, _RULER_DIR)

import ruler_gen  # noqa: E402  (faithful NVIDIA/RULER CWE/FWE/VT/NIAH generators)

_SGLANG_REPO = os.environ.get("SGLANG_REPO", "")
_REASONING_DIR = (
    os.path.join(_SGLANG_REPO, "benchmark", "reasoning_benchmark")
    if _SGLANG_REPO
    else ""
)
if _REASONING_DIR:
    sys.path.insert(0, _REASONING_DIR)
try:
    import answer_extraction  # noqa: E402
    import eval_utils  # noqa: E402
except ModuleNotFoundError:
    answer_extraction = None
    eval_utils = None

# --------------------------------------------------------------------------- #
# Paths / task registry (local extracted datasets)                            #
# --------------------------------------------------------------------------- #
DATA_ROOT = os.environ.get(
    "DASC_DATA_ROOT",
    os.path.join(_SGLANG_REPO, "dataset", "datasets_extracted")
    if _SGLANG_REPO
    else os.path.join(_REPO_ROOT, "datasets", "extracted"),
)
CHOICES = "ABCDEFGHIJ"

TASK_REGISTRY = {
    # math family: (parquet|jsonl) with a (question, short-answer) per row.
    "aime2025": dict(
        path=os.path.join(
            DATA_ROOT, "AIME_2025", "data", "train-00000-of-00001.parquet"
        ),
        loader="parquet",
        qkey="Problem",
        akey="Answer",
        family="math",
    ),
    "aime2026": dict(
        path=os.path.join(DATA_ROOT, "AIME_2026", "aime2026.jsonl"),
        loader="jsonl",
        qkey="problem",
        akey="answer",
        family="math",
    ),
    "hmmt": dict(
        path=os.path.join(
            DATA_ROOT, "hmmt_feb_2026", "data", "train-00000-of-00001.parquet"
        ),
        loader="parquet",
        qkey="problem",
        akey="answer",
        family="math",
    ),
    "imo": dict(
        path=os.path.join(
            DATA_ROOT, "imo-answerbench", "data", "train-00000-of-00001.parquet"
        ),
        loader="parquet",
        qkey="Problem",
        akey="Short Answer",
        family="math",
    ),
    # GPQA Diamond: local parquet has no MCQ options (only problem/solution/domain).
    # _build_gpqa_csv.py matches the 198 diamond questions to ankner/gpqa (448 main
    # set with Correct Answer + 3 Incorrect Answers) and writes gpqa_diamond.csv.
    # We read that CSV and build 4-choice MCQ prompts scored by letter match.
    "gpqa_diamond": dict(
        path=os.path.join(DATA_ROOT, "gpqa_diamond", "gpqa_diamond.csv"),
        loader="csv",
        family="mcq-gpqa",
    ),
    # proper 10-choice MCQ.
    "mmlu_pro": dict(
        path=os.path.join(DATA_ROOT, "MMLU-Pro", "data"),
        loader="mmlu",
        family="mcq-mmlu",
    ),
    # LoCoMo long-conversation memory QA: each conversation transcript (~13-26k
    # tokens) is a LONG shared prefix reused across its ~100-260 questions -> the
    # transcript's KDA mamba checkpoint is what the head-aware pool reconstructs
    # (and, under W_max=16, drops local columns of). SQuAD-token-F1 scored.
    "locomo": dict(
        path=os.path.join(DATA_ROOT, "locomo", "locomo10.json"),
        loader="locomo",
        family="qa-f1",
    ),
    # REAL open-source RULER (github.com/NVIDIA/RULER) synthetic aggregation /
    # retrieval tasks, generated on the fly by datasets/ruler/ruler_gen.py (faithful port
    # of the official CWE/FWE/VT/NIAH generators + verbatim templates). Each RULER
    # instance is a long context; num_tries repeats it -> shared-prefix radix hit
    # so the 2nd+ try restores the head-aware checkpoint (idea1 NORECON = dropped
    # local columns not reconstructed). Scored by RULER string_match_part recall.
    "ruler": dict(loader="ruler", family="ruler"),
}

ALL_TASKS = ["aime2025", "aime2026", "hmmt", "imo", "gpqa_diamond", "mmlu_pro"]

MATH_PROMPT_SUFFIX = (
    "\nPlease reason step by step, and put your final answer within \\boxed{}."
)

_BOXED_RE = re.compile(r"\\boxed\{(.+?)\}\s*$")


@sgl.function
def reasoning_gen(s, prompt: str):
    s += sgl.user(prompt)
    s += sgl.assistant(sgl.gen("answer"))


def _extract_boxed(text: str) -> str:
    """Pull the inner content of a trailing \\boxed{...}; else return as-is."""
    if text is None:
        return ""
    m = _BOXED_RE.search(text.strip())
    if m:
        return m.group(1).strip()
    if "\\boxed" in text:  # loose fallback: last \boxed{...} anywhere
        m2 = re.findall(r"\\boxed\{(.+?)\}", text)
        if m2:
            return m2[-1].strip()
    return text.strip()


# --------------------------------------------------------------------------- #
# Loaders -> (prompts, metas, subjects)                                       #
# --------------------------------------------------------------------------- #
def _read_rows(spec):
    if spec["loader"] == "jsonl":
        rows = []
        with open(spec["path"]) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if spec["loader"] == "csv":
        return pd.read_csv(spec["path"]).to_dict("records")
    # parquet
    df = pd.read_parquet(spec["path"])
    return df.to_dict("records")


def build_math_dataset(spec, num_tries, limit):
    rows = _read_rows(spec)
    if limit:
        rows = rows[:limit]
    boxed_gt = spec["family"] == "math_boxed_gt"
    prompts, metas = [], []
    for row in rows:
        question = str(row[spec["qkey"]])
        raw_ans = str(row[spec["akey"]])
        answer = _extract_boxed(raw_ans) if boxed_gt else raw_ans
        prompt = question + MATH_PROMPT_SUFFIX
        for _ in range(num_tries):
            prompts.append({"prompt": prompt})
            metas.append({"answer": answer, "category": "all"})
    return prompts, metas, ["all"]


# --- GPQA Diamond MCQ ----------------------------------------------------- #
GPQA_MCQ_SUFFIX = (
    "\n\nThink step by step, then finish your answer with "
    '"the answer is (X)" where X is the correct letter.'
)


def build_gpqa_mcq_dataset(spec, num_tries, limit):
    """Build 4-choice MCQ prompts from gpqa_diamond.csv (Question + 4 options)."""
    import random

    rows = _read_rows(spec)
    if limit:
        rows = rows[:limit]
    rng = random.Random(0)  # fixed seed for reproducibility
    prompts, metas = [], []
    for row in rows:
        question = str(row["Question"])
        choices = [
            str(row["Correct Answer"]),
            str(row["Incorrect Answer 1"]),
            str(row["Incorrect Answer 2"]),
            str(row["Incorrect Answer 3"]),
        ]
        perm = rng.sample(range(4), 4)
        shuffled = [choices[i] for i in perm]
        correct_idx = perm.index(0)
        correct_letter = "ABCD"[correct_idx]
        letters = "ABCD"
        lines = [f"({letters[i]}) {shuffled[i]}" for i in range(4)]
        prompt = question + "\n\n" + "\n".join(lines) + GPQA_MCQ_SUFFIX
        for _ in range(num_tries):
            prompts.append({"prompt": prompt})
            metas.append({"answer": correct_letter, "category": "all"})
    return prompts, metas, ["all"]


# --- LoCoMo long-conversation memory QA ----------------------------------- #
LOCOMO_INSTRUCTION = (
    "\n\nBased ONLY on the conversation above, answer the question with a short, "
    "specific phrase (a name, date, place, number, or brief fact). If the answer "
    'is not stated in the conversation, reply "Not mentioned".\n'
)


def _locomo_transcript(conv):
    """Flatten a LoCoMo conversation dict into an ordered multi-session transcript."""
    sess_keys = sorted(
        (k for k in conv if k.startswith("session_") and not k.endswith("date_time")),
        key=lambda x: int(x.split("_")[1]),
    )
    lines = []
    for sk in sess_keys:
        dt = conv.get(sk + "_date_time", "")
        lines.append(f"[{sk} {dt}]")
        for turn in conv[sk]:
            spk = turn.get("speaker", "")
            txt = turn.get("text", "")
            cap = turn.get("blip_caption")
            if cap:
                txt = f"{txt} [shared an image: {cap}]"
            lines.append(f"{spk}: {txt}")
    return "\n".join(lines)


def build_locomo_dataset(spec, num_tries, limit, conv_limit=0):
    with open(spec["path"]) as f:
        convs = json.load(f)
    if conv_limit:
        convs = convs[:conv_limit]
    prompts, metas = [], []
    for conv in convs:
        transcript = _locomo_transcript(conv["conversation"])
        qa_list = conv["qa"]
        if limit:
            qa_list = qa_list[:limit]
        for qa in qa_list:
            # adversarial (category 5) has the gold under adversarial_answer.
            gold = qa.get("answer", qa.get("adversarial_answer", ""))
            gold = "Not mentioned" if gold is None else str(gold)
            question = str(qa["question"])
            prompt = transcript + LOCOMO_INSTRUCTION + f"Question: {question}\nAnswer:"
            for _ in range(num_tries):
                prompts.append({"prompt": prompt})
                metas.append(
                    {"answer": gold, "category": f"cat{qa.get('category', 0)}"}
                )
    return prompts, metas, ["all"]


# --- RULER (real NVIDIA/RULER CWE/FWE/VT/NIAH via ruler_gen) --------------- #
def build_ruler_dataset(spec, num_tries, ruler_tasks, ruler_length, ruler_num_samples):
    """Each RULER instance = one long context; num_tries repeats it so the 2nd+
    try hits the shared-prefix radix and restores the head-aware checkpoint.
    Category = subtask (cwe/fwe/vt/niah); answer = the gold list (recall-scored).
    """
    subtasks = [t.strip() for t in ruler_tasks.split(",") if t.strip()]
    prompts, metas = [], []
    for sub in subtasks:
        samples = ruler_gen.build_ruler_samples(sub, ruler_length, ruler_num_samples)
        for prompt, gold in samples:
            for _ in range(num_tries):
                prompts.append({"prompt": prompt})
                metas.append({"answer": gold, "category": sub})
    return prompts, metas, subtasks


def score_ruler(question, model_output, gt_answer):
    """RULER string_match_part recall of the gold list in the model output."""
    recall = ruler_gen.score_recall(model_output, gt_answer)
    return str(model_output)[:200], ", ".join(str(g) for g in gt_answer), recall


_F1_PUNCT = re.compile(r"[^a-z0-9 ]")
_F1_ARTICLES = re.compile(r"\b(a|an|the)\b")


def _f1_normalize(s):
    s = str(s).lower()
    s = _F1_PUNCT.sub(" ", s)
    s = _F1_ARTICLES.sub(" ", s)
    return s.split()


def score_qa_f1(question, model_output, gt_answer):
    """SQuAD-style token-F1 between the model's short answer and the gold."""
    # take the model's first non-empty line as the answer (strip any reasoning).
    pred_raw = ""
    for ln in str(model_output).strip().splitlines():
        if ln.strip():
            pred_raw = ln.strip()
            break
    pred = _f1_normalize(pred_raw)
    gold = _f1_normalize(gt_answer)
    if not pred and not gold:
        return pred_raw, str(gt_answer), 1.0
    if not pred or not gold:
        return pred_raw, str(gt_answer), 0.0
    common = {}
    for t in pred:
        common[t] = min(pred.count(t), gold.count(t))
    n_same = sum(common.values())
    if n_same == 0:
        return pred_raw, str(gt_answer), 0.0
    precision = n_same / len(pred)
    recall = n_same / len(gold)
    f1 = 2 * precision * recall / (precision + recall)
    return pred_raw, str(gt_answer), f1


# --- MMLU-Pro (mirrors bench_unified.build_mmlu_dataset) ------------------- #
def mmlu_format_cot_example(example, including_answer=True):
    prompt = "Question:\n" + str(example["question"]) + "\n"
    prompt += "Options:\n"
    options = example["options"]
    if isinstance(options, str):
        options = [options]
    for i, opt in enumerate(list(options)):
        if i >= len(CHOICES):
            break
        prompt += f"{CHOICES[i]}. {str(opt).strip()}\n"
    if including_answer:
        cot = str(example.get("cot_content", "") or "").replace(
            "A: Let's think step by step.", "Answer: Let's think step by step."
        )
        prompt += cot + "\n\n"
    else:
        prompt += "Answer: Let's think step by step."
    return prompt


def build_mmlu_dataset(
    spec, num_tries, limit, ntrain, selected_subjects, question_limit=0
):
    data_dir = spec["path"]
    test = pd.read_parquet(
        os.path.join(data_dir, "test-00000-of-00001.parquet")
    ).to_dict("records")
    val = pd.read_parquet(
        os.path.join(data_dir, "validation-00000-of-00001.parquet")
    ).to_dict("records")

    test_by_cat = defaultdict(list)
    for row in test:
        test_by_cat[row["category"]].append(row)
    val_by_cat = defaultdict(list)
    for row in val:
        val_by_cat[row["category"]].append(row)

    if selected_subjects == "all":
        subjects = sorted(test_by_cat.keys())
    else:
        wanted = [s.strip() for s in selected_subjects.split(",")]
        subjects = sorted(
            cat
            for cat in test_by_cat
            if any(w.replace(" ", "_") in cat.replace(" ", "_") for w in wanted)
        )

    # Build the DISTINCT-question list first (prompt, answer, subject), so
    # question_limit caps the TOTAL number of questions across the whole task
    # (not per-category like `limit`). num_tries is expanded afterwards.
    entries = []
    for subject in subjects:
        description = (
            "The following are multiple choice questions (with answers) about "
            '{}. Think step by step and then finish your answer with "the answer '
            'is (X)" where X is the correct letter choice.\n\n'.format(subject)
        )
        fewshot = "".join(
            mmlu_format_cot_example(ex, including_answer=True)
            for ex in val_by_cat[subject][:ntrain]
        )
        curr_list = test_by_cat[subject]
        if limit:
            curr_list = curr_list[:limit]
        for curr in curr_list:
            prompt = description + fewshot + mmlu_format_cot_example(curr, False)
            entries.append((prompt, curr["answer"], subject))

    if question_limit:
        entries = entries[:question_limit]

    prompts, metas = [], []
    for prompt, answer, subject in entries:
        for _ in range(num_tries):
            prompts.append({"prompt": prompt})
            metas.append({"answer": answer, "category": subject})
    used_subjects = sorted({s for _, _, s in entries})
    return prompts, metas, used_subjects


def mmlu_extract_answer(text):
    m = re.search(r"answer is \(?([ABCDEFGHIJ])\)?", text)
    if m:
        return m.group(1)
    m = re.search(r".*[aA]nswer:\s*\(?([A-J])\)?", text)
    if m:
        return m.group(1)
    m = re.search(r"\b([A-J])\b(?!.*\b[A-J]\b)", text, re.DOTALL)
    if m:
        return m.group(1)
    return None


# --------------------------------------------------------------------------- #
# Scoring                                                                     #
# --------------------------------------------------------------------------- #
def score_math(question, model_output, gt_answer):
    if answer_extraction is None or eval_utils is None:
        raise RuntimeError(
            "Math scoring requires SGLANG_REPO=/path/to/sglang so that "
            "benchmark/reasoning_benchmark is importable."
        )
    gt = answer_extraction.strip_string(str(gt_answer))
    pred = answer_extraction.extract_math_answer(question, model_output, "limo")
    pred = pred[-1] if isinstance(pred, list) else pred
    is_correct = 1 if eval_utils.math_equal(pred, gt) else 0
    return pred, gt, is_correct


def make_scorer(family):
    if family in ("math", "math_boxed_gt"):
        return score_math
    if family == "qa-f1":
        return score_qa_f1
    if family == "ruler":
        return score_ruler
    if family in ("mcq-mmlu", "mcq-gpqa"):

        def _s(question, model_output, gt_answer):
            pred = mmlu_extract_answer(model_output)
            return pred, gt_answer, (1 if pred == gt_answer else 0)

        return _s
    raise ValueError(f"Unknown family: {family}")


def build_dataset(task, spec, args, num_tries):
    family = spec["family"]
    if family in ("math", "math_boxed_gt"):
        return build_math_dataset(spec, num_tries, args.limit)
    if family == "qa-f1":
        return build_locomo_dataset(
            spec, num_tries, args.limit, conv_limit=args.locomo_conv_limit
        )
    if family == "ruler":
        return build_ruler_dataset(
            spec, num_tries, args.ruler_tasks, args.ruler_length, args.ruler_num_samples
        )
    if family == "mcq-mmlu":
        return build_mmlu_dataset(
            spec,
            num_tries,
            args.limit,
            args.ntrain,
            args.selected_subjects,
            question_limit=args.mmlu_question_limit,
        )
    if family == "mcq-gpqa":
        return build_gpqa_mcq_dataset(spec, num_tries, args.limit)
    raise ValueError(f"Unknown family: {family}")


# --------------------------------------------------------------------------- #
# Per-task run                                                                #
# --------------------------------------------------------------------------- #
def run_task(task, args):
    spec = dict(TASK_REGISTRY[task])
    family = spec["family"]
    num_tries = args.num_tries or 1
    prompts, metas, subjects = build_dataset(task, spec, args, num_tries)
    num_questions = len(prompts) // num_tries if num_tries else len(prompts)
    print(
        f"\n[{task}] arm={args.arm} Evaluating {len(prompts)} prompts "
        f"({num_questions} questions x {num_tries} tries) "
        f"over {len(subjects)} subject(s)"
    )

    scorer = make_scorer(family)
    tic = time.perf_counter()
    states = reasoning_gen.run_batch(
        prompts,
        num_threads=args.parallel,
        progress_bar=True,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
    )
    latency = time.perf_counter() - tic

    per_category = defaultdict(lambda: {"corr": 0, "wrong": 0})
    outcomes, output_lengths, records = [], [], []
    for i, state in enumerate(states):
        category = metas[i]["category"]
        gt_answer = metas[i]["answer"]
        question = prompts[i]["prompt"]
        model_output = state["answer"]
        try:
            completion_tokens = state.get_meta_info("answer")["completion_tokens"]
        except Exception:
            completion_tokens = None
        try:
            pred, gt_norm, is_correct = scorer(question, model_output, gt_answer)
        except Exception as e:
            print(f"Error extracting answer: {e}")
            pred, gt_norm, is_correct = None, gt_answer, 0
        outcomes.append(is_correct)
        output_lengths.append(completion_tokens)
        per_category[category]["corr" if is_correct else "wrong"] += 1
        records.append(
            {
                "index": i,
                "category": category,
                "question": question,
                "model_output": model_output,
                "output_length": completion_tokens,
                "pred_answer": pred,
                "gt_answer": gt_norm,
                "is_correct": is_correct,
            }
        )

    out_file = args.output_file or f"{task}_{args.arm}_outputs.jsonl"
    with open(out_file, "w") as fout:
        for record in records:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} model outputs to {out_file}")

    overall_accuracy = float(np.mean(outcomes)) if outcomes else 0.0
    if len(subjects) > 1:
        print("------ per-category accuracy ------")
        for category in subjects:
            rec = per_category[category]
            total = rec["corr"] + rec["wrong"]
            acc = rec["corr"] / total if total else 0.0
            print(f"{category}: {acc:.4f} ({rec['corr']}/{total})")
    print(f"[{task}] arm={args.arm} Overall Accuracy: {overall_accuracy:.4f}")

    valid_lengths = [x for x in output_lengths if x is not None]
    avg_output_length = float(np.mean(valid_lengths)) if valid_lengths else 0.0
    print(f"[{task}] Average answer length: {avg_output_length:.1f} tokens")

    mean_se = None
    warmup_acc = None
    replay_acc = None
    if num_tries > 1:
        arr = np.array(outcomes).reshape(-1, num_tries)
        se = np.std(arr, axis=1, ddof=1) / np.sqrt(num_tries)
        mean_se = float(se.mean())
        # Split by try index: try 0 = warmup (cache-miss, full prefill);
        # try 1+ = replay (cache-hit, head-aware pool restores/drops state).
        warmup_acc = float(arr[:, 0].mean())
        replay_acc = float(arr[:, 1:].mean())
        print(f"[{task}] Mean SE of accuracy across questions: {mean_se:.4f}")
        print(
            f"[{task}] arm={args.arm} Warmup acc (try 0, cache-miss): {warmup_acc:.4f}  "
            f"Replay acc (try 1-{num_tries-1}, cache-hit): {replay_acc:.4f}"
        )

    with open(args.result_file, "a") as fout:
        fout.write(
            json.dumps(
                {
                    "task": task,
                    "arm": args.arm,
                    "backend": args.backend,
                    "latency": round(latency, 3),
                    "overall_accuracy": round(overall_accuracy, 4),
                    "warmup_accuracy": (
                        round(warmup_acc, 4) if warmup_acc is not None else None
                    ),
                    "replay_accuracy": (
                        round(replay_acc, 4) if replay_acc is not None else None
                    ),
                    "avg_output_length": round(avg_output_length, 1),
                    "mean_se_accuracy": (
                        round(mean_se, 4) if mean_se is not None else None
                    ),
                    "num_questions": num_questions,
                    "num_tries": num_tries,
                    "parallel": args.parallel,
                    "wmax": args.wmax,
                    "sampling": {
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "top_k": args.top_k,
                        "max_new_tokens": args.max_new_tokens,
                    },
                    "per_category": (
                        {c: per_category[c] for c in subjects}
                        if len(subjects) > 1
                        else None
                    ),
                }
            )
            + "\n"
        )

    return {
        "task": task,
        "arm": args.arm,
        "overall_accuracy": overall_accuracy,
        "mean_se": mean_se,
        "num_questions": num_questions,
        "num_tries": num_tries,
        "avg_output_length": avg_output_length,
        "warmup_acc": warmup_acc,
        "replay_acc": replay_acc,
    }


def main(args):
    sgl.set_default_backend(select_sglang_backend(args))
    tasks = ALL_TASKS if args.task == "all" else [args.task]
    summaries = [run_task(t, args) for t in tasks]
    print(f"\n====== Summary (arm={args.arm}) ======")
    for s in summaries:
        se = f"{s['mean_se']:.4f}" if s["mean_se"] is not None else "n/a"
        wa = f"{s['warmup_acc']:.4f}" if s.get("warmup_acc") is not None else "n/a"
        ra = f"{s['replay_acc']:.4f}" if s.get("replay_acc") is not None else "n/a"
        print(
            f"{s['task']}: acc={s['overall_accuracy']:.4f}  warmup={wa}  replay={ra}  "
            f"mean_se={se}  (q={s['num_questions']} x{s['num_tries']})"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument(
        "--task",
        type=str,
        default="aime2025",
        choices=list(TASK_REGISTRY.keys()) + ["all"],
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="dense",
        help="Label for this run (dense|idea1_wmax16), recorded in results.",
    )
    parser.add_argument(
        "--wmax",
        type=int,
        default=0,
        help="W_max the server was launched with (0 = dense), recorded only.",
    )
    parser.add_argument(
        "--num-tries",
        type=int,
        default=4,
        help="Repeats per question (creates shared-prefix radix hits).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, first N questions per category (smoke / subset).",
    )
    parser.add_argument(
        "--mmlu-question-limit",
        type=int,
        default=0,
        help="If >0, cap the TOTAL number of distinct MMLU-Pro "
        "questions for the whole task (across all categories). "
        "Applied before num_tries expansion; mmlu_pro only.",
    )
    parser.add_argument("--ntrain", "-k", type=int, default=0)
    parser.add_argument(
        "--locomo-conv-limit",
        type=int,
        default=0,
        help="If >0, use only the first N LoCoMo conversations "
        "(locomo task only). --limit caps QA per conversation.",
    )
    parser.add_argument(
        "--ruler-tasks",
        type=str,
        default="cwe,fwe",
        help="Comma list of RULER subtasks: cwe,fwe,vt,niah " "(ruler task only).",
    )
    parser.add_argument(
        "--ruler-length",
        type=str,
        default="4k",
        choices=["4k", "8k", "16k", "32k", "64k", "40k", "128k"],
        help="RULER context length preset (ruler task only).",
    )
    parser.add_argument(
        "--ruler-num-samples",
        type=int,
        default=100,
        help="Distinct RULER instances per subtask (ruler only).",
    )
    parser.add_argument("--selected-subjects", "-sub", type=str, default="all")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--frequency-penalty", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--output-file", type=str, default=None)
    add_common_sglang_args_and_parse(parser)
    args = parser.parse_args()
    main(args)
