#!/usr/bin/env python3
"""Faithful port of the official NVIDIA/RULER synthetic task generators.

This is the REAL open-source RULER (github.com/NVIDIA/RULER) task logic —
verbatim templates + answer_prefix from RULER `constants.py`, the exact
common/uncommon-repeat structure (CWE), the zeta-distributed coded-word sampler
(FWE), the VAR-chain tracker (VT), and the magic-number needle (NIAH). It is NOT
the hand-rolled `experiments/scripts/ruler_diffuse.py` approximation.

Only difference from upstream: the haystack size is passed in directly (a
`num_words` / `num_haystack` knob) instead of running RULER's tokenizer binary
search, so generation is tokenizer-agnostic. The task content, frequency
structure, gold answers and scoring are unchanged.

Vocab: official RULER CWE uses the `wonderwords` noun/adjective/verb lists;
those three files are vendored under `datasets/ruler/data/` (fetched from the
wonderwords repo). The same directory also contains the essay, SQuAD, and HotpotQA
inputs required by the complete 13-task suite.
"""

import os
import random
import string
import uuid

import numpy as np
from scipy.special import zeta

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Verbatim from RULER scripts/data/synthetic/constants.py (TASKS).
TASKS = {
    "niah": {
        "tokens_to_generate": 128,
        "template": (
            "Some special magic {type_needle_v} are hidden within the following "
            "text. Make sure to memorize it. I will quiz you about the "
            "{type_needle_v} afterwards.\n{context}\nWhat are all the special "
            "magic {type_needle_v} for {query} mentioned in the provided text?"
        ),
        "answer_prefix": (
            " The special magic {type_needle_v} for {query} mentioned in the "
            "provided text are"
        ),
    },
    "variable_tracking": {
        "tokens_to_generate": 30,
        "template": (
            "Memorize and track the chain(s) of variable assignment hidden in "
            "the following text.\n\n{context}\nQuestion: Find all variables that "
            "are assigned the value {query} in the text above."
        ),
        "answer_prefix": (
            " Answer: According to the chain(s) of variable assignment in the "
            "text above, {num_v} variables are assigned the value {query}, they "
            "are: "
        ),
    },
    "common_words_extraction": {
        "tokens_to_generate": 120,
        "template": (
            "Below is a numbered list of words. In these words, some appear more "
            "often than others. Memorize the ones that appear most often.\n"
            "{context}\nQuestion: What are the 10 most common words in the above "
            "list?"
        ),
        "answer_prefix": (
            " Answer: The top 10 words that appear most often in the list are:"
        ),
    },
    "freq_words_extraction": {
        "tokens_to_generate": 50,
        "template": (
            "Read the following coded text and track the frequency of each coded "
            "word. Find the three most frequently appeared coded words. "
            "{context}\nQuestion: Do not provide any explanation. Please ignore "
            "the dots '....'. What are the three most frequently appeared words "
            "in the above coded text?"
        ),
        "answer_prefix": (
            " Answer: According to the coded text above, the three most "
            "frequently appeared words are:"
        ),
    },
    # Verbatim from RULER qa.py (TEMPLATE / ANSWER_PREFIX). Scored by
    # string_match_part (any gold answer as a substring of the prediction).
    "qa": {
        "tokens_to_generate": 32,
        "template": (
            "Answer the question based on the given documents. Only give me the "
            "answer and do not output any other words.\n\nThe following are given "
            "documents.\n\n{context}\n\nAnswer the question based on the given "
            "documents. Only give me the answer and do not output any other "
            "words.\n\nQuestion: {query}"
        ),
        "answer_prefix": " Answer:",
    },
}


def _words_vocab():
    vocab = []
    for f in ("nounlist", "adjectivelist", "verblist"):
        p = os.path.join(_DATA, f + ".txt")
        with open(p) as fh:
            vocab += [w.strip() for w in fh if w.strip()]
    return sorted(set(vocab))


# --------------------------------------------------------------------------- #
# CWE — common words extraction (RULER common_words_extraction.py)            #
# --------------------------------------------------------------------------- #
def _cwe_example(
    words, num_words, common_repeats, uncommon_repeats, common_nums, rng, shuffle_seed
):
    word_list_full = rng.sample(words, num_words)
    common = word_list_full[:common_nums]
    uncommon = word_list_full[common_nums:]
    word_list = common * int(common_repeats) + uncommon * int(uncommon_repeats)
    random.Random(shuffle_seed).shuffle(word_list)
    context = " ".join(f"{i + 1}. {w}" for i, w in enumerate(word_list))
    return context, common


def gen_cwe(num_words=400, num_cw=10, freq_cw=30, freq_ucw=3, num_fewshot=1, seed=42):
    """Return (prompt, gold_list). gold = the num_cw common words."""
    words = _words_vocab()
    num_words = min(num_words, len(words))  # cap at vocab size (8166)
    rng = random.Random(seed)
    tmpl = TASKS["common_words_extraction"]["template"]
    ans_prefix = TASKS["common_words_extraction"]["answer_prefix"]

    few = []
    for _ in range(num_fewshot):
        ctx_e, ans_e = _cwe_example(words, 40, 10, 3, num_cw, rng, seed)
        few.append((ctx_e, ans_e))
    context, answer = _cwe_example(
        words, num_words, freq_cw, freq_ucw, num_cw, rng, seed
    )

    shots = []
    for ctx_e, ans_e in few:
        shots.append(
            tmpl.format(context=ctx_e)
            + ans_prefix
            + " "
            + " ".join(f"{i + 1}. {w}" for i, w in enumerate(ans_e))
        )
    prompt = "\n".join(shots) + "\n" + tmpl.format(context=context) + ans_prefix
    return prompt, list(answer)


# --------------------------------------------------------------------------- #
# FWE — frequent words extraction (RULER freq_words_extraction.py)            #
# --------------------------------------------------------------------------- #
def gen_fwe(num_words=1500, vocab_size=160, alpha=2.0, coded_wordlen=6, seed=42):
    """Return (prompt, gold_list). gold = the top-3 coded words (zeta-ranked)."""
    rng = random.Random(seed)
    vocab = [
        "".join(rng.choices(string.ascii_lowercase, k=coded_wordlen))
        for _ in range(vocab_size)
    ]
    while len(set(vocab)) < vocab_size:
        vocab.append("".join(rng.choices(string.ascii_lowercase, k=coded_wordlen)))
    vocab = sorted(set(vocab))
    random.Random(seed).shuffle(vocab)
    vocab[0] = "..."  # top-ranked treated as noise (ignored)

    k = np.arange(1, len(vocab) + 1)
    sampled_cnt = num_words * (k**-alpha) / zeta(alpha)
    sampled_words = [[w] * zi for w, zi in zip(vocab, sampled_cnt.astype(int))]
    sampled_words = [x for wl in sampled_words for x in wl]
    random.Random(seed).shuffle(sampled_words)

    tmpl = TASKS["freq_words_extraction"]["template"]
    ans_prefix = TASKS["freq_words_extraction"]["answer_prefix"]
    prompt = tmpl.format(context=" ".join(sampled_words)) + ans_prefix
    return prompt, list(vocab[1:4])


# --------------------------------------------------------------------------- #
# VT — variable tracking (RULER variable_tracking.py, noise haystack)         #
# --------------------------------------------------------------------------- #
_VT_NOISE = (
    "The grass is green. The sky is blue. The sun is yellow. "
    "Here we go. There and back again."
)


def gen_vt(num_noises=280, num_chains=1, num_hops=4, seed=42):
    """Return (prompt, gold_list). gold = the num_hops+1 vars in the queried chain."""
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    k = 5
    n_need = num_chains * (num_hops + 1)
    vars_all = [
        "".join(rng.choices(string.ascii_uppercase, k=k)).upper() for _ in range(n_need)
    ]
    while len(set(vars_all)) < n_need:
        vars_all.append("".join(rng.choices(string.ascii_uppercase, k=k)).upper())

    vars_ret, chains = [], []
    for i in range(0, len(vars_all), num_hops + 1):
        tv = vars_all[i : i + num_hops + 1]
        vars_ret.append(tv)
        chain = [f"VAR {tv[0]} = {np_rng.randint(10000, 99999)}"]
        for j in range(num_hops):
            chain.append(f"VAR {tv[j + 1]} = VAR {tv[j]} ")
        chains.append(chain)

    value = chains[0][0].split("=")[-1].strip()
    sentences = [_VT_NOISE] * num_noises
    for chain in chains:
        positions = sorted(rng.sample(range(len(sentences)), len(chain)))
        for insert_pi, j in zip(positions, range(len(chain))):
            sentences.insert(insert_pi + j, chain[j])
    context = "\n".join(sentences).replace(". \n", ".\n")

    tmpl = TASKS["variable_tracking"]["template"]
    ans_prefix = TASKS["variable_tracking"]["answer_prefix"]
    prompt = tmpl.format(
        context=context, query=value, num_v=num_hops + 1
    ) + ans_prefix.format(num_v=num_hops + 1, query=value)
    return prompt, list(vars_ret[0])


# --------------------------------------------------------------------------- #
# NIAH — magic-key/value needle, 8 official RULER variants                     #
# (RULER niah.py: type_haystack {repeat,essay,needle} x k/v {words,numbers,     #
#  uuids} x num_needle_{k,v,q}). S1/S2/S3, MK1/MK2/MK3, MV, MQ.                 #
# --------------------------------------------------------------------------- #
_NIAH_NOISE = (
    "The grass is green. The sky is blue. The sun is yellow. "
    "Here we go. There and back again."
)
_NIAH_NEEDLE = "One of the special magic {tnv} for {key} is: {value}."

_ESSAY_WORDS_CACHE = None


def _essay_words():
    """Paul Graham essay corpus split into words (RULER 'essay' haystack)."""
    global _ESSAY_WORDS_CACHE
    if _ESSAY_WORDS_CACHE is None:
        p = os.path.join(_DATA, "pg_essays_corpus.txt")
        with open(p, encoding="utf-8", errors="ignore") as fh:
            _ESSAY_WORDS_CACHE = fh.read().split()
    return _ESSAY_WORDS_CACHE


def gen_niah(
    num_haystack=800,
    type_haystack="repeat",
    type_needle_k="words",
    type_needle_v="numbers",
    num_needle_k=1,
    num_needle_v=1,
    num_needle_q=1,
    seed=42,
):
    """Return (prompt, gold_list) for one RULER NIAH instance.

    Mirrors RULER niah.py: num_needle_k is expanded to max(num_needle_k,
    num_needle_q) so multiquery inserts one needle per queried key; the
    non-queried extras are in-context distractors (multikey). type_needle_v
    selects the answer surface (numbers vs uuids); type_haystack selects the
    filler (repeat noise / PG essay sentences / distractor needles).
    """
    num_needle_k = max(num_needle_k, num_needle_q)
    rng = random.Random(seed)
    words = _words_vocab()
    tnv = "uuids" if type_needle_v == "uuids" else "numbers"

    def rnd_num():
        return str(rng.randint(10**6, 10**7 - 1))

    def rnd_uuid():
        return str(uuid.UUID(int=rng.getrandbits(128), version=4))

    def make_key():
        return rnd_uuid() if type_needle_k == "uuids" else rng.choice(words)

    def make_val():
        return rnd_uuid() if type_needle_v == "uuids" else rnd_num()

    keys, values, needles, used = [], [], [], set()
    for _ in range(num_needle_k):
        key = make_key()
        while key in used:
            key = make_key()
        used.add(key)
        keys.append(key)
        val = []
        for _ in range(num_needle_v):
            v = make_val()
            val.append(v)
            needles.append(_NIAH_NEEDLE.format(tnv=tnv, key=key, value=v))
        values.append(val)
    random.Random(seed).shuffle(needles)

    if type_haystack == "essay":
        ew = _essay_words()
        text = " ".join(ew[:num_haystack])
        sentences = [s.strip() + "." for s in text.split(".") if s.strip()]
    elif type_haystack == "needle":
        sentences = [
            _NIAH_NEEDLE.format(tnv=tnv, key=make_key(), value=make_val())
            for _ in range(num_haystack)
        ]
    else:  # repeat
        sentences = [_NIAH_NOISE] * num_haystack

    idxs = sorted(rng.sample(range(len(sentences)), len(needles)), reverse=True)
    for index, needle in zip(idxs, needles):
        sentences.insert(index, needle)
    context = "\n".join(sentences)

    q_idx = sorted(rng.sample(range(num_needle_k), num_needle_q))
    q_keys = [keys[i] for i in q_idx]
    if len(q_keys) == 1:
        query = q_keys[0]
    else:
        query = ", ".join(q_keys[:-1]) + ", and " + q_keys[-1]
    gold = [v for i in q_idx for v in values[i]]

    tmpl = TASKS["niah"]["template"]
    ans_prefix = TASKS["niah"]["answer_prefix"]
    prompt = tmpl.format(
        type_needle_v=tnv, context=context, query=query
    ) + ans_prefix.format(type_needle_v=tnv, query=query)
    return prompt, gold


# The 8 official RULER NIAH variant configs (RULER synthetic.yaml). Names match
# Table-2 columns: S1/S2/S3 single, MK1/MK2/MK3 multikey, MV multivalue, MQ
# multiquery.
_NIAH_VARIANTS = {
    "s1": dict(
        type_haystack="repeat",
        type_needle_k="words",
        type_needle_v="numbers",
        num_needle_k=1,
        num_needle_v=1,
        num_needle_q=1,
    ),
    "s2": dict(
        type_haystack="essay",
        type_needle_k="words",
        type_needle_v="numbers",
        num_needle_k=1,
        num_needle_v=1,
        num_needle_q=1,
    ),
    "s3": dict(
        type_haystack="essay",
        type_needle_k="words",
        type_needle_v="uuids",
        num_needle_k=1,
        num_needle_v=1,
        num_needle_q=1,
    ),
    "mk1": dict(
        type_haystack="essay",
        type_needle_k="words",
        type_needle_v="numbers",
        num_needle_k=4,
        num_needle_v=1,
        num_needle_q=1,
    ),
    "mk2": dict(
        type_haystack="needle",
        type_needle_k="words",
        type_needle_v="numbers",
        num_needle_k=4,
        num_needle_v=1,
        num_needle_q=1,
    ),
    "mk3": dict(
        type_haystack="needle",
        type_needle_k="uuids",
        type_needle_v="uuids",
        num_needle_k=4,
        num_needle_v=1,
        num_needle_q=1,
    ),
    "mv": dict(
        type_haystack="essay",
        type_needle_k="words",
        type_needle_v="numbers",
        num_needle_k=1,
        num_needle_v=4,
        num_needle_q=1,
    ),
    "mq": dict(
        type_haystack="essay",
        type_needle_k="words",
        type_needle_v="numbers",
        num_needle_k=1,
        num_needle_v=1,
        num_needle_q=4,
    ),
}


def _gen_niah_variant(variant):
    cfg = _NIAH_VARIANTS[variant]

    def _g(num_haystack, seed=42):
        return gen_niah(num_haystack=num_haystack, seed=seed, **cfg)

    return _g


# --------------------------------------------------------------------------- #
# QA — QA-S (SQuAD) / QA-H (HotpotQA), RULER qa.py                             #
# The gold answer is inserted (via its containing document) among distractor  #
# documents sampled to reach the target length; the model must extract it.    #
# Scored by string_match_part (any gold answer as a substring of the pred).   #
# --------------------------------------------------------------------------- #
_QA_CACHE = {}  # "qa_s"/"qa_h" -> (qas_records, docs_pool)


def _read_squad():
    """RULER read_squad: global unique-paragraph pool + answerable QA records."""
    import json

    with open(os.path.join(_DATA, "squad_dev-v2.0.json")) as fh:
        data = json.load(fh)["data"]
    total_docs = sorted({p["context"] for art in data for p in art["paragraphs"]})
    doc_idx = {c: i for i, c in enumerate(total_docs)}
    qas = []
    for art in data:
        more = [doc_idx[p["context"]] for p in art["paragraphs"]]
        for p in art["paragraphs"]:
            gi = doc_idx[p["context"]]
            for qa in p["qas"]:
                if qa.get("is_impossible") or not qa["answers"]:
                    continue
                outs = sorted({a["text"] for a in qa["answers"]})
                qas.append(
                    {
                        "query": qa["question"],
                        "outputs": outs,
                        "context": [gi],
                        "more": [j for j in more if j != gi],
                    }
                )
    return qas, total_docs


def _read_hotpotqa():
    """RULER read_hotpotqa: global unique-doc pool + supporting-fact QA records.
    Each doc = 'title\\n<sentences joined>'; gold = the supporting-fact titles."""
    import pyarrow.parquet as pq

    rows = pq.read_table(
        os.path.join(_DATA, "hotpot_distractor_validation.parquet")
    ).to_pylist()
    # global pool
    pool_set = {}

    def _docstr(t, sents):
        return f"{t}\n{''.join(sents)}"

    for r in rows:
        titles = r["context"]["title"]
        sents = r["context"]["sentences"]
        for t, s in zip(titles, sents):
            pool_set.setdefault(_docstr(t, s), None)
    total_docs = sorted(pool_set)
    doc_idx = {c: i for i, c in enumerate(total_docs)}
    qas = []
    for r in rows:
        titles = r["context"]["title"]
        sents = r["context"]["sentences"]
        sup = set(r["supporting_facts"]["title"])
        local = {t: doc_idx[_docstr(t, s)] for t, s in zip(titles, sents)}
        gold = [idx for t, idx in local.items() if t in sup]
        more = [idx for t, idx in local.items() if t not in sup]
        if not gold:
            continue
        qas.append(
            {
                "query": r["question"],
                "outputs": [r["answer"]],
                "context": gold,
                "more": more,
            }
        )
    return qas, total_docs


def _qa_corpus(name):
    if name not in _QA_CACHE:
        _QA_CACHE[name] = _read_squad() if name == "qa_s" else _read_hotpotqa()
    return _QA_CACHE[name]


def _gen_qa(name):
    def _g(num_docs, seed=42):
        qas, docs = _qa_corpus(name)
        rec = qas[seed % len(qas)]
        gold_i = list(rec["context"])
        more_i = list(rec["more"])
        need = max(0, num_docs - len(gold_i))
        rng = random.Random(seed)
        if need <= len(more_i):
            chosen = gold_i + rng.sample(more_i, need)
        else:
            extra = num_docs - len(gold_i) - len(more_i)
            others = [
                i for i in range(len(docs)) if i not in gold_i and i not in more_i
            ]
            chosen = gold_i + more_i + rng.sample(others, min(extra, len(others)))
        random.Random(4).shuffle(chosen)
        context = "\n\n".join(
            f"Document {i + 1}:\n{docs[d]}" for i, d in enumerate(chosen)
        )
        tmpl = TASKS["qa"]["template"]
        ans_prefix = TASKS["qa"]["answer_prefix"]
        prompt = tmpl.format(context=context, query=rec["query"]) + ans_prefix
        return prompt, list(rec["outputs"])

    return _g


_GENERATORS = {
    "cwe": gen_cwe,
    "fwe": gen_fwe,
    "vt": gen_vt,
    "niah": gen_niah,  # legacy alias (S1-style repeat single number)
    "qa_s": _gen_qa("qa_s"),
    "qa_h": _gen_qa("qa_h"),
}
for _v in _NIAH_VARIANTS:
    _GENERATORS[_v] = _gen_niah_variant(_v)

# NIAH haystack size (in the haystack-type's native unit) per target length:
#   repeat -> #noise sentences (~23 tok/sent), essay -> #words (~1.3 tok/word),
#   needle -> #distractor needles (~14 tok/needle). Tokenizer-agnostic; the A/B
#   feeds the SAME prompt to both arms so exact length is not load-bearing.
_NIAH_HAYSTACK_SIZE = {
    "4k": {"repeat": 180, "essay": 3000, "needle": 285},
    "8k": {"repeat": 430, "essay": 6000, "needle": 570},
    "16k": {"repeat": 860, "essay": 12000, "needle": 1140},
    "32k": {"repeat": 1720, "essay": 24000, "needle": 2280},
    "64k": {"repeat": 3440, "essay": 48000, "needle": 4560},
    "40k": {"repeat": 1760, "essay": 30000, "needle": 2850},
    "128k": {"repeat": 5640, "essay": 88000, "needle": 9100},
}

# Rough size knobs to target ~4k / ~8k context tokens (tokenizer-agnostic; the
# A/B only needs a long shared prefix, exact length is not load-bearing).
_SIZE_PRESETS = {
    "4k": {
        "cwe": dict(num_words=300),
        "fwe": dict(num_words=1500),
        "vt": dict(num_noises=175),
        "niah": dict(num_haystack=180),
        "qa_s": dict(num_docs=22),
        "qa_h": dict(num_docs=23),
    },
    "8k": {
        "cwe": dict(num_words=650),
        "fwe": dict(num_words=3200),
        "vt": dict(num_noises=420),
        "niah": dict(num_haystack=430),
        "qa_s": dict(num_docs=45),
        "qa_h": dict(num_docs=46),
    },
    "16k": {
        "cwe": dict(num_words=1300),
        "fwe": dict(num_words=6400),
        "vt": dict(num_noises=840),
        "niah": dict(num_haystack=860),
        "qa_s": dict(num_docs=90),
        "qa_h": dict(num_docs=92),
    },
    # long-context presets (linear extrapolation of the ~22.7 tok/sentence niah
    # rate: 180->4137 tok, 430->9762 tok; other tasks scaled by the same factor).
    "32k": {
        "cwe": dict(num_words=2600),
        "fwe": dict(num_words=12800),
        "vt": dict(num_noises=1680),
        "niah": dict(num_haystack=1720),
        "qa_s": dict(num_docs=180),
        "qa_h": dict(num_docs=184),
    },
    "64k": {
        "cwe": dict(num_words=5200),
        "fwe": dict(num_words=25600),
        "vt": dict(num_noises=3360),
        "niah": dict(num_haystack=3440),
        "qa_s": dict(num_docs=360),
        "qa_h": dict(num_docs=368),
    },
    "40k": {
        "cwe": dict(num_words=2900),
        "fwe": dict(num_words=14500),
        "vt": dict(num_noises=1700),
        "niah": dict(num_haystack=1760),
    },
    "128k": {
        "cwe": dict(num_words=9300),
        "fwe": dict(num_words=46500),
        "vt": dict(num_noises=5400),
        "niah": dict(num_haystack=5640),
        "qa_s": dict(num_docs=720),
        "qa_h": dict(num_docs=736),
    },
}


def _niah_size_kw(variant, length):
    """num_haystack for a NIAH variant at a target length (by haystack type).

    uuid needle/values (mk3) are ~2x longer per line than word-number needles,
    so halve the 'needle' haystack count for mk3 to keep the labeled length
    (else 128k mk3 overflows the context window).
    """
    cfg = _NIAH_VARIANTS[variant]
    htype = cfg["type_haystack"]
    n = _NIAH_HAYSTACK_SIZE[length][htype]
    if htype == "needle" and cfg["type_needle_v"] == "uuids":
        n = max(1, n // 2)
    return dict(num_haystack=n)


def build_ruler_samples(task, length, num_samples, base_seed=42):
    """Yield (prompt, gold_list) for `num_samples` distinct RULER instances."""
    gen = _GENERATORS[task]
    if task in _NIAH_VARIANTS:
        size_kw = _niah_size_kw(task, length)
    else:
        size_kw = _SIZE_PRESETS[length][task]
    out = []
    for i in range(num_samples):
        prompt, gold = gen(seed=base_seed + i, **size_kw)
        out.append((prompt, gold))
    return out


def score_recall(pred_text, gold_list):
    """RULER string_match_part: fraction of gold items appearing in the pred."""
    pl = str(pred_text).lower()
    if not gold_list:
        return 0.0
    hit = sum(1.0 for g in gold_list if str(g).lower() in pl)
    return hit / len(gold_list)


if __name__ == "__main__":
    # GPU-free self-test: oracle (feed gold back) must recall 1.0; measure sizes.
    for length in ("4k", "8k"):
        for task in ("cwe", "fwe", "vt", "niah"):
            samples = build_ruler_samples(task, length, 3)
            p, g = samples[0]
            chars = len(p)
            oracle = " ".join(str(x) for x in g)
            rec = score_recall(oracle, g)
            # distinctness across samples (different seeds -> different prompts)
            distinct = len({s[0] for s in samples}) == len(samples)
            print(
                f"[{length}/{task}] chars={chars:6d} ~tok={chars // 4:5d} "
                f"gold={len(g)} oracle_recall={rec:.2f} distinct={distinct}"
            )
            assert rec == 1.0, f"{task} oracle recall != 1.0"
            assert distinct, f"{task} samples not distinct"
    # NIAH 8-variant self-test (S/MK/MV/MQ) at 4k: oracle recall==1, gold arity
    # matches variant semantics, samples distinct.
    _want_gold = {
        "s1": 1,
        "s2": 1,
        "s3": 1,
        "mk1": 1,
        "mk2": 1,
        "mk3": 1,
        "mv": 4,
        "mq": 4,
    }
    for v in ("s1", "s2", "s3", "mk1", "mk2", "mk3", "mv", "mq"):
        samples = build_ruler_samples(v, "4k", 3)
        p, g = samples[0]
        rec = score_recall(" ".join(str(x) for x in g), g)
        distinct = len({s[0] for s in samples}) == len(samples)
        htype = _NIAH_VARIANTS[v]["type_haystack"]
        print(
            f"[4k/{v:3s}] hay={htype:6s} chars={len(p):6d} ~tok={len(p)//4:5d} "
            f"gold={len(g)} oracle_recall={rec:.2f} distinct={distinct}"
        )
        assert rec == 1.0, f"{v} oracle recall != 1.0"
        assert len(g) == _want_gold[v], f"{v} gold arity {len(g)} != {_want_gold[v]}"
        assert distinct, f"{v} samples not distinct"
    # QA-S (SQuAD) / QA-H (HotpotQA): oracle (gold answer fed back) recalls 1.0,
    # gold present in the built context (answerable), samples distinct, lengths
    # land near the label.
    for v in ("qa_s", "qa_h"):
        for length in ("4k", "8k", "16k"):
            samples = build_ruler_samples(v, length, 5)
            p, g = samples[0]
            rec = score_recall(" ".join(str(x) for x in g), g)
            in_ctx = any(str(x).lower() in p.lower() for x in g)
            distinct = len({s[0] for s in samples}) == len(samples)
            print(
                f"[{length}/{v:4s}] chars={len(p):7d} ~tok={len(p)//4:6d} "
                f"gold={g!s:.40} oracle_recall={rec:.2f} in_ctx={in_ctx} "
                f"distinct={distinct}"
            )
            assert rec == 1.0, f"{v} oracle recall != 1.0"
            assert in_ctx, f"{v} gold answer not present in built context"
            assert distinct, f"{v} samples not distinct"
    print("[ruler_gen self-test] ALL PASS")
