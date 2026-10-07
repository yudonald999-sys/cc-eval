#!/usr/bin/env python3
"""Open-source models on the discriminative subset (KNO + PHO), log-likelihood multiple choice.

Eight locally run models:
  * six models via lm-evaluation-harness 0.4.13 (float32, CPU): results/harness/<model>/..__models__<model>/results_*.json
    (aggregate accuracy per shard only; no per-item records were saved). Shard selection and "latest run per shard"
    follow harness/summarize_harness.py.
  * two models via llama.cpp GGUF Q8_0 + harness/gguf_score.py: results/harness/<model>/<task>.json (per-item preds).
For every model x task: n, correct, accuracy, Wilson 95% CI, uniform-guessing baseline, constant-answer baseline,
exact tests against both. GGUF runs additionally get prediction-preference decomposition and balanced-set checks.
Kept separate from the 11-model six-family analysis (different protocol and item coverage).
Writes analysis/out/open_source_models.json and analysis/out/open_source_models.md.
"""
import json, os, math, collections
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H = os.path.join(ROOT, "results", "harness"); DATA = os.path.join(ROOT, "harness", "data")
OUT = os.path.join(ROOT, "analysis", "out"); os.makedirs(OUT, exist_ok=True)
L = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

LMEVAL = ["Qwen2.5-0.5B", "Qwen2.5-0.5B-Instruct", "Qwen2.5-1.5B-Instruct", "Qwen2.5-3B-Instruct",
          "Llama-3.2-1B-Instruct", "Llama-3.2-3B-Instruct"]
GGUF = ["Qwen2.5-7B-q8_0", "Llama-3.1-8B-q8_0"]
INFO = {
    "Qwen2.5-0.5B": ("Qwen2.5-0.5B (base)", 0.5, "lm-eval 0.4.13, float32"),
    "Qwen2.5-0.5B-Instruct": ("Qwen2.5-0.5B-Instruct", 0.5, "lm-eval 0.4.13, float32"),
    "Qwen2.5-1.5B-Instruct": ("Qwen2.5-1.5B-Instruct", 1.5, "lm-eval 0.4.13, float32"),
    "Qwen2.5-3B-Instruct": ("Qwen2.5-3B-Instruct", 3, "lm-eval 0.4.13, float32"),
    "Llama-3.2-1B-Instruct": ("Llama-3.2-1B-Instruct", 1, "lm-eval 0.4.13, float32"),
    "Llama-3.2-3B-Instruct": ("Llama-3.2-3B-Instruct", 3, "lm-eval 0.4.13, float32"),
    "Qwen2.5-7B-q8_0": ("Qwen2.5-7B-Instruct (Q8_0)", 7, "llama.cpp GGUF Q8_0 + gguf_score.py"),
    "Llama-3.1-8B-q8_0": ("Llama-3.1-8B-Instruct (Q8_0)", 8, "llama.cpp GGUF Q8_0 + gguf_score.py"),
}
TASKS = ["cceval_kno", "cceval_pho_legal", "cceval_pho_level", "cceval_pho_poly"]
# shard sets used for scoring (identical to harness/summarize_harness.py)
PARTS = {
    "cceval_kno": {
        "Qwen2.5-0.5B": [f"cceval_kno_p{i}" for i in range(1, 5)],
        "Qwen2.5-0.5B-Instruct": [f"cceval_kno_p{i}" for i in range(1, 5)],
        "Qwen2.5-1.5B-Instruct": ["cceval_kno_p1", "cceval_kno_p2"] + [f"cceval_kno_s{i}" for i in range(5, 9)],
        "Qwen2.5-3B-Instruct": [f"cceval_kno_u{i}" for i in range(1, 35)],
        "Llama-3.2-1B-Instruct": [f"cceval_kno_s{i}" for i in range(1, 9)],
        "Llama-3.2-3B-Instruct": [f"cceval_kno_u{i}" for i in range(1, 35)],
    },
    "cceval_pho_legal": {m: ["cceval_pho_legal_a", "cceval_pho_legal_b"] for m in ["Qwen2.5-3B-Instruct", "Llama-3.2-3B-Instruct"]},
    "cceval_pho_level": {m: ["cceval_pho_level_a", "cceval_pho_level_b"] for m in ["Qwen2.5-3B-Instruct", "Llama-3.2-3B-Instruct"]},
    "cceval_pho_poly": {},
}
GGUF_FILE = {"cceval_kno": "kno", "cceval_pho_legal": "pho_legal", "cceval_pho_level": "pho_level", "cceval_pho_poly": "pho_poly"}
BAL = {"cceval_kno": ("kno_bal", "cceval_kno_bal"), "cceval_pho_level": ("pho_level_bal", "cceval_pho_level_bal"),
       "cceval_pho_poly": ("pho_poly_bal", "cceval_pho_poly_bal")}


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [100 * (c - h), 100 * (c + h)]


def poisson_binomial_p(k, probs):
    """Exact two-sided p for k successes when item i is guessed correctly with probability probs[i]."""
    pmf = np.zeros(len(probs) + 1); pmf[0] = 1.0
    for p in probs:
        pmf[1:] = pmf[1:] * (1 - p) + pmf[:-1] * p; pmf[0] *= (1 - p)
    return float(min(1.0, pmf[pmf <= pmf[k] * (1 + 1e-9)].sum()))


def label(choice):
    return choice.replace("答案：", "")


def task_meta(task):
    rows = L(os.path.join(DATA, task + ".jsonl"))
    probs = [1 / len(r["choices"]) for r in rows]
    pos = collections.Counter(r["gold"] for r in rows)
    lab = collections.Counter(label(r["choices"][r["gold"]]) for r in rows)
    fixed = len({tuple(r["choices"]) for r in rows}) == 1
    # constant answer: the same option text for every item when options are fixed (KNO, level, legality);
    # for polyphone items the options differ per item, so the constant answer is the most frequent gold position
    if fixed:
        const_label, const_n = lab.most_common(1)[0]; const = dict(kind="same answer text", answer=const_label, acc=100 * const_n / len(rows))
    else:
        p, c = pos.most_common(1)[0]; const = dict(kind="same option position", answer=f"position {p + 1}", acc=100 * c / len(rows))
    return dict(rows={r["item_id"]: r for r in rows}, n=len(rows), probs=probs, chance=100 * float(np.mean(probs)),
                constant=const, fixed_options=fixed, gold_label_dist=dict(lab), gold_position_dist={str(k): v for k, v in sorted(pos.items())})


META = {t: task_meta(t) for t in TASKS + ["cceval_kno_bal", "cceval_pho_level_bal", "cceval_pho_poly_bal", "cceval_kno_v13b"]}


def summarize(k, n, meta):
    return dict(n=n, correct=k, acc=100 * k / n, wilson=wilson(k, n), chance=meta["chance"], constant_answer=meta["constant"]["acc"],
                p_vs_chance=poisson_binomial_p(k, meta["probs"]),
                p_vs_constant=float(stats.binomtest(k, n, meta["constant"]["acc"] / 100).pvalue))


def lmeval_runs(model):
    runs = collections.defaultdict(list)
    for root, _d, files in os.walk(os.path.join(H, model)):
        for fn in files:
            if fn.startswith("results_") and fn.endswith(".json"):
                r = json.load(open(os.path.join(root, fn), encoding="utf-8")); ts = fn[8:-5]
                for t, v in r["results"].items():
                    runs[t].append((ts, v["acc,none"], r["n-samples"][t]["effective"]))
    return {t: sorted(v) for t, v in runs.items()}


res = dict(protocol="log-likelihood multiple choice, 0-shot; each option prefixed with '答案：'; prediction = option with the highest summed log-probability",
           tasks={t: {k: v for k, v in META[t].items() if k not in ("rows", "probs")} for t in META}, models={}, repeat_runs={}, gguf_detail={})
for m in LMEVAL:
    runs = lmeval_runs(m); res["models"][m] = dict(label=INFO[m][0], params_b=INFO[m][1], runtime=INFO[m][2], tasks={})
    for t in TASKS:
        parts = PARTS[t].get(m, [t]); k = n = 0
        for sh in parts:
            ts, acc, nn = runs[sh][-1]; k += int(round(acc * nn)); n += nn
        assert n == META[t]["n"], (m, t, n)
        s = summarize(k, n, META[t]); s["source"] = dict(shards=parts, run="latest run of each shard"); res["models"][m]["tasks"][t] = s
        if t == "cceval_pho_legal" and len(parts) == 2:  # split by gold label: legal items (_a) / illegal items (_b)
            s["acc_by_gold"] = {"合法": 100 * runs[parts[0]][-1][1], "非法": 100 * runs[parts[1]][-1][1]}
    for sh, v in runs.items():
        if len(v) > 1:
            res["repeat_runs"].setdefault(m, {})[sh] = [dict(timestamp=a, acc=b, n=c) for a, b, c in v]
    # complete-set accuracy from the earlier pass where a shard was run twice
    for t in TASKS:
        parts = PARTS[t].get(m, [t])
        if all(len(runs[sh]) > 1 for sh in parts):
            k = sum(int(round(runs[sh][0][1] * runs[sh][0][2])) for sh in parts)
            res["models"][m]["tasks"][t]["earlier_pass_acc"] = 100 * k / META[t]["n"]

for m in GGUF:
    res["models"][m] = dict(label=INFO[m][0], params_b=INFO[m][1], runtime=INFO[m][2], tasks={}); det = {}
    def load(run, dataname):
        d = json.load(open(os.path.join(H, m, run + ".json"), encoding="utf-8")); rows = META[dataname]["rows"]
        preds = {p["item_id"]: label(rows[p["item_id"]]["choices"][p["pred"]]) for p in d["preds"] if p.get("pred") is not None}
        corr = {p["item_id"]: bool(p["correct"]) for p in d["preds"] if p.get("pred") is not None}
        assert all(corr[i] == (rows[i]["gold"] == next(p["pred"] for p in d["preds"] if p["item_id"] == i)) for i in list(corr)[:50])
        return d, preds, corr
    for t in TASKS + ["cceval_kno_v13b"]:
        run = GGUF_FILE.get(t, "kno_v13b"); d, preds, corr = load(run, t)
        k, n = sum(corr.values()), len(corr)
        s = summarize(k, n, dict(META[t], probs=[1 / len(META[t]["rows"][i]["choices"]) for i in corr]))
        s["n_items_in_set"] = META[t]["n"]; s["file_acc"] = float(d["acc"]); s["source"] = dict(file=f"results/harness/{m}/{run}.json")
        top = collections.Counter(preds.values()).most_common(1)[0]; other = [corr[i] for i in corr if preds[i] != top[0]]
        s.update(dominant_pred=top[0], dominant_share=100 * top[1] / n, n_not_dominant=len(other),
                 acc_when_not_dominant=(100 * sum(other) / len(other)) if other else None)
        if t == "cceval_pho_legal":
            g = {i: label(META[t]["rows"][i]["choices"][META[t]["rows"][i]["gold"]]) for i in corr}
            s["acc_by_gold"] = {lab: 100 * np.mean([corr[i] for i in corr if g[i] == lab]) for lab in sorted(set(g.values()))}
        res["models"][m]["tasks"][t] = s
        if t in BAL:
            run_b, data_b = BAL[t]; db, pb, cb = load(run_b, data_b); kb = sum(cb.values())
            det[t] = dict(balanced_acc=100 * kb / len(cb), balanced_n=len(cb), balanced_wilson=wilson(kb, len(cb)),
                          same_predicted_text=f"{sum(preds[i] == pb[i] for i in preds if i in pb)}/{len(preds)}",
                          same_correctness=f"{sum(corr[i] == cb[i] for i in corr if i in cb)}/{len(corr)}")
    res["gguf_detail"][m] = det

# compact markdown table
lines = ["| Model | Runtime | KNO (400) | PHO legality (60) | PHO syllable level (40) | PHO polyphone (30) |", "|---|---|---|---|---|---|"]
f = lambda s: f"{s['acc']:.1f} [{s['wilson'][0]:.1f}, {s['wilson'][1]:.1f}]"
lines.append("| Uniform guessing | – | " + " | ".join(f"{META[t]['chance']:.1f}" for t in TASKS) + " |")
lines.append("| Constant answer | – | " + " | ".join(f"{META[t]['constant']['acc']:.1f} ({META[t]['constant']['answer']})" for t in TASKS) + " |")
for m in LMEVAL + GGUF:
    lines.append(f"| {INFO[m][0]} | {INFO[m][2]} | " + " | ".join(f(res["models"][m]["tasks"][t]) for t in TASKS) + " |")
open(os.path.join(OUT, "open_source_models.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
json.dump(res, open(os.path.join(OUT, "open_source_models.json"), "w"), ensure_ascii=False, indent=1)
print("\n".join(lines))
