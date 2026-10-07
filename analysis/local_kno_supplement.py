#!/usr/bin/env python3
"""Supplementary analysis: local open-source models on the KNO task (harness, log-likelihood multiple choice).
Kept separate from the 11-model six-family analysis (different response format, KNO only)."""
import json, glob, os, collections, math
import numpy as np
from scipy import stats
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ND = ROOT; H = f"{ND}/results/harness"; OUT = os.path.join(ROOT, "analysis", "out"); os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(7)
L = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [100 * (c - h), 100 * (c + h)]


def binom(k, n, p0):
    return float(stats.binomtest(k, n, p0, alternative="two-sided").pvalue)


items = {i["item_id"]: i for i in L(f"{ND}/items/v1.0/items.jsonl") if i["task_type"] == "KNO"}
kno = {h["item_id"]: h for h in L(f"{ND}/harness/data/cceval_kno.jsonl")}
bal = {h["item_id"]: h for h in L(f"{ND}/harness/data/cceval_kno_bal.jsonl")}
v13b = {h["item_id"]: h for h in L(f"{ND}/harness/data/cceval_kno_v13b.jsonl")}
gold_txt = {i: kno[i]["choices"][kno[i]["gold"]].replace("答案：", "") for i in kno}
res = {}
dist = collections.Counter(gold_txt.values())
res["kno400_gold_distribution"] = dict(dist); res["chance"] = 100 / 7; res["majority_label"] = dist.most_common(1)[0][0]
res["majority_baseline"] = 100 * dist.most_common(1)[0][1] / len(gold_txt)

# ---- local models: 7B/8B gguf runs (per-item preds)
local = {}
for m in ["Qwen2.5-7B-q8_0", "Llama-3.1-8B-q8_0"]:
    for run, data in [("kno", kno), ("kno_bal", bal), ("kno_v13b", v13b)]:
        d = json.load(open(f"{H}/{m}/{run}.json", encoding="utf-8"))
        preds = {p["item_id"]: data[p["item_id"]]["choices"][p["pred"]].replace("答案：", "") for p in d["preds"]}
        corr = {p["item_id"]: bool(p["correct"]) for p in d["preds"]}
        local[(m, run)] = (preds, corr)
        k, n = sum(corr.values()), len(corr)
        g = {i: data[i]["choices"][data[i]["gold"]].replace("答案：", "") for i in corr}
        top = collections.Counter(preds.values()).most_common(1)[0]
        other = [corr[i] for i in corr if preds[i] != top[0]]
        res.setdefault("local_gguf", {})[f"{m}|{run}"] = dict(
            n=n, correct=k, acc=100 * k / n, wilson=wilson(k, n), p_vs_chance=binom(k, n, 1 / 7),
            majority_baseline=100 * collections.Counter(g.values()).most_common(1)[0][1] / n,
            p_vs_majority=binom(k, n, collections.Counter(g.values()).most_common(1)[0][1] / n),
            dominant_pred=top[0], dominant_share=100 * top[1] / n,
            acc_when_not_dominant=(100 * sum(other) / len(other)) if other else None, n_not_dominant=len(other),
            file_acc=d.get("acc"))
for m in ["Qwen2.5-7B-q8_0", "Llama-3.1-8B-q8_0"]:
    a, b = local[(m, "kno")][0], local[(m, "kno_bal")][0]
    res.setdefault("position_balance_check", {})[m] = dict(
        same_predicted_label=f"{sum(a[i] == b[i] for i in a)}/{len(a)}",
        same_correctness=f"{sum(local[(m,'kno')][1][i] == local[(m,'kno_bal')][1][i] for i in a)}/{len(a)}")

# ---- small models via lm-eval shard results: use the latest run of each shard, one complete shard scheme
def shard_ids(name):
    return [h["item_id"] for h in L(f"{ND}/harness/data/{name}.jsonl")]
small = {}
for md in sorted(glob.glob(f"{H}/*/..__models__*")):
    model = md.split("/")[-2]
    if model in ("smoke", "wm-check"): continue
    latest = {}
    for f in sorted(glob.glob(md + "/results_*.json")):
        d = json.load(open(f))
        for t, v in d["results"].items():
            if t.startswith("cceval_kno_"):
                latest[t] = (v["acc,none"], d["n-samples"][t]["effective"])
    # choose shards: prefer combination covering 400 unique items without overlap
    best = None
    for scheme in (["p"], ["s"], ["u"], ["p", "s"], ["t", "u"]):
        sel = [t for t in latest if t.split("_")[-1][0] in scheme]
        ids = [i for t in sel for i in shard_ids(t)]
        if len(ids) == 400 and len(set(ids)) == 400:
            best = sel; break
    if best is None:
        # greedy: try p1,p2 + s5..s8
        sel = [t for t in latest if t in ("cceval_kno_p1", "cceval_kno_p2", "cceval_kno_s5", "cceval_kno_s6", "cceval_kno_s7", "cceval_kno_s8")]
        ids = [i for t in sel for i in shard_ids(t)]
        if len(ids) == 400 and len(set(ids)) == 400: best = sel
    if best is None:
        small[model] = dict(error="no complete non-overlapping shard set", shards=sorted(latest)); continue
    k = int(round(sum(latest[t][0] * latest[t][1] for t in best))); n = sum(latest[t][1] for t in best)
    small[model] = dict(n=n, correct=k, acc=100 * k / n, wilson=wilson(k, n), p_vs_chance=binom(k, n, 1 / 7), shards=sorted(best))
res["local_lmeval_kno400"] = small
# harness reproducibility: fast scorer vs lm-eval on 0.5B p1; repeated lm-eval runs
v = json.load(open(f"{H}/validate/0.5B_kno_p1_fast.json"))
res["fast_scorer_vs_lmeval_0.5B_p1"] = dict(fast_acc=float(v["acc"]), lmeval_acc_latest=0.13, wm_check_acc=0.13)

# ---- API models on the same 400 KNO items (generative, exact match)
api = {}
R = os.path.join(ROOT, "analysis", "input", "results")
for m in sorted(os.listdir(R)):
    sc = {i: json.load(open(f"{R}/{m}/{i}.json"))["score"] for i in kno}
    k = int(sum(sc.values())); n = len(sc)
    g79 = [sc[i] for i in kno if gold_txt[i] == "7-9"]; gn = [sc[i] for i in kno if gold_txt[i] != "7-9"]
    api[m] = dict(n=n, correct=k, acc=100 * k / n, wilson=wilson(k, n), p_vs_chance=binom(k, n, 1 / 7),
                  p_vs_majority=binom(k, n, res["majority_baseline"] / 100),
                  acc_on_7_9_items=100 * np.mean(g79), acc_on_other_items=100 * np.mean(gn), scores=sc)
res["api_kno400"] = {m: {kk: vv for kk, vv in v.items() if kk != "scores"} for m, v in api.items()}
res["api_below_majority_count"] = sum(v["acc"] < res["majority_baseline"] for v in api.values())
res["api_sig_below_majority_count"] = sum((v["acc"] < res["majority_baseline"]) and v["p_vs_majority"] < 0.05 for v in api.values())
# item-level: API mean correctness per item vs local correctness
ids = sorted(kno)
api_mean = np.array([np.mean([api[m]["scores"][i] for m in api]) for i in ids])
res["api_item_mean_distribution"] = dict(n_items=len(ids), all_wrong=int((api_mean == 0).sum()), all_right=int((api_mean == 1).sum()),
                                         mean=float(api_mean.mean()))
for m in ["Qwen2.5-7B-q8_0", "Llama-3.1-8B-q8_0"]:
    c = np.array([local[(m, "kno")][1][i] for i in ids], float)
    r, p = stats.pointbiserialr(c, api_mean)
    res.setdefault("item_level_local_vs_api", {})[m] = dict(point_biserial=float(r), p=float(p),
        api_mean_on_local_correct=float(api_mean[c == 1].mean()), api_mean_on_local_wrong=float(api_mean[c == 0].mean()))
# by subtype
sub = collections.defaultdict(list)
for i in ids: sub[items[i]["sub_type"]].append(i)
res["by_subtype"] = {s: dict(n=len(v), api_mean_acc=100 * float(np.mean([api_mean[ids.index(i)] for i in v])),
                             **{m: 100 * float(np.mean([local[(m, 'kno')][1][i] for i in v])) for m in ["Qwen2.5-7B-q8_0", "Llama-3.1-8B-q8_0"]})
                     for s, v in sub.items()}
res["v13b_gold_distribution"] = dict(collections.Counter(h["choices"][h["gold"]].replace("答案：", "") for h in v13b.values()))
res["v13b_subtypes"] = dict(collections.Counter(h["sub_type"] for h in v13b.values()))
res["v13b_overlap_with_v1.0"] = sum(1 for i in v13b if i in items)
json.dump(res, open(f"{OUT}/local_kno_supplement.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in res.items() if k not in ("api_kno400",)}, indent=1, ensure_ascii=False)[:6000])
print({m: (round(v["acc"], 1), [round(x, 1) for x in v["wilson"]], round(v["acc_on_7_9_items"], 1), round(v["acc_on_other_items"], 1), round(v["p_vs_majority"], 4)) for m, v in res["api_kno400"].items()})
