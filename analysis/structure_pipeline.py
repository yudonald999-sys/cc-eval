#!/usr/bin/env python3
"""
Data-agnostic item-level analysis pipeline: score uncertainty, task-family correlations,
dimensionality, judge substitution, SCO sensitivity and repeat-run stability.

Inputs (see INPUT_SPEC.md):
  --results ROOT      directory with ROOT/<model>/<item_id>.json, each {"item_id":..., "score": float in [0,1] or null}
  --items FILE        optional items.jsonl (item_id, task_type, sub_type, scoring.type, provenance.contamination_risk)
  --judge FILE        optional second-judge file {"judge1":name,"judge2":name,"pairs":[{"model","item_id",<j1key>,<j2key>}]}
  --j1key/--j2key     keys inside judge pairs (default k3 / j2)
  --judge1-family     comma list of evaluated-model names sharing the judge-1 family (self-preference check)
  --judge2-family     comma list of evaluated-model names sharing the judge-2 family
  --sco-metrics FILE  optional JSON {"SCO_correlations": {model: {"pearson_r":...}}} (accuracy-like SCO index)
  --repeat FILE       optional JSON {model: {"pairs": {item_id: {"gold":g,"old":a,"new":b}}}} (re-run of same items)
  --summary FILE      optional leaderboard JSON {model: {"overall":..,"KNO":..}} for cross-checking
  --exclude-models    comma list of model dirs to ignore (e.g. incomplete runs)
  --out DIR           output directory (JSON + markdown tables + figures)
Nothing here hard-codes any result; every reported number is written to out/results.json.
"""
import argparse, glob, json, math, os, re, sys
from collections import defaultdict
import numpy as np
from scipy import stats

FAMILIES_DEFAULT = ["KNO", "ERR", "SCO", "GEN", "CUL", "PED"]
RNG = np.random.default_rng(20261007)


# ---------------------------------------------------------------- loading
def load_items(path):
    meta = {}
    if path and os.path.isfile(path):
        for line in open(path, encoding="utf-8"):
            it = json.loads(line)
            sc = it.get("scoring", {})
            meta[it["item_id"]] = dict(
                task=it.get("task_type"), sub=it.get("sub_type"),
                scorer=sc.get("type") if isinstance(sc, dict) else sc,
                contamination=(it.get("provenance") or {}).get("contamination_risk"))
    return meta


def task_from_id(item_id):
    m = re.match(r"^[A-Z]+-([A-Z]+)-", item_id)
    return m.group(1) if m else None


def load_results(root, exclude, items_meta, families):
    scores = defaultdict(dict)  # model -> item -> score
    missing = defaultdict(int)
    for mdir in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(mdir):
            continue
        model = os.path.basename(mdir)
        if model in exclude:
            continue
        for f in glob.glob(os.path.join(mdir, "*.json")):
            d = json.load(open(f, encoding="utf-8"))
            iid = d.get("item_id") or os.path.basename(f)[:-5]
            task = (items_meta.get(iid) or {}).get("task") or task_from_id(iid)
            if task not in families:
                continue
            s = d.get("score")
            if s is None or d.get("error"):
                missing[model] += 1
                continue
            scores[model][iid] = float(s)
    return scores, dict(missing)


# ---------------------------------------------------------------- scoring
def family_table(scores, items_meta, families):
    models = sorted(scores)
    fam = {m: {} for m in models}
    n = {m: {} for m in models}
    micro = {}
    for m in models:
        allv = []
        for t in families:
            v = [s for i, s in scores[m].items() if ((items_meta.get(i) or {}).get("task") or task_from_id(i)) == t]
            fam[m][t] = 100 * np.mean(v) if v else np.nan
            n[m][t] = len(v)
            allv += v
        micro[m] = 100 * np.mean(allv)
    macro = {m: np.nanmean([fam[m][t] for t in families]) for m in models}
    return models, fam, n, micro, macro


def item_index(scores, items_meta, families):
    """full item list per family (union over models) + models x items matrices with NaN for missing scores"""
    models = sorted(scores)
    mats = {}
    for t in families:
        ids = set()
        for m in models:
            ids |= {i for i in scores[m] if ((items_meta.get(i) or {}).get("task") or task_from_id(i)) == t}
        ids = sorted(ids)
        mats[t] = (ids, np.array([[scores[m].get(i, np.nan) for i in ids] for m in models], float))
    return models, mats


def bootstrap_scores(models, mats, families, B=2000):
    """stratified item bootstrap (same resampled items for all models = paired); NaN-aware means"""
    k = len(models)
    fam_bs = {t: np.empty((B, k)) for t in families}
    micro_bs = np.empty((B, k)); macro_bs = np.empty((B, k))
    for b in range(B):
        sums = np.zeros(k); cnts = np.zeros(k)
        for t in families:
            M = mats[t][1]
            idx = RNG.integers(0, M.shape[1], M.shape[1])
            S = M[:, idx]
            fam_bs[t][b] = 100 * np.nanmean(S, axis=1)
            sums += np.nansum(S, axis=1); cnts += (~np.isnan(S)).sum(axis=1)
        micro_bs[b] = 100 * sums / cnts
        macro_bs[b] = np.mean([fam_bs[t][b] for t in families], axis=0)
    return fam_bs, micro_bs, macro_bs


def pairwise_diff(bs, models, point):
    """paired bootstrap CI of the difference between adjacent models in the point ranking"""
    order = sorted(range(len(models)), key=lambda j: -point[models[j]])
    out = []
    for a, b in zip(order[:-1], order[1:]):
        d = bs[:, a] - bs[:, b]
        out.append(dict(higher=models[a], lower=models[b], diff=float(point[models[a]] - point[models[b]]), ci=ci(d),
                        excludes_zero=bool(np.percentile(d, 2.5) > 0)))
    return out


def rank_summary(bs, models):
    ranks = np.argsort(np.argsort(-bs, axis=1), axis=1) + 1  # 1 = best
    out = {}
    for j, m in enumerate(models):
        r = ranks[:, j]
        out[m] = dict(median=float(np.median(r)), lo=int(np.percentile(r, 2.5)), hi=int(np.percentile(r, 97.5)),
                      p_top=float(np.mean(r == 1)))
    return out


def ci(a):
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


# ---------------------------------------------------------------- correlation / dimensionality
def corr_table(X, names):
    n = X.shape[0]
    out = []
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            x, y = X[:, a], X[:, b]
            r, p = stats.pearsonr(x, y)
            z = np.arctanh(np.clip(r, -0.999999, 0.999999)); se = 1 / math.sqrt(n - 3)
            rho, prho = stats.spearmanr(x, y)
            loo = [stats.pearsonr(np.delete(x, i), np.delete(y, i))[0] for i in range(n)]
            out.append(dict(a=names[a], b=names[b], r=float(r), p=float(p), loo_range=[float(min(loo)), float(max(loo))],
                            ci=[float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))],
                            rho=float(rho), p_rho=float(prho)))
    # Holm adjustment over all pairs
    ps = np.array([o["p"] for o in out]); order = np.argsort(ps); m = len(ps)
    adj = np.empty(m); run = 0
    for rank, i in enumerate(order):
        run = max(run, min(1, (m - rank) * ps[i])); adj[i] = run
    for o, a in zip(out, adj):
        o["p_holm"] = float(a)
    return out


def eig_share(R):
    ev = np.sort(np.linalg.eigvalsh(R))[::-1]
    return ev, ev / ev.sum()


def parallel_analysis(n, p, reps=5000, q=95, method="pearson"):
    sims = np.empty((reps, p))
    for r in range(reps):
        Z = RNG.standard_normal((n, p))
        R = np.corrcoef(Z, rowvar=False) if method == "pearson" else stats.spearmanr(Z).statistic
        sims[r] = np.sort(np.linalg.eigvalsh(R))[::-1]
    return np.percentile(sims, q, axis=0), sims.mean(axis=0)


def dimensionality(X, names, reps=5000):
    n, p = X.shape
    res = {}
    for method in ("pearson", "spearman"):
        R = np.corrcoef(X, rowvar=False) if method == "pearson" else stats.spearmanr(X).statistic
        ev, share = eig_share(R)
        thr95, thrmean = parallel_analysis(n, p, reps, 95, method)
        retained = 0
        for k in range(p):
            if ev[k] > thr95[k]:
                retained += 1
            else:
                break
        # PC1 loadings (sign-aligned so that most loadings positive)
        w, V = np.linalg.eigh(R); v1 = V[:, np.argmax(w)]
        if v1.sum() < 0: v1 = -v1
        load = v1 * math.sqrt(max(w))
        res[method] = dict(eigenvalues=ev.tolist(), share=share.tolist(), pa_95=thr95.tolist(), pa_mean=thrmean.tolist(),
                           retained_horn95=retained, kaiser_count=int((ev > 1).sum()),
                           pc1_loadings=dict(zip(names, load.tolist())),
                           offdiag_mean_r=float(R[np.triu_indices(p, 1)].mean()),
                           n_neg=int((R[np.triu_indices(p, 1)] < 0).sum()), n_pairs=int(p * (p - 1) / 2))
    # leave-one-model-out stability of PC1 share
    loo = []
    for i in range(n):
        Xi = np.delete(X, i, axis=0)
        loo.append(eig_share(np.corrcoef(Xi, rowvar=False))[1][0])
    res["loo_pc1_share_range"] = [float(min(loo)), float(max(loo))]
    # model-resampling bootstrap of PC1 share (descriptive; models are not a random sample)
    bs = []
    for _ in range(2000):
        idx = RNG.integers(0, n, n)
        Xb = X[idx]
        if np.any(Xb.std(axis=0) == 0): continue
        bs.append(eig_share(np.corrcoef(Xb, rowvar=False))[1][0])
    res["model_bootstrap_pc1_share_ci"] = ci(np.array(bs))
    return res


def family_reliability(fam_bs, X_point, families):
    """between-model reliability of each family score: 1 - mean sampling var / observed between-model var"""
    out = {}
    for j, t in enumerate(families):
        C = fam_bs[t] - fam_bs[t].mean(axis=1, keepdims=True)   # paired: remove shared item-sample shift
        samp_var = C.var(axis=0, ddof=1).mean()
        obs_var = X_point[:, j].var(ddof=1)
        out[t] = dict(sampling_var=float(samp_var), between_model_var=float(obs_var),
                      reliability=float(1 - samp_var / obs_var) if obs_var > 0 else None)
    return out


# ---------------------------------------------------------------- judges
def judge_analysis(path, j1key, j2key, scores, items_meta, families, fam1, fam2):
    J = json.load(open(path, encoding="utf-8"))
    pairs = J["pairs"]
    a = np.array([p[j1key] for p in pairs], float); b = np.array([p[j2key] for p in pairs], float)
    def agree(x, y):
        return dict(n=int(len(x)), pearson=float(stats.pearsonr(x, y)[0]), spearman=float(stats.spearmanr(x, y)[0]),
                    mad=float(np.mean(np.abs(x - y))), within_015=float(np.mean(np.abs(x - y) <= 0.15 + 1e-9)),
                    mean_j1=float(x.mean()), mean_j2=float(y.mean()))
    out = dict(judge1=J.get("judge1"), judge2=J.get("judge2"), overall=agree(a, b), by_task={})
    tasks = defaultdict(list)
    for k, p in enumerate(pairs):
        tasks[(items_meta.get(p["item_id"]) or {}).get("task") or task_from_id(p["item_id"])].append(k)
    for t, ks in tasks.items():
        out["by_task"][t] = agree(a[ks], b[ks])
    # QC: stored scores should equal judge-1 scores
    mism = sum(1 for p in pairs if p["model"] in scores and p["item_id"] in scores[p["model"]]
               and abs(scores[p["model"]][p["item_id"]] - p[j1key]) > 1e-9)
    out["qc_stored_vs_judge1_mismatch"] = int(mism)
    # judge-2 substituted scores
    s2 = {m: dict(v) for m, v in scores.items()}
    for p in pairs:
        if p["model"] in s2:
            s2[p["model"]][p["item_id"]] = float(p[j2key])
    # self-preference: (j1 - j2) on own-family outputs vs others, per judge
    def selfpref(fam, sign, task=None):
        sel = [p for p in pairs if task is None or ((items_meta.get(p["item_id"]) or {}).get("task") or task_from_id(p["item_id"])) == task]
        own = [sign * (p[j1key] - p[j2key]) for p in sel if p["model"] in fam]
        oth = [sign * (p[j1key] - p[j2key]) for p in sel if p["model"] not in fam]
        if not own or not oth: return None
        own, oth = np.array(own), np.array(oth)
        d = own.mean() - oth.mean()
        bs = [RNG.choice(own, len(own)).mean() - RNG.choice(oth, len(oth)).mean() for _ in range(5000)]
        return dict(own_mean_gap=float(own.mean()), other_mean_gap=float(oth.mean()), diff=float(d), ci=ci(np.array(bs)),
                    n_own=int(len(own)), n_other=int(len(oth)))
    out["self_preference_judge1"] = selfpref(set(fam1), +1)   # judge1 more lenient on own family?
    out["self_preference_judge2"] = selfpref(set(fam2), -1)   # judge2 more lenient on own family?
    out["self_preference_judge1_selfonly"] = selfpref({J.get("judge1_eval_name") or J.get("judge1")}, +1)
    out["self_preference_judge2_selfonly"] = selfpref({J.get("judge2_eval_name") or J.get("judge2")}, -1)
    out["self_preference_by_task"] = {}
    for t in sorted(tasks):
        out["self_preference_by_task"][t] = dict(
            judge1_family=selfpref(set(fam1), +1, t), judge2_family=selfpref(set(fam2), -1, t),
            judge1_self=selfpref({J.get("judge1_eval_name") or J.get("judge1")}, +1, t),
            judge2_self=selfpref({J.get("judge2_eval_name") or J.get("judge2")}, -1, t))
    out["within_015_strict_float"] = float(np.mean(np.abs(a - b) <= 0.15))
    out["within_015_strict_float_by_task"] = {t: float(np.mean(np.abs(a[ks] - b[ks]) <= 0.15)) for t, ks in tasks.items()}
    # per model x task mean scores under both judges
    pm = defaultdict(lambda: defaultdict(list))
    for k, p in enumerate(pairs):
        t = (items_meta.get(p["item_id"]) or {}).get("task") or task_from_id(p["item_id"])
        pm[p["model"]][t].append((a[k], b[k])); pm[p["model"]]["ALL"].append((a[k], b[k]))
    out["model_task_means"] = {m: {t: dict(n=len(v), j1=float(np.mean([x[0] for x in v])), j2=float(np.mean([x[1] for x in v])))
                                   for t, v in d.items()} for m, d in pm.items()}
    return out, s2


def ranks_of(d):
    ms = sorted(d, key=lambda m: -d[m]); return {m: i + 1 for i, m in enumerate(ms)}


# ---------------------------------------------------------------- repeat runs
def band_score(pred, gold):
    d = abs(pred - gold); return 1.0 if d <= 10 else (0.5 if d <= 20 else 0.0)


def repeat_analysis(path, sco_items=None):
    R = json.load(open(path, encoding="utf-8")); out = {}
    for m, v in R.items():
        prs = v.get("pairs", {})
        old = np.array([x["old"] for x in prs.values()], float); new = np.array([x["new"] for x in prs.values()], float)
        gold = np.array([x["gold"] for x in prs.values()], float)
        out[m] = dict(n=int(len(old)), retest_r=float(stats.pearsonr(old, new)[0]), mad=float(np.mean(np.abs(old - new))),
                      r_gold_run1=float(stats.pearsonr(old, gold)[0]), r_gold_run2=float(stats.pearsonr(new, gold)[0]),
                      band_run1=float(100 * np.mean([band_score(o, g) for o, g in zip(old, gold)])),
                      band_run2=float(100 * np.mean([band_score(o, g) for o, g in zip(new, gold)])),
                      band_item_agreement=float(np.mean([band_score(o, g) == band_score(w, g) for o, w, g in zip(old, new, gold)])),
                      exact_same_score=float(np.mean(old == new)))
        b1 = np.array([band_score(o, g) for o, g in zip(old, gold)]); b2 = np.array([band_score(w, g) for w, g in zip(new, gold)])
        bs = [100 * (b2[ix] - b1[ix]).mean() for ix in (RNG.integers(0, len(b1), len(b1)) for _ in range(5000))]
        out[m]["band_diff_run2_minus_run1_ci"] = ci(np.array(bs))
        out[m]["mean_pred_run1"] = float(old.mean()); out[m]["mean_pred_run2"] = float(new.mean())
        if sco_items and m in sco_items:
            match = [abs(sco_items[m][i]["pred"] - x["old"]) < 1e-9 for i, x in prs.items() if i in sco_items[m]]
            out[m]["run1_matches_v1_outputs"] = f"{sum(match)}/{len(match)}"
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True); ap.add_argument("--items"); ap.add_argument("--judge")
    ap.add_argument("--j1key", default="k3"); ap.add_argument("--j2key", default="j2")
    ap.add_argument("--judge1-family", default=""); ap.add_argument("--judge2-family", default="")
    ap.add_argument("--sco-metrics"); ap.add_argument("--repeat"); ap.add_argument("--summary")
    ap.add_argument("--exclude-models", default=""); ap.add_argument("--families", default=",".join(FAMILIES_DEFAULT))
    ap.add_argument("--B", type=int, default=2000); ap.add_argument("--pa-reps", type=int, default=5000)
    ap.add_argument("--sco-items"); ap.add_argument("--out", required=True)
    A = ap.parse_args(); os.makedirs(A.out, exist_ok=True)
    families = A.families.split(",")
    meta = load_items(A.items)
    scores, missing = load_results(A.results, set(filter(None, A.exclude_models.split(","))), meta, families)
    models, fam, n, micro, macro = family_table(scores, meta, families)
    res = dict(models=models, families=families, n_models=len(models), missing_scores=missing,
               family_pct=fam, family_n=n, micro=micro, macro=macro)
    if A.summary:
        S = json.load(open(A.summary)); diffs = {}
        for m in models:
            if m in S:
                diffs[m] = {t: round(fam[m][t] - S[m][t], 2) for t in families if t in S[m]}
                if "overall" in S[m]: diffs[m]["overall_vs_micro"] = round(micro[m] - S[m]["overall"], 2)
        res["crosscheck_vs_summary"] = diffs
    # item bootstrap
    _, mats = item_index(scores, meta, families)
    res["common_items_per_family"] = {t: len(mats[t][0]) for t in families}
    fam_bs, micro_bs, macro_bs = bootstrap_scores(models, mats, families, A.B)
    res["ci_family"] = {m: {t: ci(fam_bs[t][:, j]) for t in families} for j, m in enumerate(models)}
    res["ci_micro"] = {m: ci(micro_bs[:, j]) for j, m in enumerate(models)}
    res["ci_macro"] = {m: ci(macro_bs[:, j]) for j, m in enumerate(models)}
    res["rank_micro"] = rank_summary(micro_bs, models); res["rank_macro"] = rank_summary(macro_bs, models)
    res["adjacent_diff_micro"] = pairwise_diff(micro_bs, models, micro)
    res["adjacent_diff_macro"] = pairwise_diff(macro_bs, models, macro)
    res["kendall_micro_vs_macro"] = float(stats.kendalltau([ranks_of(micro)[m] for m in models], [ranks_of(macro)[m] for m in models])[0])
    # auto-scored items only (no LLM judge): families with >= 1 auto item
    auto = {m: {i: v for i, v in sc.items() if (meta.get(i) or {}).get("scorer") != "llm_rubric"} for m, sc in scores.items()}
    fams_auto = [t for t in families if any(((meta.get(i) or {}).get("task") == t) for i in auto[models[0]])]
    _, famA, nA, microA, macroA = family_table(auto, meta, fams_auto)
    XA = np.array([[famA[m][t] for t in fams_auto] for m in models])
    res["auto_only"] = dict(families=fams_auto, family_pct=famA, family_n={m: nA[m] for m in models[:1]}, micro=microA, macro=macroA,
                            correlations=corr_table(XA, fams_auto), dimensionality=dimensionality(XA, fams_auto, A.pa_reps))
    # correlation / dimensionality
    X = np.array([[fam[m][t] for t in families] for m in models])
    res["correlations"] = corr_table(X, families)
    res["dimensionality_all"] = dimensionality(X, families, A.pa_reps)
    res["family_reliability"] = family_reliability(fam_bs, X, families)
    if "SCO" in families:
        keep = [t for t in families if t != "SCO"]; Xs = X[:, [families.index(t) for t in keep]]
        res["correlations_noSCO"] = corr_table(Xs, keep)
        res["dimensionality_noSCO"] = dimensionality(Xs, keep, A.pa_reps)
        if A.sco_items:
            SI = json.load(open(A.sco_items)); sr = {}
            for m in models:
                pr = np.array([v["pred"] for v in SI[m].values()]); gd = np.array([v["gold"] for v in SI[m].values()])
                sr[m] = dict(n=int(len(pr)), pearson_r=float(stats.pearsonr(pr, gd)[0]), spearman_rho=float(stats.spearmanr(pr, gd)[0]),
                             pred_mean=float(pr.mean()), gold_mean=float(gd.mean()), pred_sd=float(pr.std()), mean_bias=float((pr - gd).mean()))
            res["sco_recomputed"] = sr
            Xr = X.copy(); Xr[:, families.index("SCO")] = [100 * sr[m]["pearson_r"] for m in models]
            res["correlations_SCO_as_r"] = corr_table(Xr, families)
            res["dimensionality_SCO_as_r"] = dimensionality(Xr, families, A.pa_reps)
            fa = res["auto_only"]["families"]; XAr = np.array([[res["auto_only"]["family_pct"][m][t] if t != "SCO" else 100 * sr[m]["pearson_r"] for t in fa] for m in models])
            res["auto_only_SCO_as_r"] = dict(correlations=corr_table(XAr, fa), dimensionality=dimensionality(XAr, fa, A.pa_reps))
            sb = X[:, families.index("SCO")]; ab = np.array([abs(sr[m]["mean_bias"]) for m in models])
            res["sco_band_vs_abs_bias"] = dict(zip(["r", "p"], map(float, stats.pearsonr(sb, ab))))
            res["sco_proximity_vs_r"] = dict(zip(["r", "p"], map(float, stats.pearsonr(X[:, families.index("SCO")], [sr[m]["pearson_r"] for m in models]))))
        if A.sco_metrics:
            SM = json.load(open(A.sco_metrics))["SCO_correlations"]
            res["sco_metrics_user"] = {m: SM.get(m) for m in models}
        if False:
            if all(m in SM for m in models):
                Xr = X.copy(); Xr[:, families.index("SCO")] = [100 * SM[m]["pearson_r"] for m in models]
                res["correlations_SCO_as_r"] = corr_table(Xr, families)
                res["dimensionality_SCO_as_r"] = dimensionality(Xr, families, A.pa_reps)
                fa = res["auto_only"]["families"]; XAr = np.array([[res["auto_only"]["family_pct"][m][t] if t != "SCO" else 100 * sr[m]["pearson_r"] for t in fa] for m in models])
            res["auto_only_SCO_as_r"] = dict(correlations=corr_table(XAr, fa), dimensionality=dimensionality(XAr, fa, A.pa_reps))
            res["sco_proximity_vs_r"] = dict(zip(["r", "p"], map(float, stats.pearsonr(
                    X[:, families.index("SCO")], [SM[m]["pearson_r"] for m in models]))))
    # judges
    if A.judge:
        ja, s2 = judge_analysis(A.judge, A.j1key, A.j2key, scores, meta, families,
                                [x for x in A.judge1_family.split(",") if x], [x for x in A.judge2_family.split(",") if x])
        _, fam2, _, micro2, macro2 = family_table(s2, meta, families)
        ja["family_pct_judge2"] = fam2; ja["micro_judge2"] = micro2; ja["macro_judge2"] = macro2
        r1m, r2m = ranks_of(micro), ranks_of(micro2); r1M, r2M = ranks_of(macro), ranks_of(macro2)
        ja["ranks"] = {m: dict(micro_j1=r1m[m], micro_j2=r2m[m], macro_j1=r1M[m], macro_j2=r2M[m]) for m in models}
        ja["kendall_tau_micro"] = float(stats.kendalltau([r1m[m] for m in models], [r2m[m] for m in models])[0])
        ja["kendall_tau_macro"] = float(stats.kendalltau([r1M[m] for m in models], [r2M[m] for m in models])[0])
        ja["top_micro"] = [min(r1m, key=r1m.get), min(r2m, key=r2m.get)]
        ja["top_macro"] = [min(r1M, key=r1M.get), min(r2M, key=r2M.get)]
        X2 = np.array([[fam2[m][t] for t in families] for m in models])
        ja["correlations_judge2"] = corr_table(X2, families)
        ja["dimensionality_judge2"] = dimensionality(X2, families, A.pa_reps)
        ja["family_delta_j2_minus_j1"] = {m: {t: float(fam2[m][t] - fam[m][t]) for t in families} for m in models}
        res["judges"] = ja
    if A.repeat:
        res["repeat_runs"] = repeat_analysis(A.repeat, json.load(open(A.sco_items)) if A.sco_items else None)
    json.dump(res, open(os.path.join(A.out, "results.json"), "w"), indent=1, ensure_ascii=False, default=float)
    write_tables(res, A.out); make_figures(res, X, A.out)
    print("wrote", A.out)


def write_tables(res, out):
    fams = res["families"]; L = []
    L.append("| Model | Micro [95% CI] | Macro [95% CI] | " + " | ".join(fams) + " |")
    L.append("|" + "---|" * (3 + len(fams)))
    for m in sorted(res["models"], key=lambda m: -res["micro"][m]):
        c1, c2 = res["ci_micro"][m], res["ci_macro"][m]
        cells = [f"{res['family_pct'][m][t]:.1f} [{res['ci_family'][m][t][0]:.1f}, {res['ci_family'][m][t][1]:.1f}]" for t in fams]
        L.append(f"| {m} | {res['micro'][m]:.1f} [{c1[0]:.1f}, {c1[1]:.1f}] | {res['macro'][m]:.1f} [{c2[0]:.1f}, {c2[1]:.1f}] | " + " | ".join(cells) + " |")
    L.append("\n| Pair | r [95% CI] | p | p (Holm) | rho | p (rho) |\n|---|---|---|---|---|---|")
    for c in res["correlations"]:
        L.append(f"| {c['a']}-{c['b']} | {c['r']:.3f} [{c['ci'][0]:.2f}, {c['ci'][1]:.2f}] | {c['p']:.3f} | {c['p_holm']:.3f} | {c['rho']:.3f} | {c['p_rho']:.3f} |")
    open(os.path.join(out, "tables.md"), "w").write("\n".join(L) + "\n")


def make_figures(res, X, out):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception:
        return
    d = res["dimensionality_all"]["pearson"]; p = len(d["eigenvalues"])
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(range(1, p + 1), d["eigenvalues"], "o-", label="Observed")
    ax.plot(range(1, p + 1), d["pa_95"], "s--", label="Parallel analysis (95th pct.)")
    ax.axhline(1, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("Component"); ax.set_ylabel("Eigenvalue"); ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_scree_pa.png"), dpi=300); plt.close(fig)
    ms = sorted(res["models"], key=lambda m: -res["micro"][m])
    fig, ax = plt.subplots(figsize=(5.5, 0.35 * len(ms) + 1))
    for i, m in enumerate(ms):
        r = res["rank_micro"][m]; ax.plot([r["lo"], r["hi"]], [i, i], "k-"); ax.plot(r["median"], i, "ko")
    ax.set_yticks(range(len(ms))); ax.set_yticklabels(ms); ax.invert_yaxis(); ax.set_xlabel("Rank (item bootstrap, 95% interval)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_rank_ci.png"), dpi=300); plt.close(fig)


if __name__ == "__main__":
    main()
