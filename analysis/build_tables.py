import json, os
A = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(A, "out", "results.json")))
K = json.load(open(os.path.join(A, "out", "local_kno_supplement.json")))
NAME = {"deepseek": "deepseek-chat", "tchub-dsv4f": "deepseek-v4-flash", "hunyuan": "hy3", "doubao": "doubao-seed-1-6",
        "ernie": "ernie-4.5-turbo", "grok": "grok-3-mini-fast", "qwen": "qwen-flash", "stepfun": "step-3.7-flash",
        "zhipu": "glm-4-flash", "k3-agent": "k3-agent", "k2d6-agent": "k2d6-agent"}
fams = R["families"]; models = sorted(R["models"], key=lambda m: -R["micro"][m])
def pv(p):
    return "< .001" if p < .001 else f"{p:.3f}".replace("0.", ".", 1) if p < 1 else "1.000"
def r3(x): return f"{x:.3f}".replace("0.", ".", 1) if x >= 0 else f"−{abs(x):.3f}".replace("0.", ".", 1)
def r2(x): return f"{x:.2f}".replace("0.", ".", 1) if x >= 0 else f"−{abs(x):.2f}".replace("0.", ".", 1)
def sg(x, d=1):
    v = f"{x:+.{d}f}"; return v.replace("-", "−")
def c1(a): return f"[{a[0]:.1f}, {a[1]:.1f}]"
T = {}
# Table 2
L = ["| Model | Micro | Macro | " + " | ".join(fams) + " |", "|" + "---|" * (3 + len(fams))]
for m in models:
    L.append(f"| {NAME[m]} | {R['micro'][m]:.1f} {c1(R['ci_micro'][m])} | {R['macro'][m]:.1f} {c1(R['ci_macro'][m])} | " +
             " | ".join(f"{R['family_pct'][m][t]:.1f} {c1(R['ci_family'][m][t])}" for t in fams) + " |")
T["TABLE2"] = "\n".join(L)
# Table 3
L = ["| Pair | r [95% CI] | p | p (Holm) | Spearman ρ (p) | Leave-one-out range of r |", "|---|---|---|---|---|---|"]
for c in R["correlations"]:
    L.append(f"| {c['a']}–{c['b']} | {r3(c['r'])} [{r2(c['ci'][0])}, {r2(c['ci'][1])}] | {pv(c['p'])} | {pv(c['p_holm'])} | {r3(c['rho'])} ({pv(c['p_rho'])}) | {r2(c['loo_range'][0])} to {r2(c['loo_range'][1])} |")
T["TABLE3"] = "\n".join(L)
# Table 4 dimensionality
specs = [("A. Six families, SCO as band proximity (primary)", R["dimensionality_all"]),
         ("B. SCO excluded", R["dimensionality_noSCO"]),
         ("C. SCO as Pearson r with human marks", R["dimensionality_SCO_as_r"]),
         ("D. Judge-2 rubric scores substituted", R["judges"]["dimensionality_judge2"]),
         ("E. Automatically scored items only", R["auto_only"]["dimensionality"]),
         ("F. As E, SCO as Pearson r", R["auto_only_SCO_as_r"]["dimensionality"])]
L = ["| Specification | Mean r (negative r / pairs) | λ1 / PA95 | λ2 / PA95 | PC1 share | Retained: Horn (Kaiser) | PC1 share, leave-one-out | PC1 share, model bootstrap 95% |", "|---|---|---|---|---|---|---|---|"]
for lab, d in specs:
    p = d["pearson"]
    L.append(f"| {lab} | {r2(p['offdiag_mean_r'])} ({p['n_neg']}/{p['n_pairs']}) | {p['eigenvalues'][0]:.2f} / {p['pa_95'][0]:.2f} | {p['eigenvalues'][1]:.2f} / {p['pa_95'][1]:.2f} | {100*p['share'][0]:.1f}% | {p['retained_horn95']} ({p['kaiser_count']}) | {100*d['loo_pc1_share_range'][0]:.1f}–{100*d['loo_pc1_share_range'][1]:.1f}% | {100*d['model_bootstrap_pc1_share_ci'][0]:.1f}–{100*d['model_bootstrap_pc1_share_ci'][1]:.1f}% |")
T["TABLE4"] = "\n".join(L)
# Table 5 judge agreement
J = R["judges"]
L = ["| Items | Pairs | Pearson r | Spearman ρ | Mean absolute difference | Within ±0.15 | Mean, judge 1 | Mean, judge 2 |", "|---|---|---|---|---|---|---|---|"]
for lab, a in [("All rubric items", J["overall"])] + [(t, J["by_task"][t]) for t in ("GEN", "CUL", "PED")]:
    L.append(f"| {lab} | {a['n']:,} | {r3(a['pearson'])} | {r3(a['spearman'])} | {a['mad']:.3f} | {a['within_015']:.3f} | {a['mean_j1']:.3f} | {a['mean_j2']:.3f} |")
T["TABLE5"] = "\n".join(L)
# Table 6 model-level under judges
L = ["| Model | Micro, J1 (rank) | Micro, J2 (rank) | Macro, J1 (rank) | Macro, J2 (rank) | CUL, J1 | CUL, J2 | GEN, J2 − J1 | PED, J2 − J1 |", "|---|---|---|---|---|---|---|---|---|"]
for m in models:
    rk = J["ranks"][m]; d = J["family_delta_j2_minus_j1"][m]
    L.append(f"| {NAME[m]} | {R['micro'][m]:.1f} ({rk['micro_j1']}) | {J['micro_judge2'][m]:.1f} ({rk['micro_j2']}) | {R['macro'][m]:.1f} ({rk['macro_j1']}) | {J['macro_judge2'][m]:.1f} ({rk['macro_j2']}) | {R['family_pct'][m]['CUL']:.1f} | {J['family_pct_judge2'][m]['CUL']:.1f} | {sg(d['GEN'])} | {sg(d['PED'])} |")
T["TABLE6"] = "\n".join(L)
# Table 7 self-preference
def sp(x): return f"{100*x['diff']:+.1f} [{100*x['ci'][0]:+.1f}, {100*x['ci'][1]:+.1f}]".replace("-", "−")
L = ["| Contrast | All rubric items | GEN | CUL | PED |", "|---|---|---|---|---|"]
bt = J["self_preference_by_task"]
rows = [("Judge 1 (k3-agent), own family (k3-agent, k2d6-agent)", J["self_preference_judge1"], "judge1_family"),
        ("Judge 1 (k3-agent), own outputs only", J["self_preference_judge1_selfonly"], "judge1_self"),
        ("Judge 2 (deepseek-chat), own family (deepseek-chat, deepseek-v4-flash)", J["self_preference_judge2"], "judge2_family"),
        ("Judge 2 (deepseek-chat), own outputs only", J["self_preference_judge2_selfonly"], "judge2_self")]
for lab, a, k in rows:
    L.append(f"| {lab} | {sp(a)} | {sp(bt['GEN'][k])} | {sp(bt['CUL'][k])} | {sp(bt['PED'][k])} |")
T["TABLE7"] = "\n".join(L)
# Table 8 SCO repeat
L = ["| Model | Items | Test–retest r | Same band credit | Band score, run 1 | Band score, run 2 | Run 2 − run 1 [95% CI] | r with human marks, run 1 / run 2 |", "|---|---|---|---|---|---|---|---|"]
for m in ["deepseek", "tchub-dsv4f", "hunyuan"]:
    v = R["repeat_runs"][m]
    L.append(f"| {NAME[m]} | {v['n']} | {r2(v['retest_r'])} | {100*v['band_item_agreement']:.0f}% | {v['band_run1']:.1f} | {v['band_run2']:.1f} | {sg(v['band_run2']-v['band_run1'])} [{sg(v['band_diff_run2_minus_run1_ci'][0])}, {sg(v['band_diff_run2_minus_run1_ci'][1])}] | {r2(v['r_gold_run1'])} / {r2(v['r_gold_run2'])} |")
T["TABLE8"] = "\n".join(L)
# Table 9: eight open-source models on the discriminative subset (KNO + PHO)
O = json.load(open(os.path.join(A, "out", "open_source_models.json")))
TK = ["cceval_kno", "cceval_pho_legal", "cceval_pho_level", "cceval_pho_poly"]
def cell(v):
    mark = "‡" if (v["p_vs_constant"] < .05 and v["acc"] > v["constant_answer"]) else ("†" if (v["p_vs_chance"] < .05 and v["acc"] > v["chance"]) else "")
    n = "" if v["n"] == O["tasks"][t]["n"] else f" (n = {v['n']})"
    return f"{v['acc']:.1f} [{v['wilson'][0]:.1f}, {v['wilson'][1]:.1f}]{mark}{n}"
L = ["| Model | Size (B) | Route | KNO level (400) | PHO syllable legality (60) | PHO syllable level (40) | PHO polyphone reading (30) |", "|---|---|---|---|---|---|---|"]
L.append("| Uniform guessing | – | – | " + " | ".join(f"{O['tasks'][t]['chance']:.1f}" for t in TK) + " |")
cl = {"cceval_kno": "“7–9”", "cceval_pho_legal": "either label", "cceval_pho_level": "“1”", "cceval_pho_poly": "first option"}
L.append("| Constant answer | – | – | " + " | ".join(f"{O['tasks'][t]['constant']['acc']:.1f} ({cl[t]})" for t in TK) + " |")
route = {"lm-eval 0.4.13, float32": "lm-eval, FP32", "llama.cpp GGUF Q8_0 + gguf_score.py": "llama.cpp, Q8_0"}
for m, mv in O["models"].items():
    cells = []
    for t in TK:
        cells.append(cell(mv["tasks"][t]))
    L.append(f"| {mv['label'].replace(' (Q8_0)', '')} | {mv['params_b']:g} | {route[mv['runtime']]} | " + " | ".join(cells) + " |")
T["TABLE9"] = "\n".join(L)
# Table 10: response preferences of the 7B/8B models (per-item records)
L = ["| Model | Task and item set | Accuracy % [95% CI] | p vs chance | Most frequent answer (share) | Accuracy when not giving that answer |", "|---|---|---|---|---|---|"]
lab = {"Qwen2.5-7B-q8_0": "Qwen2.5-7B-Instruct", "Llama-3.1-8B-q8_0": "Llama-3.1-8B-Instruct"}
for m in ["Qwen2.5-7B-q8_0", "Llama-3.1-8B-q8_0"]:
    for run, iset in [("kno", "KNO, original 400"), ("kno_bal", "KNO, original 400, options position-balanced"), ("kno_v13b", "KNO, new balanced 399")]:
        v = K["local_gguf"][f"{m}|{run}"]
        L.append(f"| {lab[m]} | {iset} | {v['acc']:.1f} [{v['wilson'][0]:.1f}, {v['wilson'][1]:.1f}] | {pv(v['p_vs_chance'])} | “{v['dominant_pred']}” ({v['dominant_share']:.1f}%) | {v['acc_when_not_dominant']:.1f}% (n = {v['n_not_dominant']}) |")
    for t, iset in [("cceval_pho_legal", "PHO syllable legality, 60"), ("cceval_pho_level", "PHO syllable level, 40")]:
        v = O["models"][m]["tasks"][t]
        nd = f"{v['acc_when_not_dominant']:.1f}% (n = {v['n_not_dominant']})" if v["acc_when_not_dominant"] is not None else "– (n = 0)"
        L.append(f"| {lab[m]} | {iset} | {v['acc']:.1f} [{v['wilson'][0]:.1f}, {v['wilson'][1]:.1f}] | {pv(v['p_vs_chance'])} | “{v['dominant_pred']}” ({v['dominant_share']:.1f}%) | {nd} |")
T["TABLE10"] = "\n".join(L).replace("“非法”", "“illegal”").replace("“合法”", "“legal”")
json.dump(T, open(os.path.join(A, "out", "summary_tables.json"), "w"), ensure_ascii=False, indent=1)
for k, v in T.items(): print(k); print(v); print()
