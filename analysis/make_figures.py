import json, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
A = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(A, "out", "results.json")))
K = json.load(open(os.path.join(A, "out", "local_kno_supplement.json")))
OUT = os.path.join(A, "out", "figures") + os.sep; os.makedirs(OUT, exist_ok=True)
NAME = {"deepseek": "deepseek-chat", "tchub-dsv4f": "deepseek-v4-flash", "hunyuan": "hy3", "doubao": "doubao-seed-1-6",
        "ernie": "ernie-4.5-turbo", "grok": "grok-3-mini-fast", "qwen": "qwen-flash", "stepfun": "step-3.7-flash",
        "zhipu": "glm-4-flash", "k3-agent": "k3-agent", "k2d6-agent": "k2d6-agent"}
fams = R["families"]; models = sorted(R["models"], key=lambda m: -R["micro"][m])

# Fig 1: family scores with item-bootstrap CIs
fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0), sharey=True)
for ax, t in zip(axes.flat, fams):
    for i, m in enumerate(models):
        lo, hi = R["ci_family"][m][t]; ax.plot([lo, hi], [i, i], color="0.3", lw=1); ax.plot(R["family_pct"][m][t], i, "o", color="black", ms=3)
    ax.set_title(f"{t} (n = {R['common_items_per_family'][t]})"); ax.set_xlabel("Score (0-100)")
    ax.set_yticks(range(len(models))); ax.set_yticklabels([NAME[m] for m in models]); None
axes[0,0].invert_yaxis(); fig.tight_layout(); fig.savefig(OUT + "fig1_family_scores.png", dpi=300); plt.close(fig)

# Fig 2: rank intervals micro / macro
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
for ax, key, lab in zip(axes, ["rank_micro", "rank_macro"], ["Micro composite (item-weighted)", "Macro composite (family-weighted)"]):
    for i, m in enumerate(models):
        r = R[key][m]; ax.plot([r["lo"], r["hi"]], [i, i], color="0.3", lw=1.2); ax.plot(r["median"], i, "o", color="black", ms=3.5)
    ax.set_xlim(0.5, 11.5); ax.set_xticks(range(1, 12)); ax.set_xlabel("Rank (median and 95% bootstrap interval)"); ax.set_title(lab)
    ax.set_yticks(range(len(models))); ax.set_yticklabels([NAME[m] for m in models])
axes[0].invert_yaxis(); fig.tight_layout(); fig.savefig(OUT + "fig2_rank_intervals.png", dpi=300); plt.close(fig)

# Fig 3: scree + parallel analysis, four specifications
specs = [("dimensionality_all", "Six families (SCO as band proximity)"), ("dimensionality_noSCO", "SCO excluded"),
         ("dimensionality_SCO_as_r", "SCO as correlation with human marks"), (("auto_only", "dimensionality"), "Automatically scored items only")]
fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.4))
for ax, (k, lab) in zip(axes, specs):
    d = (R[k[0]][k[1]] if isinstance(k, tuple) else R[k])["pearson"]; p = len(d["eigenvalues"]); x = range(1, p + 1)
    ax.plot(x, d["eigenvalues"], "o-", color="black", ms=3, label="Observed"); ax.plot(x, d["pa_95"], "s--", color="0.55", ms=3, label="PA 95th pct.")
    ax.axhline(1, color="0.75", lw=0.7, ls=":"); ax.set_title(lab, fontsize=7); ax.set_xticks(list(x)); ax.set_xlabel("Component")
axes[0].set_ylabel("Eigenvalue"); axes[0].legend(frameon=False, fontsize=6)
fig.tight_layout(); fig.savefig(OUT + "fig3_scree_parallel.png", dpi=300); plt.close(fig)

# Fig 4: SCO band score vs |mean bias| and vs Pearson r
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
sr = R["sco_recomputed"]
for ax, key, xl in [(axes[0], "bias", "|Mean predicted - human score| (points)"), (axes[1], "r", "Pearson r with human scores")]:
    for m in models:
        x = abs(sr[m]["mean_bias"]) if key == "bias" else sr[m]["pearson_r"]; y = R["family_pct"][m]["SCO"]
        ax.plot(x, y, "o", color="black", ms=3.5); ax.annotate(NAME[m], (x, y), fontsize=6, xytext=(3, 2), textcoords="offset points")
    ax.set_xlabel(xl); ax.set_ylabel("SCO band-proximity score")
c1, c2 = R["sco_band_vs_abs_bias"], R["sco_proximity_vs_r"]
axes[0].set_title(f"r = {c1['r']:.2f}, p < .001"); axes[1].set_title(f"r = {c2['r']:.2f}, p = {c2['p']:.2f}")
fig.tight_layout(); fig.savefig(OUT + "fig4_sco_band_bias.png", dpi=300); plt.close(fig)

# Fig 5: KNO accuracy, API vs local, with chance and majority baselines
rows = [(NAME[m] + " (commercial)", v["acc"], v["wilson"]) for m, v in sorted(K["api_kno400"].items(), key=lambda kv: -kv[1]["acc"])]
loc = [("Qwen2.5-7B-Instruct Q8", K["local_gguf"]["Qwen2.5-7B-q8_0|kno"]), ("Llama-3.1-8B-Instruct Q8", K["local_gguf"]["Llama-3.1-8B-q8_0|kno"])]
loc += [(k, v) for k, v in K["local_lmeval_kno400"].items() if "acc" in v]
rows += [(n + " (open-source)", v["acc"], v["wilson"]) for n, v in loc]
fig, ax = plt.subplots(figsize=(6.0, 4.6))
for i, (n, a, w) in enumerate(rows):
    ax.plot(w, [i, i], color="0.3", lw=1); ax.plot(a, i, "o" if "commercial" in n else "s", color="black" if "commercial" in n else "0.45", ms=3.5)
ax.axvline(K["chance"], color="0.5", ls=":", lw=1); ax.axvline(K["majority_baseline"], color="0.2", ls="--", lw=1)
ax.text(K["chance"] + 0.5, len(rows) - 0.3, "chance (1/7)", fontsize=6); ax.text(K["majority_baseline"] + 0.5, len(rows) - 0.3, 'always "7-9"', fontsize=6)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis(); ax.set_xlabel("Accuracy on the 400 KNO items (%, 95% Wilson CI)")
fig.tight_layout(); fig.savefig(OUT + "fig5_kno_local_api.png", dpi=300); plt.close(fig)
print("ok")
