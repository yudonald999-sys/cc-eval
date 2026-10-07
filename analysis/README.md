# analysis/ — item-level reanalysis of the v1.0 results (19 models)

Eleven commercial models were run on all 1,555 items with the generative protocol; they enter the six-family analysis.
Eight open-source models were run on the discriminative subset (KNO + PHO) with the log-likelihood protocol; they are analysed separately and never pooled with the eleven-model matrix.

Run `bash analysis/run_all.sh` from the repository root (Python 3.10+, numpy, scipy, matplotlib).
All random seeds are fixed; outputs in `analysis/out/` are reproduced exactly.

| File | Purpose |
|---|---|
| `convert.py` | Normalizes the raw per-item files (`results/<model>/<item_id>.json`, unpacked from `results/results_v1.0_raw.tar.gz`): string scores to numbers, errors flagged, SCO predicted scores re-extracted with the runner's own rule, judge pairs copied from `results/judge_reliability_full.json`. Writes `analysis/input/` (not tracked). |
| `structure_pipeline.py` | Main analysis: stratified paired item bootstrap (B = 2000) for family scores, micro/macro composites, rank intervals and adjacent differences; family reliability; Pearson/Spearman correlations with Fisher-z intervals, Holm adjustment and leave-one-out ranges; Horn parallel analysis (5,000 replications) vs Kaiser; sensitivity analyses (SCO excluded, SCO as Pearson r with human scores, second judge, automatically scored items only); judge substitution and self-preference contrasts; SCO repeat-run stability. |
| `local_kno_supplement.py` | KNO-only supplement: API models and local open-source models (llama.cpp GGUF log-likelihood scoring and lm-eval runs) against chance and constant-answer baselines; balanced and v1.3b item sets. Kept separate from the 11-model matrix. |
| `open_source_models.py` | Eight open-source models (0.5B–8B) on the discriminative subset (KNO 400 + PHO 130): accuracy, Wilson intervals, uniform-guessing and constant-answer baselines with exact tests, response preferences and balanced-set checks for the two GGUF models. lm-eval shard selection follows `harness/summarize_harness.py` (latest run of each shard). Writes `out/open_source_models.json` and `.md`. |
| `build_tables.py`, `make_figures.py` | Summary tables (`out/summary_tables.json`) and figures (`out/figures/`). |

Model folder names: `deepseek` = deepseek-chat, `tchub-dsv4f` = deepseek-v4-flash, `hunyuan` = hy3, `doubao` = doubao-seed-1-6,
`ernie` = ernie-4.5-turbo, `grok` = grok-3-mini-fast, `qwen` = qwen-flash, `stepfun` = step-3.7-flash, `zhipu` = glm-4-flash.
Judge 1 is `k3-agent`; judge 2 is the deepseek-chat endpoint. Both are also evaluated models. `gemini` (24 items) is excluded.
