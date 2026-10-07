#!/usr/bin/env bash
# Reproduce the v1.0 item-level analysis from the files in this repository.
# Requirements: Python 3.10+, numpy, scipy, matplotlib.
set -euo pipefail
cd "$(dirname "$0")/.."
# 1. Unpack the raw per-item outputs (11 models + 1 incomplete run) into results/<model>/
[ -d results/k3-agent ] || tar xzf results/results_v1.0_raw.tar.gz -C results
# 2. Normalize scores, re-extract SCO predictions, collect judge pairs -> analysis/input/
python3 analysis/convert.py > /dev/null
# 3. Main analysis -> analysis/out/results.json, analysis/out/tables.md
cd analysis
python3 structure_pipeline.py \
  --results input/results --items input/items.jsonl --judge input/judge.json \
  --judge1-family k3-agent,k2d6-agent --judge2-family deepseek,tchub-dsv4f \
  --sco-items input/sco_items.json --sco-metrics ../results/analysis_metrics.json \
  --repeat ../results/sco_resample_50.json --summary ../results/multimodel_summary.json \
  --out out
# 4. Supplementary KNO analysis with local open-source models -> out/local_kno_supplement.json
python3 local_kno_supplement.py
# 5. Eight open-source models on the discriminative subset (KNO + PHO) -> out/open_source_models.json, .md
python3 open_source_models.py
# 6. Summary tables and figures -> out/summary_tables.json, out/figures/
python3 build_tables.py
python3 make_figures.py
