#!/usr/bin/env python3
"""Convert the v1.0 item-level result tree (results/<model>/<item_id>.json) into the pipeline input format.
Run from any directory; paths are resolved relative to the repository root.
- Scores in the raw files are stored as strings ('0.5', 'None'); errors as 'True'. Normalized to float / null / bool.
- Item metadata: items/v1.0/items.jsonl (exact 1,555-ID match, unwatermarked prompts; v1.1/v1.2 are supersets with watermarked prompts).
- SCO per-item predicted scores are re-extracted from raw outputs with the benchmark's own rule (runner/run_eval.py score_numeric).
"""
import json, glob, os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ND = ROOT
OUT = os.path.join(ROOT, "analysis", "input")
EXCLUDE = {"gemini", "harness"}


def norm_score(s):
    if s is None: return None
    if isinstance(s, (int, float)): return float(s)
    s = str(s).strip()
    if s in ("None", "", "null", "nan"): return None
    return float(s)


def truthy(v):
    return v is True or str(v).strip().lower() == "true"


def clean_output(s):
    return re.sub(r"<think>.*?</think>", "", s, flags=re.S).strip()


def extract_score(out):
    m = re.search(r'"score"\s*[:：]\s*(\d+)', out)
    if not m: m = re.search(r"(\d{1,3})\s*分", out)
    if not m: m = re.search(r"\b(\d{1,3})\b", clean_output(out))
    return float(m.group(1)) if m else None


def main():
    os.makedirs(OUT, exist_ok=True)
    items = {}
    with open(f"{ND}/items/v1.0/items.jsonl", encoding="utf-8") as f, open(f"{OUT}/items.jsonl", "w", encoding="utf-8") as g:
        for l in f:
            it = json.loads(l); items[it["item_id"]] = it
            g.write(json.dumps({k: it[k] for k in ("item_id", "task_type", "sub_type", "scoring", "provenance", "reference")}, ensure_ascii=False) + "\n")
    stats = {}; sco = {}
    for mdir in sorted(glob.glob(f"{ND}/results/*/")):
        m = os.path.basename(mdir.rstrip("/"))
        if m in EXCLUDE: continue
        os.makedirs(f"{OUT}/results/{m}", exist_ok=True)
        st = dict(files=0, scored=0, null=0, error=0, api_model=set(), judge=set()); sp = {}
        for fp in glob.glob(f"{mdir}*.json"):
            d = json.load(open(fp, encoding="utf-8")); iid = d["item_id"]
            err = truthy(d.get("error")); s = None if err else norm_score(d.get("score"))
            st["files"] += 1; st["error"] += err; st["null"] += (s is None and not err); st["scored"] += (s is not None)
            st["api_model"].add(d.get("model")); 
            if d.get("judge"): st["judge"].add(d.get("judge"))
            json.dump({"item_id": iid, "model": m, "score": s, "error": err}, open(f"{OUT}/results/{m}/{iid}.json", "w"))
            if items[iid]["task_type"] == "SCO" and not err:
                p = extract_score(d.get("output", "") or "")
                if p is not None: sp[iid] = {"pred": p, "gold": float(items[iid]["reference"]["answer"])}
        st["api_model"] = sorted(x for x in st["api_model"] if x); st["judge"] = sorted(st["judge"])
        stats[m] = st; sco[m] = sp
    json.dump(stats, open(f"{OUT}/conversion_log.json", "w"), indent=1)
    json.dump(sco, open(f"{OUT}/sco_items.json", "w"))
    # judge pairs: copy verbatim
    J = json.load(open(f"{ND}/results/judge_reliability_full.json", encoding="utf-8"))
    # judge1 (k3-agent) and judge2 (deepseek-chat endpoint) are also evaluated models: map to their result-folder names
    json.dump({"judge1": J["judge1"], "judge2": J["judge2"], "pairs": J["pairs"],
               "judge1_eval_name": "k3-agent", "judge2_eval_name": "deepseek"}, open(f"{OUT}/judge.json", "w"))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
