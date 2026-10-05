# -*- coding: utf-8 -*-
"""补测：基座 0.5B 在 seen_words（训练词切片）上的零样本先验（chat 口径）。
结果写入 injection/results_inject.json 的 base.seen_chat。
"""
import json, os, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(14)
BASE = os.path.join(ROOT, 'models/Qwen2.5-0.5B')
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32)
model.eval()

CHOICES = ['答案：1', '答案：2', '答案：3', '答案：4', '答案：5', '答案：6', '答案：7-9']


def loglikelihood(context, continuation):
    ids_ctx = tok(context, add_special_tokens=False)['input_ids']
    ids_cont = tok(continuation, add_special_tokens=False)['input_ids']
    ids = ids_ctx + ids_cont
    with torch.no_grad():
        logits = model(torch.tensor([ids])).logits[0]
    lp = 0.0
    for i, cid in enumerate(ids_cont):
        lp += torch.log_softmax(logits[len(ids_ctx) + i - 1], dim=-1)[cid].item()
    return lp


def clean0(t):
    for ch in ['​', '‌', '‍', '﻿', '⁠']:
        t = t.replace(ch, '')
    return t


recs = json.load(open(os.path.join(ROOT, 'injection/data/seen_words.jsonl'), encoding='utf-8'))
print('seen_words:', len(recs))
correct, n, t0 = 0, 0, time.time()
for i, r in enumerate(recs):
    instr = clean0(r['instruction'])
    ctx = f"<|im_start|>user\n{instr}<|im_end|>\n<|im_start|>assistant\n"
    lps = [loglikelihood(ctx, c) for c in CHOICES]
    if max(range(len(lps)), key=lambda k: lps[k]) == r['gold']:
        correct += 1
    n += 1
    if n % 100 == 0:
        print(f'{n}/{len(recs)} acc={correct/n*100:.1f}%', flush=True)
acc = correct / n * 100
print(f'基座/seen_chat: {acc:.1f}%（n={n}，{(time.time()-t0)/60:.1f} 分钟）')

RP = os.path.join(ROOT, 'injection/results_inject.json')
res = json.load(open(RP, encoding='utf-8'))
res.setdefault('base', {})['seen_chat'] = acc
json.dump(res, open(RP, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('已写入 results_inject.json base.seen_chat')
