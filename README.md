# CC-Eval · 国际中文教育大模型评测基准

**CC-Eval**（Chinese Language Education Evaluation）是面向**国际中文教育**（Teaching Chinese to Speakers of Other Languages）垂直场景的大语言模型实用价值评测基准。它以权威标准文件为骨架、真实学习者语料为素材，回答一个朴素的问题：**通用大模型在国际中文教育的具体工作中，到底能不能用、哪里能用、哪里不能用？**

作者：***（***）· 首次跑量：2026-09 · 版本：v1.0

**施测模型共 19 个，分两种协议：**

- **11 个商用模型**（API 调用）× v1.0 全量 1,555 题，**生成式协议**：模型自由作答，1,357 题由自动评分器评分，198 道主观题由 LLM 评委按量表评分（第二评委复评见 `results/judge_reliability_full.json`）；
- **8 个开源模型**（本地运行，0.5B–8B）× **判别式子集**（KNO 等级知识 400 题 + PHO 语音 130 题），**对数似然协议**：不生成文本，比较各候选答案的对数概率，取最高者为预测。

两种协议的作答方式与题目覆盖不同，两组分数**不可直接比较**（见下文“测量等价性说明”）。

## 商用模型排行榜（11 个模型，v1.0 全量 1,555 题，生成式协议，百分制；总分为全部题目得分的平均，即按题量加权）

| 排名 | 模型 | 总分 | KNO 等级知识 | ERR 偏误识别 | SCO 作文评分 | GEN 教学生成 | CUL 文化国情 | PED 标准定位 |
|---|---|---|---|---|---|---|---|---|
| 1 | Kimi k3-agent | 46.7 | 29.0 | 33.3 | 78.6 | 48.7 | 94.9 | 45.0 |
| 2 | DeepSeek deepseek-chat | 44.2 | 35.2 | 31.4 | 75.8 | 36.6 | 84.7 | 41.7 |
| 3 | deepseek-v4-flash（TokenHub） | 42.9 | 22.5 | 30.5 | 70.8 | 50.8 | 88.9 | 42.5 |
| 4 | 腾讯混元 hy3（TokenHub） | 42.2 | 23.8 | 32.8 | 48.6 | 59.7 | 92.0 | 43.3 |
| 5 | Grok grok-3-mini-fast | 41.2 | 24.8 | 28.0 | 58.6 | 53.7 | 87.4 | 40.3 |
| 6 | Kimi k2d6-agent | 39.4 | 21.2 | 36.1 | 55.8 | 40.0 | 85.6 | 44.1 |
| 7 | 百度 ernie-4.5-turbo | 38.4 | 20.2 | 24.1 | 76.8 | 35.0 | 82.4 | 42.5 |
| 8 | 阿里 qwen-flash | 38.2 | 12.2 | 30.8 | 74.2 | 38.4 | 80.4 | 43.8 |
| 9 | 字节 doubao-seed-1-6 | 37.9 | 20.2 | 34.0 | 44.4 | 46.6 | 86.3 | 41.1 |
| 10 | 阶跃 step-3.7-flash | 37.0 | 15.8 | 32.0 | 63.3 | 37.4 | 77.5 | 40.8 |
| 11 | 智谱 glm-4-flash | 33.4 | 12.0 | 21.5 | 74.2 | 31.4 | 70.5 | 37.1 |

> 完整分析见 [results/REPORT_v1.0_multimodel.md](results/REPORT_v1.0_multimodel.md)；机读数据见 [results/multimodel_summary.json](results/multimodel_summary.json)；逐题原始输出打包于 `results/results_v1.0_raw.tar.gz`。

**核心结论（商用模型）**：作文辅助评分 / 文化国情问答 / 场景化教学设计——可直接用；偏误归因 / 等级判断——需外挂标准检索；限级生成（最高仅 15%）/ 标准条文合规定位（最高 8.9%）——暂不可用。

## 开源模型结果（8 个模型，判别式子集，对数似然协议）

准确率 %，方括号内为 95% Wilson 置信区间。† 显著高于均匀猜测基线（精确检验 p < .05）；‡ 同时显著高于恒定作答基线。

| 模型 | 推理路线 | KNO 等级知识（400） | PHO 音节合法性（60） | PHO 音节定级（40） | PHO 多音字定音（30） |
|---|---|---|---|---|---|
| 均匀猜测基线 | – | 14.3 | 50.0 | 14.3 | 47.8 |
| 恒定作答基线 | – | 39.0（恒答「7-9」） | 50.0（恒答任一标签） | 32.5（恒答「1」） | 53.3（恒选第 1 项） |
| Qwen2.5-0.5B（基座） | lm-eval / FP32 | 6.5 [4.5, 9.4] | 50.0 [37.7, 62.3] | 32.5 [20.1, 48.0]† | 66.7 [48.8, 80.8]† |
| Qwen2.5-0.5B-Instruct | lm-eval / FP32 | 6.5 [4.5, 9.4] | 50.0 [37.7, 62.3] | 30.0 [18.1, 45.4]† | 60.0 [42.3, 75.4] |
| Qwen2.5-1.5B-Instruct | lm-eval / FP32 | 15.8 [12.5, 19.6] | 50.0 [37.7, 62.3] | 5.0 [1.4, 16.5] | 80.0 [62.7, 90.5]‡ |
| Qwen2.5-3B-Instruct | lm-eval / FP32 | 11.8 [9.0, 15.3] | 50.0 [37.7, 62.3] | 32.5 [20.1, 48.0]† | 90.0 [74.4, 96.5]‡ |
| Llama-3.2-1B-Instruct | lm-eval / FP32 | 6.2 [4.3, 9.1] | 50.0 [37.7, 62.3] | 32.5 [20.1, 48.0]† | 73.3 [55.6, 85.8]‡ |
| Llama-3.2-3B-Instruct | lm-eval / FP32 | 10.5 [7.9, 13.9] | 50.0 [37.7, 62.3] | 15.0 [7.1, 29.1] | 70.0 [52.1, 83.3]† |
| Qwen2.5-7B-Instruct（Q8_0） | llama.cpp / Q8_0 | 14.0 [10.9, 17.7] | 85.0 [73.9, 91.9]‡ | 25.0 [14.2, 40.2] | 83.3 [66.4, 92.7]‡ |
| Llama-3.1-8B-Instruct（Q8_0） | llama.cpp / Q8_0 | 33.5 [29.1, 38.3]† | 50.0 [37.7, 62.3] | 10.0 [4.0, 23.1] | 72.4 [54.3, 85.3]‡（n=29） |

- 0.5B–3B 六个模型用 lm-evaluation-harness 0.4.13（float32，CPU，0-shot）运行，仅保存分片汇总准确率，置信区间按题数计算；7B/8B 两个模型为 GGUF Q8_0 量化，经 llama.cpp + `harness/gguf_score.py` 按同一对数似然规则计分，保存逐题预测。
- KNO 400 题与商用模型所用的 v1.0 KNO 题目相同（题号、题干与标准答案一致；判别式数据由 v1.1 导出，题干含不可见的零宽水印字符）；PHO 130 题来自 v1.1，商用模型未施测。
- 7B/8B 另测了选项位置均衡版与 v1.3b 新题（399 题，每级 57 题）：Qwen2.5-7B 19.8% [16.2, 24.0]，Llama-3.1-8B 18.8% [15.3, 22.9]。Llama-3.1-8B 在原版 KNO 上的 33.5% 来自对「7-9」的作答偏好（76.8% 的题答「7-9」，其余题命中 12.9%），不代表等级知识。
- Llama-3.1-8B 多音字题有 1 题评分出错，按 29 题计。
- 复现：`python analysis/open_source_models.py`（输出 `analysis/out/open_source_models.json` / `.md`）。

### 测量等价性说明

两组模型的分数不处于同一量尺：① 作答协议不同——商用模型生成自由文本、经答案抽取与自动或评委评分，开源模型只在给定候选中按对数概率选择，后者不受格式遵循与答案抽取影响，也无法“拒答”；② 题目覆盖不同——商用模型覆盖六类任务 1,555 题，开源模型只覆盖 KNO 与 PHO；③ 运行条件不同——7B/8B 为 8 bit 量化，0.5B–3B 为 float32；判别式题干含零宽水印字符，可能影响分词。因此本仓库不把 19 个模型合并排名。唯一题目完全相同的是 KNO 400 题，可在同一组基线（均匀猜测 14.3%、恒答「7-9」39.0%）下分别解读两组结果，但不宜据此比较两组模型的高低。

## 基准设计

六类任务（v1.0 共 1,555 题）：

| 代码 | 任务 | 题量 | 素材来源 |
|---|---|---|---|
| KNO | 等级知识（音节/汉字/词汇/语法点定级） | 400 | GF 0025-2021 四张等级表 |
| ERR | 学习者偏误识别与纠正 | 377 | HSK 动态作文语料库（偏误标注） |
| SCO | 作文评分对齐（以人工分数为锚） | 250 | HSK 动态作文语料库（分数） |
| GEN | 教学生成（限级改写/用词成段/语法造句） | 300 | 词表约束程序化构造 |
| CUL | 文化国情与语用推理 | 78 | 《文化和国情教学参考框架》 |
| PED | 标准条文定位与教学设计 | 150 | 教材评价标准/教师能力标准/职业中文标准 |

评分方式：exact_match / numeric_proximity / set_overlap / deterministic_rule 四类自动评分器（1,357 题）+ LLM 评委按量表评分（198 题）。任务与评分明细见 [schema/](schema/) 与 `items/`。

## 目录结构

```
standard/    四份权威标准的结构化数据与全书级校验记录（含 errata.json）
schema/      题目 JSON Schema、评分器定义、示例
items/       题库（v1.0 完整版 items.jsonl 1,555 题、items_public.jsonl 928 题 + 生成器 generate_items.py）
runner/      评测运行器 run_eval.py / 评委补评 judge_rubric.py / 汇总 summarize.py
results/     报告、机读汇总、逐题原始输出压缩包、评委信度/SCO 重测/统计文件、8 个开源模型的原始结果（results/harness/）
harness/     开源模型判别式评测（lm-eval 任务定义、GGUF 计分脚本、题目数据 harness/data/）
analysis/    v1.0 逐题再分析脚本与输出（见下文“复现分析”）
corpus/      语料库清单与使用申请说明（不含语料原文）
```

## 快速开始

```bash
# 1. 安装依赖（仅需 Python 3.10+ 标准库）
# 2. 配置被测模型的 OpenAI 兼容接口（自行创建 runner/providers.json，格式见 runner/run_provider.py）
# 3. 跑测（断点续跑，按 results/<name>/<item_id>.json 跳过已完成项）
python runner/run_eval.py --name mymodel --model <model-name> \
    --base <https://.../v1> --key <API_KEY> --workers 10
# 4. LLM 评委补评主观题
python runner/judge_rubric.py mymodel <judge-model>
# 5. 汇总
python runner/summarize.py mymodel
```

## 数据合规说明

- ERR/SCO 共 627 道题目中的学习者文本取自 **HSK 动态作文语料库**（北京语言大学），该语料库面向研究开放使用；本仓库不包含语料库原始数据文件。相关题目仅供研究使用，语料著作权归原权利方。完整 1,555 题见 `items/v1.0/items.jsonl`，也可由 `items/generate_items.py` 重建（种子 20260905，结果可复现）。
- `standard/` 下结构化数据整理自公开出版标准（GF 0025-2021 等），仅供研究使用，著作权归原发布机构。
- 评测所用 API 凭证不包含在本仓库中。

## 复现分析（v1.0 逐题再分析）

`analysis/` 包含基于 v1.0 逐题结果的再分析：题目自助法置信区间与排名区间、六类任务间相关（Holm 校正）、平行分析维度检验、
第二评委替换与自评偏好检验、SCO 评分规则敏感性、SCO 重测稳定性（以上针对 11 个商用模型），以及 8 个开源模型在判别式子集（KNO/PHO）上的单独分析。

```bash
pip install numpy scipy matplotlib
bash analysis/run_all.sh
```

脚本会先把 `results/results_v1.0_raw.tar.gz` 解压到 `results/<model>/`，再依次运行：

| 脚本 | 作用 | 输出 |
|---|---|---|
| `analysis/convert.py` | 规范化逐题分数、按评测器规则重抽 SCO 预测分、整理评委配对 | `analysis/input/`（不入库） |
| `analysis/structure_pipeline.py` | 主分析（B=2000 分层配对自助法；平行分析 5000 次） | `analysis/out/results.json`、`tables.md` |
| `analysis/local_kno_supplement.py` | KNO 基线分析：11 个商用模型与开源模型在同一 400 题上的准确率、作答偏好（不并入六任务矩阵） | `analysis/out/local_kno_supplement.json` |
| `analysis/open_source_models.py` | 8 个开源模型 × KNO/PHO 四个任务：准确率、Wilson 区间、均匀猜测与恒定作答基线、均衡版核对 | `analysis/out/open_source_models.json`、`.md` |
| `analysis/build_tables.py` / `make_figures.py` | 汇总表与图 | `analysis/out/summary_tables.json`、`analysis/out/figures/` |

输入文件：`items/v1.0/items.jsonl`、逐题结果压缩包、`results/judge_reliability_full.json`（第二评委配对分）、
`results/sco_resample_50.json`（SCO 重测）、`results/analysis_metrics.json`、`results/multimodel_summary.json`、
`harness/data/cceval_kno*.jsonl` 与 `results/harness/`（本地模型）。随机种子固定，重复运行结果一致。
gemini 仅完成 24 题，不纳入分析。

## 引用

```bibtex
@misc{yu2026cceval,
  author = {*** (***)},
  title  = {CC-Eval: 国际中文教育大模型评测基准},
  year   = {2026},
  howpublished = {\url{https://github.com/***/cc-eval}}
}
```

## 许可

代码与原创题库：[MIT License](LICENSE)。第三方标准数据与语料的著作权归原权利方。


## v1.1 更新：PHO 语音规范维度（新增 180 题）

- `items/v1.1/items.jsonl` = v1.0 全部 1555 题 + PHO 语音维度 180 题（共 1735 题）
- 四个子任务：音节合法性判断（60）、音节定级（40）、词语实际读音/变调轻声儿化（50）、多音字语境定音（30）
- 全部 exact_match 确定性评分；拼音答案经声调符→数字归一化比对（见 `runner/run_eval.py` 的 `canon_pinyin`）
- 生成器：`items/gen_pho_items.py`（种子 20260906，可复现）
- 施测：`python runner/run_eval.py --items items/v1.1/items.jsonl ...`

## v1.2 更新：三 Bench 榜单维度（新增 441 题，36 维全覆盖）

对接《国际中文教育大模型测评榜单建设规划》（TeacherBench / LearnerBench / ResearchBench 三专项模块）：
- `items/v1.2/items.jsonl` = v1.1 全部 1735 题 + 榜单专项 361 题（共 2096 题）
- 一批（`items_bench.jsonl`，186 题）：知识讲解（60）、教学目标设计（20）、课堂活动设计（20）、
  练习生成（20）、文献检索（6）、文献理解（20）、论文评审（20）、学术真实性（20，1 真 3 假判真伪）
- 二批（`items_bench_b.jsonl`，100 题，纯现有数据源衍生）：阅读理解（30，文化概要 FMM 分级）、
  个性化学习（20，HSK 真实画像）、个性化教学（20，偏误档案归因）、口语任务设计（30，职业情境卡）
- 生成器：`items/gen_bench_items.py` / `items/gen_bench_items_b.py`（种子 20260907，可复现）；ID 用 level=8 专用段
- 三批（`items_bench_c.jsonl`，75 题，公开官方资料衍生）：路径规划（16，语法大纲教材序列引用）、翻译（29，政府工作报告官方中英对照）、听力理解文本面（30，HSK 官方真题改编）
- 四批（`items_bench_d.jsonl`，80 题，ResearchBench 八维专家量表题）：研究选题/问题设计/文献综述/研究设计/数据分析/语料计算/学术写作/结果解释各 10 题，量表体系 RB-Rubric-1.0（***个人设计，见 `schema/RB八维专家量表体系.md`）——**至此三 Bench 36 维全部有题**
- 维度匹配全景（36 个二级维度 × CC-Eval 现有题/缺口/需补数据源）：`report/三Bench维度匹配矩阵.md`
- 施测：`python runner/run_eval.py --items items/v1.2/items.jsonl ...`
