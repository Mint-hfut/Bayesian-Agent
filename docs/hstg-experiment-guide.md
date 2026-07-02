# HSTG 对比实验操作指南

本指南描述如何用 DeepSeek v4 跑通 HSTG(层级时空贝叶斯)与原始 categorical Bayesian backend 的对照实验,并用分析脚本产出论文级别的对比指标。方法本身的公式与实现说明见 [HSTG Backend](hstg.md)。

## 1. 实验设计总览

对照实验的唯一变量是 `--evolution-algorithm`,其余参数(模型、benchmark、mode)保持一致:

| 实验组 | 参数 | 说明 |
|---|---|---|
| baseline | `--mode baseline` | 无进化层的裸 harness,作为下界(已有历史 artifact 可复用) |
| frequentist(可选) | `--evolution-algorithm frequentist` | 频率计数控制组 |
| categorical(原始版本) | `--evolution-algorithm categorical_bayes` | v0.5 默认 Bayesian Evidence Model |
| **hstg(新版本)** | `--evolution-algorithm hstg` | 核加权层级时空先验 |
| hstg + embedding(可选) | `hstg` + `--similarity-backend embedding` | 用语义 embedding 核替换默认词法核 |

**主战场选 RealFin**:40 个任务、语义家族多样(MACD / RSI / 布林带 / 动量组合),空间核才有区分度。SOP-Bench 和 Lifelong AgentBench 任务同质性高,作为次要验证。

**关键提醒**:

- 每组必须用**不同的 `--out-root`**,否则 belief store 和 `results.json` 会互相覆盖。
- RealFin 一组 bayesian_full 约消耗 8–11M tokens(flash),先确认预算,再决定跑几组。
- 建议先用 `deepseek-v4-flash` 验证有效性,再用 `deepseek-v4-pro` 复现。

## 2. 前置条件

```bash
cd Bayesian-Agent
export DEEPSEEK_API_KEY="sk-..."
```

数据集默认在仓库上层的 `GA-Technical-Report/datasets`;不在默认位置时用 `--data-root /path/to/datasets` 显式指定。

## 3. Smoke 测试(必做,几分钟)

先用 `--limit 2` 确认全链路能跑通,再烧全量 token:

```bash
python experiments/run_benchmarks.py \
  --harness bayesian-agent --model deepseek-v4-flash \
  --mode bayesian-full --bench realfin --limit 2 \
  --evolution-algorithm hstg \
  --out-root temp/smoke_hstg
```

跑完检查两件事:

1. `temp/smoke_hstg/results.json` 存在且 `summaries` 有数;
2. `temp/smoke_hstg/skill_evolution/realfin_benchmark/*/belief_before.json` 里有 `hstg_audit` 字段(`w_local`、`alpha_local`、`neighbors`)——有就说明核加权在生效。

## 4. 全量对照实验

### 4.1 对照组:原始版本(categorical_bayes)

```bash
MODEL=deepseek-v4-flash

python experiments/run_benchmarks.py \
  --harness bayesian-agent --model $MODEL \
  --mode bayesian-full --bench realfin \
  --evolution-algorithm categorical_bayes \
  --out-root results/hstg_ablation_${MODEL//-/_}/categorical
```

### 4.2 实验组:HSTG(词法核)

```bash
python experiments/run_benchmarks.py \
  --harness bayesian-agent --model $MODEL \
  --mode bayesian-full --bench realfin \
  --evolution-algorithm hstg \
  --out-root results/hstg_ablation_${MODEL//-/_}/hstg
```

### 4.3 可选实验组:HSTG + Embedding 核

DeepSeek 平台没有 embeddings API,需要另配一个 OpenAI 兼容的 embedding 端点(DashScope/Qwen、OpenAI,或本地 vLLM/TEI 服务均可):

```bash
export EMBEDDING_API_KEY="sk-..."

python experiments/run_benchmarks.py \
  --harness bayesian-agent --model $MODEL \
  --mode bayesian-full --bench realfin \
  --evolution-algorithm hstg \
  --similarity-backend embedding \
  --embedding-model text-embedding-v4 \
  --embedding-base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --out-root results/hstg_ablation_${MODEL//-/_}/hstg_embedding
```

参数缺失或 API key 未设置时脚本会在启动时报错,不会跑到一半失败。向量按任务文本缓存,每个不同文本只调用一次 embedding API。

### 4.4 可选:frequentist 控制组

```bash
python experiments/run_benchmarks.py \
  --harness bayesian-agent --model $MODEL \
  --mode bayesian-full --bench realfin \
  --evolution-algorithm frequentist \
  --out-root results/hstg_ablation_${MODEL//-/_}/frequentist
```

### 4.5 可选:增量修复模式对比

验证"接在已有 Agent 后面只修失败任务"的场景。两组用**同一份** baseline 结果,保证可比:

```bash
BASELINE=results/native_harness_deepseek_v4_flash_full/realfin/baseline/results.json

for ALGO in categorical_bayes hstg; do
  python experiments/run_benchmarks.py \
    --harness bayesian-agent --model $MODEL \
    --mode bayesian-incremental --bench realfin \
    --evolution-algorithm $ALGO \
    --baseline-results $BASELINE \
    --out-root results/hstg_ablation_${MODEL//-/_}/incremental_${ALGO}
done
```

## 5. 结果分析:compare_hstg.py

所有组跑完后,一条命令产出对比报告。注意路径要指到**包含 `results.json` 的那一层**(`--mode bayesian-full` 时是 `<out-root>/bayesian_full`):

```bash
python experiments/compare_hstg.py \
  --run categorical=results/hstg_ablation_${MODEL//-/_}/categorical/bayesian_full \
  --run hstg=results/hstg_ablation_${MODEL//-/_}/hstg/bayesian_full \
  --run hstg_emb=results/hstg_ablation_${MODEL//-/_}/hstg_embedding/bayesian_full \
  --first-k 10 \
  --out temp/hstg_compare.md \
  --json-out temp/hstg_compare.json
```

`--run 标签=路径` 可以重复任意次,标签自定义,会出现在报告列名里。`--benchmark realfin_benchmark` 可以限定只分析一个 benchmark(不传则分析路径下的全部)。

## 6. 报告怎么读

Markdown 报告(`--out`)分四节,对应论文里的四类证据:

### 6.1 Final Metrics

accuracy / tokens / efficiency 基本盘。注意:40 题只差 2–3 题时单看这张表说服力弱,要结合下面的结构性指标。

### 6.2 Cold Start And Patch Behavior(HSTG 的靶心)

| 列 | 含义 | HSTG 预期表现 |
|---|---|---|
| First-K Accuracy | 前 K 个任务(执行顺序)的成功率 | **应高于 categorical**——这是方法声称改善的冷启动窗口 |
| First Patch Activation | 第一个 prompt 里带上 failure-mode patch 的任务序号 | **应更早**(相同/近邻任务的单次失败即可激活) |
| Prompts With Patches | 带 patch 的 prompt 占比 | 视任务分布,不必更多 |
| Mean Patch Modes/Prompt | 平均每个 prompt 激活几条 patch | **可以更低**——语义无关的 patch 被抑制,prompt 更干净 |

### 6.3 Cumulative Accuracy Curve

逐任务累计准确率。论文画图:两条曲线前段(冷启动区)分离、后段收敛,就是"HSTG 把增益前移"的直接图证。JSON 里的 `cumulative_accuracy` 数组可直接喂给画图脚本。

### 6.4 w_local Trajectory

HSTG 组独有。`w_local` 从 0 随近邻证据积累单调上升,对应论文里"动态置信度回退"的机制验证图。JSON 里的 `w_local_curve` 数组同样可直接画图(非 HSTG 组为 `null`)。

## 7. 解读注意事项

- **LLM 非确定性**:`temperature=0` 也不保证完全可复现,单次运行 ±2 题波动正常。预算允许时每组跑 2–3 遍取均值;不允许时把结论重心放在冷启动曲线和 patch 激活时机这类结构性证据上。
- **token 成本对比要谨慎**:HSTG 更早注入 patch 会略微增加 prompt token;报告 efficiency 时同时给出 accuracy,避免"便宜但失败"被误读为高效(参见 `docs/experiments/bayesian-vs-frequentist-realfin.md` 里的教训)。
- **词法核 vs embedding 核**:同质 benchmark(SOP)上两者差异会很小;差异主要出现在 RealFin 这类任务描述多样的场景。论文叙事用 "semantic manifold" 时,以 embedding 组为准,词法组作为零依赖的 fallback 消融。
- **超参敏感性**:`C_stable`(默认 4.0)、激活阈值(1.0)、相关度下限(0.1)在 `bayesian_agent/benchmarks/evolution.py` 和 `bayesian_agent/core/algorithms/hstg.py` 中定义,做敏感性扫描时直接改常量重跑即可。

## 8. 产物清单

一轮完整实验后,论文素材对应关系:

| 产物 | 位置 | 用途 |
|---|---|---|
| 各组 `results.json` / `table.md` | 各 `--out-root` 下 | 主结果表 |
| 对比报告 | `temp/hstg_compare.md` | 附录对比表 |
| 曲线数据 | `temp/hstg_compare.json` | 冷启动曲线图、w_local 机制图 |
| 每任务信念快照 | 各组 `skill_evolution/` | 案例分析(某个失败如何被近邻激活的 patch 修复) |
| belief store | 各组 `bayesian_skill_beliefs.json` | 最终后验状态审计 |
