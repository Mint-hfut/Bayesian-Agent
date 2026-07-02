# HSTG: Hierarchical Spatio-Temporal Belief Backend

`algorithm="hstg"` is an experimental belief backend that targets the
cold-start weakness of from-scratch self-evolution: with an empty registry,
the categorical posterior is just a uniform prior, and count-based patch
gating (`failure_mode count >= 2`) cannot act on the first occurrence of a
failure even when it is highly diagnostic for the current task.

HSTG addresses this with two shrinkage axes fused into one hierarchical
dynamic prior.

## Spatial Axis: Kernel-Weighted Local Evidence

Each verified trajectory stores a short `task_text`. For the current task
`x_t`, historical evidence is weighted by a semantic kernel
`K(x_t, x_i) in [0, 1]`:

```text
alpha_local(x_t) = alpha_0 + sum_i K(x_t, x_i) * 1[y_i = success]
beta_local(x_t)  = beta_0  + sum_i K(x_t, x_i) * 1[y_i = failure]
```

The default kernel is lexical (unigram + bigram Jaccard over latin tokens
and CJK characters) and depends only on the standard library, preserving
the core package's stdlib-only runtime. An embedding-backed kernel can be
injected through `bayesian_agent.core.similarity.EmbeddingSimilarity`.

## Temporal Axis: Confidence-Weighted Fallback

A local confidence weight grows with the kernel mass in the current task's
neighborhood:

```text
S(x_t)       = sum_i K(x_t, x_i)
w_local(x_t) = S(x_t) / (S(x_t) + C_stable)
```

`w_local` is computed from kernel mass only, with pseudo-counts excluded.
With no similar history (`S = 0`), `w_local = 0` and the model falls back
exactly to the global class prior, so HSTG reduces to the categorical
backend's cold-start behavior instead of hallucinating local structure.

## Hierarchical Fusion: Dynamic Class Prior

The fused prior replaces the static Laplace class prior inside the
categorical likelihood product:

```text
pi_local(x_t)      = alpha_local / (alpha_local + beta_local)
pi'(success | x_t) = w_local * pi_local + (1 - w_local) * pi_global
P(y | h, x_t, x)   proportional to pi'(y | x_t) * prod_j P(x_j | y, h)
```

`pi_global` is the Laplace-smoothed global class probability already
maintained by the categorical evidence model.

## Kernel-Weighted Patch Activation

With `algorithm="hstg"` and a task text available, benchmark patch gating
switches from the global count threshold to kernel-weighted support:

```text
support(mode, x_t) = sum_{i: failure, mode} K(x_t, x_i)
activate when support(mode, x_t) >= 1.0
         or  (count(mode) >= 2 and support(mode, x_t) >= 0.1)
```

This changes cold-start behavior in both directions:

- one semantically near-identical failure can activate a patch for the
  current task without waiting for a second global occurrence;
- repeated failures on semantically distant tasks no longer inject
  irrelevant patch text into the current prompt.

The second clause keeps the original count-based rule as a floor for
modes that already recur globally, but requires minimal semantic
relevance to the current task, so kernel gating never activates slower
than count gating on related tasks while still suppressing unrelated
patches.

Without a task text, gating degrades gracefully to the original
count-based rule (`count >= 2`).

## Honest Framing

HSTG is a kernel-weighted hierarchical shrinkage scheme, not an exact
conjugate posterior: kernel weights break exchangeability, so
`Beta(alpha_local, beta_local)` should be read as a local pseudo-likelihood
belief in the spirit of local likelihood and power-prior methods.
`C_stable` (default 4.0) and the activation threshold (default 1.0) are
hyperparameters that deserve sensitivity analysis before strong claims.

## Usage

```bash
python experiments/run_benchmarks.py \
  --harness bayesian-agent \
  --model "$MODEL" \
  --mode bayesian-full \
  --bench realfin \
  --evolution-algorithm hstg
```

### Embedding Kernel

The default kernel is lexical. To use a semantic embedding kernel, point
the runner at any OpenAI-compatible `/embeddings` endpoint (DeepSeek does
not serve embeddings; DashScope/Qwen, OpenAI, or a local server such as
vLLM/TEI all work):

```bash
export EMBEDDING_API_KEY="sk-..."
python experiments/run_benchmarks.py \
  --harness bayesian-agent \
  --model "$MODEL" \
  --mode bayesian-full \
  --bench realfin \
  --evolution-algorithm hstg \
  --similarity-backend embedding \
  --embedding-model text-embedding-v4 \
  --embedding-base-url https://dashscope.aliyuncs.com/compatible-mode/v1
```

Embedding vectors are cached per task text in-process, so kernel cost is
one API call per distinct text. The same provider can be installed
programmatically with `set_default_similarity(EmbeddingSimilarity(...))`.

### Comparing Ablation Arms

`experiments/compare_hstg.py` reads two or more run roots and reports
final metrics, cold-start window accuracy, patch activation timing, the
cumulative accuracy curve, and the `w_local` trajectory:

```bash
python experiments/compare_hstg.py \
  --run categorical=results/hstg_ablation/categorical/bayesian_full \
  --run hstg=results/hstg_ablation/hstg/bayesian_full \
  --first-k 10 \
  --out temp/hstg_compare.md \
  --json-out temp/hstg_compare.json
```

```python
from bayesian_agent import BayesianSkillRegistry, TrajectoryEvidence

registry = BayesianSkillRegistry("temp/beliefs.json", algorithm="hstg")
registry.record(
    TrajectoryEvidence(
        task_id="task_04_consecutive_rise",
        skill_id="benchmark/realfin_benchmark",
        context="realfin_benchmark",
        outcome="failure",
        failure_mode="blank_ohlcv_field_crashed_calculation",
        task_text="task_04 consecutive rise: find stocks with 5 consecutive rising closes",
    )
)
belief = registry.get("benchmark/realfin_benchmark")
print(belief.hstg.audit("task_29 momentum reversal: rising closes with volume confirmation"))
```

Per-task audit values (`w_local`, `alpha_local`, `beta_local`, `pi_local`,
`pi_global`, nearest neighbors) are persisted in the
`skill_evolution/*/belief_*.json` artifacts under `hstg_audit`, which is
the data source for cold-start fallback curves in experiment reports.
