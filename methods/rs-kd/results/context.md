# Context: cached sparse knowledge distillation for LLM pre-training (circa 2024)

## Research question

We want to distill a large, well-trained teacher LLM into a smaller student in the *offline
logits* setting: run the teacher once over the pre-training corpus, cache its output
distribution per token, and then reuse that cache to train a family of students of various
sizes. This setting is enormously attractive for LLMs — the expensive teacher runs a single
time on cheap compute without fast interconnects; the student trains on small clusters with no
teacher memory footprint; and small-scale design experiments and ablations stop paying the
constant overhead of re-running the teacher. The obstacle is storage. Distribution-matching
distillation needs the teacher's soft target — its probability vector over the vocabulary — at
every training token, and modern vocabularies are huge. Storing the *full* distribution for a
trillion tokens at one byte per probability runs to roughly `128` PB for a `100k`-vocab model;
this is infeasible. So the teacher signal must be stored *sparsely*: only a small subset of
each token's probability vector can be cached. The precise problem is to choose, per token, a
sparse summary of the teacher distribution — ideally on the order of ten entries — such that a
student trained against the cached summary learns essentially what it would have learned from
the full teacher distribution: the same loss, the same calibration, the same downstream
quality, at a tiny fraction of the storage and with negligible compute overhead over plain
cross-entropy training.

## Background

**Knowledge distillation and soft targets (Hinton, Vinyals & Dean 2015).** A small student is
trained to match a large teacher's output distribution rather than (or in addition to) the
one-hot labels. The teacher's *soft targets* — the relative probabilities it assigns across the
wrong classes — carry similarity structure ("dark knowledge") that a one-hot label discards. A
temperature `T` softens the softmax, `p_i = exp(z_i/T) / sum_j exp(z_j/T)`, exposing more of the
non-target mass. Crucially, soft targets give the student a *lower-variance* gradient signal
than hard labels, which is part of why distillation helps. The student is fit to the soft
targets with a KL (or, equivalently up to a teacher-only constant, a cross-entropy) loss.

**The softmax-KL logit gradient.** For a target distribution `t` and student probabilities
`p = softmax(x)`, the KL loss `L = sum_i t_i log(t_i / p_i)` has a clean gradient at the
logits. Using the softmax Jacobian `dp_i/dx_j = p_i (1{i=j} - p_j)`,
```
dL/dx_j = -sum_i t_i (1/p_i) dp_i/dx_j = (sum_i t_i) p_j - t_j.
```
When the target is a proper distribution (`sum_i t_i = 1`, as for the full teacher or for
cross-entropy on a one-hot label) this collapses to `dL/dx_j = p_j - t_j`, whose unique zero is
`p = t`: the student converges to exactly the target distribution. This `(sum t) p_j - t_j`
form is the lens through which any sparse-target scheme must be read, because the factor
`sum_i t_i` need not be `1` once the target is truncated.

**Calibration as a diagnostic.** Expected Calibration Error (Guo, Pleiss, Sun & Weinberger
2017) measures whether predicted confidence matches empirical accuracy. Pre-trained
(non-instruction-tuned) LLMs are observed to be nearly perfectly calibrated, and so are
students trained with plain cross-entropy or with the full teacher distribution. Miscalibration
of a distilled student is therefore a visible symptom that its training target has been
distorted, not a property of the data — a fact prior work on cached distillation already
exploits (Shum et al. 2024, "FIRST", note that students trained on a handful of cached teacher
probabilities come out over-confident).

**Token distributions are heavy-tailed.** Real language token probabilities are not uniform;
they follow a hyperbolic, Zipf-like law (`p_i ~ 1/i`). A teacher's per-token distribution
typically has a few high-probability tokens and a long, low-probability tail that nonetheless
holds the ground-truth token for rare events. Discarding that tail is known to degrade models
trained on the resulting distribution (Shumailov et al. 2024 observe collapse when models are
trained on tail-deprived, degraded distributions).

**Estimating an expectation under one distribution by sampling from another.** A standard tool
exists for approximating `E_{x~p}[f(x)] = sum_x f(x) p(x)` when drawing from `p` directly is
inconvenient: draw from a *proposal* `q` and reweight (importance sampling; Goodfellow, Bengio &
Courville 2016, ch. 17; Elvira & Martino 2021). The estimator
`s_q = (1/n) sum_i f(x_i) p(x_i)/q(x_i)` with `x_i ~ q` has expectation `E_q[s_q] = E_p[f]` for
*any* proposal `q` that is non-zero wherever `p f` is non-zero — and its variance,
`Var[s_q] = Var[p f / q]/n`, depends strongly on how well `q` is chosen; the variance-minimizing
proposal is `q*(x) ∝ p(x)|f(x)|`, which for a non-negative integrand even reaches zero variance
with a single sample (though `q*` requires knowing the normalizer one is trying to compute).
Two facts from this framework are load-bearing here: the estimate is unbiased for any
sufficiently-supported proposal, and it is *biased* the moment the proposal assigns zero
probability somewhere the integrand does not.

## Baselines

These are the prior sparse-target schemes a new method is measured against, and the ceiling and
floor that bracket them.

**Full distillation (FullKD) — the unstorable ceiling.** Fit the student to the entire teacher
distribution `t` with forward KL; logit gradient `p_j - t_j`; students come out essentially
perfectly calibrated and reach the best quality. This is the target to match. *Limitation:* it
requires the full `|V|`-dimensional distribution at every token — about `10` PB for `100`B
tokens at one byte per probability — so it cannot be cached or stored at LLM scale. It defines
the goal but is not itself a storable method.

**Cross-entropy only (CE) — the floor.** Train on the one-hot ground truth, no teacher signal;
calibrated, cheap, but it throws away all of the teacher's dark knowledge. One logit and a few
bytes per token.

**Top-K caching (Raman, Mani, Liang & Lipton 2023; Peng, Lv, Bai et al. 2024).** Keep the `K`
highest-probability tokens of the teacher and zero the rest: `t^s_i = t_i` for `i` in the top-K
set, `t^s_i = 0` otherwise (note `sum_i t^s_i != 1`), optionally re-normalized, optionally
combined with Top-p (dynamic `K` to capture a fixed mass `p`). It is the intuitive choice, and
it is *locally* optimal in a precise sense: among all `K`-element subsets, the top-`K`
(re-normalized) minimizes the `L1` distance to `t` for a single token — the error is `2(1 - a)`
with `a = sum_{i in K} t_i`, minimized by taking the largest `a`. *Observed limitations:* (1)
**It distorts the target.** Truncating to the top-K and applying the KL loss makes the logit
gradient `(sum_{i in K} t_i) p_j - t_j` rather than `p_j - t_j`; the gradient now vanishes when
the in-K student probabilities are a *scaled-up* copy of the teacher's,
`p_i = t_i / (sum_{j in K} t_j)`, and when the out-of-K probabilities are driven to zero. The
student becomes over-confident on the kept tokens and under-confident on the rest, and the
calibration error grows as `K` shrinks. Prior sparse-logit recipes therefore compensate by
keeping many more tokens than the storage target can afford, and small-student distillation can
even fall below the plain CE floor. (2) **It deletes the tail.** Every token outside the top-K,
including a rare ground-truth token, is assigned probability zero, so the student gets no
supervision there — a weaker signal than even CE provides for rare tokens.

**Partial patches to Top-K** (each repairs one symptom and stalls on another):
- *Label smoothing* — spread the residual mass `1 - sum_{i in K} t_i` uniformly over all tokens.
  This restores calibration but *degrades* quality, because real token mass is heavy-tailed
  (Zipf), not uniform, so a flat tail is a poor approximation.
- *Ghost token* — add one extra bucket holding the residual mass for both teacher
  (`t^s_ghost = 1 - sum_{i in K} t_i`) and student (`p_ghost = 1 - sum_{i in K} p_i`), and take
  KL over the `K+1` buckets. This makes the in-K gradient exactly `p_j - t_j` (matching FullKD on
  the kept tokens) and gives the out-of-K tokens a gradient proportional to student confidence,
  `(sum_{i in K}(t_i - p_i)/(1 - sum_{i in K} p_i)) p_j`, fixing the up-scaling and most of the
  calibration error. *Limitation:* the entire tail is collapsed into a single undifferentiated
  bucket, so there is no *per-token* supervision in the tail; it preserves the kept-token
  gradient but still cannot tell the student how to distribute mass among tail tokens.
- *Naive fix* — assign the residual mass to the ground-truth token. This gives one tail token a
  signal, but it still does not reconstruct the teacher's tail distribution.

The common thread across the top-K family and its patches: they either give a *biased* summary
of the teacher distribution, or they fail to supply real supervision in the tail, or both — and
the storage they need to compensate is large.

## Evaluation settings

The yardsticks already in use for cached LLM distillation, all pre-method:

- **Models and data.** LLaMA-style students (≈`100`M / `300`M / `1`B / `3`B) distilled from a
  larger teacher (`3`B internal, or open Llama-3-8B), and a Qwen-style `0.5`B student (the
  Qwen2.5-0.5B architecture) from the same teacher; pre-training on web data / Fineweb-edu;
  training budgets from `10`B to `100`B tokens (multiples of the Chinchilla-optimal count).
- **Loss.** Forward KL distillation, optionally mixed with cross-entropy as
  `L = alpha * L_KD + (1 - alpha) * L_CE`. Standard large-model stack (Megatron / DeepSpeed /
  flash-attention).
- **Metrics.** Language-modeling loss on the pre-training set; Expected Calibration Error;
  speculative-decoding acceptance rate (top-1 agreement with the teacher, a correlate of
  quality); zero-shot NLU via LM-Eval-Harness (HellaSwag, ARC-Easy, LAMBADA, PiQA); and, after
  Tulu SFT, instruction-following quality scored by an LLM-as-judge (Llama-3.1-405B) on Dolly /
  SelfInst / Vicuna / S-NI / UnNI. A "% CE→FullKD" summary reports the fraction of the gap
  between the CE floor and the FullKD ceiling that a method closes.
- **Comparators.** CE (floor), FullKD (ceiling), Top-K and Top-p caching, at matched numbers of
  stored tokens, plus storage in TB and tokens/sec throughput.

## Code framework

The method plugs into a standard cached-distillation training step that already exists. A frozen
teacher runs under `no_grad` and produces a full per-token probability vector; this vector is
turned into a *sparse cached target* by a downsampling function; the student runs and produces
logits; a distillation loss compares the student to the cached sparse target, optionally mixed
with cross-entropy on the ground-truth labels. Everything here is settled *except* how to turn a
full teacher distribution into the sparse summary that will be cached, and the loss that consumes
it — that pair is the single empty slot.

```python
import torch
import torch.nn.functional as F


def downsample(teacher_probs):
    """Turn a full teacher distribution [B, V] into a SPARSE cached target [B, V]
    (mostly zeros) that is cheap to store and reuse. This is the slot to design."""
    pass


def distillation_loss(student_logits, sparse_target):
    """Compare the student to the (sparse) cached teacher target.
    Pad / prompt positions are excluded from the reduction upstream."""
    pass


# existing cached-distillation training step the slot plugs into
def train_step(inputs, labels, teacher_model, student_model, alpha=0.5):
    with torch.no_grad():                                   # teacher runs once, frozen
        teacher_logits = teacher_model(inputs)
        teacher_probs = F.softmax(teacher_logits, dim=-1)   # full distribution [B, V]

    # TODO: define the sparse-target construction and matching loss.
    sparse_target = downsample(teacher_probs)               # cache-able sparse summary

    student_logits = student_model(inputs)                  # student forward [B, V]
    ce_loss = F.cross_entropy(student_logits, labels)       # ground-truth signal
    kd_loss = distillation_loss(student_logits, sparse_target)
    return alpha * kd_loss + (1 - alpha) * ce_loss          # combined objective
```

The downsampled target is what gets written to disk and re-read across experiments; the loss is
what the student is fit against. The contribution lives entirely in those two stubs.
