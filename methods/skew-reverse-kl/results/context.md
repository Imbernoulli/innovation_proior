# Context: choosing the divergence for distilling autoregressive language models (circa 2023-2024)

## Research question

Compress a large autoregressive language teacher `p(y|x)` into a much smaller student
`q_θ(y|x)` that keeps as much of the teacher's task ability as possible, so inference is cheap
while quality stays high. The teacher is queryable — its full next-token distribution is readable
at any context. Distillation is framed as minimizing a divergence `D(p, q_θ)` between the two
per-token distributions, averaged over the positions of a sequence. The hard part is that the
student is *capacity-limited*: it cannot represent the teacher's full distribution, so the *choice
of divergence* — and the way the training sequences are sampled — decides whether the small student
ends up smeared and incoherent, collapsed onto too few behaviors, or genuinely good. The objective
must be one that optimizes *stably* and *fast* for an under-capacity student, especially when it is
trained on sequences the teacher has never seen.

## Background

By this time, knowledge distillation (Bucilua et al. 2006; Hinton, Vinyals & Dean 2015) is the
standard compression tool: a trained teacher's softened output carries the relative probabilities of
the wrong tokens, a richer signal than the one-hot label, and matching it lets a student learn from
less data. For autoregressive language models the loss is applied per token, averaged over a
sequence, with the training pairs `(x, y)` drawn either from a fixed ground-truth dataset (Hinton et
al. 2015) or from teacher-generated outputs (Kim & Rush 2016).

Two facts about this landscape are load-bearing.

First, **the divergence direction matters under capacity mismatch.** The default objective is the
forward KL `D_KL(p, q_θ) = Σ_v p(v)·log(p(v)/q_θ(v))`, weighted by the *teacher's* probability. It
is *mass-covering*: wherever the teacher puts mass, the term blows up unless the student also covers
it, so an under-capacity student is forced to spread its mass over the teacher's entire support,
including low-probability tokens. When the student cannot match all of the teacher's modes — i.e.
there exist `(x, y)` with `p(y|x) ≫ 0` but `q_θ(y|x) ≈ 0` — this produces a *mode-averaging*
problem: an over-smooth student that hedges and generates incoherently (Wen et al. 2023; Gu et al.
2023). The reverse KL `D_RKL(p, q_θ) = D_KL(q_θ, p)`, weighted by the *student's* probability, is
*mode-seeking* (zero-forcing): it concentrates the student on the teacher's dominant modes and drops
the tail. It cures the over-smoothing but can overshoot into *mode collapse* for a low-capacity
student. The generalized Jensen-Shannon divergence (Agarwal et al. 2023) interpolates the two
directions through a mixture parameter `β`, with `D_JSD^β(p, q_θ) = β·D_KL(p, M) + (1−β)·D_KL(q_θ,
M)`, `M = β·p + (1−β)·q_θ`. In that convention, the gradient behavior approaches forward KL as
`β → 0` and reverse KL as `β → 1`, so the same parameter controls both the mixture and the
forward-vs-reverse emphasis.

Second, the gradient of a KL-type loss has a known instability. Following the per-sequence form
(Ji et al. 2023), the gradient of the forward KL with respect to `θ` is
`∇_θ D_KL(p, q_θ) = − r_{p,q_θ}·∇_θ q_θ(y|x)`, where `r_{p1,p2}` is the ratio between two
distributions — the negative gradient of the model probability, weighted *inversely* by that
probability. When `q_θ(y|x) ≈ 0`, the ratio `r_{p,q_θ}` explodes and the gradient norm becomes very
large, producing a big, potentially noisy parameter step. For the minimized reverse KL,
`∇_θ D_KL(q_θ, p) = (log r_{q_θ,p} + 1)·∇_θ q_θ(y|x)`, whose coefficient `log r_{q_θ,p}` likewise
blows up where the teacher probability `p(y|x) ≈ 0`. Some implementations accumulate the negative
KL and then negate it at reduction time; the final minimized loss has this sign.

Third, **where you sample the training sequences matters.** Training on a fixed dataset is
*off-policy*: the prefixes the student conditions on in training are not the prefixes it generates
at inference, a *training-inference mismatch* that compounds errors over the sequence (the
exposure-bias / imitation-learning picture). Training on **student-generated outputs (SGOs)** —
prompting the student to generate, then having the teacher score those self-generated sequences —
addresses the mismatch by training on the student's own familiar states (Lin et al. 2020; Agarwal et
al. 2023). But SGOs introduce their own difficulty, and it interacts with the gradient instability
above: on a self-generated sequence the teacher may assign near-zero probability to what the student
produced — `p(y|x) ≈ 0` on the student's own samples — precisely the regime where the reverse-KL
gradient coefficient explodes. So the very data that fixes the mismatch *triggers* the gradient
blow-up, and at LLM scale generating SGOs every step is also extremely expensive (up to ~80% of
training time), so any practical setup has to reckon with both SGO frequency and stability.

## Baselines

The prior objectives a new distillation loss would be measured against and reacts to.

**Forward KL / supervised KD (Hinton et al. 2015; Kim & Rush 2016).** Match the teacher's per-token
distribution by forward KL (equivalently cross-entropy against the soft teacher). Rich signal, easy
to optimize. **Gap:** mass-covering, so an under-capacity student mode-averages into an over-smooth,
incoherent distribution; and its gradient coefficient `r_{p,q_θ}` explodes wherever the student
probability is near zero.

**Reverse KL (Gu et al. 2023; Agarwal et al. 2023).** Flip the direction to `D_KL(q_θ, p)`,
mode-seeking, so the student commits to the teacher's dominant modes instead of smearing.
**Gap:** can overshoot into mode collapse for a small student; its gradient coefficient
`log r_{q_θ,p}` blows up where the teacher assigns near-zero probability — exactly what happens on
the student's own self-generated sequences — producing large, noisy steps; and the policy-gradient
realizations need a battery of stabilizers.

**Generalized JSD (Agarwal et al. 2023).** Interpolate forward and reverse KL via a probability-space
mixture `M = β·p + (1−β)·q_θ` and the weighted sum `β·D_KL(p, M) + (1−β)·D_KL(q_θ, M)`. **Gap:** the
same `β` changes both the mixture and the weighting of the two KL directions, so one dial must handle
mode averaging, mode collapse, and task-dependent behavior at once.

**Total variation distance (Wen et al. 2023).** A symmetric `f`-divergence, `½·Σ_v |p(v) − q_θ(v)|`,
sitting between the two KL directions. **Gap:** still a single fixed yardstick against a fixed
teacher; it does not address the gradient-explosion-on-SGO problem and offers no analytical handle on
optimization stability or estimation error.

**On-policy SGO training (Lin et al. 2020; Agarwal et al. 2023).** Train on student-generated
sequences scored by the teacher, fixing the training-inference mismatch. **Gap:** the teacher is
unfamiliar with off-distribution SGOs and can give noisy feedback (low loss to short wrong answers,
high loss to long correct ones); generating SGOs every step is computationally dominant; and on SGOs
the teacher probability is often near zero, which is precisely where the reverse-KL gradient
coefficient explodes.

## Evaluation settings

Natural yardsticks for autoregressive distillation include:

- **Instruction following** — distill a student from a larger instruct teacher; evaluate generated
  responses against held-out instructions (e.g. with ROUGE-L against references, or a judged
  win-rate), with greedy or sampled decoding.
- **Task-specific generation** — summarization (ROUGE), machine translation (BLEU), and arithmetic or
  instruction-following tasks scored by task-specific exact-match, overlap, or judged-preference metrics.
- **Protocol** — a small student distilled from a single larger teacher of the same family; the
  student first supervised-fine-tuned so it generates adequate sequences; training data drawn from a
  fixed dataset, from teacher-generated outputs, and/or from student-generated outputs; ablations over
  the divergence and over the fraction of self-generated data.

## Code framework

The standard distillation loop runs the student and the frozen teacher forward over the same tokens
to get two logit tensors `[B, T, V]`, and feeds them, with a label mask (`-100` on prompt/padding),
to a scalar per-token loss that is backpropagated into the student. The scaffold leaves the token-level
divergence as the one empty slot.

```python
import torch
import torch.nn.functional as F


def distill_loss(logits, teacher_logits, no_model_batch):
    """Map two token distributions over the same sequence positions to a scalar."""
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)

    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    # TODO: define the token-level divergence between teacher and student.
    vocab_terms = ...                                # [B, T, V]

    vocab_terms = torch.masked_fill(vocab_terms, inf_mask, 0)
    per_token = torch.sum(vocab_terms, dim=-1).view(-1)
    return torch.sum(per_token * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)


# distillation training loop the loss plugs into
def train(student, teacher, data_loader, optimizer):
    teacher.eval()
    for batch in data_loader:
        student_logits = student(**batch).logits
        with torch.no_grad():
            teacher_logits = teacher(**batch).logits
        loss = distill_loss(student_logits, teacher_logits, batch)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
```
