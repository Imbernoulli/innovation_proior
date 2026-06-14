**Problem.** Under the fixed on-policy loop, decide the per-token *label* the student is pushed
toward at each visited state. The crudest defensible choice — the imitation-learning floor — is to
use the teacher's *chosen action* (a hard target), discarding the rest of the teacher's
distribution. This isolates the ladder's central axis: how much of the teacher's soft distribution
the loss uses.

**Key idea (DAgger loss).** At each completion position, the teacher's "expert action" is its top-1
(argmax) next token; train the student by plain cross-entropy to predict that token. No KL,
no temperature, no advantage weighting, unit token weight. The on-policy state distribution is
supplied upstream by the trainer's `lmbda` mixing; this loss only swaps the soft-target supervision
for a hard target at each token.

**Why this is the floor.** Hard-target cross-entropy collapses the teacher's full vocabulary
distribution to a one-hot vector at its mode, throwing away the "dark knowledge" (relative mass on
the alternatives) that soft distillation exists to exploit. Worse, where the teacher is genuinely
uncertain (flat distribution over interchangeable continuations — common in math prose), the argmax
picks an arbitrary winner and cross-entropy demands one-hot confidence on it, manufacturing a sharp,
partly spurious gradient. Under the 0.5B-student / 7.6B-teacher capacity gap, all-or-nothing token
supervision is the brittle regime: a near-miss earns no partial credit. So any soft-distribution loss
should beat it, and the gap measures what the soft information is worth.

**What is NOT included.** DAgger's aggregate-and-refit (Follow-The-Leader) and decaying expert-mix
`β_i` live in the data layer, which is the trainer's static `lmbda` mixing here, not the loss. The
loss body cannot even see whether a batch is student- or dataset-generated. This lands DAgger's
*loss* on the trainer's static on-policy mixing, not its full no-regret schedule.

**Hyperparameters.** None beyond the fixed signature. Temperature is deliberately unused (a hard
target has no soft distribution to soften); `ignore_index=-100` preserves the trainer's
prompt/padding mask; `batchmean` averages over valid completion tokens.

**What to watch.** Should function (short GSM8K chains have unambiguous teacher argmax) but land at
or near the bottom on the reliable metrics (GSM8K, MATH-500), trailing the soft losses — especially
on MATH-500 where chains are longer and the teacher's per-token distribution is more multi-modal.
AMC23 (40 problems) is noise. If it underperforms, the next rung stops discarding the teacher's
distribution and gives soft per-token targets.

```python
import torch
import torch.nn.functional as F


def compute_distill_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor = None,
    beta: float = 0.5,
    temperature: float = 1.0,
    reduction: str = "batchmean",
    step: int = 0,
    total_steps: int = 0,
    lmbda: float = 0.5,
) -> torch.Tensor:
    # DAgger (arXiv 2605.12913) — cross-entropy on teacher's top-1 action.
    # Eq.12-13 + §A: plain CE with unit token weight (w(s,a)=1), no temperature
    # rescaling on either side. The state distribution comes from upstream
    # GKDTrainer's lmbda mixing (with prob lmbda, inputs are student-generated;
    # otherwise dataset). This loss replaces OPD's reverse-KL soft target with
    # a hard-target supervision signal at each token position.

    # No temperature scaling: paper specifies plain log π_θ(ã|s).
    # Teacher's argmax action at each position is the deterministic
    # instantiation of the teacher's chosen action at that state.
    target_tokens = teacher_logits.argmax(dim=-1)  # [B, T]

    if labels is not None:
        # Preserve the -100 padding mask from `labels`.
        target_tokens = torch.where(labels == -100, labels, target_tokens)

    # Token-level CE.
    per_token = F.cross_entropy(
        student_logits.reshape(-1, student_logits.size(-1)),
        target_tokens.reshape(-1),
        ignore_index=-100,
        reduction="none",
    )  # [B*T]

    if labels is not None:
        valid_mask = (labels != -100).reshape(-1)
        denom = valid_mask.sum().clamp_min(1)
        loss_sum = per_token.sum()
    else:
        denom = torch.tensor(max(per_token.numel(), 1), device=per_token.device, dtype=per_token.dtype)
        loss_sum = per_token.sum()

    if reduction == "batchmean":
        return loss_sum / denom
    elif reduction == "sum":
        return loss_sum
    elif reduction == "mean":
        return per_token.mean()
    return per_token
```
