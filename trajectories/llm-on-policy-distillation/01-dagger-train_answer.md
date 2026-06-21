The substrate hands me an on-policy loop with one editable region — the per-token loss that turns the student and teacher logit tensors into a scalar — and the scaffold default is forward KL against the full teacher distribution. Before I reach for any soft-distillation refinement, I want to fix the floor of the whole experiment: the crudest loss that still respects the loop's one real lesson, which is to train on the states the student itself visits. The trainer already supplies those states (with probability $\lambda$ it swaps in the student's own rollouts before calling me), so the cascade that dooms plain behavior cloning — one early slip lands the model in a context the expert never saw and the error compounds to a cost growing like $O(T^2\varepsilon)$ over the horizon — is structurally handled upstream. What is left for me to decide is not the data but the *label*: given a visited state, which target do I push the student toward? Starting from the loss that throws away the *most* of the teacher's signal is deliberate, because it isolates the exact axis the rest of the ladder will move along — how much of the teacher's distribution the loss is allowed to use — and gives a clean reading of what the soft losses buy on top of it.

I propose the **DAgger loss**: hard-target imitation on the teacher's chosen action. Map the imitation reduction onto tokens — the "state" is the partial sequence the student has generated, the "expert" is the frozen teacher, and the teacher's chosen action at that state is its most likely next token, the argmax of its conditional distribution. So at each completion position I take $\tilde a = \arg\max_v \texttt{teacher\_logits}_v$, a $[B,T]$ tensor of target token ids, and train the student by ordinary cross-entropy to predict exactly that token: minimize $-\log \pi_\theta(\tilde a \mid s)$ summed over the valid completion positions. There is no KL term, no temperature, no advantage weighting; the per-token weight is one. This is DAgger's loss in its most literal autoregressive form — collect the learner's states (the trainer does this), label each with the expert's top-1 action, minimize cross-entropy.

What makes it the *floor*, and not just *a* loss, is precisely what it discards. The teacher does not merely have a preferred token; it has a full distribution over the $\sim$152k-entry vocabulary, and that distribution carries the relative plausibility of every alternative — the "dark knowledge" that soft distillation exists to exploit. Hard-target cross-entropy collapses that distribution to a single one-hot vector at its mode, so everything the teacher knows about the second-best continuation, about its own confidence, about which wrong tokens are near-misses versus absurd, is thrown away. This loss therefore uses strictly less of the available signal than any divergence-based loss could: if a soft loss cannot beat it, the dark-knowledge premise is wrong for this teacher–student pair; if it can, the gap measures exactly what the soft information is worth.

The honest weakness I expect to bite is that a hard target is simultaneously *low information* and *high noise*. Where the teacher's distribution is flat — two or three continuations almost equally likely, as happens constantly in mathematical prose ("therefore" vs "hence" vs "so") — the argmax picks one of them by a hair of logit, essentially at random, and cross-entropy then commands the student to put all its mass on that arbitrary winner and treat the equally-good alternatives as errors. A soft KL would have told the student "these three are interchangeable"; the hard target manufactures a sharp, partly spurious gradient. And under the capacity gap — a 0.5B base student against a 7.6B math-tuned teacher — this is the brittle regime: a hard target asks the student to become as sharp as the teacher's mode at every position with no credit for the surrounding mass, so a near-miss earns nothing to interpolate toward, whereas soft losses degrade gracefully through the distribution.

Two operational details matter for the literal edit. First, I preserve the trainer's padding convention: positions where $\texttt{labels} = -100$ are prompt or padding and must not contribute, so I overwrite the argmax targets with $-100$ there and let cross-entropy's `ignore_index=-100` drop them, with the denominator clamped to at least one so an all-masked batch cannot divide by zero. Second, there is **no temperature scaling on either side**. The imitation loss is plain $\log \pi_\theta(\tilde a \mid s)$; the argmax that defines $\tilde a$ is temperature-invariant anyway, so dividing logits by a temperature would only needlessly rescale the student's logits. The temperature knob the signature exposes is simply unused here, on purpose, because a hard target has no soft distribution for temperature to soften.

I should be clear about what this is *not*. DAgger's aggregate-and-refit (Follow-The-Leader over a growing dataset) and its decaying expert-mixing schedule $\beta_i \to 0$ live in the *data* layer, which here is the trainer's static $\lambda$ mixing — it replaces the batch per step rather than aggregating rounds, and the loss body cannot even see whether a given batch came from the student or the dataset. So what I land is DAgger's *loss* — hard cross-entropy on the expert's action at the learner's visited states — on top of the trainer's static on-policy mixing, not its full no-regret schedule. That is fine for a floor: the point is the label, and the label is the teacher's top-1 token regardless of where the state came from.

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
