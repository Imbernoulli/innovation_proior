On-policy distillation tries to train a small student language model to match a large teacher by exposing it to the prefixes it will actually generate at deployment. The supervised-learning floor is to train only on dataset prompts and the teacher's outputs, but that ignores the compounding-error problem: when the student produces its own chain-of-thought, it can drift into states the teacher never visited, and a loss that only knows the teacher's behavior on the original data has no recovery signal. The mismatch between the training distribution and the student's own induced distribution is the same disease that makes plain behavior cloning in robotics fail after the first mistake. The bound is sharp: one early disagreement with probability on the order of the per-state error can throw the policy off the demonstration manifold, after which it pays maximal cost for the remaining steps, giving an extra cost that scales quadratically in the horizon rather than linearly.

Existing soft-distillation approaches try to solve this by carrying the teacher's full probability distribution, but before measuring the value of that soft information we need the hard-target baseline. The crudest defensible choice is to discard everything except the teacher's chosen action and train the student to predict that action with plain cross-entropy. This isolates the central axis of how much of the teacher's soft distribution the loss should use, and it anchors the bottom of the ladder that any soft loss must beat. It also answers the state-distribution question in the simplest possible way: the on-policy prefixes are supplied by the trainer's existing mixing mechanism, and the loss only has to decide what label to place on each visited state.

The method is DAgger, short for Dataset Aggregation. In its original imitation-learning form it iteratively rolls out the current policy, labels every visited state with the expert's action, aggregates those pairs into a growing dataset, and retrains a single policy on the aggregate. Refitting on all data seen so far is Follow-The-Leader, which makes the procedure a reduction of imitation learning to no-regret online learning and drives the loss under the policy's own state distribution to a small value.

In the on-policy distillation setting, the trainer's lmbda mixing already supplies the on-policy state distribution: with probability lmbda the batch is replaced by student-generated prefixes. DAgger's loss then reduces to a plain cross-entropy target at each token, where the expert action is the teacher's argmax token. At each completion position we take the teacher logits, compute the argmax to get the deterministic expert action, and train the student to predict that token with standard cross-entropy. The -100 padding mask is preserved so prompts and padding do not contribute, no temperature is applied because a hard target has no soft distribution to sharpen, and the loss is averaged over valid completion tokens.

This hard-target supervision is the imitation-learning floor. It collapses the teacher's full vocabulary distribution to a one-hot vector at its mode, throwing away dark knowledge and manufacturing sharp gradients wherever the teacher is genuinely uncertain. Under a large teacher-small student gap it is expected to trail soft-distribution losses, especially on long multi-step reasoning chains where the teacher is multi-modal. Its purpose is to establish the baseline and make the next move clear: stop discarding the teacher's distribution and give the student soft per-token targets.

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
    """DAgger (Dataset Aggregation) hard-target loss for on-policy distillation.

    The teacher's chosen action at each completion position is its argmax token.
    The student is trained by plain cross-entropy to predict that token. The on-policy
    state distribution comes from the trainer's lmbda mixing upstream; this loss only
    swaps the soft-target supervision for a hard target.
    """
    # Teacher's deterministic expert action at each position.
    target_tokens = teacher_logits.argmax(dim=-1)  # [B, T]

    if labels is not None:
        # Preserve the -100 padding/prompt mask from labels.
        target_tokens = torch.where(labels == -100, labels, target_tokens)

    # Token-level cross-entropy against the teacher's top-1 action.
    per_token = F.cross_entropy(
        student_logits.reshape(-1, student_logits.size(-1)),
        target_tokens.reshape(-1),
        ignore_index=-100,
        reduction="none",
    )  # [B*T]

    if labels is not None:
        valid_mask = (labels != -100).reshape(-1)
        per_token = per_token[valid_mask]
        denom = valid_mask.sum().clamp_min(1)
    else:
        denom = torch.tensor(
            max(per_token.numel(), 1),
            device=per_token.device,
            dtype=per_token.dtype,
        )

    if reduction == "batchmean":
        return per_token.sum() / denom
    elif reduction == "sum":
        return per_token.sum()
    elif reduction == "mean":
        return per_token.mean() if per_token.numel() else per_token.sum()
    return per_token
```
