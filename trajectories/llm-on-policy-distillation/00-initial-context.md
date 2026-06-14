## Research question

Distill a large math-tuned teacher into a small student under an **on-policy** training loop —
the student generates rollouts, the frozen teacher scores those same tokens, and the student is
updated to match the teacher on its own samples — and ask **which distillation loss transfers the
teacher's reasoning ability best**. The whole training loop is fixed: the trainer owns the
on-policy generation, the teacher forward pass, the optimizer step, and the mixing between
student-generated and dataset batches. The *only* thing being designed is the per-step
**token-level loss** that turns the two logit tensors into a scalar. Everything else is frozen, so
leaderboard differences reflect the loss formulation alone.

## Prior art before the first rung (distillation / imitation lineage)

The first rung reacts to the line of distillation and imitation-learning ideas that came before
it. These are the ancestors the ladder starts from; the editable loss below is what they all
collapse to.

- **Supervised knowledge distillation (Hinton, Vinyals & Dean 2015; Sanh et al. 2019,
  DistilBERT).** Train the student to match the teacher's *softened* token distribution on a fixed
  corpus, minimizing the forward KL `KL(p_T ‖ p_S)` (equivalently cross-entropy against the soft
  teacher) at every position. The soft targets carry "dark knowledge" — the relative mass on the
  wrong tokens — a richer, lower-variance signal than the one-hot label. Gap: the data is *fixed
  and off-policy*, so the prefixes the student trains on are not the prefixes it conditions on at
  inference; and forward KL is *mass-covering*, so an under-capacity student smears probability
  over teacher-unlikely tokens.
- **Sequence-level KD (Kim & Rush 2016).** Approximate the teacher's whole-sequence distribution
  by its beam-search mode and train the student by plain negative log-likelihood on that single
  output — behavior cloning on teacher text. Gap: still fixed, teacher-generated, single-mode, and
  off-policy; it discards the per-token distribution entirely.
- **Behavior cloning and its compounding error (Pomerleau 1989, ALVINN; Ross & Bagnell 2010).**
  Fitting a policy to an expert's `(state, action)` pairs minimizes loss under the *expert's* state
  distribution but is deployed under its *own*; one early slip lands it in a state the expert never
  visited, and the error cascades. The reduction analysis makes it quantitative: a cloned policy
  with per-step error `ε` pays cost `O(T²ε)` over a horizon `T`, quadratic rather than linear. The
  cure is to train on the states the learner itself visits.
- **DAgger (Ross, Gordon & Bagnell 2011).** Keep one policy and fix the *data*: roll out a mixture
  `π_i = β_i π* + (1−β_i) π̂_i` of the expert and the current learner, collect the visited states,
  have the expert label them, aggregate, and refit (Follow-The-Leader → no-regret). It schedules
  `β_i → 0` and reaches a bound linear in `T`. Gap: generic imitation learning stated for control,
  learning from the expert's *chosen action* (a hard target), not the full distribution the
  queryable teacher could provide.

## The fixed substrate

A TRL `GKDTrainer` (v1.4.0) on-policy distillation loop is frozen and must not be touched. It owns:
the student forward, the frozen-teacher forward under `no_grad` over the same (prompt + completion)
tokens, the logit shift that aligns position `n`'s logits with the token they predict, the
vocab-alignment crop (the teacher head is padded to 152064; it is sliced to the student's 151936),
and the optimizer step. Crucially it also owns the **on-policy mixing**: before each step, with
probability `lmbda` the trainer replaces the dataset batch with the student's own sampled
completions (relabeled, `-100` on padding), otherwise it uses the dataset batch. The loss body sees
*only* the resulting two logit tensors and `labels`; it is **not** told which source produced the
current batch. Helpers exposed for reference: a `generalized_jsd_loss` at the top of
`gkd_trainer.py` (a sanity reference), and the config meanings of `beta`, `lmbda`, `temperature`,
`step`, `total_steps`.

## The editable interface

Exactly one region is editable — the body of `compute_distill_loss` in
`trl/experimental/gkd/custom_distill_loss.py`, between the `EDITABLE START` and `EDITABLE END`
markers (lines 42–65). The signature is fixed. The loss receives student/teacher logits `[B,T,V]`
over the same completion tokens, `labels` (`-100` on prompt/padding, which MUST be excluded from the
reduction), the interpolation knob `beta`, the softmax `temperature`, the `reduction`, the current
`step` / `total_steps` (for curricula), and `lmbda` (the static on-policy fraction the trainer
applies upstream). It returns a scalar that becomes the training loss. Every method on the ladder
is a fill of this one body.

The starting point is the scaffold default: **forward KL** `KL(p_T ‖ p_S)`, the supervised-KD
divergence, evaluated on whatever batch the trainer's `lmbda` coin produced.

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
    # ================== EDITABLE START ==================
    # Default = forward KL D(teacher || student). Replace this body with a
    # novel distillation loss. Keep the signature unchanged.
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    # KL(p_T || p_S), summed over vocab -> per-token KL of shape [B, T]
    kl = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True).sum(dim=-1)

    if labels is not None:
        mask = labels != -100
        kl = kl[mask]
        denom = mask.sum().clamp_min(1)
    else:
        denom = torch.tensor(max(kl.numel(), 1), device=kl.device, dtype=kl.dtype)

    if reduction == "batchmean":
        return kl.sum() / denom
    elif reduction == "sum":
        return kl.sum()
    elif reduction == "mean":
        return kl.mean()
    return kl
    # ================== EDITABLE END ==================
```

## Evaluation settings

- **Student**: `Qwen/Qwen2.5-0.5B` (494M, base). **Teacher**: `Qwen/Qwen2.5-Math-7B-Instruct`
  (7.6B, math-tuned). **Training prompts**: a 10k subset of `open-r1/OpenR1-Math-220k`.
- **Trainer**: TRL `GKDTrainer` v1.4.0, defaults `lmbda=0.5`, `beta=0.5`, `temperature=0.9`,
  `max_steps=2000`, `per_device_train_batch_size=4`, gradient accumulation 4, bf16, gradient
  checkpointing, 2 GPUs. Single seed {42}.
- **Evaluation**: vLLM-served student generation. **GSM8K** (1319 problems) and **MATH-500** (500
  problems) use greedy decoding (temperature 0, n=1); **AMC23** (40 problems) uses avg@8 at
  temperature 0.6, top_p 0.95 (8 samples per problem, averaged). Final answers extracted from
  `\boxed{...}` and graded with `math-verify`.
- **Metrics** (all higher-is-better): `gsm8k_accuracy`, `math500_accuracy`, `amc_accuracy`.
