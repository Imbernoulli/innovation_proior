## Research question

Distill a large math-tuned teacher into a small student under an **on-policy** training loop — the student generates rollouts, the frozen teacher scores those same tokens, and the student is updated to match the teacher on its own samples. The only design choice is the per-step **token-level loss** that turns the two logit tensors into a scalar. The trainer owns generation, the teacher forward pass, optimizer step, and on-policy mixing, so leaderboard differences reflect the loss formulation alone.

## Prior art / Background / Baselines

- **Supervised knowledge distillation (Hinton, Vinyals & Dean 2015; Sanh et al. 2019).** Train the student to match the teacher's softened token distribution on a fixed corpus via forward KL `KL(p_T ‖ p_S)`.
- **Sequence-level KD (Kim & Rush 2016).** Approximate the teacher's whole-sequence distribution by its beam-search mode and train the student by negative log-likelihood on that single output.
- **Behavior cloning (Pomerleau 1989, ALVINN; Ross & Bagnell 2010).** Fit a policy to expert `(state, action)` pairs.
- **DAgger (Ross, Gordon & Bagnell 2011).** Roll out a mixture of the expert and current learner, collect visited states, label them with the expert, aggregate, and refit while scheduling the expert mix to zero.

## Fixed substrate / Code framework

A TRL `GKDTrainer` (v1.4.0) on-policy distillation loop is frozen and must not be touched. It owns the student forward, the frozen-teacher forward under `no_grad` over the same `(prompt + completion)` tokens, the logit shift that aligns position `n`'s logits with the token they predict, the vocab-alignment crop (teacher head 152064 is sliced to student 151936), and the optimizer step. It also owns the **on-policy mixing**: before each step, with probability `lmbda` the trainer replaces the dataset batch with the student's own sampled completions (relabeled, `-100` on padding); otherwise it uses the dataset batch. The loss body sees only the two logit tensors and `labels`, and is not told which source produced the batch. Helpers exposed for reference: a `generalized_jsd_loss` in `gkd_trainer.py` and the config meanings of `beta`, `lmbda`, `temperature`, `step`, `total_steps`.

## Editable interface

Exactly one region is editable — the body of `compute_distill_loss` in `trl/experimental/gkd/custom_distill_loss.py`, between the `EDITABLE START` and `EDITABLE END` markers. The signature is fixed. The loss receives student/teacher logits `[B,T,V]` over the same completion tokens, `labels` (`-100` on prompt/padding, which must be excluded from reduction), the interpolation knob `beta`, the softmax `temperature`, the `reduction`, the current `step` / `total_steps` (for curricula), and `lmbda` (the static on-policy fraction the trainer applies upstream). It returns a scalar that becomes the training loss.

The starting point is the scaffold default: **forward KL** `KL(p_T ‖ p_S)`, the supervised-KD divergence, evaluated on whatever batch the trainer's `lmbda` coin produced.

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

- **Student**: `Qwen/Qwen2.5-0.5B` (494M, base). **Teacher**: `Qwen/Qwen2.5-Math-7B-Instruct` (7.6B, math-tuned). **Training prompts**: a 10k subset of `open-r1/OpenR1-Math-220k`.
- **Trainer**: TRL `GKDTrainer` v1.4.0, defaults `lmbda=0.5`, `beta=0.5`, `temperature=0.9`, `max_steps=2000`, `per_device_train_batch_size=4`, gradient accumulation 4, bf16, gradient checkpointing, 2 GPUs. Single seed {42}.
- **Evaluation**: vLLM-served student generation. **GSM8K** (1319 problems) and **MATH-500** (500 problems) use greedy decoding (temperature 0, n=1); **AMC23** (40 problems) uses avg@8 at temperature 0.6, top_p 0.95 (8 samples per problem, averaged). Final answers extracted from `\boxed{...}` and graded with `math-verify`.
- **Metrics** (all higher-is-better): `gsm8k_accuracy`, `math500_accuracy`, `amc_accuracy`.
