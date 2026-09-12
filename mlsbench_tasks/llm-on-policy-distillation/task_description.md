# LLM On-Policy Distillation — Custom Loss

## Research Question
**What distillation loss best transfers reasoning ability from a large
math-tuned teacher to a small student under an on-policy training loop?**

On-policy distillation (OPD) — the student generates rollouts, the teacher
scores them, and the student is updated to match the teacher on its own
samples — is the dominant 2025/2026 recipe for compressing frontier
reasoning models (Lu et al. *On-Policy Distillation*, Thinking Machines
2025; Qwen3 Tech Report 2025; TAID, ICLR'25 Spotlight; DistiLLM-2,
ICML'25 oral; RS-KD, ACL'25 oral; SKD, ICLR'25). The training loop and
the optimization knobs (β, λ, temperature) are well-understood — the
**loss function** is where competing methods disagree.

This task isolates that loss function.

## Task
Modify the body of `compute_distill_loss` in
`trl/trl/experimental/gkd/custom_distill_loss.py` (lines 38–61, between the
`EDITABLE START` and `EDITABLE END` markers). The signature is fixed:

```python
def compute_distill_loss(
    student_logits: torch.Tensor,    # [B, T, V]
    teacher_logits: torch.Tensor,    # [B, T, V]
    labels: torch.Tensor = None,     # [B, T]; -100 on padding/prompt
    beta: float = 0.5,
    temperature: float = 1.0,
    reduction: str = "batchmean",
    step: int = 0,                   # current training step
    total_steps: int = 0,            # total planned steps
    lmbda: float = 0.5,              # static on-policy mixing fraction
) -> torch.Tensor:
```

## What this surface supports (and doesn't)

Methods implementable as a single `compute_distill_loss` body:

- **Loss-function variants**: forward/reverse KL, generalized JSD, divergence
  mixtures (Hellinger, Tsallis, …), contrastive (DistiLLM-2), top-K + tail
  bucket (RS-KD).
- **Curriculum / scheduled losses**: anything that varies with `step` and
  `total_steps` — TAID-style adaptive interpolation, λ ramp, temperature
  annealing.
- **DAgger-style on/off-policy weighting**: use `lmbda` (the static mixing
  fraction the trainer applies upstream) together with `step` to design a
  hybrid schedule, e.g. reverse-KL early when on-policy fraction matters
  more, forward-KL late.

Methods that require *framework-level* changes (and are therefore **out of
scope** for this task as currently designed):

- **Reference-policy KL** (Stable-OPD, arXiv 2604.08527) — needs a third
  frozen network in compute_loss.
- **Offline teacher-logprob caching** (Lightning-OPD, arXiv 2604.13010) —
  needs a precomputation pass + dataloader modification.
- **Outcome-weighted batching** (Uni-OPD, arXiv 2605.03677) — needs reward
  signal flowing into the loss.
- **Cold-start two-stage** (Rethinking-OPD, arXiv 2604.13016) — needs an
  SFT phase before OPD.
- **Multi-turn DAgger** (Revisiting-DAgger, arXiv 2605.12913) — only
  meaningful for multi-turn agent tasks, not single-turn math.

`student_logits` and `teacher_logits` are returned by the student and
teacher forward passes over the same (prompt + completion) tokens.
`labels` marks the completion positions (`-100` everywhere else). Padding
positions MUST be excluded from the reduction. The result is a scalar
that becomes the training loss.

The function is invoked once per gradient step from
`trl/experimental/gkd/gkd_trainer.py::compute_loss` (read-only). The
agent can also inspect `gkd_config.py` for the meanings of `beta`,
`lmbda`, `temperature`, etc.

## Setup (fixed)
- **Student**: `Qwen/Qwen2.5-0.5B` (494M params)
- **Teacher**: `Qwen/Qwen2.5-Math-7B-Instruct` (7.6B params, math-tuned)
- **Training prompts**: 10k subset of `open-r1/OpenR1-Math-220k`
- **Trainer**: TRL `GKDTrainer` (v1.4.0) — handles on-policy generation,
  teacher forward, optimizer step. Default `lmbda=0.5`, `beta=0.5`,
  `temperature=0.9`, `max_steps=2000`, `per_device_train_batch_size=4`,
  gradient accumulation = 4, bf16, gradient checkpointing on 2 GPUs.
- **Evaluation**: vLLM-served student generation. GSM8K + MATH-500 use
  greedy (temperature 0, n=1); AMC23 uses **avg@8** with temperature 0.6,
  top_p 0.95 (8 independent samples per problem, averaged) — matches the
  small-eval-set protocol DeepSeek-R1 uses for AIME/AMC. Final answer
  extracted from `\boxed{...}` and graded with `math-verify`.

## Evaluation
Three reasoning benchmarks (each a separate `test_cmd`):
- **GSM8K** — grade-school math, 1319 problems
- **MATH-500** — competition-style problems, 500 problems
- **AMC23** — American Math Competition 2023, 40 problems

Reported metrics (all "higher is better"):
- `gsm8k_accuracy` — exact-answer accuracy on GSM8K test
- `math500_accuracy` — exact-answer accuracy on MATH-500
- `amc_accuracy` — exact-answer accuracy on AMC23

## Baselines
Four single-loss reference implementations are provided:

| Baseline | Reference | Loss family |
|---|---|---|
| `gkd` | Agarwal et al., ICLR'24 (arXiv 2306.13649) | Generalized JSD, β=0.5 |
| `opd` | Lu et al., Thinking Machines 2025 / Qwen3 Tech Report | Per-token reverse KL on student samples |
| `taid` | Shing et al., ICLR'25 Spotlight (arXiv 2501.16937) | Forward KL to adaptive teacher–student mixture (logit-space) |
| `rs_kd` | Anshumann et al., ACL'25 oral (arXiv 2503.16870) | Random-sparse top-K KL (unbiased) |
| `dagger` | arXiv 2605.12913 (Revisiting DAgger) | Cross-entropy on teacher's top-1 action (hard target) |

All four implement the same `compute_distill_loss` signature and run on
the identical training loop, so leaderboard differences reflect the
*loss formulation* alone.

## Hints
- The reference `generalized_jsd_loss` lives at the top of
  `gkd_trainer.py` (around line 225) — useful as a sanity reference.
- `step` and `total_steps` are exposed for curriculum-style losses (e.g.
  the TAID λ schedule).
- For top-K methods, prefer **importance-corrected** sampling (RS-KD) over
  naive top-K caching (which is biased; see Anshumann et al.).
- Numerical stability: clamp probabilities/logsumexp; mask `-100` labels
  before the reduction; consider `bf16`-friendly arithmetic (no inf/nan).
- The student is a base model (not instruct), so the chat template used
  in training is the simple `Question: … Answer: …` format defined by
  the data collator. Be careful not to over-fit to the prompt template.
- Wall-clock budget per `test`: ~4h on 2 GPUs for training + ~45 min per
  eval split. Plan loss variants that are cheap to compute per step.
