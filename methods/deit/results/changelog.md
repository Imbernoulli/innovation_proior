# Changelog

## 2026-08-18 — obs-fix(V)
- Removed narrator-run experiment outcomes from reasoning.md and answer.md/train_answer.md: the
  optimizer-choice numbers (SGD vs AdamW top-1), the augmentation knockout-ablation numbers (Mixup+CutMix,
  repeated augmentation, dropout), the hard-vs-soft distillation head-to-head, the distillation-token vs
  duplicate-class-token control (cosine-similarity numbers), the convnet-vs-transformer teacher
  comparison, and the bilinear-vs-bicubic fine-tune accuracy claim. All of these presented completed
  ImageNet-1k training/fine-tuning runs and their results as if the narrator had already executed them —
  not available to a single-turn proposal.
- Rewrote each as hypothesis -> discriminating test design (matched training budget, controls held fixed)
  -> prediction -> decision rule, keeping the mechanism reasoning (why AdamW should suit the architecture,
  why Mixup/CutMix should matter most, why the distillation token should stay distinct, why a convnet
  teacher should transfer more useful inductive bias, why bicubic should preserve positional-embedding
  norm) intact. Numbers removed; no content trimmed otherwise.
- Left untouched: the on-page deterministic computations (the tau^2 gradient finite-difference check
  against synthetic 1000-class logit vectors; the bilinear-averaging vector-norm geometry check) — these
  are desk-scale math/synthetic checks the narrator can actually do while writing, not claimed real
  experiments, per the obs-fix rule's allowed list.
- No change to the landing (final method + code): the distillation token, hard-label distillation,
  convnet teacher choice, and bicubic resize were already correct; only the justification/evidence framing
  changed from claimed observations to design + prediction + validation plan.
