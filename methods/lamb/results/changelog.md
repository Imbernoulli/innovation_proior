# Changelog — lamb

## 2026-08-18 — svfix(epistemic)
- **Fixed the framing of a self-supplied observation introduced by the
  svfix(B_selfaccount_candidate) pass** (commit `c6f263f66`, "grounded the
  global-LR-fails two-layer example in real AlexNet conv1.1/fc6.1
  weight-vs-gradient-norm ratios"). A single-turn method unit is a proposal:
  the method's own experiments have not happened at that point in the
  frame, so reasoning.md must not have the narrator run an experiment and
  report its result — real numbers or not. The prior pass replaced the
  invented round-number two-layer example (weight norm 100/update norm 0.1
  vs. weight norm 1/update norm 1.0) with real per-layer weight/gradient
  norms from AlexNet (conv1.1 bias ratio ≈ 20, fc6.1 bias ratio ≈ 3690,
  sourced to Yang You's PhD thesis EECS-2020-136), but introduced it with
  "I logged the actual per-layer weight and gradient norms from an AlexNet
  run at batch 4k, first epoch" — a first-person claim that the narrator
  personally ran the training and logged the measurement.
- This example is not a test of LAMB (LAMB does not exist yet at this point
  in the derivation); it is the motivating diagnostic for why a single
  global learning rate breaks at all, and the numbers pre-date the method
  and exist in the record (You et al.'s earlier LARS-era work). That makes
  it a prior-work known fact, which is allowed — the only defect was the
  "I ran/logged it" framing. Reworded to "per-layer weight and gradient
  norms measured on an AlexNet training run at batch 4k, first epoch, show
  exactly this mismatch," dropping the first-person experiment claim while
  keeping every number, the ratio comparison, and the downstream algebra
  (`η ≈ 36.9`, the `0.181`-vs-`0.098` step-explosion check) unchanged.
- No numbers were removed, so the landing (per-layer trust-ratio update)
  and the worked-example algebra that follows it are unaffected; nothing
  needs to move to a trajectory observation turn for this passage.
- No other svfix-diff passages in this method (answer.md/train_answer.md
  untouched by the B_selfaccount_candidate pass); scope was this one
  sentence.
