# Changelog

- 2026-08-19 `methods/signsgd/results/reasoning.md` toy-quadratic density-ratio paragraph: removed a
  self-supplied experimental observation (invented final losses `0.025`/`0.020` at invented learning
  rates `0.03`/`0.1`, contradicting the primary's own reported values `0.001`/`0.01` for this exact
  toy problem) and rewrote it as hypothesis -> discriminating-experiment design -> opposing
  predictions -> decision rule, per the fix-prompt's HARD RULE against stating experiment outcomes
  claimed to be run mid-reasoning. Propagated the matching fix to the "just demonstrated"
  back-reference in the following paragraph. See `notes/sources.md`.
- 2026-08-19 `methods/signsgd/results/reasoning.md`, `results/answer.md`, `results/train_answer.md`
  real-network gradient/noise-density passage: removed the assertion that the Welford
  density-measurement finding (gradient and noise densities are dense and of the same order on
  Resnet-20/CIFAR-10) was personally observed by the narrator/write-up; this is the primary's own
  reported experimental result and per the HARD RULE may not be asserted as fact in any single-turn
  channel. Rewrote all three to state the measurement protocol and the two opposing predictions it
  discriminates between, without asserting the outcome. See `notes/sources.md`.
- 2026-08-19 `methods/signsgd/notes/sources.md` added, recording the quality-gate verification that
  the TRIAGE-hinted decisive step (coordinate-wise smoothness + Gauss's-inequality sign-flip bound)
  is genuinely derived on the page and backed by primary's own self-contained proofs — left
  untouched, `sound_as_is` for that step — and the two self-supplied-observation fixes above.
