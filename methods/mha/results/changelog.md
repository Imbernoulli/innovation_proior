# mha changelog

## 2026-08-18 — epistemic correction (svfix)
- `results/reasoning.md` (the paragraph right after the toy `d_k = 4/64/256` Jacobian simulation,
  before the `√d_k` fix is derived) and `results/train_answer.md` (the matching sentence in the
  variance-derivation paragraph): the prior svfix pass (`8ef962943`, grounding the "lines up with
  the reported behavior" gesture with real Britz et al. 2017 newstest2013 BLEU numbers) wrote the
  grounding in self-run-experiment voice — "I want to know whether the same crossover shows up
  when someone actually trains the two scores head-to-head instead of me simulating it. Sweep the
  score width from 128 up to 1024 ... at 128 the two land close together ... By 1024 the
  dot-product run has fallen to 18.22" — narrating the comparison as if being executed and
  watched in real time within the trace, rather than recalled as an already-published finding.
- Fix: reworded to reported-fact voice ("an empirical sweep ... reports the two landing close
  together ... the reported dot-product score fall[s] to 18.22 while additive still holds
  22.10"), matching how `context.md`'s pre-existing citation of the same Britz et al. 2017 result
  is already voiced (a prior-work fact that pre-dates this method, not the method's own
  observation). The real BLEU numbers (22.03/22.23 at width 128, 22.33/22.33 at 256, 18.22/22.10
  at 1024) and the honest hedge added by the later svfix pass (`613575700` — "consistent with the
  saturation account rather than proof of it", "one sweep ... is not a controlled ablation") are
  both kept unchanged; only the narrator's claimed authorship of the run was removed. The toy
  Jacobian simulation earlier in the same paragraph is the narrator's own on-page computation and
  was left untouched.
- The landing (dividing by `√d_k`) is not left unjustified: it was already fully derived from the
  variance algebra and the toy-row Jacobian numbers before this sentence, both of which are
  on-page computation, not observation. No trajectory-conversion queue entry needed.
- `results/answer.md` and `results/context.md`: no svfix diff touches either file for this
  method, so nothing to check there.
