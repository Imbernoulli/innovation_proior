# Changelog — gptq

## 2026-08-18 — svfix(epistemic)
- **Removed a self-supplied observation introduced by the svfix(D_candidate)
  pass** (commit `ebb0d5e00`, "arbitrary-order step now checked not argued").
  A single-turn method unit is a proposal: the method's own experiments have
  not happened at that point in the frame, so reasoning.md must not have the
  narrator run an experiment and report its result — real numbers or not.
  The prior pass replaced an unverified intuition ("greedy order barely
  separates from fixed order on giant layers... I'd want to confirm on real
  layers, not just argue") with a claimed BERT-base/SQuAD + OPT-125M/WikiText2
  greedy-vs-fixed-order run at 4-bit and 3-bit, complete with F1/perplexity
  numbers narrated as observed ("BERT-base F1 is 88.23 greedy versus 88.18
  fixed... the number settles it"). That is the narrator running the
  method's own diagnostic experiment mid-derivation and stating the outcome,
  which the frame does not allow regardless of whether the numbers are real.
- Rewrote the passage to keep the hypothesis (does per-row greedy order earn
  its cost on large layers), the discriminating-experiment design (same two
  models/datasets, greedy vs. fixed order, matched at both 4-bit and 3-bit
  grids), each hypothesis's prediction (4-bit: the two should track closely;
  3-bit: where greedy's early-rounding advantage should show up clearest if
  it matters at all), and the decision rule (whichever ordering survives the
  comparison at both bit-widths is the one built around) — without asserting
  an observed outcome.
- Landing (shared fixed column order + single downdated inverse) is now
  conditional on that comparison's result rather than pre-settled by it; the
  downstream paragraph already reads as conditional ("if I'm allowed to
  quantize all rows in the same fixed order..."), so no other text needed to
  change. The specific empirical question — does fixed order actually match
  or beat greedy on real transformer layers at 3-bit — is still open in
  proposal voice; it belongs in a trajectory observation turn.
- No other svfix-diff passages in this method (answer.md/train_answer.md
  untouched by the D_candidate pass); scope was this one paragraph.
