# Changelog — awq

## 2026-08-18 — svfix(D_candidate)
- **Fixed fabrication in results/reasoning.md (decisive-step scale-vs-error table).**
  The trace previously described a synthetic "200k trials on random 8-weight
  groups" Monte Carlo experiment with invented error/Δ'/Δ numbers, presented
  as something the scientist actually ran. None of those numbers correspond
  to src/ or any saved source. Replaced with the real experiment on
  OPT-6.7B recorded in src/text/3_approach.tex §3.2 and src/figure_text/tab_scale_study.tex:
  the actual measured proportion of groups with Δ'≠Δ, average Δ'/Δ, average
  (Δ'/Δ)(1/s), and WikiText-2 PPL for s ∈ {1, 1.25, 1.5, 2, 4}. This also
  fixes the trace's closing recap paragraph, which cited the fabricated
  "climb to 3×" figure — now cites the real measured value (~1.2× at s=4).
- **Grounded two previously unsourced design choices with author self-account
  (GitHub issue replies from Ji Lin / tonylins, AWQ first author, on
  mit-han-lab/llm-awq):** (1) why the search-space activation statistic is
  the per-channel *average* magnitude rather than the max — src/text/3_approach.tex
  doesn't explain this, only the code does it silently; issue #58 has Lin's
  stated rationale. (2) that combining weight magnitude into the scale
  search (on top of activation magnitude) was tried and dropped because it
  didn't move the loss — issue #110. Sources logged in notes/sources.md;
  saved locally at refs/github_issues_scale_design.txt.
- No change to the landing (method + code): the scale-search formula,
  α-grid-search procedure, and quantize_block implementation are unchanged.

## 2026-08-18 — svfix(epistemic)
- **Removed self-supplied observations introduced by the svfix(D_candidate)
  pass above.** A single-turn method unit is a proposal: the method's own
  experiments have not happened at that point in the frame, so reasoning.md
  must not have the narrator run the OPT-6.7B scale study (or the
  weight-vs-activation loss check) and report a result — real, sourced
  numbers or not. Both additions from the prior pass did exactly that:
  (1) the OPT-6.7B Δ'/Δ-vs-PPL table plus the two paragraphs narrating its
  readout ("PPL falls off a cliff...", "PPL is best at s=2 and gets worse
  again at s=4...") and the two downstream references to it ("climb past
  1.2×...", "the perplexity confirms it... the same run exposed..."); (2)
  "multiply the activation term by a weight-magnitude term... rerun it. The
  loss doesn't move" (issue #110 ablation, narrated as self-run).
- Rewrote both passages to keep the hypothesis, the discriminating-experiment
  design (model, scale range, matched 1% salient budget, metric), each
  hypothesis's prediction, and the decision rule — without asserting an
  observed outcome. The Δ'≥Δ inequality (scaling a channel can only enlarge,
  never shrink, a group's max) is kept as on-page algebra, not an
  experiment, and now carries the "≈ is optimistic" argument that the table
  used to carry empirically. The weight-magnitude paragraph now declines the
  extra term by inference from the earlier (pre-existing, untouched)
  large-weight diagnostic rather than by citing a rerun.
- Landing (whole-layer output-MSE objective; one-parameter α-grid-search
  scored by real per-layer output MSE at calibration time) is unchanged and
  remains justified by the algebra + hypothesis alone. The specific
  empirical question the removed table answered — does the predicted
  Δ'/Δ-vs-PPL crossover actually occur on OPT-6.7B — is still open in
  proposal voice; it belongs in a trajectory observation turn.
- Sources named in the prior entry (Ji Lin / tonylins GitHub issues #58,
  #110) still ground the avg-not-max and drop-the-weight-term design
  *choices* themselves — only the "I ran it and here's the number" framing
  was removed.

## 2026-08-18 — obs-fix
- **train_answer.md still had a narrator-run-experiment claim the epistemic
  pass above only fixed in reasoning.md**: "Diagnostic experiments show that
  keeping the weight channels that multiply the largest-magnitude input
  features in FP16 recovers accuracy, while keeping channels selected by
  weight magnitude barely helps" reported a result in the proposal's own
  voice (obs_scan_v3 flag `abl_shows`).
- Rewrote to hypothesis + discriminating-test design + prediction: saliency
  is set by activations not weight magnitude (the premise); the test is to
  quantize a layer low-bit, FP16-protect a small fraction of channels
  chosen by activation magnitude vs. by weight magnitude at the same
  budget, and compare recovered accuracy; the activation-selected set is
  predicted to recover most of the loss, the weight-magnitude set to barely
  beat a random selection, because raw weight size carries no signal about
  a channel's contribution to the output. No numbers removed (none were
  present beyond the qualitative claim); mechanism explanation (why
  activation magnitude matters) kept.
- answer.md's parallel sentence ("Keeping only a small activation-selected
  fraction of weight channels in FP16 is diagnostic evidence that these
  channels matter") was already result-free; left untouched. context.md
  unaffected — it carries no result claims here.
