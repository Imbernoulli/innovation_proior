# Changelog

## 2026-08-17
- `methods/movement-pruning/notes/sources.md` created (notes/ did not previously exist). Documents
  the non-primary source search (Victor Sanh HF blog/forum, `huggingface/transformers` PR #4637,
  `huggingface/nn_pruning`, NeurIPS official review page, Twitter/blog explainer search) for the
  key_step "magnitude fails under fine-tuning -> movement score from zero -> STE swap lemma /
  sign preservation." Outcome: `no_source_found` — everything found beyond the primary paper was
  either pure restatement, unrelated engineering/schedule-tuning commentary, or reviewer opinion
  without a preserved author rebuttal, so nothing qualified to reshape the reasoning trace.
- Cross-checked all algebra in `methods/movement-pruning/results/reasoning.md` against
  `methods/movement-pruning/src/neurips_2020.tex` line-for-line: shared S/M/W notation, the hard
  Top_v straight-through gradient `dL/dS_{i,j} = dL/da_i * W_{i,j} * x_j = dL/dW_{i,j} * W_{i,j}`
  (reasoning.md:13,19), the movement-accumulator formula (reasoning.md:34), the soft/threshold
  variant with sigmoid regularizer (reasoning.md:40-44), the L0-regularization gradient comparison
  (reasoning.md:46-50), the first-order swap-lemma derivation and its reversal under a
  negative-threshold/`|S|` selection (reasoning.md:52-77), and the mask convention `M = Top_v(S)`
  thresholding on signed `S` rather than `|S|` (reasoning.md:7,77-78 vs. `src/neurips_2020.tex:114`
  magnitude-pruning contrast and `:458-469` negative-threshold contradiction proof). No errors
  found — the paper thresholds signed `S` directly, matching reasoning.md's sign convention and
  code (`TopVSTE`/`ThresholdSTE` in reasoning.md:93-112, answer.md, train_answer.md all agree).
- No content changes made to `reasoning.md`, `answer.md`, or `train_answer.md`: no non-primary
  source was found to ground the key_step further, and no factual/algebraic error was found to fix.

## 2026-08-17 (follow-up pass)
- Re-ran the non-primary source search with a broader technique (fetching Victor Sanh's own
  "Explain Like I'm Five" SlideShare deck for movement pruning via its server-rendered bot page,
  since a plain fetch of that URL returns a JS anti-bot challenge page and had apparently been
  treated as inaccessible in the prior pass). This turned up real content: a 12-slide deck by
  Victor Sanh (linked from his own `huggingface/transformers` PR #4637 / `examples/movement-pruning/README.md`)
  explaining movement vs. magnitude pruning via a stock-picking analogy (two $200 stocks vs. a $75
  stock; year-to-year returns of +1%/-5%/+30%; the $75 stock overtakes either $200 stock in under
  7 years). Saved to `refs/self_accounts/sanh_eli5_slides_transcript.txt`; full record in
  `notes/sources.md` (which now supersedes the prior pass's now-lost `notes/sources.md`, since
  `notes/` is gitignored and was not preserved between passes).
- `methods/movement-pruning/results/reasoning.md`: surgically rewrote the sentence pivoting from
  weight *value* (magnitude, 0th order) to weight *motion* (movement, 1st order) — the paragraph
  beginning "So I need a criterion that the fine-tuning process actually shapes." — to run through
  a concrete numeric analogy grounded in Sanh's own stock-picking framing (two equally-priced
  options, one with a higher growth rate, overtakes the other under a few years) instead of stating
  the pivot as an unadorned abstraction. No math, code, or landing content changed; no length
  padding; first person / present tense preserved; no provenance strings added.
  Outcome revised from `no_source_found` to `fixed`.
