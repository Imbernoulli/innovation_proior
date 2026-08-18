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
