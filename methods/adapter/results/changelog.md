# Changelog

## 2026-08-18 — svfix(D_candidate)

Grounded the near-identity-initialization decisive step (reasoning.md) beyond the primary
paper. Found the official code repo `google-research/adapter-bert` and, on it, GitHub issue
#10 ("How a near-identity initialization is implemented"), answered by paper co-author Andrea
Gesmundo (`agesmundo`): the shipped `init_scale=1e-3` default is chosen to be "smaller than
default initialization," pushing the "residual adapters" output "closer to zero."

Cross-checked against the repo's `modeling.py`: BERT's own dense/attention/embedding layers
default to `initializer_range=0.02`. This gives a concrete, checkable number that reasoning.md
did not previously have: relative to that 0.02 floor (not just relative to the activation
|x|, which is all the trace computed before this fix), the s=1e-2 default reported for the
main experiments is only ~2x smaller, while the code default s=1e-3 is ~20x smaller. Rewrote
the tail of the initialization-scale paragraph in reasoning.md to derive and use this
comparison as the real reason to fix s=1e-3 as the code default, rather than the previous
vague "more conservative."
Propagated the same 0.02-vs-1e-2-vs-1e-3 comparison into answer.md and train_answer.md so the
three files stay consistent.

No factual errors found in the existing derivation, code, or landing; init_scale=1e-3 as the
code default was independently re-confirmed correct against the actual released
`modeling.py` (`feedforward_adapter(..., init_scale=1e-3)`).

Sources: methods/adapter/notes/sources.md, methods/adapter/refs/adapter-bert_github_issues.txt.
