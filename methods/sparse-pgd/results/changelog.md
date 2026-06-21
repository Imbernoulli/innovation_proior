# Sparse-PGD File:Line Changelog

Line numbers refer to the post-edit files.

- `results/context.md:1` rewrote the context title to avoid naming the target method.
- `results/context.md:3` replaced the research question with pre-method sparse-attack requirements only.
- `results/context.md:11` removed answer leakage about continuous masks and straight-through top-`k`; kept only the pre-method geometry/background.
- `results/context.md:19` replaced hindsight-heavy baselines with source-grounded prior attacks and their gaps.
- `results/context.md:31` reset evaluation settings around fixed-budget spatial `L0`, robust accuracy, and valid-candidate requirements.
- `results/context.md:39` replaced the scaffold with a generic empty sparse step instead of a partial target implementation.
- `results/reasoning.md:1` rewrote the reasoning as continuous first-person present-tense prose with no markdown headers.
- `results/reasoning.md:3` derived the `p * m` decomposition from the support/value split.
- `results/reasoning.md:7` fixed the attack objective sign convention to maximize loss.
- `results/reasoning.md:9` distinguished projected (`MaskingB`) and unprojected (`MaskingA`) magnitude-gradient cases.
- `results/reasoning.md:11` corrected the mask-gradient derivation, including channel aggregation and sigmoid derivative.
- `results/reasoning.md:21` corrected default step constants and documented the paper/code small-gradient cutoff mismatch.
- `results/reasoning.md:23` corrected the native validity projection order for `p`.
- `results/reasoning.md:25` corrected the loss description: paper CE vs reference low-confidence margin fallback.
- `results/reasoning.md:27` corrected best-candidate tracking to larger attack loss under ascent.
- `results/reasoning.md:29` corrected mask-stall reinitialization details.
- `results/reasoning.md:31` added the structured sparse extension without letting it dominate the unstructured core.
- `results/reasoning.md:35` replaced the old code with a reference-faithful unstructured core.
- `results/answer.md:1` rewrote the final artifact around the canonical Sparse-PGD formulation.
- `results/answer.md:11` corrected the core algorithm formulas and ascent signs.
- `results/answer.md:42` added constants and every projected/unprojected case.
- `results/answer.md:54` added the structured extension summary.
- `results/answer.md:58` replaced the old margin-descent implementation with reference-faithful code.
- `results/answer.md:190` added the invariant warning about flipping signs when using the opposite margin.
- `results/.codex_review.json:1` replaced the previous manual-review claim with an explicit independent-review-not-run record.
- `notes/source_matrix.md:1` added the strict evidence matrix with primary, ancestor, explainer, self-account, and code sources.
- `notes/discovery_synthesis.md:1` added source-grounded reconstruction notes and the audit rationale for the rewrite.
