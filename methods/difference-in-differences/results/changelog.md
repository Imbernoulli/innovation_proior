# Changelog

- `methods/difference-in-differences/results/answer.md:16` added the ATT plus non-parallel-trends-bias decomposition, making the sign and sign-flip cases explicit.
- `methods/difference-in-differences/results/answer.md:36` added all four saturated-regression cell means and verified that the interaction coefficient equals the treated change minus comparison change.
- `methods/difference-in-differences/results/answer.md:47` added the two-wave first-difference implementation and the Card-Krueger `NJ_i`/`GAP_i` cases.
- `methods/difference-in-differences/results/reasoning.md:15` added in-frame reasoning for why first-differenced microdata and exposure-coded treatment preserve the same estimator.
- `methods/difference-in-differences/refs/final_artifact/did-estimand.md:20` added the potential-outcomes decomposition and the non-parallel-trends bias sign caveat.
- `methods/difference-in-differences/refs/final_artifact/did-estimand.md:40` added all four regression cells.
- `methods/difference-in-differences/refs/final_artifact/did-estimand.md:49` added the canonical first-difference implementation and Card-Krueger survey timing/exposure details.
- `methods/difference-in-differences/notes/discovery_synthesis.md:9` corrected the Card-Krueger timing constants and recorded the primary-paper first-difference implementation.
- `methods/difference-in-differences/notes/discovery_synthesis.md:11` added the source-grounded ATT plus bias-term synthesis.
- `methods/difference-in-differences/notes/source_matrix.md:5` tightened the primary-source evidence row around survey timing and the `Delta E_i`/`GAP_i` regression implementation.
- `methods/difference-in-differences/notes/source_matrix.md:8` recorded Cunningham's ATT plus non-parallel-trends-bias decomposition as load-bearing evidence.
- `methods/difference-in-differences/results/.codex_review.json:4` corrected the review metadata to avoid claiming that the absent strict checker ran.

Verification:

- `detect_leakage.py` on `results/context.md`, `results/reasoning.md`, and `results/answer.md`: 0 regex suspects.
- `lint_inframe.py` on the same files: 0 hits across all categories.
- Structural shell check: `context.md` has exactly five `##` sections, `reasoning.md` has zero markdown headers, and `context.md` has zero target-name hits.
- `scripts/check_strict_method.py`: not present in this workspace; this is recorded in `.codex_review.json`.
