# CRPO Deliverable Changelog

- `methods/crpo/results/context.md:48` softened the final setup section so it states the missing primal safe-RL requirement without leaking the current-feasibility switch or naming the method.
- `methods/crpo/results/reasoning.md:47` removed posterior implementation discussion from the in-frame derivation and kept the convergence argument in first-person present tense.
- `methods/crpo/results/reasoning.md:59` added the neural softmax case: temperature-scaled parameters, `alpha`, `K_in`, `eta`, and the fourth-root log factor.
- `methods/crpo/results/answer.md:36` expanded the tabular guarantee with the TD inner-loop scale, `eta`, `alpha`, and the output-from-`N_0` high-probability bounds.
- `methods/crpo/results/answer.md:55` added the neural update signs and theorem-scale parameters, including the finite-width `m^(-1/8)` log term.
- `methods/crpo/results/answer.md:71` anchored code faithfulness to OmniSafe commit `15603dd7a654a991d0a4648216b69d60b81a6366`, the upstream paths, and the byte-for-byte local match.
- `methods/crpo/notes/source_matrix.md:46` updated the code evidence rows with upstream OmniSafe commit/path provenance and local byte-for-byte verification.
- `methods/crpo/notes/discovery_synthesis.md:29` corrected the neural theorem synthesis beyond a loose `1/sqrt(T)` summary.
- `methods/crpo/notes/discovery_synthesis.md:38` recorded upstream OmniSafe verification and the single-constraint/on-policy implementation caveat.
- `methods/crpo/results/.codex_review.json:9` refreshed the review record to describe the manual audit against upstream OmniSafe and the retrieved source bundle.
- `methods/crpo/notes/strict_check_output.txt:1` updated the local structural check output; `scripts/check_strict_method.py` is still absent from this checkout.
