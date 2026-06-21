# RAdam Review Changelog

- `methods/radam/results/context.md:120` corrected the NMT benchmark line from `WMT'14` to the paper's `WMT'16 En-De` setting and kept context free of the target method name.
- `methods/radam/results/reasoning.md:11` corrected the exact scaled-inverse-chi-square moment domains: `E[sqrt(x)]` exists for `rho > 1`, and exact `Var[sqrt(x)]` is finite for `rho > 2`.
- `methods/radam/results/reasoning.md:17` separated the exact variance result from the delta-method approximation, making `rho > 4` the implementation/approximation threshold because `Var[x]` is finite only there.
- `methods/radam/results/reasoning.md:28` checked the rectifier sign and ratio: `r_t = sqrt(Var_inf / Var_t)`, so the early multiplier is `<= 1`; the inverted ratio would be wrong.
- `methods/radam/results/reasoning.md:34` aligned branch behavior with the canonical PyTorch implementation: adaptive branch at `N_sma >= 5`, optional momentum fallback under `degenerated_to_sgd=True`, default skipped inactive updates under `False`.
- `methods/radam/results/reasoning.md:36` fixed the `beta2 <= 0.6` edge case to distinguish the paper's momentum fallback from the canonical default skip behavior.
- `methods/radam/results/answer.md:30` changed the final answer's variance-domain statement from "finite only for rho > 4" to exact `rho > 2`, with `rho > 4` reserved for the stable approximation.
- `methods/radam/results/answer.md:45` rebuilt the optimizer code to match the canonical reference semantics: `degenerated_to_sgd=False` default, per-group buffer support, `N_sma >= 5`, cached step-size formula, sparse-gradient rejection, and direct parameter weight decay.
- `methods/radam/results/answer.md:98` kept PyTorch-compatible modern in-place calls while preserving canonical math for the EMA updates and update branches.
- `methods/radam/results/.codex_review.json:3` replaced the stale errored review marker with a completed review record for this run.
- `methods/radam/notes/source_matrix.md:5` added the evidence matrix with primary source, canonical code, ancestors, explainer, and self-account search sources.
- `methods/radam/notes/discovery_synthesis.md:18` recorded the corrected exact-versus-approximate derivation and the code-faithfulness findings.
- `methods/radam/refs/self_accounts/search_log.md:5` documented the author self-account search and the absence of a separate long-form author retrospective beyond the canonical README/author page.
- `methods/radam/notes/synthesis.md:37` updated the older synthesis note so it no longer contradicts the corrected deliverables on moment domains and canonical defaults.

Verification:

- `context.md` has exactly five `##` sections.
- `reasoning.md` has no markdown headers and no target-paper artifact phrasing.
- The Python code block in `answer.md` parses with `ast.parse`.
- A toy PyTorch smoke test matched canonical branch behavior: default `degenerated_to_sgd=False` skipped the first inactive step; `True` applied the momentum fallback.
