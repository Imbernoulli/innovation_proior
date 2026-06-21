# Changelog

- `methods/goldreich-levin/results/context.md:29` removed hindsight phrasing that effectively concluded the reduction and reframed it as the pre-method missing bridge.
- `methods/goldreich-levin/results/reasoning.md:9` made the sampled-label step faithful to the author proof by choosing polynomially many subset masks via `ell = ceil(log2(m+1))`, so enumeration remains polynomial.
- `methods/goldreich-levin/results/reasoning.md:11` tightened the recovery argument to use pairwise independence, Chebyshev-style concentration, and a union bound rather than an unqualified high-probability majority claim.
- `methods/goldreich-levin/results/answer.md:17` changed the theorem reduction wording from a bare `epsilon` predictor to a non-negligible-advantage predictor and added the polynomial sample/list-size parameterization.
- `methods/goldreich-levin/results/answer.md:29` clarified that candidate testing identifies the preimage when it is present in the enumerated polynomial-size list.
- `methods/goldreich-levin/notes/discovery_synthesis.md:21` and `methods/goldreich-levin/notes/discovery_synthesis.md:23` replaced ambiguous `r+e_i` / addition notation with explicit xor over `F_2`.
- `methods/goldreich-levin/notes/discovery_synthesis.md:27` and `methods/goldreich-levin/notes/discovery_synthesis.md:29` fixed the seed-mask constant accounting and changed the coordinate vote from `+ G(...)` to `xor G(...)`.
- `methods/goldreich-levin/notes/discovery_synthesis.md:31` and `methods/goldreich-levin/notes/discovery_synthesis.md:40` recorded the concentration/union-bound case and the `ceil(log2(m+1))` seed-mask construction in the synthesis notes.
- `methods/goldreich-levin/refs/final_artifact/goldreich_levin_theorem_artifact.md:19`, `methods/goldreich-levin/refs/final_artifact/goldreich_levin_theorem_artifact.md:22`, and `methods/goldreich-levin/refs/final_artifact/goldreich_levin_theorem_artifact.md:33` aligned the canonical proof artifact with Goldreich's later Algorithm A/A0 presentation.
- `methods/goldreich-levin/results/.codex_review.json:4` corrected the review metadata so it no longer claims that a missing strict checker was run.
- `methods/goldreich-levin/notes/strict_check_output.txt:3` through `methods/goldreich-levin/notes/strict_check_output.txt:9` recorded the unavailable checker and the manual replacement checks.
