# Changelog

- `results/context.md:3` Rewrote the setup as exactly five `##` sections with
  pre-method framing only.
- `results/context.md:39` Kept VAE, MDN, and CMA-ES as prior primitives while
  avoiding the final V/M/C wiring.
- `results/context.md:73` Replaced answer-leaking implementation details with a
  TODO scaffold.
- `results/reasoning.md:1` Rewrote the reasoning as first-person present-tense
  prose with no markdown headers.
- `results/reasoning.md:7` Corrected the MDN likelihood derivation to the
  factorized scalar mixture used by the official implementation.
- `results/reasoning.md:14` Corrected controller feature cases and parameter
  counts for car racing and Doom.
- `results/reasoning.md:20` Reframed temperature as the anti-exploitation knob
  with the implementation's sampling behavior.
- `results/answer.md:18` Corrected VAE math to `logvar`, `exp(logvar / 2)`, and
  KL tolerance.
- `results/answer.md:43` Replaced the previous joint diagonal-GMM code with the
  official scalar-per-coordinate MDN-RNN shape and NLL.
- `results/answer.md:61` Corrected Doom to predict restart/done only, with
  weighted BCE and survival reward supplied by the wrapper.
- `results/answer.md:83` Corrected the car and Doom controller cases, action
  post-processing, and parameter counts.
- `results/answer.md:122` Added reference-faithfulness notes for signs, shapes,
  and environment-specific cases.
- `notes/source_matrix.md:3` Added the strict evidence matrix with primary,
  ancestor, explainer, self-account, and code artifacts.
- `notes/source_matrix.md:17` Added the author self-account search log.
- `notes/discovery_synthesis.md:16` Added math/sign/constant audit notes.
- `notes/discovery_synthesis.md:64` Added canonical-code faithfulness findings.
- `notes/discovery_synthesis.md:89` Documented posterior/hindsight leak cleanup
  and scaffold-purity checks.
- `notes/discovery_synthesis.md:105` Documented the missing strict-check script
  and fallback structural checks.
- `results/.codex_review.json:1` Replaced stale errored review metadata with an
  explicit not-run record.
- `notes/strict_check_output.txt:1` Recorded the failed strict-checker lookup.
