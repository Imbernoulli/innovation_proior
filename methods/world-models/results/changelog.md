# Changelog

- 2026-08-18 `results/reasoning.md` — epistemic correction: the 2026-08-18
  svfix pass above had the narrator run a controller against the dream and
  report the outcome ("running a controller against the dream shows me it is
  not hypothetical... τ=0.1: dream ~2086, real 193, worse than random ~210;
  τ=1.15: dream 918, real 1092"). A single-turn proposal's own experiments
  have not happened yet, so that observation and its numbers are removed.
  Kept: the hazard hypothesis, the mode-collapse mechanism now stated as a
  prediction ("the mixture could collapse... any policy trained against it
  would carry the same blind spot"), the τ knob math, and the decision rule
  ("I can only settle it by sweeping τ... measuring real-environment survival
  against a random-policy baseline: whichever τ gives the best real transfer
  is the one I ship"). answer.md/train_answer.md were not touched by the
  svfix commit (`git diff` confirms) so are out of scope and left as-is.

- 2026-08-18 `results/reasoning.md` — grounded the temperature/exploit paragraph
  (previously "that creates a hazard I have to take seriously... the policy
  may discover hidden-state trajectories where projectiles disappear or never
  launch") in the self-account material already on disk at
  `refs/self_accounts/worldmodels_github_io.html` ("Cheating the World Model"):
  the failure is now the documented mixture mode collapse at τ=0.1 (monsters
  never form/fire, dream score ~2086 but real-environment transfer only 193,
  worse than the ~210 random-policy baseline), and the resolution is closed
  with the actual swept numbers (τ=1.15: dream 918, real 1092) instead of an
  unresolved "I can only settle this by sweeping" hedge. Landing (temperature
  knob, τ≈1.15 best transfer) unchanged — no factual error found, so
  answer.md/train_answer.md left as-is. See `notes/sources.md` for the quote
  and provenance.

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
