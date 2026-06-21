# Changelog - IMPALA strict review

- `methods/impala/results/context.md:59` removed the posterior empirical claim that lag degradation was monotonic in controlled experiments, replacing it with the pre-method distribution-mismatch constraint.
- `methods/impala/results/reasoning.md:11` softened the controlled-lag result language into an in-frame expectation and removed the hindsight comparison against epsilon correction as an observed outcome.
- `methods/impala/results/reasoning.md:48` fixed the contraction proof's non-negativity step by conditioning on the history through `x_t` before using `E_mu[rho_t | x_t] <= 1`.
- `methods/impala/results/reasoning.md:146` added `off_policy_targets(...)` so the final code fills the `context.md` scaffold slot directly.
- `methods/impala/results/reasoning.md:171` changed the learner code to consume `(value_targets, pg_advantages)` from `off_policy_targets`, matching the context scaffold.
- `methods/impala/results/answer.md:193` added the same scaffold-compatible `off_policy_targets(...)` wrapper around the reference-faithful `from_logits` implementation.
- `methods/impala/results/answer.md:221` changed `learner_loss` to use the scaffold wrapper and pass `value_targets - values` into the canonical baseline loss.
- `methods/impala/notes/synthesis.md:105` corrected the same contraction-proof expectation step in the pre-existing synthesis notes.
- `methods/impala/notes/source_matrix.md:1` added the strict source matrix covering primary, ancestors, explainer, code, and self-account search artifacts.
- `methods/impala/notes/discovery_synthesis.md:1` added the source-grounded reconstruction and audit notes.
- `methods/impala/refs/self_accounts/search_log.md:1` documented the author self-account search and the unusable Reddit retrieval attempt.
- `methods/impala/results/.codex_review.json:1` refreshed the review marker for this completed in-place repair pass.
