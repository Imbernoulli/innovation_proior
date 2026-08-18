# Changelog

## 2026-08-18 — svfix(W3_notes_unclear)
- **Fixed**: `results/reasoning.md` para 15 (the alpha-beta-vs-MCTS averaging
  illustration) stated the `K=50` max-value as `≈ +1.09` in one sentence, then
  restated the same `K=50` point as `1.04` two sentences later in the explicit
  `K=5,50,500 -> 1.04, 1.04, 0.87` sequence — an internal self-contradiction.
  Changed the first mention from `≈ +1.09` to `≈ +1.04` so both statements of the
  `K=50` case agree.
- **Not rewritten**: the decisive step (`π ∝ N^{1/τ}` as a provable policy
  improvement over the network's prior `p`) was reviewed against TRACK
  `W3_notes_unclear`. Real search (web + on-disk + `SELF_ACCOUNT_SOURCES.md`) found
  no self-account/interview/thesis material beyond the primary paper that
  specifically grounds this claim without becoming a name-drop; see
  `notes/sources.md` for the full search log and the two adjacent ancestor papers
  that were found but not used. Outcome: `no_source_found`.
