# Changelog

- 2026-08-18 `results/reasoning.md` — grounded the decisive-step paragraph (naive
  separate-batch BatchNorm in the critic, traced with a self-generated numeric
  toy, "BatchNorm has a bad reputation in RL, though...") in the self-account
  material now on disk at `refs/bhatt2019_crossnorm_v1.pdf/.txt` — the same
  authors' 2019 predecessor paper under the same arXiv id (1902.05605v1,
  "CrossNorm: Normalization for Off-Policy TD Reinforcement Learning"). The
  obvious first move (forward `(s,a)` then separately forward `(s',a')`
  through the same BatchNorm critic, both in training mode) is now stated as
  the exact configuration that earlier paper names "a common pitfall,"
  diagnosed there with the specific, checkable mechanism "the statistics of
  each batch oscillate between two modes" — not a vague "bad reputation." The
  resolution (concatenate the two batches, one joint forward pass, then
  split) is now tied explicitly to that earlier paper's own resolution
  (there named CrossNorm), rather than reading as a clean forward derivation
  invented for this trace. The hand-computed numeric toy is kept as a
  secondary mechanistic check, not the sole justification. No factual errors
  found; landing (BatchRenorm twin critics, joint forward pass, no target
  network, UTD=1) unchanged, so answer.md/train_answer.md left as-is. See
  `notes/sources.md` for the quotes and provenance.

- 2026-08-18 (epistemic svfix) `results/reasoning.md` — the grounding pass above
  overshot: it recast the documented "common pitfall" (Bhatt et al. 2019) as the
  narrator's own past experimental history — "it's not hearsay to me — I hit this
  exact wall once already, dropping a BatchNorm layer straight into a DDPG critic
  ... I hadn't built any other mode for it," "destabilization I already watched
  happen once," and "this is exactly the fix that worked the first time I hit
  this, on the DDPG critic, when nothing fancier did." That is a narrator
  claiming to have run an experiment (naive separate-batch BatchNorm on a DDPG
  critic) and reporting its outcome (failure, then a fix that "worked") — a
  single-turn proposal has no such history to report. Removed all three claims;
  kept the documented mechanism stated as a known fact ("a documented failure
  mode, not just folklore" / "statistics ... oscillate between two modes"), the
  on-page numeric toy trace (unchanged, allowed as on-page computation), and the
  concatenate-then-split fix design (unchanged, allowed as design). No prediction
  or decision-rule sentence existed in this passage to preserve — this paragraph
  is diagnosis-of-a-known-pitfall + fix-design, not a discriminating experiment
  with a landing that depended on the removed text, so `needs_traj` is not
  triggered by this fix. Lint clean (`tools/lint_inframe.py`); answer.md /
  train_answer.md had no svfix changes in this method, so untouched.
