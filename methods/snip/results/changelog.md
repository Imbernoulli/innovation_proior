# Changelog

- 2026-08-18 `results/reasoning.md` — epistemic fix (svfix pass on
  2026-08-18): the earlier svfix commit (`svfix(D_candidate)`, grounding the
  joint (c,w)-optimization dismissal in an OpenReview B1VZqjAcYX author
  rebuttal about alternating-optimization instability) had the narrator
  *run* an alternating (c,w) optimization scheme and *report* its outcome
  ("in some runs this lands on a sparse network with accuracy close to the
  reference net"; "run it again, nudge the seed, and the two blocks start
  chasing each other ... the loss curve stops behaving like ordinary
  training"). That is the method's own not-yet-run experimental result
  stated in proposal voice, which the frame rule forbids regardless of
  whether the underlying claim is historically documented (results belong
  only in trajectory observation turns). Rewrote the passage to keep the
  alternating-scheme *design* (freeze c, step w; freeze w, step c; back and
  forth) and a structural, on-page *reasoning* for why it is at risk of not
  settling (each block's update chases a target the other block just moved,
  with no fixed-point guarantee absent extra machinery), without claiming
  any of it was executed or observed. Landing (park the joint optimization,
  fall back to the decoupled dial c_j and its per-connection sensitivity)
  unchanged and was already justified independently of the removed
  observation — not contingent on it. No corresponding svfix diff existed
  in answer.md/train_answer.md this pass, so both left untouched.
