# Changelog

- 2026-08-20 `methods/relationnet/results/answer.md` "Why it works / relation to prior methods"
  paragraph (~L61-63): removed the paper's own Fig-8 synthetic-ablation outcome stated as
  accomplished fact ("Euclidean-NN, a learned Mahalanobis metric, and even an
  MLP-embedding-then-Mahalanobis all fail, but the deep relation module solves it" —
  `refs/primary/oneshot_V23_LZ.tex:645-657`, an experiment the authors ran, not a fact derivable
  on the page). Reworded to the capacity claim the method unit is already allowed to make: any
  fixed-distance comparator (Euclidean/cosine/Mahalanobis, even after a learned nonlinear
  embedding warp) classifies by thresholding a distance, so its matched region is always a convex
  sublevel set and cannot carve a non-convex region such as an annulus, whereas a relation module
  with a hidden nonlinearity is not bound by that convexity. This mirrors the annulus/convex
  -sublevel-set argument already derived on the page in `results/reasoning.md` (own three-point
  proof, not an experimental report), so the "Why it works" section no longer asserts an
  experimental pass/fail result the method unit never ran in-loop. `results/train_answer.md`
  already used the capacity/derivation framing only and needed no change.

## Commit-scope note (added 2026-08-20)
This fix (and this changelog entry) was originally landed in commit `c72b5918b`, whose commit
message described only a `none` (3DGS) repair and did not disclose that the same commit also
carried this relationnet fix — an undisclosed scope violation (two methods bundled into one
commit, one of them unmentioned). The content itself is unchanged from that commit; this note
re-files it under its own accurately-scoped commit so the audit trail attributes relationnet's
repair to its own commit rather than `none`'s.
