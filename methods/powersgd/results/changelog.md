# Changelog

## 2026-08-19 — svfix(W3_primary_plus_ancestors) quality-gate verification pass
- Quality-gate review of the decisive step (drop the optimal-but-nonlinear
  rank-r approximation — SVD/top-k/sign all fail because their encodings are
  nonlinear functions of the local gradient, so averaging the encodings is
  not the encoding of the average — and instead use one step of
  warm-started subspace iteration, `P = MQ` then `Q = M^T P`, with the two
  matrix multiplications separated by two ordered all-reduces so that
  distributed execution is provably identical to running the same update on
  the averaged gradient; reasoning.md paragraphs 2-6): confirmed genuinely
  derived, not asserted.
  - The "SVD is not linear" failure reason is stated verbatim in the primary:
    `refs/primary/powersgd_1905.13727.txt` ("computing the actual top
    eigenvectors of the stochastic gradients is very computationally
    expensive, and more-over is not linear (and hence does not support
    reduce)"), and the exact two-collective ordering the trace derives
    matches Algorithm 1's pseudocode line-for-line (`P<-MQ; ALL_REDUCE_MEAN(P);
    ORTHOGONALIZE(P); Q<-M^T P; ALL_REDUCE_MEAN(Q)`).
  - The trace's own numerical checks (2-worker toy matrices verifying the
    two-all-reduce procedure agrees with subspace iteration on the averaged
    gradient to machine precision; a "one-shot" single-all-reduce shortcut
    tested on the same toy matrices and shown strictly worse) are honest,
    checkable, self-contained linear-algebra computations on random toy
    inputs — not fabricated results about the method's own real training
    runs, and not something a single-turn reasoning trace needs external
    material to produce.
- TRIAGE (class B) pointed at `refs/self_accounts/` (epfml README,
  tvogels Observable benchmark) as on-disk-but-unused material. Both were
  reviewed: the README's rank-vs-compress_ratio / min_compression_rate
  config knobs are already the peripheral passage answer.md's
  "rank-interface question" paragraph matches (not the decisive step); the
  Observable benchmark page is almost entirely notebook/JS boilerplate with
  no narrative struggle content once stripped. Neither contains
  documented-struggle material that the decisive step is missing, so no
  source was grafted onto the already-grounded derivation.
- No rewrite to reasoning.md/answer.md/train_answer.md. No factual errors
  found in the reviewed passages. See `notes/source_matrix.md` for the full
  source inventory.
