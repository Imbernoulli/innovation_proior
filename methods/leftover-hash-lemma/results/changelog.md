# Leftover Hash Lemma changelog

## 2026-08-17 — source-value recheck
- `results/reasoning.md`, the decisive "why 2-universality is enough" step: the trace told it as a
  forward search ("I look for the smallest property that can control the evidence of nonuniformity"),
  which is the reverse of how it happened. Impagliazzo's Berkeley PhD thesis, located this pass
  (`refs/self_accounts/impagliazzo_phd_thesis_prg_for_crypto_and_randomized_algorithms.pdf`,
  acknowledgements; quote in `notes/sources.md`), records that the lemma "was proved for a specific
  hash function, and Moni pointed out that any universal hash function would suffice". The passage
  now runs through that order: prove it for one concrete family, then notice — on its being pointed
  out — that the family appears in exactly one line of the proof, the bound on distinct inputs
  colliding, so the statement is about that pairwise bound and nothing else. The "a fully random
  function is more than the proof consumes" observation is kept, moved to where it belongs.
- Scoped an implicit overclaim: the trace presented `m = k - 2 log(1/epsilon)` as though the factor of
  two were an artifact of the L2-to-L1 conversion. It now says explicitly that this is an upper bound
  on what must be sacrificed by this route and that nothing here settles whether the two is intrinsic.
  (It is: Radhakrishnan–Ta-Shma's counting lower bound, saved in `refs/self_accounts/`, gives
  `k + d - m >= 2 log(1/eps) - O(1)` for extractors versus `log log(1/eps)` for dispersers — a 2000
  third-party result, so it is used to scope the claim rather than narrated in-frame.)
- No other factual errors found; the collision-probability split, the `(1/D)(1/K + 1/M)` bound, the
  Cauchy–Schwarz step and the landing are unchanged.
