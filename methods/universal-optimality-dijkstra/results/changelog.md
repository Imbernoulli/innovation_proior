# universal-optimality-dijkstra changelog

## 2026-08-18 — source-value recheck
- `results/reasoning.md` (doubly-exponential inner-heap sizing, 3 passages): the trace made
  contradictory claims about which requirement forces doubly- rather than singly-exponential heap
  growth — one passage said singly-exponential is "fine for delete-min" but "kills the insert
  amortization"; a later passage said the opposite, that under singly-exponential "the sum still
  converges" (insert fine) "but the delete-min side fails." Working the algebra through for a
  singly-exponential schedule (`|H_i| ≈ 2^i`) shows neither bound is actually broken — both the
  delete-min log-ratio and the insert charge series stay finite-constant under that schedule too,
  by the same charging argument, just with a `Θ(log n)` rather than `O(log log n)` heap count.
  Reworded all three passages to drop the false "kills X / X fails" necessity claims and instead
  motivate doubly-exponential growth by what it actually, verifiably buys in one move: a clean
  multiplicative delete-min ratio (`/8`, worked out in the surviving math) and the `O(log log n)`
  heap-count cap that `results/reasoning.md`'s own later routing-task-2 section (single-machine-word
  bit vector) depends on — matching `src/main-JACM.tex`'s own stated rationale ("this gives us the
  working-set bound for DeleteMin operations, and it limits the number of inner heaps to be doubly
  logarithmic ... since the number of inner heaps is very small we can use bit-vector techniques").
  No formulas, invariants, lemma statements, code, or the landing changed — only the exploratory
  "why doubly- not singly-exponential" narrative around them. `answer.md`/`train_answer.md` do not
  repeat the contradictory claim (train_answer.md's parallel sentence already only says the
  singly-exponential ratio "collapse[s] to an additive gap", which is correct and needed no change).
- Grounding verdict on the decisive step (Iacono/Fredman wall forcing the new outer-heap
  construction): sound as-is, left untouched. `src/main-JACM.tex` states the identical wall
  verbatim — a locality-sensitive heap (Iacono's pairing-heap working-set bound) exists only
  "if DecreaseKey is not a supported operation," and Dijkstra needs `O(1)` DecreaseKey, "so these
  results do not help us here" (main-JACM.tex line 262). `refs/iacono2000-pairing.pdf` (already on
  disk) independently confirms the companion fact the trace cites — Fredman's result that pairing
  heaps cannot do `O(1)` DecreaseKey ("the amortized cost of Decrease-Key must be `Ω(log log n)`
  amortized", p.2/p.3). The obstacle is genuinely derived and both halves are backed by material
  already present before this pass; no new source was needed and none was grafted on.
