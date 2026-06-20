Measured result — `construct:greedy-lex`, deterministic (no seed). Every returned set was checked
by the `O(|cap|^2 n)` verifier, and independently by an `O(|cap|^3)` triple scan for `n ≤ 6`; all
pass.

| `n` | `|cap|` | valid | brute-check | known max | gap to max |
|---|---|---|---|---|---|
| 1 | 2 | ✓ | ✓ | 2 | 0 |
| 2 | 4 | ✓ | ✓ | 4 | 0 |
| 3 | 8 | ✓ | ✓ | 9 | −1 |
| 4 | 16 | ✓ | ✓ | 20 | −4 |
| 5 | 32 | ✓ | ✓ | 45 | −13 |
| 6 | 64 | ✓ | ✓ | 112 | −48 |
| 7 | 128 | ✓ | (skip) | 236 | −108 |

Reference points: trivial `2^n` floor, proven optima `20/45/112/236` at `n = 4..7`, FunSearch
record `512` at `n = 8`.

Notes: the cap size is exactly `2^n` at every `n`, confirming the rigidity predicted — lexicographic
greedy reproduces the trivial power-of-two baseline and nothing more. It is optimal only at `n = 1,
2` (where `2^n` *is* the optimum); from `n = 3` on the gap to the optimum opens and widens
monotonically (`−1, −4, −13, −48, −108`), exactly because `2^n` grows far slower than the true
`~2.756^n`. Validity is never in question — the construction is correct by design — so the entire
deficit is attributable to the *order*. The geometry-blind counting order is the whole weakness, and
the cheapest way to attack it is to try many orders and keep the best, which is the next rung.
