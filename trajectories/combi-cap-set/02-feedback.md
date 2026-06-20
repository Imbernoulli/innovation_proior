Measured result — `construct:random-multistart`, `seed = 0`, best cap over `starts` random orders.
Every returned cap verified by the `O(|cap|^2 n)` check (all pass).

| `n` | starts | `|cap|` | valid | greedy-lex floor | known max | gap to max |
|---|---|---|---|---|---|---|
| 4 | 5000 | 20 | ✓ | 16 | 20 | **0** |
| 5 | 5000 | 39 | ✓ | 32 | 45 | −6 |
| 6 | 3000 | 77 | ✓ | 64 | 112 | −35 |
| 7 | 1000 | 147 | ✓ | 128 | 236 | −89 |

Reference points: `2^n` floor, proven optima `20/45/112/236`, FunSearch record `512` at `n = 8`.

Notes: random multi-start clears the lexicographic floor at every `n` (`+4, +7, +13, +19`) and
**reaches the proven optimum `20` at `n = 4`** — the first rung to hit a known-optimal value, bought
purely by sampling orders. But the gap to the optimum *grows* with `n` exactly as anticipated: `39`
vs `45`, `77` vs `112`, `147` vs `236`. The cap-size distribution under random greedy concentrates
below the optimum and its right tail thins out fast, so additional restarts buy little — this is a
lottery over orders, and a lottery cannot exploit the algebraic regularity the large caps are built
from. The method removed the *bias* of a fixed order but not its *blindness*: every order is still
uniform noise. To close the growing gap the order must be *biased* by a structured priority that
reflects the symmetry of `F_3^n`, which is the next rung.
