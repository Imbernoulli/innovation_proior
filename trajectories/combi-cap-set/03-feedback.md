Measured result — `construct:structured-priority`, deterministic. Every returned cap verified by
the `O(|cap|^2 n)` check (all pass).

| `n` | `|cap|` | valid | lex floor | random multi-start | known max |
|---|---|---|---|---|---|
| 4 | 18 | ✓ | 16 | 20 | 20 |
| 5 | 36 | ✓ | 32 | 39 | 45 |
| 6 | 64 | ✓ | 64 | 77 | 112 |
| 7 | 138 | ✓ | 128 | 147 | 236 |

Reference points: `2^n` floor, proven optima `20/45/112/236`, FunSearch record `512` at `n = 8`.

Notes: the structured priority beats the lexicographic floor at `n = 4, 5, 7` (`+2, +4, +10`) and
ties it at `n = 6`, but it does **not** consistently beat best-of-thousands random multi-start —
`18 < 20`, `36 < 39`, `64 < 77`, `138 < 147` — it lands in the same band and is in fact *below* the
random best at every `n` shown. This is the intended lesson, made concrete: having a structured
priority is not enough; it must be the *right* priority, and a human guessing which symmetries to
reward (reflection pairs + a weight-mod-3 layer) cannot reliably out-do brute sampling of orders.
The greedy-priority *skeleton* is clearly correct — it is the exact machinery the strong
constructions use — but the hand-designed priority plugged into it plateaus. What reaches the
records is a priority *discovered by search* over the function space, tuned to the specific `n`,
encoding regularities no one would hand-write. That is the endpoint: the FunSearch-evolved priority.
