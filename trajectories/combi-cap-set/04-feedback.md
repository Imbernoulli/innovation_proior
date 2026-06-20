Measured result — `construct:funsearch-evolved` (the priority discovered by FunSearch, verbatim
from the repo), deterministic. Every returned cap verified by the `O(|cap|^2 n)` check; `n ≤ 6` also
cross-checked by the independent `O(|cap|^3)` triple scan. All pass.

| `n` | `|cap|` | valid | brute-check | lex floor | structured | random best | known max |
|---|---|---|---|---|---|---|---|
| 4 | 16 | ✓ | ✓ | 16 | 18 | 20 | 20 |
| 5 | 32 | ✓ | ✓ | 32 | 36 | 39 | 45 |
| 6 | 64 | ✓ | ✓ | 64 | 64 | 77 | 112 |
| 7 | 128 | ✓ | — | 128 | 138 | 147 | 236 |
| **8** | **512** | **✓** | — | 256 | — | — | **512** |

Reference points: `2^n` floor, proven optima `20/45/112/236`, FunSearch record **`512` at `n = 8`**
(previous best construction `496`).

Notes: exactly as predicted, the discovered function is **dimension-specialized**. At `n = 4..7` it
lands back at the `2^n` floor (`16, 32, 64, 128`) — at or *below* both the hand-designed structured
priority and best-of-thousands random multi-start — because it was evolved specifically for `n = 8`
and carries no guarantee elsewhere. Its entire value is at `n = 8`, where it builds a cap of
**exactly `512`**, valid, matching the FunSearch record and improving the previous best construction
of `496`.

Verification of genuineness (two independent checks, both pass):
- The `n = 8` cap has size `512` and passes the incremental cap verifier — a real cap, not a claimed
  size.
- The set of `512` points produced by the greedy fill **equals, as a set, the explicit `512`-cap
  stored in the FunSearch repository** (`cap_set/n8_size512.txt`): all 512 points coincide. So the
  run reconstructs the exact object the search discovered, not merely the number.

This is the honest ceiling. `512` at `n = 8` is reproduced, not beaten — it *is* the record this
construction set. The companion discovered function for `n = 9` (also in the repo) reaches the
known-best `1082` in nine dimensions by the same skeleton (separately confirmed: `solve(9) → 1082`,
valid), corroborating that the mechanism generalizes. There is no rung above this, because going
further would mean running the evolutionary search itself (millions of LLM samples under evaluation)
to discover a new priority — a different computation, not a single constructor. The ladder's arc:
blind order `→ 2^n`; random multi-start `→` best of a thin lottery (optimum only at `n = 4`);
hand-designed structured priority `→` same band, no higher; and only a priority *discovered by
program search* reaches the record `512`. The skeleton was right throughout; the discovery was the
function.
