Measured result — `construct:uniform` (flat half-density `v_i ≡ 1/2`), AlphaEvolve App. B.5 evaluator.
Deterministic (no seed).

| Instance | pieces `n` | `Σ v` (target `n/2`) | `C` (upper bound) |
|---|---|---|---|
| flat-10 | 10 | 5.0 | 0.500000 |
| flat-100 | 100 | 50.0 | 0.500000 |
| flat-600 | 600 | 300.0 | 0.500000 |

Reference points: White lower bound `0.379005`, AutoEvolver record `0.38086945`, TTT-Discover
`0.38087532`, AlphaEvolve `0.380924`, Haugland `0.380927`.

Notes: the bound is exactly `0.5` — Erdős's own 1955 value — for every piece count, confirming the
discretization-invariance argued by hand: the worst shift is `k = 0`, where all `n` cells self-align at
`(1/2)(1/2) = 1/4` each, giving `n/4`, rescaled by `2/n` to `1/2`. The profile sits exactly on the
balance constraint (`Σ v = n/2`) and is a strict, rigid floor: it has no internal degree of freedom, and
the overlap is locked to `1/2` by perfect self-alignment. The whole gap to the record (`0.5 → 0.3809`)
must be bought by search that breaks the flat symmetry into a near-binary profile — which is the next
rung.
