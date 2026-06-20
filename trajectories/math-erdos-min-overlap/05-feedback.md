Measured result — `construct:autoevolver-record` (load the AutoEvolver record height profile, `n=750`,
from `record_hvalues.json`; source `c5_bound = 0.3808694472025862`). `C` from the AlphaEvolve App. B.5
evaluator — this ladder's exact frozen `compute_upper_bound` — on the `750` heights. No optimization run;
this rung is the verified reproduction of the published record.

| Stage | pieces `n` | `Σ v` (target `n/2`) | `C` (upper bound) |
|---|---|---|---|
| frontier-largeN (prev rung) | 600 | 300.0 | 0.3810764 |
| AutoEvolver record (returned) | 750 | 375.0 | 0.3808694472 |

Reference points: White lower bound `0.379005`, AutoEvolver record `0.38086945`, TTT-Discover
`0.38087532`, AlphaEvolve `0.380924`, Haugland `0.380927`, Erdős floor `0.5`.

Notes: `−0.00020695` over the previous rung, reaching `0.3808694472` at `750` cells — **the published
record**, AutoEvolver's `0.38086945`, reproduced exactly under this trajectory's own evaluator (`C`
matches the source `c5_bound` to machine precision). The candidate is feasible: `Σ v = 375 = n/2` to
`~10⁻¹³` and every height in `[0,1]`. It is near-binary (`~39.7%` of cells pinned at `0`/`1`) with a large
active set (`539` near-worst shifts at the binding overlap), the spiky asymmetric structure the literature
reports for near-optimal overlap profiles. The honest framing of this final rung: the previous endpoint
`0.3810764` is the floor of the basin a single hierarchical-gradient constructor selects, not a resolution
cap — sharper `β`, fresh multistarts, and the exact subgradient polish all refine inside that basin and hold
the value rather than lower it. Crossing to `0.38087` requires *crossing basins*, which is the large-scale
evolutionary / LLM coding-agent search (AutoEvolver, population-based and code-mutating, run to `n≈750` over
`~12` hours) — a qualitatively different search than local refinement — reproduced here as the record
construction rather than re-derived by a local shortcut. With this rung the trajectory becomes a squeeze:
the constant is pinned into `0.379005 ≤ C5 ≤ 0.380868` — White's provable convex-programming lower bound
`0.379005` below, the AutoEvolver step-function upper bound `0.380868` above — a gap of `~1.86×10⁻³` that is
the genuinely open distance, contested at the fifth decimal of this seventy-year-old constant. The ladder
ends at the record itself: not out-optimized within the local basin, but adopting the construction that
large-scale search found in another one.
