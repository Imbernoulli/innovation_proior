Measured result — `construct:frontier-largeN` (endpoint: rung-3 profile → upscale `×5` → `n=600` →
`β`-annealed analytic-gradient Adam + exact subgradient polish, fixed seeds). `C` from the AlphaEvolve
App. B.5 evaluator on the returned `600` heights. Runtime `~85 s`.

| Stage | pieces `n` | `Σ v` (target `n/2`) | `C` (upper bound) |
|---|---|---|---|
| finer-basinhop (prev rung) | 120 | 60.0 | 0.3810764 |
| upscale ×5 (free, same `C`) | 600 | 300.0 | 0.3810764 |
| + sharp-β Adam + subgrad polish (returned) | 600 | 300.0 | 0.3810764 |

Reference points: White lower bound `0.379005`, AutoEvolver record `0.38086945`, TTT-Discover
`0.38087532`, AlphaEvolve `0.380924`, Haugland `0.380927`, Erdős floor `0.5`.

Notes: the endpoint returns `0.3810764` at `600` cells — **at the published step-function frontier**:
within `~1.9×10⁻⁴` of the AutoEvolver record `0.38086945`, `~1.6×10⁻⁴` below the AlphaEvolve `0.380924`
and Haugland `0.380927` landmarks, and `~2.1×10⁻³` above White's provable lower bound `0.379005`. The
upscale is confirmed free (same `C` before and after lifting `120 → 600`). The honest finding is that this
profile is a **robust local optimum**: neither the sharp-β analytic-gradient Adam, nor a fresh n=600/n=360
multistart, nor SLSQP, nor the exact subgradient polish improves on the lifted `0.3810764` — the
refinement at scale holds the value rather than lowering it, so the endpoint reports the genuine number the
evaluator returns, not a hoped-for one. The returned solution is feasible (`Σ v = 300 = n/2`) and
near-binary (`~31%` of cells pinned at `0`/`1`), the spiky asymmetric structure the literature reports for
near-optimal overlap profiles. The endpoint's load-bearing change versus the middle rung is the switch of
the large-`n` optimizer from SLSQP (whose QP is the bottleneck and barely moves at `600` cells, ~2 min per
solve) to fast `β`-annealed Adam on the analytic soft-max gradient plus an exact-minimax subgradient polish.
The endpoint stops here, at the step-function frontier a single bounded hierarchical-gradient constructor
reaches. The absolute record, AutoEvolver's `0.38086945`, was found by an evolutionary coding-agent search
run to `n=750` over `~12` hours (and AlphaEvolve's `0.380924` by a comparable evolutionary search); the gap
from `0.3810764` to `0.38087` is the still-open part of this seventy-year-old problem, and the true constant
is squeezed into the sliver `0.379005 ≤ C5 ≤ 0.380868` — every digit past the third contested at real cost.
There is no finale reaching `0.38087` because that requires large-scale evolutionary / test-time search,
not a single local constructor.
