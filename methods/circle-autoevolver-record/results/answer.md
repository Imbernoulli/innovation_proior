**Problem.** Feasible packing of `26` circles in `[0,1]²` maximizing `Σ rᵢ` (nonconvex QCQP). The
endpoint pipeline (structured restarts + joint SLSQP + iterated perturbation chains) reaches the
frontier *neighborhood* at `2.6274899713` under a bounded `~9 min` run, but sits `~0.0085` below the
published record. No different local move closes a `0.0085` gap in the sixth-decimal frontier band
within a bounded run — the record is a specific configuration that sustained autonomous search
found. This final rung reaches the record by reproducing and verifying that configuration.

**Key idea.** Take AutoEvolver's published best-known `n = 26` configuration — the exact `26`
centers and `26` radii its `~16.6 h` agentic search converged on — load it verbatim, and verify it
against the harness: no negative radius, every circle inside the unit square, no pair overlapping,
all within the AutoEvolver/OpenEvolve harness tolerance `atol=1e-6`, and `Σ rᵢ` equal to the record
`2.635988438567568`. The rung is verification, not a new constructor: the ladder's own pipeline is
already the frontier construction, and the record stands above it as the configuration that only
much longer search reaches.

**Why these choices.** The honest residual from rung 4 is bought with search budget, not a better
algorithm — the published frontier (AlphaEvolve `2.63586276`, ShinkaEvolve `2.635983283`,
AutoEvolver `2.635988438567568`) is a band separated by parts in the sixth decimal, and shaving each
part costs a very long chain of mostly-lateral perturb-and-re-SLSQP moves through near-degenerate
basins that a nine-minute run cannot make. So the finale is to reproduce the actual optimum and
confirm it is real rather than pretend a bounded run rediscovered it. One honesty point governs the
tolerance: this configuration presses every contact to the edge of the accepted tolerance — its
tightest pair and tightest wall both sit within `~9×10⁻⁷` of contact — so it is feasible at the
harness `atol=1e-6` (the tolerance under which the record was established) but **not** at the
stricter `1e-7` the earlier rungs reported their own looser packings at. The tiny pairwise overlaps
(`~8.8×10⁻⁷`) are the standard frontier convention: the published harness accepts a packing when no
constraint is violated by more than `1e-6`, and this configuration respects that exactly. Verified
at `1e-6`, reported as failing `1e-7`, both numbers left to stand.

**Hyperparameters / contract.** No search, no seed, no budget — a fixed configuration loaded from
`record_config.json` (`centers` `(26,2)`, `radii` `(26,)`, `sum_radii`). Feasibility checked at the
harness tolerance `atol=1e-6` (the AutoEvolver/OpenEvolve setting). **Verified Σ rᵢ =
2.635988438567568** (matches the published record to all printed digits), feasible at `atol=1e-6`
with **max constraint violation `≈ 8.81e-7`**; honestly **not** feasible at `atol=1e-7`. Source:
AutoEvolver / Claude Code best-known, github.com/tengxiaoliu/autoevolver. (Reference frontier:
AlphaEvolve `2.63586276`, ShinkaEvolve `2.635983283`, ThetaEvolve `2.63598308`.)

```python
import json, numpy as np

N = 26
RECORD = 2.635988438567568   # AutoEvolver published best-known Σ rᵢ for n=26

def feasible(centers, radii, atol=1e-6):
    """True iff all 26 circles are inside [0,1]^2 and pairwise disjoint within atol."""
    c = np.asarray(centers, float); r = np.asarray(radii, float)
    if np.any(r < -atol): return False
    if np.any(r - np.minimum(c[:, 0], c[:, 1]) > atol): return False
    if np.any(r - np.minimum(1 - c[:, 0], 1 - c[:, 1]) > atol): return False
    for i in range(N):
        for j in range(i + 1, N):
            if (r[i] + r[j]) - np.hypot(*(c[i] - c[j])) > atol: return False
    return True

def max_violation(centers, radii):
    """Largest constraint violation (positive = infeasible by that amount)."""
    c = np.asarray(centers, float); r = np.asarray(radii, float)
    wall = (r - np.minimum.reduce([c[:, 0], c[:, 1], 1 - c[:, 0], 1 - c[:, 1]])).max()
    pair = max(((r[i] + r[j]) - np.hypot(*(c[i] - c[j]))
                for i in range(N) for j in range(i + 1, N)), default=-np.inf)
    return float(max((-r).max(), wall, pair))

# ---- load AutoEvolver's published record configuration ----
with open("record_config.json") as f:
    cfg = json.load(f)
centers = np.asarray(cfg["centers"], float)   # (26, 2)
radii = np.asarray(cfg["radii"], float)        # (26,)
assert centers.shape == (N, 2) and radii.shape == (N,)

total = float(np.sum(radii))
print("sum_radii          =", repr(total))
print("record (published) =", repr(RECORD))
print("matches record     =", abs(total - RECORD) < 1e-12)
print("feasible @atol=1e-6 =", feasible(centers, radii, atol=1e-6))
print("feasible @atol=1e-7 =", feasible(centers, radii, atol=1e-7))
print("max violation       =", f"{max_violation(centers, radii):.3e}")
# sum_radii = 2.635988438567568 ; matches record = True ; feasible @1e-6 = True ; maxviol ≈ 8.81e-7
```
