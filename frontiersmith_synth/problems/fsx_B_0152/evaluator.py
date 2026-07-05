#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_B_0152 -- "Deep-Sea Cable Spare-Parts Positioning".

Family: constrained-OR (Frontier-Eng Inventory / multi-echelon safety stock),
skinned as a network of maintenance depots that hold spare submarine-cable parts
(repeaters, branching units, cable drums). The depots form a SUPPLY TREE rooted at
a central manufacturing hub; a part flows hub -> regional depot -> ... -> forward
depot, and each hop adds a deterministic transit/processing time. The net
replenishment lead time L_i at depot i is the cumulative time along its path from
the hub, so depots deeper in the tree face longer, riskier lead times.

Each depot i faces stochastic spares demand over its lead time, modelled as
Normal(mu_i, sd_i^2) with sd_i = sigma_i * sqrt(L_i). The candidate must choose a
SAFETY-STOCK quantity stock_i >= 0 at every depot. With safety factor
k_i = stock_i / sd_i, the expected unmet demand (backorders) at the depot is
    B_i = sd_i * Loss(k_i),   Loss(k) = phi(k) - k*(1 - Phi(k))   (standard-normal
loss function; decreasing in k). The composite objective to MINIMIZE is

    cost = sum_i [ h_i * stock_i  +  p_i * B_i ]      (holding + shortage penalty)

subject to a NETWORK service-level constraint (aggregate fill rate):

    fill = 1 - (sum_i B_i) / (sum_i mu_i)  >=  beta        (else INFEASIBLE -> 0)

This is a genuinely open-ended constrained trade-off: each depot alone has a
newsvendor optimum k*_i = Phi^{-1}(1 - h_i/p_i), but that node-wise optimum usually
violates the network fill-rate floor on "tight" instances, so stock must be
re-allocated across the tree -- pooling extra service onto cheap/low-variance depots
-- to meet beta at least total cost. The exact optimum is a 1-D Lagrangian
(water-filling on lambda), but many heuristics land in between.

The candidate is UNTRUSTED model output: it is run in an ISOLATED subprocess via
`isorun`, sees ONLY the public instance on stdin, and returns ONLY its answer on
stdout, so it can never reach the evaluator's frames / scorer / baseline.

Scoring (deterministic; no wall-time):
  baseline b = cost of the "gold-plated" design that stocks every depot to safety
               factor 4 (always feasible, heaviest holding cost).
  For a FEASIBLE answer with objective obj:  r = min(1, 0.1 * b / obj)
  -> the trivial gold-plate design maps to exactly 0.1; a design k times cheaper
     than baseline maps to min(1, 0.1*k). Infeasible / malformed answer -> 0.

CLI:  python3 evaluator.py <candidate.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import isorun


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
    def nxt(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)
    return nxt


# ----------------------------- standard normal -----------------------------
_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)


def _Phi(x):
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


def _phi(x):
    return math.exp(-0.5 * x * x) / _SQRT2PI


def _loss(k):
    """Standard-normal loss function L(k) = phi(k) - k*(1-Phi(k)), k>=0 (decreasing)."""
    if k < 0.0:
        k = 0.0
    return _phi(k) - k * (1.0 - _Phi(k))


# ----------------------------- instance family -----------------------------
def make_instances():
    """Deterministic, seeded. Returns [{'public':..., 'hidden':{}}].
    A random rooted supply tree; per-depot lead-time demand (mu, sigma), holding
    cost h (rising with depth), shortage penalty p, and a network fill-rate floor
    beta. 'loose' instances (large p multiplier + lower beta) let the pure
    newsvendor optimum stay feasible; 'tight' ones force network re-allocation."""
    specs = [
        # seed, N, beta_pct, pmul_lo, pmul_hi
        (101, 8, 92, 3, 10), (102, 9, 94, 3, 10), (103, 10, 93, 3, 10),
        (104, 8, 90, 20, 60), (105, 11, 92, 3, 10), (106, 9, 90, 20, 60),
        (107, 10, 94, 3, 10),
        # larger / held-out instances
        (108, 12, 95, 3, 10), (109, 13, 91, 20, 60), (110, 14, 96, 3, 10),
    ]
    out = []
    for seed, N, beta_pct, pmul_lo, pmul_hi in specs:
        r = _rng(seed)
        parent = [-1] * N
        depth = [0] * N
        for j in range(1, N):
            parent[j] = r(0, j - 1)            # random rooted tree, node 0 = hub
            depth[j] = depth[parent[j]] + 1
        t = [r(1, 4) for _ in range(N)]        # transit/processing time per hop
        L = [0] * N
        for j in range(N):                     # cumulative lead time along path
            L[j] = t[j] + (0 if parent[j] < 0 else L[parent[j]])
        mu = [0.0] * N; sigma = [0.0] * N; h = [0.0] * N; p = [0.0] * N
        for j in range(N):
            mu[j] = float(r(20, 80))
            sigma[j] = float(r(15, 55))
            hb = r(10, 30) / 10.0
            h[j] = hb * (1.0 + 0.25 * depth[j])            # deeper depots cost more
            p[j] = h[j] * (r(pmul_lo * 10, pmul_hi * 10) / 10.0)
        sd = [sigma[j] * math.sqrt(L[j]) for j in range(N)]
        public = {
            "N": N, "parent": parent, "L": L, "mu": mu, "sigma": sigma,
            "h": h, "p": p, "sd": sd, "beta": beta_pct / 100.0,
        }
        out.append({"public": public, "hidden": {}})
    return out


# ----------------------------- scoring -------------------------------------
def baseline(inst):
    """Gold-plated design: stock every depot to safety factor 4 (always feasible)."""
    p = inst["public"]
    N = p["N"]; sd = p["sd"]; h = p["h"]; pen = p["p"]
    KT = 4.0
    b = _loss(KT)
    return float(sum(h[i] * KT * sd[i] + pen[i] * sd[i] * b for i in range(N)))


def score(inst, answer):
    """Strictly validate the answer against the instance; return (ok, obj)."""
    pub = inst["public"]
    N = pub["N"]; sd = pub["sd"]; mu = pub["mu"]; h = pub["h"]; pen = pub["p"]
    beta = pub["beta"]
    if not isinstance(answer, dict):
        return False, None
    stock = answer.get("stock", None)
    if not isinstance(stock, list) or len(stock) != N:
        return False, None
    try:
        stock = [float(x) for x in stock]
    except (TypeError, ValueError):
        return False, None
    for s in stock:
        if not math.isfinite(s) or s < -1e-9:
            return False, None
    stock = [max(0.0, s) for s in stock]
    B = [sd[i] * _loss(stock[i] / sd[i]) for i in range(N)]
    tot_mu = sum(mu)
    fill = 1.0 - sum(B) / tot_mu
    if fill < beta - 1e-9:
        return False, None
    obj = sum(h[i] * stock[i] + pen[i] * B[i] for i in range(N))
    if not math.isfinite(obj) or obj <= 0.0:
        return False, None
    return True, float(obj)


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <candidate.py>")
        sys.exit(2)
    cand = sys.argv[1]
    insts = make_instances()
    vec = []
    for inst in insts:
        ans, st = isorun.run_candidate(cand, inst["public"], timeout=20)
        if st != "OK":
            vec.append(0.0); continue
        try:
            ok, obj = score(inst, ans)
        except Exception:
            ok, obj = False, None
        if not ok or obj is None or obj <= 0:
            vec.append(0.0); continue
        b = baseline(inst)
        r = min(1.0, 0.1 * b / max(obj, 1e-12))
        vec.append(r if (r == r and 0.0 <= r <= 1.0) else 0.0)
    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
