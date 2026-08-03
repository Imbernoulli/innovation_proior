#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_B_0168 -- "Winter Provisioning of a Polar Research Base".

Family: constrained-OR (Frontier-Eng Inventory / multi-echelon safety stock),
skinned as a polar research programme. A coastal logistics HUB (node 0) resupplies a
tree of interior stations: main base -> traverse depots -> deep-field camps. A resupply
travels hub -> ... -> camp and each leg adds a deterministic transit time, so the net
replenishment lead time L_i at station i is the cumulative time along its path from the
hub. Deep-field camps therefore face the longest, riskiest lead times.

Over its lead time each station i faces stochastic consumable demand (fuel, food,
medical, spares) modelled as Normal(mu_i, sd_i^2) with sd_i = sigma_i * sqrt(L_i). The
candidate chooses a SAFETY-STOCK quantity stock_i >= 0 at every station. With safety
factor k_i = stock_i / sd_i the expected over-winter shortfall (backorders) is
    B_i = sd_i * Loss(k_i),   Loss(k) = phi(k) - k*(1 - Phi(k))    (standard-normal
loss function, decreasing in k). The composite objective to MINIMIZE is

    cost = sum_i [ h_i * stock_i  +  p_i * B_i ]      (holding + shortage penalty)

subject to TWO COUPLED service constraints:

  (1) a PROGRAMME-WIDE fill-rate floor
          fill = 1 - (sum_i B_i) / (sum_i mu_i)  >=  beta
  (2) for every designated LIFE-SUPPORT station c (crit[c]==1) a LOCAL fill-rate floor
          fill_c = 1 - B_c / mu_c  >=  alpha        (a med/power/fuel post may not run
                                                     out even if the network is fine)

Any answer violating (1) or (2), or malformed / negative / non-finite, is INFEASIBLE -> 0.

Why this is genuinely open-ended (and NOT a single-Lagrangian problem): the classic fix
for constraint (1) alone is uniform penalty water-filling on one multiplier lambda. But
that allocation starves expensive / high-variance life-support stations, breaking their
LOCAL floors (2). The min-cost feasible design must first pin each critical station to
its local requirement k_c^min = Loss^{-1}((1-alpha) mu_c / sd_c), THEN water-fill the
residual network deficit over the remaining stations. Node-wise newsvendor
k*_i = Phi^{-1}(1 - h_i/p_i) is cheapest but usually violates both floors; many
heuristics land in between.

The candidate is UNTRUSTED model output: it is run in an ISOLATED subprocess via
`isorun`, sees ONLY the public instance on stdin, and returns ONLY its answer on stdout,
so it can never reach the evaluator's frames / scorer / baseline.

Scoring (deterministic; no wall-time):
  baseline b = cost of the "top-off-every-locker" design that stocks every station to
               safety factor 4 (always feasible for both floors, heaviest holding).
  For a FEASIBLE answer with objective obj:  r = min(1, 0.1 * b / obj)
  -> the trivial top-off design maps to exactly 0.1; a design k times cheaper than
     baseline maps to min(1, 0.1*k). Infeasible / malformed answer -> 0.

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
    """Standard-normal loss L(k) = phi(k) - k*(1-Phi(k)), k>=0 (decreasing)."""
    if k < 0.0:
        k = 0.0
    return _phi(k) - k * (1.0 - _Phi(k))


# ----------------------------- instance family -----------------------------
def make_instances():
    """Deterministic, seeded. Returns [{'public':..., 'hidden':{}}].
    A random rooted supply tree of polar stations; per-station lead-time demand
    (mu, sigma), holding cost h (rising with depth: deep-field storage is dear),
    shortage penalty p, a programme fill-rate floor beta, and a set of LIFE-SUPPORT
    stations each carrying a LOCAL fill floor alpha. Instances vary from 'no critical /
    loose' (pure network problem) to 'many tight critical' (needs local pinning)."""
    specs = [
        # seed,  N, beta%, pmul_lo, pmul_hi, alpha%, ncrit
        (201, 28, 90,  3, 10,  0, 0),
        (202, 32, 94,  3, 10,  0, 0),
        (203, 30, 92, 20, 60, 90, 4),
        (204, 35, 92,  3, 10, 98, 5),
        (205, 40, 95,  3, 10, 97, 6),
        (206, 26, 90, 20, 60, 98, 3),
        (207, 38, 93,  3, 10,  0, 0),
        # larger / held-out instances
        (208, 45, 96,  3, 10, 98, 7),
        (209, 42, 91, 20, 60, 97, 5),
        (210, 48, 94,  3, 10, 98, 6),
        (211, 50, 95,  3, 10, 96, 8),
        (212, 36, 92, 20, 60,  0, 0),
    ]
    out = []
    for seed, N, beta_pct, pmul_lo, pmul_hi, alpha_pct, ncrit in specs:
        r = _rng(seed)
        parent = [-1] * N
        depth = [0] * N
        for j in range(1, N):
            parent[j] = r(0, j - 1)            # random rooted tree, node 0 = hub/port
            depth[j] = depth[parent[j]] + 1
        t = [r(1, 4) for _ in range(N)]        # transit/processing time per leg
        L = [0] * N
        for j in range(N):                     # cumulative lead time along path
            L[j] = t[j] + (0 if parent[j] < 0 else L[parent[j]])
        mu = [0.0] * N; sigma = [0.0] * N; h = [0.0] * N; p = [0.0] * N
        for j in range(N):
            mu[j] = float(r(20, 80))
            sigma[j] = float(r(15, 55))
            hb = r(10, 30) / 10.0
            h[j] = hb * (1.0 + 0.25 * depth[j])            # deeper stations cost more
            p[j] = h[j] * (r(pmul_lo * 10, pmul_hi * 10) / 10.0)
        sd = [sigma[j] * math.sqrt(L[j]) for j in range(N)]
        # choose life-support (critical) stations deterministically
        pool = list(range(N)); chosen = []
        for _ in range(min(ncrit, N)):
            idx = r(0, len(pool) - 1)
            chosen.append(pool.pop(idx))
        cset = set(chosen)
        crit = [1 if j in cset else 0 for j in range(N)]
        public = {
            "N": N, "parent": parent, "L": L, "mu": mu, "sigma": sigma,
            "h": h, "p": p, "sd": sd,
            "beta": beta_pct / 100.0,
            "alpha": alpha_pct / 100.0,
            "crit": crit,
        }
        out.append({"public": public, "hidden": {}})
    return out


# ----------------------------- scoring -------------------------------------
_BASE_K = 4.0  # "top off every locker" safety factor


def baseline(inst):
    """Top-off design: stock every station to safety factor 4 (always feasible)."""
    p = inst["public"]
    N = p["N"]; sd = p["sd"]; h = p["h"]; pen = p["p"]
    b = _loss(_BASE_K)
    return float(sum(h[i] * _BASE_K * sd[i] + pen[i] * sd[i] * b for i in range(N)))


def score(inst, answer):
    """Strictly validate the answer against the instance; return (ok, obj)."""
    pub = inst["public"]
    N = pub["N"]; sd = pub["sd"]; mu = pub["mu"]; h = pub["h"]; pen = pub["p"]
    beta = pub["beta"]; alpha = pub["alpha"]; crit = pub["crit"]
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
    # (1) programme-wide fill-rate floor
    fill = 1.0 - sum(B) / sum(mu)
    if fill < beta - 1e-9:
        return False, None
    # (2) per-station local floors on life-support stations
    for i in range(N):
        if crit[i]:
            local = 1.0 - B[i] / mu[i]
            if local < alpha - 1e-9:
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
