#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_B_0176 -- "Glacier Sensor Net: Spare-Module Staging on a
Supply Tree with Storage-Locker Caps".

Family: constrained-OR (Frontier-Eng Inventory / multi-echelon safety stock),
skinned as a glacier-monitoring sensor network. Field sensor stations (GPS pingers,
seismographs, weather masts) are resupplied with spare hardware modules (battery
packs, comms boards, heater cartridges) flown out from a base camp. The stations
form a SUPPLY TREE rooted at base camp (node 0): a spare flows base camp -> relay
cache -> ... -> forward station, and every hop across the ice adds a deterministic
transit/staging delay. The net replenishment lead time L_i at station i is the
cumulative delay along its path from base camp, so stations deep on the glacier
face longer, riskier lead times.

Over its lead time each station faces stochastic module demand, modelled as
Normal(mu_i, sd_i^2) with sd_i = sigma_i * sqrt(L_i). The candidate chooses a
SAFETY-STOCK quantity stock_i for every station. With safety factor
k_i = stock_i / sd_i the expected unmet demand (backorders) is
    B_i = sd_i * Loss(k_i),   Loss(k) = phi(k) - k*(1 - Phi(k))
(the standard-normal loss function; decreasing in k). The composite objective to
MINIMIZE is the holding-plus-shortage cost

    cost = sum_i [ h_i * stock_i  +  p_i * B_i ].

THE NOVELTY vs a plain safety-stock tree: every station has a HARD STORAGE-LOCKER
CAPACITY cap_i (a heated equipment box of fixed size), so

    0 <= stock_i <= cap_i         (per-station capacity)

and a subset of the CHEAPEST, lowest-variance caches -- exactly where an
unconstrained design would want to pool extra service -- have TIGHT lockers. The
whole network must still meet an aggregate availability (fill-rate) floor

    fill = 1 - (sum_i B_i) / (sum_i mu_i)  >=  beta        (else INFEASIBLE -> 0).

Because the cheap caches saturate their lockers, a naive uniform water-filling
overflows them; service must instead be re-routed to more expensive stations, so
the capacity caps genuinely reshape the constrained optimum. This is an open-ended
trade-off with several viable strategies (node-wise newsvendor, capped Lagrangian
water-filling, greedy marginal allocation) and no easy closed form once lockers bind.

The candidate is UNTRUSTED model output: it runs in an ISOLATED subprocess via
`isorun`, sees ONLY the public instance on stdin, and returns ONLY its answer on
stdout, so it can never reach the evaluator's frames / scorer / baseline / hidden.

Scoring (deterministic; no wall-time):
  baseline b = cost of the "full-locker gold-plate" design that stocks every station
               to safety factor 4 CLAMPED to its locker (stock_i = min(4*sd_i, cap_i)).
               This is always feasible by construction (beta is set below its fill).
  For a FEASIBLE answer with objective obj:  r = min(1, 0.1 * b / obj)
  -> the trivial full-locker design maps to exactly 0.1; a design k times cheaper
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

    def uf():  # uniform float in [0,1)
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return (state >> 11) / float(1 << 53)

    nxt.uf = uf
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
def _fill_of_stock(stock, sd, mu):
    tot_mu = sum(mu)
    B = sum(sd[i] * _loss(stock[i] / sd[i]) for i in range(len(sd)))
    return 1.0 - B / tot_mu


def make_instances():
    """Deterministic, seeded. Returns [{'public':..., 'hidden':{}}].

    Builds a random rooted supply tree (node 0 = base camp), per-station lead-time
    demand (mu, sigma -> sd = sigma*sqrt(L)), holding cost h (rising with depth),
    shortage penalty p, and a HARD locker capacity cap_i. The cheapest / lowest-sd
    stations get TIGHT lockers (cap ~ 1.3..2.2 sd); the rest get roomy lockers
    (cap ~ 3.5..6 sd). The availability floor beta is placed a fraction `frac` of the
    way from the do-nothing (all-zero) fill up to the full-locker gold-plate fill, so
    that (i) the all-zero design is ALWAYS infeasible, (ii) the gold-plate baseline is
    ALWAYS feasible, and (iii) low `frac` = loose (node-wise newsvendor survives),
    high `frac` = tight (service must be re-routed off the saturated lockers)."""
    specs = [
        # seed, N, frac, pmul_lo, pmul_hi   (pmul = p/h multiplier range in 0.1 units)
        (211, 6, 0.40, 25, 70), (212, 7, 0.62, 25, 70), (213, 8, 0.55, 25, 70),
        (214, 7, 0.80, 25, 70), (215, 9, 0.70, 25, 70), (216, 8, 0.86, 25, 70),
        (217, 9, 0.50, 25, 70),
        # larger / held-out instances (deeper trees, tighter floors)
        (218, 10, 0.88, 25, 70), (219, 11, 0.74, 25, 70), (220, 12, 0.92, 25, 70),
    ]
    out = []
    for seed, N, frac, pmul_lo, pmul_hi in specs:
        r = _rng(seed)
        parent = [-1] * N
        depth = [0] * N
        for j in range(1, N):
            parent[j] = r(0, j - 1)            # random rooted tree, node 0 = base camp
            depth[j] = depth[parent[j]] + 1
        t = [r(1, 4) for _ in range(N)]        # transit/staging delay per hop
        L = [0] * N
        for j in range(N):                     # cumulative lead time along path
            L[j] = t[j] + (0 if parent[j] < 0 else L[parent[j]])
        mu = [0.0] * N; sigma = [0.0] * N; h = [0.0] * N; p = [0.0] * N
        for j in range(N):
            mu[j] = float(r(20, 90))
            sigma[j] = float(r(15, 55))
            hb = r(10, 30) / 10.0
            h[j] = round(hb * (1.0 + 0.25 * depth[j]), 4)   # deeper stations cost more
            p[j] = round(h[j] * (r(pmul_lo, pmul_hi) / 10.0), 4)
        sd = [round(sigma[j] * math.sqrt(L[j]), 6) for j in range(N)]

        # A Lagrangian water-filling pools extra service onto the LOWEST-holding-cost
        # stations (they minimize h_i/(p_i+lambda), so they get pushed to the highest
        # safety factor). We give exactly those cheapest caches the TIGHTEST lockers,
        # so the naive optimum overflows them and service must be re-routed onto
        # roomier, pricier stations. The rest get generous lockers.
        order = sorted(range(N), key=lambda i: (h[i], i))
        n_tight = max(1, N // 3)
        tight = set(order[:n_tight])
        cap = [0.0] * N
        for j in range(N):
            if j in tight:
                cmul = 0.8 + 0.6 * r.uf()      # 0.8 .. 1.4  (binds under water-filling)
            else:
                cmul = 3.5 + 2.5 * r.uf()      # 3.5 .. 6.0
            cap[j] = round(cmul * sd[j], 6)

        # availability floor between all-zero fill and full-locker fill
        zero_stock = [0.0] * N
        gold_stock = [min(4.0 * sd[i], cap[i]) for i in range(N)]
        f0 = _fill_of_stock(zero_stock, sd, mu)
        fg = _fill_of_stock(gold_stock, sd, mu)
        beta = round(f0 + frac * (fg - f0), 4)

        public = {
            "N": N, "parent": parent, "L": L, "mu": mu, "sigma": sigma,
            "h": h, "p": p, "sd": sd, "cap": cap, "beta": beta,
        }
        out.append({"public": public, "hidden": {}})
    return out


# ----------------------------- scoring -------------------------------------
def baseline(inst):
    """Full-locker gold-plate: stock_i = min(4*sd_i, cap_i). Always feasible."""
    pub = inst["public"]
    N = pub["N"]; sd = pub["sd"]; h = pub["h"]; pen = pub["p"]; cap = pub["cap"]
    tot = 0.0
    for i in range(N):
        s = min(4.0 * sd[i], cap[i])
        tot += h[i] * s + pen[i] * sd[i] * _loss(s / sd[i])
    return float(tot)


def score(inst, answer):
    """Strictly validate the answer against the instance; return (ok, obj)."""
    pub = inst["public"]
    N = pub["N"]; sd = pub["sd"]; mu = pub["mu"]; h = pub["h"]
    pen = pub["p"]; cap = pub["cap"]; beta = pub["beta"]
    if not isinstance(answer, dict):
        return False, None
    stock = answer.get("stock", None)
    if not isinstance(stock, list) or len(stock) != N:
        return False, None
    try:
        stock = [float(x) for x in stock]
    except (TypeError, ValueError):
        return False, None
    for i, s in enumerate(stock):
        if not math.isfinite(s):
            return False, None
        if s < -1e-9:
            return False, None
        if s > cap[i] + 1e-6:          # hard locker capacity
            return False, None
    stock = [min(max(0.0, stock[i]), cap[i]) for i in range(N)]
    B = [sd[i] * _loss(stock[i] / sd[i]) for i in range(N)]
    fill = 1.0 - sum(B) / sum(mu)
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
