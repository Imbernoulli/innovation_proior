#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_B_0160 -- "Metro Spare-Parts Safety-Stock Placement".

Family: constrained-OR (Frontier-Eng Inventory / multi-echelon safety stock),
skinned as a subway (metro) maintenance spare-parts supply tree.

A single critical spare (say, a traction-motor brush kit) is stocked across a
supply TREE:
    depot (root) -> regional maintenance yards -> line stores -> stations (leaves).
Each node i holds SAFETY STOCK of the part. Station (leaf) demand is random with a
known standard deviation; internal-node demand is the aggregate of the stations it
serves (independent demand => variances add, so sigma_i = sqrt(sum of leaf var)).

We use the classical Graves-Willems GUARANTEED-SERVICE model. Every node i quotes an
outbound service time S_i (whole days it promises to deliver to its customers) and
receives an inbound service time SI_i from its single supplier (its parent):
    SI_i = S_parent(i)          (SI_root = Sext, the external supplier's quote)
The node's NET REPLENISHMENT TIME is
    tau_i = SI_i + T_i - S_i     (T_i = local processing/lead time)
and it must hold enough safety stock to cover demand variability over tau_i at the
required service level. With safety factor k (k = z for the target cycle service
level, e.g. k=1.645 for ~95%), the safety-stock HOLDING COST at node i is
    cost_i = h_i * k * sigma_i * sqrt(tau_i).
The service-level constraint is satisfied by construction for ANY feasible S (the k
factor bakes it in); feasibility is the guaranteed-service structure itself:
    S_i >= 0,   tau_i = SI_i + T_i - S_i >= 0,   and   S_leaf <= s_max_leaf
(the station's quoted service to maintenance crews may not exceed its cap s_max).

DECISION (what the candidate outputs): the integer service-time vector S (one per
node, in the node order of the public instance). The candidate is UNTRUSTED and runs
in an ISOLATED subprocess via `isorun`; it only sees the public instance and can
never reach the evaluator's frames/scorer.

There is no easy optimum: quoting S_i=0 everywhere ("hold full safety stock at every
node for its own lead time") is the naive decoupled policy and is the evaluator's
baseline; pushing service times up pools risk upstream (sqrt savings) but the station
caps s_max and the downstream-increasing holding cost create a genuine trade-off.
The exact optimum is a dynamic program over the tree (Graves-Willems).

Scoring (deterministic; no wall-time):
  cost(S)   = sum_i h_i * k * sigma_i * sqrt(tau_i)
  baseline b = cost(all-zero decoupled design)   (naive "stock everywhere")
  feasible answer with objective obj:  r = min(1, 0.1 * b / obj)
  -> the trivial all-zero design maps to exactly 0.1; a design x times cheaper than
     baseline maps to min(1, 0.1*x). Infeasible / malformed answer -> 0.

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


# ----------------------------- instance family -----------------------------
K = 1.645          # safety factor (target cycle service level ~ 95%)
_BIG = 10 ** 9     # "no cap" sentinel for internal nodes


def _build(seed, R, L, Sp):
    """Build a 4-level supply tree: depot -> R yards -> L stores each -> Sp
    stations each. Holding cost rises downstream; only stations carry demand."""
    r = _rng(seed)
    parent = [-1]; T = [r(6, 12)]; h = [1.0]; level = [0]
    leafsig = [0.0]; smax = [_BIG]

    def add(p, lv, Tlo, Thi, hb, leaf=False):
        parent.append(p)
        T.append(r(Tlo, Thi))
        h.append(round(hb * (r(85, 115) / 100.0), 4))
        level.append(lv)
        leafsig.append(float(r(4, 20)) if leaf else 0.0)
        smax.append(r(3, 8) if leaf else _BIG)
        return len(parent) - 1

    for _ in range(R):
        y = add(0, 1, 2, 5, 1.5)
        for _ in range(L):
            st = add(y, 2, 1, 4, 2.0)
            for _ in range(Sp):
                add(st, 3, 1, 3, 3.0, leaf=True)

    n = len(parent)
    ch = [[] for _ in range(n)]
    for i in range(1, n):
        ch[parent[i]].append(i)
    sigma = [0.0] * n
    for i in sorted(range(n), key=lambda i: -level[i]):
        sigma[i] = leafsig[i] if not ch[i] else round(
            math.sqrt(sum(sigma[c] ** 2 for c in ch[i])), 6)

    public = {
        "n": n, "parent": parent, "children": ch, "level": level,
        "T": T, "h": h, "sigma": sigma, "smax": smax, "Sext": 0, "k": K,
    }
    return {"public": public, "hidden": {}}


def make_instances():
    """Deterministic, seeded. 12 medium-scale metro networks of growing size;
    the last few are larger held-out generalization instances."""
    specs = [
        (101, 2, 2, 2), (102, 2, 2, 3), (103, 3, 2, 3), (104, 2, 3, 3),
        (105, 3, 2, 4), (106, 2, 3, 4), (107, 3, 3, 3), (108, 3, 3, 4),
        # larger / held-out
        (109, 4, 3, 3), (110, 3, 4, 3), (111, 4, 3, 4), (112, 4, 4, 4),
    ]
    return [_build(*s) for s in specs]


# ----------------------------- scoring -------------------------------------
def _cost(p, S):
    """Total safety-stock holding cost, or None if S is infeasible."""
    parent = p["parent"]; T = p["T"]; h = p["h"]; sigma = p["sigma"]
    smax = p["smax"]; k = p["k"]; Sext = p["Sext"]
    tot = 0.0
    for i in range(p["n"]):
        Si = S[i]
        if Si < 0 or Si > smax[i]:
            return None
        SI = Sext if parent[i] < 0 else S[parent[i]]
        tau = SI + T[i] - Si
        if tau < -1e-9:
            return None
        tot += h[i] * k * sigma[i] * math.sqrt(max(tau, 0.0))
    return tot


def baseline(inst):
    return _cost(inst["public"], [0] * inst["public"]["n"])


def score(inst, answer):
    """Strictly validate the answer against the instance; return (ok, obj)."""
    p = inst["public"]
    n = p["n"]
    if not isinstance(answer, dict):
        return False, None
    S = answer.get("S", None)
    if not isinstance(S, list) or len(S) != n:
        return False, None
    Sint = []
    for x in S:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return False, None
        xf = float(x)
        if xf != xf or xf in (float("inf"), float("-inf")):
            return False, None
        xr = round(xf)
        if abs(xf - xr) > 1e-6:          # service times must be whole days
            return False, None
        Sint.append(int(xr))
    obj = _cost(p, Sint)
    if obj is None or obj != obj or obj < 0:
        return False, None
    return True, obj


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
