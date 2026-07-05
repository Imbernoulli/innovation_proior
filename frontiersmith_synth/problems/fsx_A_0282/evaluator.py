#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0282 -- "Greenway Depot: Two-Constraint Collection Routes"
(family: online-heuristic-simulator; format B, quality-metric; theme: recycling depot routes).

THEME.  A municipal recycling depot dispatches identical collection trucks to clear
a stream of curbside containers.  Every container has TWO independent limits that
matter: its MASS (kg of compacted recyclables) and its BULK (litres of loose
volume -- crushed cans are dense, foam and cardboard are light-but-huge).  Each
truck can carry at most W kilograms AND at most V litres; a load is legal only when
BOTH limits hold.  Dispatching a truck on a route costs one dispatch (a full loop
of the depot circuit + tipping fee).  The depot wants to collect EVERY container
using as FEW dispatched trucks as possible.

This is 2-D *vector* bin packing skinned as depot routing.  Containers = items with
a 2-vector (mass, bulk); a truck = a bin with a 2-vector capacity (W, V); "trucks
dispatched" = bins used, which we MINIMIZE.  The vector twist is what makes it hard:
packing tightly on mass can waste bulk and vice versa, and the L1 bound (max over
the two 1-D bounds) is loose, so even excellent packers leave headroom.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "W": int, "V": int, "n": int,
             "mass":  [m_0, ..., m_{n-1}],   # 1 <= m_i <= W
             "bulk":  [b_0, ..., b_{n-1}]}   # 1 <= b_i <= V
  stdout: ONE JSON object:
            {"assign": [t_0, ..., t_{n-1}]}
          where t_i >= 0 is the truck index container i is loaded onto.  Truck
          indices need not be contiguous; a truck "exists" iff >=1 container loads
          onto it, and the number of DISTINCT non-empty trucks is the dispatch count.

  A plan is VALID iff `assign` is a list of exactly n non-negative integers and no
  truck's total mass exceeds W and no truck's total bulk exceeds V.  Invalid output,
  wrong length, an overloaded truck, a crash, a timeout, or non-JSON -> instance 0.0.

SCORING (deterministic; no wall-time).  Per instance we compute three references:
    q_lb   = max( ceil(sum(mass)/W), ceil(sum(bulk)/V) )   # L1 vector lower bound
    q_base = trucks used by the internal NEXT-FIT operator  # weak online reference
    q_cand = trucks used by the candidate plan
  and normalize with an affine anchor (weak baseline -> 0.1, L1 ideal -> 1.0):
    r = clamp( 0.1 + 0.9 * (q_base - q_cand) / max(1e-9, q_base - q_lb), 0, 1 )
  A candidate matching next-fit scores ~0.1; a candidate reaching the (generally
  unreachable) L1 vector bound scores 1.0; doing worse than next-fit scores < 0.1.

ISOLATION.  The candidate is untrusted and runs in a FRESH SANDBOXED SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  The references
(L1, next-fit baseline) are computed by THIS parent process, so a frame-walking /
introspecting / source-reading candidate learns nothing useful.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json
import isorun


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt_int(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt_int


# ----------------------------- instance family -----------------------------
def _build_containers(seed, n, W, V, dist):
    """Return (mass[], bulk[]): n containers with per-truck-legal 2-vectors."""
    ni = _rng(seed)
    mass, bulk = [], []
    for _ in range(n):
        if dist == "bal":                       # both dims mid-range, independent
            m = ni(W // 5, (4 * W) // 5)
            b = ni(V // 5, (4 * V) // 5)
        elif dist == "anti":                    # anti-correlated: heavy => small bulk
            m = ni((3 * W) // 10, (9 * W) // 10)
            b = V + (V // 10) - (m * V) // W + ni(-(V // 8), V // 8)
        elif dist == "heavy":                   # mass usually binds
            m = ni((W // 2), (19 * W) // 20)
            b = ni(V // 10, V // 2)
        elif dist == "bulky":                   # bulk usually binds
            m = ni(W // 10, W // 2)
            b = ni((V // 2), (19 * V) // 20)
        elif dist == "spiky":                   # many tiny + a few near-full on each axis
            if ni(0, 99) < 60:
                m = ni(1, max(1, W // 6)); b = ni(1, max(1, V // 6))
            else:
                m = ni((3 * W) // 5, (9 * W) // 10); b = ni((3 * V) // 5, (9 * V) // 10)
        else:
            m = ni(1, W); b = ni(1, V)
        if m < 1: m = 1
        if m > W: m = W
        if b < 1: b = 1
        if b > V: b = V
        mass.append(m)
        bulk.append(b)
    return mass, bulk


def _build_instances():
    """Deterministic instance family. (seed, n, W, V, dist)."""
    specs = [
        (701, 26, 100, 100, "bal"),
        (702, 28, 100, 100, "anti"),
        (703, 30, 120,  90, "bal"),
        (714, 26, 100, 100, "spiky"),
        (705, 32, 100, 100, "anti"),
        (720, 30,  90, 110, "bulky"),
        (707, 28, 100, 100, "heavy"),
        (708, 30, 110, 100, "anti"),
        # harder / larger held-out instances
        (811, 44, 100, 100, "anti"),
        (810, 42, 120,  90, "spiky"),
        (811 + 7, 48, 100, 100, "bal"),
        (812, 52, 100, 110, "anti"),
    ]
    out = []
    for seed, n, W, V, dist in specs:
        mass, bulk = _build_containers(seed, n, W, V, dist)
        out.append({"name": f"route{seed}", "W": W, "V": V, "n": n,
                    "mass": mass, "bulk": bulk, "dist": dist})
    return out


# ----------------------------- references ----------------------------------
def _l1(mass, bulk, W, V):
    return max(-(-sum(mass) // W), -(-sum(bulk) // V))    # max of two ceil bounds


def _next_fit(mass, bulk, W, V):
    """Weak online operator: keep loading the current truck; the moment a container
    breaks EITHER limit, dispatch it and start a fresh truck.  Never looks back."""
    trucks = 1
    rm, rb = W, V
    for m, b in zip(mass, bulk):
        if m <= rm and b <= rb:
            rm -= m; rb -= b
        else:
            trucks += 1
            rm = W - m; rb = V - b
    return trucks


# ----------------------------- validation ----------------------------------
def _dispatches(inst, answer):
    """Validate answer against the instance. Return truck count or None."""
    if not isinstance(answer, dict):
        return None
    assign = answer.get("assign")
    if not isinstance(assign, list):
        return None
    mass, bulk = inst["mass"], inst["bulk"]
    W, V, n = inst["W"], inst["V"], inst["n"]
    if len(assign) != n:
        return None
    lm, lb = {}, {}
    for i, t in enumerate(assign):
        if isinstance(t, bool) or not isinstance(t, int):
            return None
        if t < 0:
            return None
        lm[t] = lm.get(t, 0) + mass[i]
        lb[t] = lb.get(t, 0) + bulk[i]
        if lm[t] > W or lb[t] > V:
            return None
    return len(lm)          # number of distinct non-empty trucks


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        W, V = inst["W"], inst["V"]
        mass, bulk = inst["mass"], inst["bulk"]
        q_lb = _l1(mass, bulk, W, V)
        q_base = _next_fit(mass, bulk, W, V)
        denom = q_base - q_lb
        if denom < 1e-9:
            denom = 1e-9
        public = {"name": inst["name"], "W": W, "V": V, "n": inst["n"],
                  "mass": list(mass), "bulk": list(bulk)}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            q_cand = _dispatches(inst, ans)
        except Exception:
            q_cand = None
        if q_cand is None:
            vec.append(0.0)
            continue
        r = 0.1 + 0.9 * (q_base - q_cand) / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        if r < 0.0:
            r = 0.0
        elif r > 1.0:
            r = 1.0
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
