#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0922 -- "Jam Storm: Anticipatory Packet Routing"
(family: congestion-routing-policy; format B, quality-metric).

THEME.  A dispatcher must route a fully-known traffic plan -- a sequence of
ROUNDS, each a batch of packets that must simultaneously choose one of K
parallel links to a shared destination -- across a network whose link delay
grows SUPERLINEARLY with load (a BPR-style function).  Every link also carries
a QUEUE: whatever a round pushes past a link's service capacity only partially
drains (a fixed DECAY fraction) and haunts the NEXT round as extra starting
load.  The dispatcher sees the whole plan (all rounds' packet weights, all
link parameters) up front and must commit to ONE routing decision per packet.

MECHANISM COMPOSITION (this is deliberate, not a single textbook algorithm):
  - superlinear-congestion:   L_e(x) = t0_e * (1 + (x/cap_e)^p_e), p_e > 1.
  - frozen-evaluator-policy:  the natural per-round decision signal is the
    link state AS OF THE START of the round (before this round's own
    packets land on it) -- a policy that reads only that frozen snapshot and
    applies it identically to every packet in the round is blind to the load
    IT ITSELF is about to pile on.
  - feedback-oscillation:     dumping a whole round onto "whichever link
    looked cheapest a moment ago" floods that link; its queue then dominates
    next round's snapshot, so the *next* round's naive decision swings to a
    different link -- a herding / ping-pong dynamic that compounds via the
    carried-over queue into real congestion collapse.

INNOVATION HOOK.  Myopic least-latency routing (pick the single link that
*currently* looks fastest, send everyone there) induces exactly this
oscillation.  A policy that internalizes its OWN marginal contribution --
i.e. accounts for the load it is concurrently committing within the round,
splitting traffic so every used link's MARGINAL delay is equalized rather
than chasing the single lowest AVERAGE delay -- damps the feedback and stays
close to the (unreachable) continuous water-filling optimum.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "k": K (int),
             "edges": [{"t0": float>0, "cap": float>0, "p": int>=2}, ... K],
             "decay": float in [0,1],
             "rounds": [{"n": N_r (int), "weights": [w_0,...,w_{N_r-1}]}, ...]}
  stdout: ONE JSON object:
            {"routes": [[e_0,...,e_{N_r-1}], ...]}   # one row per round,
          where each e_i is an integer link index in [0, K) -- the link
          packet i of that round is sent on.  ALL rows/decisions are
          committed in this single shot (the dispatcher plans the whole
          traffic schedule at once; the evaluator then REPLAYS it through the
          queueing dynamics below to get the true cost).

  `routes` is VALID iff it has exactly len(rounds) rows, row r has exactly
  rounds[r]["n"] integer entries, and every entry is in [0, K).  Invalid
  shape/type, a crash, a timeout, or non-JSON output -> that instance scores
  0.0.  (There is no capacity-feasibility rejection: overloading a link is
  legal, it just costs a lot via the superlinear queue below.)

DYNAMICS (deterministic; computed by the PARENT, never sent to the
candidate).  Per link e, backlog B_e starts at 0.  For round r:
    load_e = B_e + sum(weights of packets routed to e this round)
    cost   += load_e * L_e(load_e)                      # total delay-time
    B_e(next) = max(0, load_e - cap_e) * decay           # partial queue carry
Total objective = sum of cost over all rounds and links (MINIMIZE).

SCORING.  Per instance the evaluator computes, itself, an unreachable
CONTINUOUS floor: for each round independently (queue reset to 0, packets
treated as an infinitely-divisible fluid), the exact convex-optimal split
across links found by equalizing marginal delay d/dx[x*L_e(x)] via bisection
on the marginal-cost threshold. Summed over rounds this floor is a genuine
lower bound no discrete, queue-respecting policy can beat.
    r = clamp( 0.1 + 0.9 * HEADROOM * (floor / max(obj, eps)) ** GAMMA, 0, 1 )
Matching the floor exactly scores ~0.1 + 0.9*HEADROOM (< 1, by design: the
floor is unreachable so headroom stays open above any real solution); any
*valid* answer scores strictly above 0.1 and decays smoothly toward that
floor as obj grows arbitrarily large -- only an invalid answer (bad
shape/type, a crash, a timeout, or non-JSON output) scores exactly 0.0.
GAMMA < 1 compresses the wide (orders-of-magnitude)
dynamic range spanned by the ladder so a herding collapse is punished
smoothly rather than everything indistinguishably saturating at 0 or 1.

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  The floor
(continuous water-filling bound) is computed by THIS parent process only, so
a frame-walking / introspecting candidate learns nothing useful.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json
import isorun

GAMMA = 0.75
HEADROOM = 0.85
EPS = 1e-9


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt_int(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt_int


def _build_rounds(seed, T, n_lo, n_hi, w_lo, w_hi):
    ni = _rng(seed)
    rounds = []
    for _ in range(T):
        n = ni(n_lo, n_hi)
        weights = [ni(w_lo, w_hi) for _ in range(n)]
        rounds.append({"n": n, "weights": weights})
    return rounds


# ----------------------------- instance family ------------------------------
# (name, edges=[(t0,cap,p),...], decay, T, n_lo,n_hi, w_lo,w_hi, seed)
_SPECS = [
    ("mild_asym",             [(1, 24, 2), (2, 20, 3), (3, 16, 3)],                          0.35, 10, 10, 16, 1, 3, 501),
    ("mild_symmetric",        [(2, 22, 3), (2, 22, 3), (2, 22, 3)],                          0.35, 10, 14, 20, 1, 3, 502),
    ("trap_symmetric_burst",  [(2, 20, 3), (2, 20, 3), (2, 20, 3), (2, 20, 3)],               0.55, 12, 20, 32, 1, 3, 511),
    ("trap_twoedge_pingpong", [(2, 18, 3), (2, 18, 3)],                                       0.60, 12, 14, 22, 1, 3, 512),
    ("trap_capacity_illusion",[(1, 10, 3), (3, 34, 2), (3, 34, 2)],                           0.45, 12, 12, 20, 1, 4, 513),
    ("trap_herd_manyedge",    [(2, 16, 3), (2, 16, 3), (2, 16, 3), (2, 16, 3), (2, 16, 3)],    0.55, 14, 24, 36, 1, 3, 514),
    ("mixed_moderate",        [(1, 22, 2), (3, 22, 3), (2, 16, 4)],                           0.45, 10, 14, 22, 1, 4, 521),
    ("held_out_larger",       [(2, 22, 3), (2, 22, 3), (2, 22, 3), (2, 22, 3)],                0.55, 16, 26, 38, 1, 4, 611),
    ("held_out_capshift",     [(1, 30, 2), (2, 14, 4), (3, 14, 4), (2, 18, 3)],                0.45, 12, 16, 24, 1, 4, 612),
    ("held_out_extreme",      [(2, 14, 4), (2, 14, 4), (2, 14, 4), (2, 14, 4), (2, 14, 4)],    0.65, 16, 30, 44, 1, 3, 613),
]


def make_instances():
    out = []
    for name, edges, decay, T, n_lo, n_hi, w_lo, w_hi, seed in _SPECS:
        edges_d = [{"t0": float(t0), "cap": float(cap), "p": int(p)} for (t0, cap, p) in edges]
        rounds = _build_rounds(seed, T, n_lo, n_hi, w_lo, w_hi)
        public = {"name": name, "k": len(edges_d), "edges": edges_d,
                  "decay": decay, "rounds": rounds}
        out.append({"public": public, "hidden": {}})
    return out


# ----------------------------- dynamics -------------------------------------
def _latency(e, x):
    if x <= 0:
        return e["t0"]
    return e["t0"] * (1.0 + (x / e["cap"]) ** e["p"])


def _simulate(edges, decay, rounds, choices):
    """Replay the queueing dynamics for a full committed routing plan.
    Returns total delay-time cost (float)."""
    K = len(edges)
    backlog = [0.0] * K
    total = 0.0
    for rnd, ch in zip(rounds, choices):
        load = list(backlog)
        for w, e in zip(rnd["weights"], ch):
            load[e] += w
        for e in range(K):
            x = load[e]
            if x > 0:
                total += x * _latency(edges[e], x)
            backlog[e] = max(0.0, x - edges[e]["cap"]) * decay
    return total


# ------------------------- unreachable continuous floor ---------------------
def _water_fill_floor_round(edges, N):
    """Exact convex-optimal (fractional, queue-free) cost of splitting N units
    of arriving load across `edges` this round, via bisection on the marginal
    -cost threshold tau.  Marginal cost of link e at load x is
    t0*(1+(p+1)*(x/cap)^p); its inverse gives x_e(tau) in closed form."""
    if N <= 0:
        return 0.0
    lo = 0.0
    hi = max(e["t0"] for e in edges) + 1.0

    def total_x(tau):
        s = 0.0
        for e in edges:
            if tau <= e["t0"]:
                continue
            val = (tau - e["t0"]) / (e["t0"] * (e["p"] + 1))
            s += e["cap"] * (val ** (1.0 / e["p"]))
        return s

    tries = 0
    while total_x(hi) < N and tries < 200:
        hi *= 2.0
        tries += 1
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if total_x(mid) < N:
            lo = mid
        else:
            hi = mid
    tau = hi
    cost = 0.0
    for e in edges:
        if tau <= e["t0"]:
            x = 0.0
        else:
            val = (tau - e["t0"]) / (e["t0"] * (e["p"] + 1))
            x = e["cap"] * (val ** (1.0 / e["p"]))
        if x > 0:
            cost += x * _latency(e, x)
    return cost


def _floor_cost(edges, rounds):
    return sum(_water_fill_floor_round(edges, sum(r["weights"])) for r in rounds)


def baseline(inst):
    """The evaluator's own reference: the unreachable continuous floor."""
    pub = inst["public"]
    return _floor_cost(pub["edges"], pub["rounds"])


# ----------------------------- validation ------------------------------------
def _validate_routes(public, answer):
    if not isinstance(answer, dict):
        return None
    routes = answer.get("routes")
    if not isinstance(routes, list):
        return None
    rounds = public["rounds"]
    K = public["k"]
    if len(routes) != len(rounds):
        return None
    for row, rnd in zip(routes, rounds):
        if not isinstance(row, list) or len(row) != rnd["n"]:
            return None
        for e in row:
            if isinstance(e, bool) or not isinstance(e, int):
                return None
            if e < 0 or e >= K:
                return None
    return routes


def score(inst, answer):
    """Validate `answer` against inst; return (ok, objective_cost)."""
    public = inst["public"]
    routes = _validate_routes(public, answer)
    if routes is None:
        return False, None
    cost = _simulate(public["edges"], public["decay"], public["rounds"], routes)
    if not (cost == cost) or cost in (float("inf"), float("-inf")):
        return False, None
    return True, cost


# ----------------------------- scoring driver --------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = make_instances()

    vec = []
    for inst in instances:
        floor = baseline(inst)
        ans, st = isorun.run_candidate(cand, inst["public"], timeout=5)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            ok, obj = score(inst, ans)
        except Exception:
            ok = False
        if not ok:
            vec.append(0.0)
            continue
        r = 0.1 + 0.9 * HEADROOM * (floor / max(obj, EPS)) ** GAMMA
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        vec.append(max(0.0, min(1.0, r)))

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
