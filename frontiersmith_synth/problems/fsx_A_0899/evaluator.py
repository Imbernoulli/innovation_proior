#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0899 -- "Cable Trucks to the Substations"
(family: online-fiedler-secretary; format B, quality-metric).

THEME.  An engineer is wiring a set of substations into a resilient power network.
A cheap BACKBONE of cables is already installed for free (it guarantees the network
starts connected) but it is a fragile design: two well-knit clusters joined by a
single thin bridge.  Over the course of the build, a fixed manifest of `m` extra
candidate cables arrives ONE TRUCK AT A TIME.  The engineer has a total cash BUDGET.

THIS IS A GENUINELY ONLINE / IRREVOCABLE PROTOCOL, NOT A DRESSED-UP OFFLINE ONE.
The candidate program is invoked FRESH, ONCE PER TRUCK, in arrival order -- it is
NEVER shown the rest of the manifest, only: the truck now at the gate (u, v, cost),
the network built so far (backbone + every cable bought so far), how much budget
remains, how many trucks total there are (`m`) and which index this one is (`t`),
and whatever private JSON "state" blob it chose to hand itself on the previous call
(the evaluator only ever echoes this back verbatim; it never inspects or uses it).
The candidate answers accept/reject for THIS truck only, immediately and forever --
there is no way to defer the decision, peek ahead, or revisit it later. This is what
makes "reserve budget for a later truck" a real constraint rather than an artifact of
processing a fixed subset in a fixed order: a policy that spends too eagerly early
has, at the time it commits, no way to know what it is giving up.

OBJECTIVE.  Maximize the algebraic connectivity (Fiedler value, the second-smallest
Laplacian eigenvalue lambda_2) of the final network (backbone + purchased cables).
lambda_2 is a standard proxy for how robust the network is to a single link failure;
it is monotone non-decreasing as edges are added (Weyl's inequality), so buying more
can never directly hurt -- the only real tension is the opportunity cost of spending
budget on a mediocre truck instead of holding it for a better one you cannot yet see.

WHY "ACCEPT ANYTHING THAT HELPS" IS A TRAP.  Each backbone is two cliquish clusters
joined by one thin bridge -- its lambda_2 is bottlenecked by that bridge. Early
trucks mostly offer cheap intra-cluster "decoy" cables: each one gives a real, exactly
measurable, but SMALL lambda_2 gain (it slightly hardens a cluster that is already
adequate). A handful of trucks concentrated in the LAST quarter of the manifest offer
cross-cluster "bridge" cables that attack the actual bottleneck and are worth far
more. A rule that buys every truck whose immediate exact marginal gain is positive is
individually defensible at every single step, yet collectively burns the budget on
the decoy crowd and has nothing left when the high-leverage bridges finally arrive.

CANDIDATE CONTRACT (isolated stdin -> stdout program, invoked ONCE PER TRUCK).
  stdin : ONE JSON object (the PUBLIC per-step instance):
            {"n": N, "m": M, "t": t, "budget_total": B0, "remaining": R,
             "backbone": [[u,v], ...],      # free edges, constant across the manifest
             "accepted": [[u,v], ...],      # cables bought at steps 0..t-1 (backbone excluded)
             "u": u_t, "v": v_t, "cost": cost_t,   # THIS truck's cable
             "state": <whatever this program returned as "state" last call, or null at t=0>}
  stdout: ONE JSON object:
            {"accept": true/false, "state": <any JSON value, <= 4000 chars serialized>}
          "state" is your own private scratchpad across calls (e.g. leverage samples
          observed so far); the evaluator only stores and replays it back verbatim.

Any malformed response AT ANY STEP (wrong type, oversized state, non-JSON, a crash,
or a timeout) invalidates the WHOLE instance -> that instance scores 0.0. The graph
built from steps 0..t-1 is otherwise never revisited.

SCORING (deterministic; no wall-time).  Per instance we compute two references purely
from the (evaluator-known) full manifest:
    lam_base = lambda_2 reached by an internal MYOPIC-GREEDY rule: walk the manifest
               causally, buying truck t iff it is affordable right now AND its exact
               marginal lambda_2 gain (recomputed exactly at every step) is positive.
               This is the "obvious" online rule -- it has no pacing discipline.
    lam_ceil = lambda_2 reached by buying EVERY truck's cable, ignoring the budget --
               an unreachable ideal (monotonicity guarantees it upper-bounds anything
               a budget-respecting policy, online or not, can ever reach).
  and normalize with an affine anchor (myopic-greedy -> 0.1, unreachable ideal -> 1.0):
    r = clamp( 0.1 + 0.9 * (lam_cand - lam_base) / max(1e-9, lam_ceil - lam_base), 0, 1)
  Matching myopic-greedy scores ~0.1; the unreachable ceiling scores 1.0; doing worse
  than myopic-greedy scores below 0.1.

ISOLATION.  Each per-truck call runs the untrusted candidate in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees that step's public JSON. The references
(myopic-greedy, unreachable-ideal) are computed by THIS parent process directly from
the full manifest, so a frame-walking / introspecting candidate learns nothing useful.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json
import numpy as np
import isorun

MAX_STATE_CHARS = 4000


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt_int(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt_int


# ----------------------------- spectral helpers -----------------------------
def _laplacian(n, edges):
    L = np.zeros((n, n))
    for u, v, w in edges:
        L[u, u] += w
        L[v, v] += w
        L[u, v] -= w
        L[v, u] -= w
    return L


def _lam2(n, edges):
    L = _laplacian(n, edges)
    ev = np.linalg.eigvalsh(L)
    return float(sorted(ev)[1])


# ----------------------------- instance construction ------------------------
def _build_trap_instance(name, seed, n, na, m, budget, n_bridge):
    """Two cliquish clusters (each a cycle) joined by one thin backbone bridge.
    Stream: many cheap intra-cluster 'decoy' cables early (real but small marginal
    gain -- they harden an already-adequate cluster), a handful of far more valuable
    cross-cluster 'bridge' cables concentrated in the LAST quarter of the manifest."""
    ni = _rng(seed)
    backbone = []
    for i in range(na):
        backbone.append((i, (i + 1) % na))
    for i in range(na, n):
        j = na + (i - na + 1) % (n - na)
        backbone.append((i, j))
    backbone.append((na - 1, na))

    bridge_positions = set()
    while len(bridge_positions) < n_bridge:
        bridge_positions.add(ni(int(m * 0.75), m - 1))

    stream = []
    for t in range(m):
        if t in bridge_positions:
            u = ni(0, na - 1)
            v = ni(na, n - 1)
            cost = ni(6, 10)
        else:
            if ni(0, 1) == 0:
                u = ni(0, na - 1)
                v = ni(0, na - 1)
                while v == u:
                    v = ni(0, na - 1)
            else:
                u = ni(na, n - 1)
                v = ni(na, n - 1)
                while v == u:
                    v = ni(na, n - 1)
            cost = ni(1, 3)
        stream.append((u, v, cost))
    return {"name": name, "n": n, "budget": float(budget),
            "backbone": [list(e) for e in backbone], "m": m,
            "stream": [list(s) for s in stream]}


def _build_instances():
    specs = [
        # (name, seed, n, na, m, budget, n_bridge)
        ("trap-a", 2004, 16, 8, 28, 10, 3),
        ("trap-b", 2001, 16, 8, 30, 11, 4),
        ("trap-c", 2002, 18, 9, 32, 12, 4),
        ("trap-d", 2003, 20, 10, 34, 13, 5),
        ("trap-e", 2005, 22, 11, 36, 14, 5),   # harder / held-out
        ("trap-f", 2006, 24, 12, 38, 15, 5),   # harder / held-out
        ("trap-g", 2007, 14, 7, 24, 9, 3),
        ("trap-h", 2008, 20, 10, 30, 12, 4),
        ("trap-i", 2009, 26, 13, 40, 16, 6),   # harder / held-out
        ("trap-j", 2010, 28, 14, 42, 17, 6),   # harder / held-out
    ]
    return [_build_trap_instance(*s) for s in specs]


# ----------------------------- references (evaluator-only) -----------------
def _myopic_greedy(inst):
    """Walk the manifest causally; buy iff affordable AND it strictly raises the
    exact lambda_2 of the network built so far. The 'obvious' online rule."""
    n = inst["n"]
    edges = [(u, v, 1) for u, v in inst["backbone"]]
    remaining = inst["budget"]
    base = _lam2(n, edges)
    for u, v, cost in inst["stream"]:
        if cost > remaining + 1e-9:
            continue
        trial = edges + [(u, v, 1)]
        val = _lam2(n, trial)
        if val > base + 1e-9:
            edges = trial
            remaining -= cost
            base = val
    return base


def _unreachable_ceiling(inst):
    n = inst["n"]
    edges = [(u, v, 1) for u, v in inst["backbone"]]
    edges += [(u, v, 1) for u, v, _c in inst["stream"]]
    return _lam2(n, edges)


# ----------------------------- online replay driver -------------------------
def _run_instance_online(cand, inst, per_call_timeout=5):
    """Drive the candidate through the manifest ONE TRUCK AT A TIME. Returns
    (ok, final_lam2). Any malformed / crashing / timed-out step invalidates the
    whole instance."""
    n = inst["n"]
    m = inst["m"]
    budget = inst["budget"]
    edges = [(u, v, 1) for u, v in inst["backbone"]]
    accepted_pairs = []
    remaining = budget
    state = None

    for t in range(m):
        u, v, cost = inst["stream"][t]
        public = {"n": n, "m": m, "t": t, "budget_total": budget, "remaining": remaining,
                  "backbone": inst["backbone"], "accepted": accepted_pairs,
                  "u": u, "v": v, "cost": cost, "state": state}
        ans, st = isorun.run_candidate(cand, public, timeout=per_call_timeout)
        if st != "OK":
            return False, None
        if not isinstance(ans, dict):
            return False, None
        acc = ans.get("accept")
        if isinstance(acc, bool):
            accept_bool = acc
        elif isinstance(acc, int) and acc in (0, 1):
            accept_bool = bool(acc)
        else:
            return False, None
        new_state = ans.get("state", None)
        try:
            blob = json.dumps(new_state)
        except (TypeError, ValueError):
            return False, None
        if len(blob) > MAX_STATE_CHARS:
            return False, None
        state = new_state
        if accept_bool and cost <= remaining + 1e-9:
            edges.append((u, v, 1))
            accepted_pairs.append([u, v])
            remaining -= cost

    val = _lam2(n, edges)
    if not (val == val) or val in (float("inf"), float("-inf")):
        return False, None
    return True, val


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        lam_base = _myopic_greedy(inst)
        lam_ceil = _unreachable_ceiling(inst)
        denom = lam_ceil - lam_base
        if denom < 1e-9:
            denom = 1e-9

        try:
            ok, lam_cand = _run_instance_online(cand, inst)
        except Exception:
            ok, lam_cand = False, None
        if not ok:
            vec.append(0.0)
            continue

        r = 0.1 + 0.9 * (lam_cand - lam_base) / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        r = max(0.0, min(1.0, r))
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
