#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0322 -- "Ironhump Freight Yard: Single-Hump Shunting Priority"
(family: online-heuristic-simulator; format B, quality-metric).

THEME.  A railway classification yard has ONE hump (a single shunting engine).
Freight "cuts" (coupled groups of cars bound for the same outbound train) roll onto
the receiving lead over the course of a shift.  Cut i becomes AVAILABLE at time r_i
(it cannot be humped before it has arrived), needs p_i time units of hump work to be
pushed over and sorted, carries a priority weight w_i (penalty per time unit it makes
its outbound train wait), and has a scheduled cut-off d_i (the departure slot of its
outbound train).  The hump works one cut at a time and never interrupts a cut once it
starts (non-preemptive).  If cut i finishes at time C_i, its lateness penalty is
w_i * max(0, C_i - d_i).  The yardmaster wants a DISPATCH PRIORITY that minimizes the
total weighted lateness sum_i w_i * max(0, C_i - d_i).

This is the classic strongly-NP-hard single-machine weighted-tardiness problem with
release dates, 1 | r_j | sum w_j T_j, skinned as a hump yard.  The model does NOT
output a full schedule -- it outputs a *dispatch priority order* (a permutation of the
cuts), and a FIXED, NON-DELAY simulator turns that priority into an actual timetable:
whenever the hump is idle and one or more cuts have arrived, it immediately starts the
arrived cut that stands EARLIEST in the model's priority order; if nothing has arrived
yet, the hump fast-forwards to the next arrival.  This is exactly a FunSearch-style
"evolve the dispatch rule" task consumed by an ALE-Bench-style simulator.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "n": N (int), "horizon": H (int),
             "cuts": [ {"p": p_i, "w": w_i, "r": r_i, "d": d_i}, ...]}   # length N
          where p_i>=1 (hump time), w_i>=1 (weight), r_i>=0 (release), d_i (due).
  stdout: ONE JSON object:
            {"order": [j_0, j_1, ..., j_{N-1}]}
          a PERMUTATION of 0..N-1 giving the dispatch priority (j_0 = highest
          priority).  Anything that is not a permutation of exactly the N cut
          indices -- wrong length, repeats, out-of-range, non-int, a crash, a
          timeout, or non-JSON -- makes that instance score 0.0.

SCORING (deterministic; no wall-time).  Per instance the simulator computes the
total weighted tardiness of the candidate's order, q_cand.  The reference is the
first-come-first-served rule (dispatch strictly by arrival time), q_fcfs, a weak but
sensible yard default.  The normalized score is the classic quality ratio
    r = min( 1.0, 0.1 * q_fcfs / max(q_cand, 1e-9) )
so reproducing FCFS scores exactly 0.1, halving its lateness scores ~0.2, and only an
order that cuts total lateness to <=10% of FCFS approaches 1.0.  Because the instances
are congested with tight cut-offs (no zero-lateness schedule exists) and the problem is
NP-hard, even a strong local search stays well below 1.0 -> headroom.  Doing WORSE than
FCFS scores below 0.1.

ISOLATION.  The candidate is untrusted and runs in a FRESH OS-SANDBOXED SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  The FCFS reference and
the tardiness simulator are computed by THIS parent process, so a frame-walking /
introspecting candidate learns nothing it could not compute itself.

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
def _build_cuts(seed, n, P, W, spread, tight):
    """Deterministic list of N cuts (p, w, r, d). Releases accumulate small gaps
    (congestion -> tardiness); due dates are release+processing plus tight slack."""
    ni = _rng(seed)
    cuts = []
    t = 0
    for _ in range(n):
        p = ni(1, P)
        w = ni(1, W)
        t += ni(0, spread)          # cumulative arrival time
        r = t
        d = r + p + ni(0, tight)    # cut-off = ready time + a small slack
        cuts.append({"p": p, "w": w, "r": r, "d": d})
    return cuts


def _build_instances():
    """Deterministic instance family. (seed, n, P, W, spread, tight)."""
    specs = [
        (201, 14, 8, 5, 3, 10),
        (202, 16, 8, 6, 4, 12),
        (203, 14, 6, 4, 2, 8),
        (204, 18, 8, 5, 4, 14),
        (205, 15, 8, 8, 3, 10),
        (206, 16, 8, 5, 3, 12),
        (207, 14, 10, 4, 4, 10),
        (208, 17, 8, 6, 3, 12),
        (209, 20, 8, 5, 5, 16),
        (210, 16, 8, 6, 3, 10),
        (211, 18, 6, 4, 3, 10),
        (212, 15, 8, 5, 2, 9),
        # harder / larger held-out instances
        (313, 22, 8, 6, 4, 14),
        (314, 20, 10, 6, 3, 12),
    ]
    out = []
    for seed, n, P, W, spread, tight in specs:
        cuts = _build_cuts(seed, n, P, W, spread, tight)
        horizon = cuts[-1]["r"] + sum(c["p"] for c in cuts)
        out.append({"name": f"yard{seed}", "n": n, "horizon": horizon, "cuts": cuts})
    return out


# ----------------------------- simulator -----------------------------------
def _simulate(cuts, order):
    """NON-DELAY dispatch simulator.  `order` is a permutation of cut indices
    (position 0 = highest priority).  Returns total weighted tardiness."""
    n = len(cuts)
    pos = [0] * n
    for k, j in enumerate(order):
        pos[j] = k
    done = [False] * n
    t = 0
    total = 0.0
    ndone = 0
    while ndone < n:
        best = -1
        best_pos = 1 << 30
        next_r = 1 << 60
        for j in range(n):
            if done[j]:
                continue
            rj = cuts[j]["r"]
            if rj <= t:
                if pos[j] < best_pos:
                    best_pos = pos[j]
                    best = j
            elif rj < next_r:
                next_r = rj
        if best < 0:                 # nothing arrived yet -> fast-forward
            t = next_r
            continue
        t += cuts[best]["p"]
        c = cuts[best]
        lateness = t - c["d"]
        if lateness > 0:
            total += c["w"] * lateness
        done[best] = True
        ndone += 1
    return total


def _fcfs_order(cuts):
    return sorted(range(len(cuts)), key=lambda i: (cuts[i]["r"], i))


# ----------------------------- validation ----------------------------------
def _validate_order(inst, answer):
    """Return the order (list of ints) iff it is a valid permutation, else None."""
    if not isinstance(answer, dict):
        return None
    order = answer.get("order")
    if not isinstance(order, list):
        return None
    n = inst["n"]
    if len(order) != n:
        return None
    seen = [False] * n
    for v in order:
        if isinstance(v, bool) or not isinstance(v, int):
            return None
        if v < 0 or v >= n or seen[v]:
            return None
        seen[v] = True
    return order


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        cuts = inst["cuts"]
        q_fcfs = _simulate(cuts, _fcfs_order(cuts))
        public = {"name": inst["name"], "n": inst["n"],
                  "horizon": inst["horizon"],
                  "cuts": [dict(c) for c in cuts]}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            order = _validate_order(inst, ans)
        except Exception:
            order = None
        if order is None:
            vec.append(0.0)
            continue
        try:
            q_cand = _simulate(cuts, order)
        except Exception:
            vec.append(0.0)
            continue
        if q_fcfs <= 1e-9:
            r = 1.0                      # degenerate (no lateness under FCFS); not used here
        else:
            r = 0.1 * q_fcfs / max(q_cand, 1e-9)
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
