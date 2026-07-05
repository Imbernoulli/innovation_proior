#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0082 -- "Twilight Carnival: Gondola Dispatch Circuit"
(family: online-heuristic-simulator; format B, quality-metric).

THEME.  A carnival's flagship ride is a circular gondola circuit.  Thrill-seekers
queue up in *parties* (families / friend groups that insist on riding together and
must NOT be split across gondolas).  The ride operator dispatches identical
gondolas, each with a fixed seat capacity C.  A gondola may carry any number of
parties as long as the total headcount does not exceed C.  Sending out a gondola
costs one dispatch (fuel + a full loop of the circuit).  The operator wants to
clear the whole queue using as FEW dispatched gondolas as possible.

This is 1-D bin packing skinned as a carnival circuit: parties = items (integer
sizes), gondola capacity = bin capacity, "gondolas dispatched" = bins used, which
we MINIMIZE.  The operator processes the queue with a dispatch heuristic; the model
supplies the packing.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "capacity": C (int), "n": N (int),
             "parties": [s_0, ..., s_{N-1}]   # integer headcounts, 1 <= s_i <= C}
  stdout: ONE JSON object:
            {"assign": [g_0, ..., g_{N-1}]}
          where g_i >= 0 is the gondola index party i boards.  Gondola indices
          need not be contiguous; a gondola "exists" iff >=1 party boards it, and
          the number of DISTINCT non-empty gondolas is the dispatch count.

  A layout is VALID iff `assign` is a list of exactly N non-negative integers and
  no gondola's boarded headcount exceeds C.  Invalid output, wrong length, an
  overfilled gondola, a crash, a timeout, or non-JSON -> that instance scores 0.0.

SCORING (deterministic; no wall-time).  Per instance we compute three references:
    q_lb   = L1 lower bound = ceil(sum(parties) / C)            # unreachable ideal
    q_base = dispatches used by the internal NEXT-FIT operator   # weak baseline
    q_cand = dispatches used by the candidate layout
  and normalize with an affine anchor (weak baseline -> 0.1, L1 ideal -> 1.0):
    r = clamp( 0.1 + 0.9 * (q_base - q_cand) / max(1e-9, q_base - q_lb), 0, 1 )
  A candidate matching next-fit scores ~0.1; a candidate reaching the (generally
  unreachable) L1 bound scores 1.0; doing worse than next-fit scores < 0.1.

  Because L1 is a LOOSE lower bound, even excellent packers (first-fit-decreasing /
  best-fit-decreasing) stay strictly below 1.0 on most instances -> headroom.

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  The references
(L1, next-fit baseline) are computed by THIS parent process, so a frame-walking /
introspecting candidate learns nothing useful.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
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
def _build_parties(seed, n, C, dist):
    """Return a list of N integer party sizes in [1, C]. Deterministic."""
    ni = _rng(seed)
    out = []
    for _ in range(n):
        if dist == "uni":
            s = ni(1, C)
        elif dist == "medium":                       # sizes near half a gondola
            s = ni(max(1, C // 4), (3 * C) // 4)
        elif dist == "bimodal":                      # many tiny + some large parties
            s = ni(1, max(1, C // 5)) if ni(0, 99) < 55 else ni((3 * C) // 5, (9 * C) // 10)
        elif dist == "heavy":                        # mostly large parties, hard to pair
            s = ni(max(1, (2 * C) // 5), (17 * C) // 20)
        else:
            s = ni(1, C)
        if s < 1:
            s = 1
        if s > C:
            s = C
        out.append(s)
    return out


def _build_instances():
    """Deterministic instance family. (seed, n, C, dist)."""
    specs = [
        (101, 24, 20, "medium"),
        (102, 28, 20, "medium"),
        (103, 30, 24, "medium"),
        (114, 26, 20, "bimodal"),
        (205, 34, 24, "medium"),
        (220, 30, 18, "medium"),
        (107, 32, 20, "medium"),
        (108, 28, 24, "heavy"),
        # harder / larger held-out instances
        (311, 45, 22, "medium"),
        (110, 44, 20, "bimodal"),
        (111, 48, 24, "medium"),
        (112, 52, 20, "heavy"),
    ]
    out = []
    for seed, n, C, dist in specs:
        parties = _build_parties(seed, n, C, dist)
        out.append({"name": f"circuit{seed}", "capacity": C, "n": n,
                    "parties": parties, "dist": dist})
    return out


# ----------------------------- references ----------------------------------
def _l1(parties, C):
    return -(-sum(parties) // C)          # ceil(sum / C)


def _next_fit(parties, C):
    """Weak online operator: keep filling the current gondola; open a new one
    the moment a party doesn't fit.  Never looks back."""
    bins = 1
    rem = C
    for s in parties:
        if s <= rem:
            rem -= s
        else:
            bins += 1
            rem = C - s
    return bins


# ----------------------------- validation ----------------------------------
def _dispatches(inst, answer):
    """Validate answer against the instance. Return dispatch count or None."""
    if not isinstance(answer, dict):
        return None
    assign = answer.get("assign")
    if not isinstance(assign, list):
        return None
    parties = inst["parties"]
    C = inst["capacity"]
    N = inst["n"]
    if len(assign) != N:
        return None
    load = {}
    for i, g in enumerate(assign):
        if isinstance(g, bool) or not isinstance(g, int):
            return None
        if g < 0:
            return None
        load[g] = load.get(g, 0) + parties[i]
        if load[g] > C:
            return None
    return len(load)          # number of distinct non-empty gondolas


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()
    n_inst = len(instances)

    vec = []
    for inst in instances:
        C = inst["capacity"]
        parties = inst["parties"]
        q_lb = _l1(parties, C)
        q_base = _next_fit(parties, C)
        denom = q_base - q_lb
        if denom < 1e-9:
            denom = 1e-9
        public = {"name": inst["name"], "capacity": C,
                  "n": inst["n"], "parties": list(parties)}
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
