#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0102 -- "Solstice Festival: Main-Stage Slotting"
(family: online-heuristic-simulator; format B, quality-metric).

THEME.  A large open-air music festival lays out its evening lineup on a bank of
IDENTICAL main stages.  Acts (bands, DJs, dance troupes) are booked in a fixed
running order.  Every act reserves a "footprint" of stage resources -- backline
power draw, crew, floor area -- summarised as one integer.  A single stage can host
several acts across the evening as long as (a) the acts' combined footprint does not
exceed the stage's resource capacity C, AND (b) no more than K acts are slotted on
that stage (only K soundcheck / changeover windows exist per stage).  Rolling out a
stage -- trucks, rigging, a full sound system, a licensed crew -- costs one
"stage".  The festival wants to place every booked act using as FEW stages as
possible.

This is CARDINALITY-CONSTRAINED 1-D bin packing skinned as a festival: acts = items
(integer sizes), stage capacity C = bin capacity, per-stage act cap K = bin
cardinality limit, "stages rolled out" = bins used, which we MINIMIZE.  The extra
count limit K is what distinguishes this from plain bin packing: a stage can be
"full" either because it is out of resource capacity OR because it already holds K
acts, so a good layout must balance BOTH constraints at once.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "capacity": C (int), "max_acts": K (int), "n": N (int),
             "acts": [s_0, ..., s_{N-1}]   # integer footprints, 1 <= s_i <= C}
  stdout: ONE JSON object:
            {"assign": [g_0, ..., g_{N-1}]}
          where g_i >= 0 is the stage index act i is slotted on.  Stage indices need
          not be contiguous; a stage "exists" iff >=1 act is slotted on it, and the
          number of DISTINCT non-empty stages is the stage count.

  A layout is VALID iff `assign` is a list of exactly N non-negative integers and,
  for every stage, (i) the total footprint of its acts does not exceed C, and
  (ii) the number of acts on it does not exceed K.  Invalid output, wrong length,
  an over-capacity stage, a stage with > K acts, a crash, a timeout, or non-JSON
  -> that instance scores 0.0.

SCORING (deterministic; no wall-time).  Per instance we compute three references:
    q_lb   = combined L1 lower bound = max( ceil(sum(acts)/C), ceil(N/K) )   # ideal
    q_base = stages used by the internal NEXT-FIT operator (a weak online rule)
    q_cand = stages used by the candidate layout
  and normalize with an affine anchor (weak baseline -> 0.1, ideal -> 1.0):
    r = clamp( 0.1 + 0.9 * (q_base - q_cand) / max(1e-9, q_base - q_lb), 0, 1 )
  A candidate matching next-fit scores ~0.1; a candidate reaching the (generally
  unreachable) combined L1 bound scores 1.0; doing worse than next-fit scores < 0.1.

  Because the L1 bound is LOOSE (it ignores how sizes actually combine), even strong
  packers (first-fit-decreasing / best-fit-decreasing with the count cap) stay
  strictly below 1.0 on most instances -> real headroom, no easy optimum.

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  The references
(L1, next-fit baseline) are computed by THIS parent process, so a frame-walking /
introspecting candidate learns nothing useful.

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
def _build_acts(seed, n, C, dist):
    """Return a list of N integer act footprints in [1, C]. Deterministic."""
    ni = _rng(seed)
    out = []
    for _ in range(n):
        if dist == "uni":
            s = ni(1, C)
        elif dist == "medium":                       # footprints near half a stage
            s = ni(max(1, C // 4), (3 * C) // 4)
        elif dist == "bimodal":                      # many small headliners + big rigs
            s = ni(1, max(1, C // 5)) if ni(0, 99) < 55 else ni((3 * C) // 5, (9 * C) // 10)
        elif dist == "heavy":                        # mostly big stage rigs, hard to pair
            s = ni(max(1, (2 * C) // 5), (17 * C) // 20)
        elif dist == "light":                        # many small acts -> count cap bites
            s = ni(1, max(1, C // 6))
        else:
            s = ni(1, C)
        if s < 1:
            s = 1
        if s > C:
            s = C
        out.append(s)
    return out


def _build_instances():
    """Deterministic instance family. (seed, n, C, K, dist)."""
    specs = [
        (401, 40, 30, 6, "medium"),
        (402, 44, 30, 5, "medium"),
        (403, 48, 28, 6, "bimodal"),
        (404, 42, 32, 4, "medium"),
        (405, 50, 30, 7, "heavy"),
        (406, 46, 24, 5, "light"),
        (407, 52, 30, 6, "medium"),
        (408, 48, 28, 4, "bimodal"),
        # harder / larger held-out instances
        (511, 72, 32, 6, "medium"),
        (512, 80, 30, 5, "bimodal"),
        (513, 66, 28, 4, "light"),
        (514, 88, 30, 6, "heavy"),
    ]
    out = []
    for seed, n, C, K, dist in specs:
        acts = _build_acts(seed, n, C, dist)
        out.append({"name": f"fest{seed}", "capacity": C, "max_acts": K,
                    "n": n, "acts": acts, "dist": dist})
    return out


# ----------------------------- references ----------------------------------
def _l1(acts, C, K, N):
    cap_lb = -(-sum(acts) // C)            # ceil(sum / C)
    cnt_lb = -(-N // K)                    # ceil(N / K)
    return max(cap_lb, cnt_lb)


def _next_fit(acts, C, K):
    """Weak online operator: keep loading the current stage; open a new stage the
    moment the next act does not fit -- either it would exceed capacity C or the
    stage already holds K acts.  Never looks back."""
    stages = 1
    rem = C
    cnt = 0
    for s in acts:
        if s <= rem and cnt < K:
            rem -= s
            cnt += 1
        else:
            stages += 1
            rem = C - s
            cnt = 1
    return stages


# ----------------------------- validation ----------------------------------
def _stages(inst, answer):
    """Validate answer against the instance. Return stage count or None."""
    if not isinstance(answer, dict):
        return None
    assign = answer.get("assign")
    if not isinstance(assign, list):
        return None
    acts = inst["acts"]
    C = inst["capacity"]
    K = inst["max_acts"]
    N = inst["n"]
    if len(assign) != N:
        return None
    load = {}
    cnt = {}
    for i, g in enumerate(assign):
        if isinstance(g, bool) or not isinstance(g, int):
            return None
        if g < 0:
            return None
        load[g] = load.get(g, 0) + acts[i]
        cnt[g] = cnt.get(g, 0) + 1
        if load[g] > C:
            return None
        if cnt[g] > K:
            return None
    return len(load)          # number of distinct non-empty stages


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        C = inst["capacity"]
        K = inst["max_acts"]
        acts = inst["acts"]
        N = inst["n"]
        q_lb = _l1(acts, C, K, N)
        q_base = _next_fit(acts, C, K)
        denom = q_base - q_lb
        if denom < 1e-9:
            denom = 1e-9
        public = {"name": inst["name"], "capacity": C, "max_acts": K,
                  "n": N, "acts": list(acts)}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            q_cand = _stages(inst, ans)
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
