#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0302 -- "Aperture Array: Exposure-Block Scheduling"
(family: online-heuristic-simulator; format B, quality-metric).

THEME.  An aperture-synthesis radio telescope ARRAY works through a queue of
observation requests.  Each request has an integer EXPOSURE DURATION (in tracking
minutes).  The correlator can only integrate a fixed budget `capacity` of exposure
minutes into a single EXPOSURE BLOCK before it must be flushed and a fresh block
opened.  Requests arrive in a fixed order (the night's dynamically-scheduled queue)
and every request must be assigned to some exposure block; the total duration packed
into any one block may never exceed `capacity`.

Opening a block costs correlator reconfiguration + calibration overhead, so the array
operator wants to complete the whole queue using the FEWEST exposure blocks.  This is
online 1-D bin packing skinned as a telescope-array scheduling contest: the arrival
order and the hard per-block capacity are fixed by the instance, and the candidate
must decide, for every request, which block it lands in.

    objective(assignment) = number of DISTINCT exposure blocks used   (MINIMIZE)

The tension is fragmentation vs. reuse: a myopic "keep one block open, flush when the
next request won't fit" policy (Next-Fit) wastes capacity because a small request that
would have topped off an earlier block instead forces a fresh one.  Reusing earlier
blocks (First-Fit), and reordering by duration (Best-Fit-Decreasing) pack far tighter.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str,
             "capacity": C (int),          # exposure-minute budget per block
             "items": [d0, d1, ...]}       # request durations in ARRIVAL order, each 1<=d<=C
  stdout: ONE JSON object:
            {"assign": [b0, b1, ...]}       # b_i = exposure-block index for request i
          `assign` has EXACTLY len(items) entries; each b_i is an integer >= 0.  A block's
          total packed duration must not exceed `capacity`.  The number of blocks USED is
          the count of DISTINCT indices appearing in `assign` (indices need not be
          contiguous).  Wrong length, a non-integer / negative index, any block over
          capacity, a crash, a timeout, or non-JSON  ->  that instance scores 0.0.

SCORING (deterministic; no wall-time).  Per instance we compute two references:
    B_nf = blocks used by NEXT-FIT (one open block; flush and open a new one whenever the
           next request does not fit).  This overlap-blind online policy is the weak
           baseline.
    L1   = ceil( sum(items) / capacity ).  The area lower bound: no packing can use fewer
           blocks.  Generally UNREACHABLE (fragmentation), so even excellent schedules
           stay below 1.0 -> headroom.
    B    = blocks used by the candidate's (validated) assignment.
  normalized with an affine anchor (weak baseline -> 0.1, area-optimal -> 1.0):
    r = clamp( 0.1 + 0.9 * (B_nf - B) / max(1e-9, B_nf - L1), 0, 1 )
  Reproducing Next-Fit scores ~0.1; using more blocks scores < 0.1; packing tighter
  scores higher, capped at 1.0.  Final score = mean of r over all instances (a mix of
  sizes, capacities and duration distributions, including harder held-out queues whose
  near-half-capacity requests fragment badly).

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  Both references and all
validation happen in THIS parent process, so a frame-walking / introspecting candidate
learns nothing that helps it pack.

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
def _gen_items(seed, n, lo, hi):
    ni = _rng(seed)
    return [ni(lo, hi) for _ in range(n)]


def _build_instances():
    """Deterministic queue family: (seed, n, lo, hi, capacity).

    Duration ranges span uniform-wide, near-half-capacity (heavy fragmentation),
    and mixed regimes; capacities vary.  The last few are harder / larger held-out
    queues whose requests cluster just above C/2 so blocks rarely fill.
    """
    specs = [
        (101, 30, 1, 100, 100),
        (102, 40, 1, 100, 100),
        (103, 50, 20, 80, 100),
        (104, 40, 30, 100, 100),
        (105, 60, 1, 100, 100),
        (107, 45, 40, 90, 100),
        (108, 60, 10, 70, 100),
        (110, 50, 25, 75, 100),
        (111, 70, 1, 100, 100),
        # harder / larger held-out queues (near-half-capacity -> fragment badly)
        (112, 55, 45, 95, 100),
        (113, 48, 50, 99, 120),
        (114, 64, 30, 120, 150),
    ]
    out = []
    for (seed, n, lo, hi, cap) in specs:
        items = _gen_items(seed, n, lo, hi)
        out.append({"name": f"queue{seed}", "capacity": cap, "items": items})
    return out


# ----------------------------- references / scoring ------------------------
def _next_fit_blocks(items, cap):
    blocks = 1
    cur = 0
    for x in items:
        if cur + x <= cap:
            cur += x
        else:
            blocks += 1
            cur = x
    return blocks


def _l1_bound(items, cap):
    return int(math.ceil(sum(items) / cap))


def _validate(inst, answer):
    """Validate assignment; return blocks_used (int) or None if infeasible."""
    if not isinstance(answer, dict):
        return None
    assign = answer.get("assign")
    if not isinstance(assign, list):
        return None
    items = inst["items"]
    cap = inst["capacity"]
    if len(assign) != len(items):
        return None
    loads = {}
    for i, b in enumerate(assign):
        if isinstance(b, bool) or not isinstance(b, int):
            return None
        if b < 0:
            return None
        loads[b] = loads.get(b, 0) + items[i]
        if loads[b] > cap:
            return None
    return len(loads)


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        items = inst["items"]
        cap = inst["capacity"]
        b_nf = _next_fit_blocks(items, cap)
        l1 = _l1_bound(items, cap)
        denom = b_nf - l1
        if denom < 1e-9:
            denom = 1e-9
        public = {"name": inst["name"], "capacity": cap, "items": list(items)}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            blocks = _validate(inst, ans)
        except Exception:
            blocks = None
        if blocks is None:
            vec.append(0.0)
            continue
        r = 0.1 + 0.9 * (b_nf - blocks) / denom
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
