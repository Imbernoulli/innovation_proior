#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0092 -- "Rift Valley Geothermal: Brine Reinjection Loops"
(family: online-heuristic-simulator; format B, quality-metric).

THEME.  A geothermal power station taps a rift-valley field.  After the turbines,
spent brine from many production WELLS must be reinjected back into the ground
through a set of injection LOOPS (pump-and-pipe circuits).  Every loop has the same
maximum hydraulic THROUGHPUT capacity C (integer, litres/s).  Several wells may
share one loop as long as their combined flow does not exceed C.

BUT brine is hot, and mixing streams of very different temperature in one loop
cracks the pipes through thermal shock.  So a loop is only physically safe if the
temperatures of ALL wells sharing it lie within a fixed BAND of width `band`
(i.e. max_temp - min_temp <= band).  Firing up a loop (its pump + a full pipe
circuit) costs one unit regardless of how full it is.  The operator wants to
reinject every well's brine using as FEW active loops as possible.

This is 1-D bin packing WITH A COMPATIBILITY (conflict) CONSTRAINT skinned as a
geothermal field:  wells = items with an integer size (flow) AND a temperature
tag; loops = bins of capacity C; a bin is feasible only if it is not overfilled
AND its temperature spread stays within `band`.  We MINIMIZE the number of active
loops.  The temperature band is what makes this NOT plain bin packing: two wells
that would happily share on size can be forbidden to share on temperature, so a
size-only packer is easily beaten by one that also clusters on temperature.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "capacity": C (int), "band": B (int), "n": N (int),
             "flow": [f_0, ..., f_{N-1}],   # integer flows, 1 <= f_i <= C
             "temp": [t_0, ..., t_{N-1}]}   # integer temperatures, >= 0
  stdout: ONE JSON object:
            {"assign": [g_0, ..., g_{N-1}]}
          where g_i >= 0 is the loop index well i is connected to.  Loop indices
          need not be contiguous; a loop "exists" iff >=1 well joins it, and the
          number of DISTINCT non-empty loops is the active-loop count.

  A layout is VALID iff `assign` is a list of exactly N non-negative integers and,
  for EVERY loop, (a) the total flow of its wells does not exceed C and (b) the
  temperature spread (max - min) of its wells does not exceed `band`.  Invalid
  output, wrong length, an overfilled loop, a temperature-band violation, a crash,
  a timeout, or non-JSON -> that instance scores 0.0.

SCORING (deterministic; no wall-time).  Per instance we compute three references:
    q_lb   = L1 lower bound = ceil(sum(flow) / C)                 # unreachable ideal
    q_base = loops used by the internal constrained NEXT-FIT rule   # weak baseline
    q_cand = active loops used by the candidate layout
  and normalize with an affine anchor (weak baseline -> 0.1, L1 ideal -> 1.0):
    r = clamp( 0.1 + 0.9 * (q_base - q_cand) / max(1e-9, q_base - q_lb), 0, 1 )
  A candidate matching next-fit scores ~0.1; a candidate reaching the (generally
  unreachable) L1 bound scores 1.0; doing worse than next-fit scores < 0.1.

  Because L1 IGNORES the temperature band, it is a LOOSE lower bound: even an
  excellent temperature-aware packer stays strictly below 1.0 on most instances,
  which leaves real headroom.

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  The references
(L1, constrained next-fit) are computed by THIS parent process, so a frame-walking
/ introspecting candidate learns nothing useful.

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
def _build_wells(seed, n, C, fdist, tdist, tmax):
    """Return (flow[], temp[]) of N wells. Deterministic."""
    ni = _rng(seed)
    flow = []
    temp = []
    for _ in range(n):
        # ---- flow (hydraulic size) ----
        if fdist == "uni":
            f = ni(1, C)
        elif fdist == "medium":                       # near half a loop
            f = ni(max(1, C // 4), (3 * C) // 4)
        elif fdist == "bimodal":                      # many tiny + some large
            f = ni(1, max(1, C // 5)) if ni(0, 99) < 55 else ni((3 * C) // 5, (9 * C) // 10)
        elif fdist == "heavy":                         # mostly large, hard to pair
            f = ni(max(1, (2 * C) // 5), (17 * C) // 20)
        else:
            f = ni(1, C)
        f = 1 if f < 1 else (C if f > C else f)
        # ---- temperature tag ----
        if tdist == "spread":                          # spread across full range
            t = ni(0, tmax)
        elif tdist == "clustered":                     # a handful of hot spots
            centers = (10, 45, 80, 120)
            c = centers[ni(0, len(centers) - 1)]
            t = c + ni(-8, 8)
        else:
            t = ni(0, tmax)
        if t < 0:
            t = 0
        flow.append(f)
        temp.append(t)
    return flow, temp


def _build_instances():
    """Deterministic instance family. (seed, n, C, fdist, tdist, tmax, band)."""
    specs = [
        (101, 22, 20, "medium",  "clustered", 130, 24),
        (102, 26, 20, "medium",  "spread",    120, 30),
        (103, 28, 24, "medium",  "clustered", 130, 22),
        (114, 24, 20, "bimodal", "clustered", 130, 26),
        (205, 30, 24, "medium",  "spread",    120, 28),
        (220, 26, 18, "uni",     "clustered", 130, 24),
        (107, 28, 20, "medium",  "spread",    120, 26),
        (108, 26, 24, "heavy",   "clustered", 130, 30),
        # harder / larger held-out instances
        (311, 40, 22, "medium",  "spread",    120, 24),
        (110, 42, 20, "bimodal", "clustered", 130, 22),
        (111, 44, 24, "medium",  "spread",    120, 28),
        (112, 46, 20, "heavy",   "clustered", 130, 26),
    ]
    out = []
    for seed, n, C, fdist, tdist, tmax, band in specs:
        flow, temp = _build_wells(seed, n, C, fdist, tdist, tmax)
        out.append({"name": f"field{seed}", "capacity": C, "band": band,
                    "n": n, "flow": flow, "temp": temp,
                    "fdist": fdist, "tdist": tdist})
    return out


# ----------------------------- references ----------------------------------
def _l1(flow, C):
    return -(-sum(flow) // C)          # ceil(sum / C); loose (ignores temperature)


def _next_fit(flow, temp, C, band):
    """Weak online operator: keep filling the CURRENT loop; open a new one the
    moment a well doesn't fit -- on flow OR temperature band.  Never looks back."""
    loops = 0
    rem = -1        # remaining capacity of current loop; -1 = no loop open yet
    tmin = tmax = 0
    for f, t in zip(flow, temp):
        if rem >= 0 and f <= rem:
            nmin = t if t < tmin else tmin
            nmax = t if t > tmax else tmax
            if nmax - nmin <= band:
                rem -= f
                tmin, tmax = nmin, nmax
                continue
        # open a new loop
        loops += 1
        rem = C - f
        tmin = tmax = t
    return loops


# ----------------------------- validation ----------------------------------
def _active_loops(inst, answer):
    """Validate answer against the instance. Return active-loop count or None."""
    if not isinstance(answer, dict):
        return None
    assign = answer.get("assign")
    if not isinstance(assign, list):
        return None
    flow = inst["flow"]
    temp = inst["temp"]
    C = inst["capacity"]
    band = inst["band"]
    N = inst["n"]
    if len(assign) != N:
        return None
    load = {}
    tlo = {}
    thi = {}
    for i, g in enumerate(assign):
        if isinstance(g, bool) or not isinstance(g, int):
            return None
        if g < 0:
            return None
        load[g] = load.get(g, 0) + flow[i]
        if load[g] > C:
            return None
        ti = temp[i]
        if g not in tlo:
            tlo[g] = thi[g] = ti
        else:
            if ti < tlo[g]:
                tlo[g] = ti
            if ti > thi[g]:
                thi[g] = ti
        if thi[g] - tlo[g] > band:
            return None
    return len(load)


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
        band = inst["band"]
        flow = inst["flow"]
        temp = inst["temp"]
        q_lb = _l1(flow, C)
        q_base = _next_fit(flow, temp, C, band)
        denom = q_base - q_lb
        if denom < 1e-9:
            denom = 1e-9
        public = {"name": inst["name"], "capacity": C, "band": band,
                  "n": inst["n"], "flow": list(flow), "temp": list(temp)}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            q_cand = _active_loops(inst, ans)
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
