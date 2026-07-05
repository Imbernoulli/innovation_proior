#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0112 -- "Contact-Net Cohorting: Quarantine Pod Assignment"
(family: online-heuristic-simulator; format B, quality-metric).

THEME.  During an outbreak, a public-health team must sort newly-flagged contacts
into physical **quarantine pods**.  Each contact carries an integer **viral load**
w (a triage number, 1..C) and belongs to a known **strain lineage** s (a small
integer label).  A pod has:
  * a total-load capacity C (sum of loads of its occupants must not exceed C), and
  * a *cross-contamination* limit K: a pod may host occupants of at most K DISTINCT
    strain lineages (mixing more lineages risks recombination and is forbidden).
Opening a pod costs one facility.  The team wants to house EVERY contact using as
FEW pods as possible.

This is 1-D bin packing with an added CLASS/COLOR constraint (a.k.a.
class-constrained bin packing): contacts = items with a size AND a color, pod
capacity = bin capacity, the K-limit caps distinct colors per bin, "pods" = bins,
which we MINIMIZE.  The color limit makes pure weight heuristics leave lineages
stranded, so it is genuinely harder than plain bin packing.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "capacity": C (int), "max_strains": K (int),
             "n": N (int),
             "loads":   [w_0, ..., w_{N-1}],   # integer viral loads, 1<=w_i<=C
             "strains": [s_0, ..., s_{N-1}]}    # integer lineage labels >=0
  stdout: ONE JSON object:
            {"assign": [p_0, ..., p_{N-1}]}
          where p_i >= 0 is the pod index contact i is housed in.  Pod indices
          need not be contiguous; a pod "exists" iff >=1 contact lives in it, and
          the number of DISTINCT non-empty pods is the facility count.

  A layout is VALID iff `assign` is a list of exactly N non-negative integers and
  every non-empty pod satisfies BOTH: total load <= C AND number of distinct
  strain lineages <= K.  Invalid output, wrong length, an over-capacity pod, a pod
  mixing more than K lineages, a crash, a timeout, or non-JSON -> that instance
  scores 0.0.

SCORING (deterministic; no wall-time).  Per instance we compute three references:
    q_lb   = L1 lower bound = ceil(sum(loads) / C)             # unreachable ideal
                                                               #  (ignores colors)
    q_base = pods used by the internal COLOR-AWARE NEXT-FIT operator  # weak base
    q_cand = pods used by the candidate layout
  and normalize with an affine anchor (weak baseline -> 0.1, L1 ideal -> 1.0):
    r = clamp( 0.1 + 0.9 * (q_base - q_cand) / max(1e-9, q_base - q_lb), 0, 1 )
  A candidate matching next-fit scores ~0.1; a candidate reaching the (generally
  unreachable) L1 bound scores 1.0; doing worse than next-fit scores < 0.1.

  Because L1 IGNORES the color limit, even excellent class-aware packers stay
  strictly below 1.0 on the color-dominated instances -> real headroom.

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
def _build_people(seed, n, C, n_strains, dist):
    """Return (loads, strains): N integer loads in [1,C] and N strain labels in
    [0, n_strains).  Deterministic."""
    ni = _rng(seed)
    loads, strains = [], []
    for _ in range(n):
        if dist == "small":                          # many tiny loads
            w = ni(1, max(1, C // 4))
        elif dist == "medium":                       # near half a pod
            w = ni(max(1, C // 4), (3 * C) // 4)
        elif dist == "mixed":                        # broad spread
            w = ni(1, C // 2)
        elif dist == "heavy":                        # mostly large loads
            w = ni(max(1, (2 * C) // 5), (4 * C) // 5)
        else:
            w = ni(1, C)
        if w < 1:
            w = 1
        if w > C:
            w = C
        loads.append(w)
        strains.append(ni(0, n_strains - 1))
    return loads, strains


def _build_instances():
    """Deterministic instance family. (seed, n, C, K, n_strains, dist)."""
    specs = [
        (101, 24, 20, 2, 5, "small"),
        (102, 26, 20, 2, 4, "medium"),
        (103, 28, 24, 2, 5, "small"),
        (114, 26, 20, 3, 6, "mixed"),
        (205, 30, 24, 2, 4, "medium"),
        (220, 28, 18, 2, 5, "small"),
        (107, 30, 20, 3, 6, "mixed"),
        (108, 26, 24, 2, 3, "heavy"),
        # harder / larger held-out instances
        (311, 42, 22, 2, 6, "small"),
        (110, 40, 20, 2, 5, "medium"),
        (111, 46, 24, 3, 7, "mixed"),
        (112, 48, 20, 2, 4, "heavy"),
    ]
    out = []
    for seed, n, C, K, n_strains, dist in specs:
        loads, strains = _build_people(seed, n, C, n_strains, dist)
        out.append({"name": f"pods{seed}", "capacity": C, "max_strains": K,
                    "n": n, "loads": loads, "strains": strains, "dist": dist})
    return out


# ----------------------------- references ----------------------------------
def _l1(loads, C):
    return -(-sum(loads) // C)            # ceil(sum / C)


def _next_fit(loads, strains, C, K):
    """Weak online operator: keep the current pod open; admit each arriving
    contact iff the load still fits AND it would not push the pod above K distinct
    lineages; otherwise close it and open a fresh pod.  Never looks back."""
    pods = 1
    rem = C
    cur = set()
    for w, s in zip(loads, strains):
        ok_load = w <= rem
        ok_color = (s in cur) or (len(cur) < K)
        if ok_load and ok_color:
            rem -= w
            cur.add(s)
        else:
            pods += 1
            rem = C - w
            cur = {s}
    return pods


# ----------------------------- validation ----------------------------------
def _pods_used(inst, answer):
    """Validate answer against the instance. Return pod count or None."""
    if not isinstance(answer, dict):
        return None
    assign = answer.get("assign")
    if not isinstance(assign, list):
        return None
    loads = inst["loads"]
    strains = inst["strains"]
    C = inst["capacity"]
    K = inst["max_strains"]
    N = inst["n"]
    if len(assign) != N:
        return None
    load = {}
    colors = {}
    for i, p in enumerate(assign):
        if isinstance(p, bool) or not isinstance(p, int):
            return None
        if p < 0:
            return None
        load[p] = load.get(p, 0) + loads[i]
        if load[p] > C:
            return None
        cs = colors.setdefault(p, set())
        cs.add(strains[i])
        if len(cs) > K:
            return None
    return len(load)          # number of distinct non-empty pods


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
        K = inst["max_strains"]
        loads = inst["loads"]
        strains = inst["strains"]
        q_lb = _l1(loads, C)
        q_base = _next_fit(loads, strains, C, K)
        denom = q_base - q_lb
        if denom < 1e-9:
            denom = 1e-9
        public = {"name": inst["name"], "capacity": C, "max_strains": K,
                  "n": inst["n"], "loads": list(loads), "strains": list(strains)}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            q_cand = _pods_used(inst, ans)
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
