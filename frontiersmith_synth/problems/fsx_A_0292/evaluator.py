#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0292 -- "The Grand Atrium: Docent Gallery Tours"
(family: online-heuristic-simulator; format B, quality-metric).

THEME.  A large art museum clears its morning queue by grouping visitors into
GUIDED GALLERY TOURS, each led by one docent.  Visitors arrive already clustered
into *tour groups* (families, school cohorts, guided-club parties) that insist on
staying together and therefore must be assigned ENTIRELY to a single tour.  A tour
is constrained by TWO independent resources:

  * CROWD capacity C -- the fire-code head limit for one docent-led party moving
    through the galleries: the total number of PEOPLE on a tour may not exceed C.
  * DOCENT-MINUTE budget T -- one docent's shift provides T minutes of dedicated
    narration / accessibility support; each group demands f_i minutes of that
    attention, and the total demanded by a tour may not exceed T.

Every group must ride exactly one tour; a tour is valid iff its total headcount
<= C AND its total docent-minute demand <= T.  The museum wants to clear the whole
queue using as FEW guided tours (docent shifts) as possible.

This is 2-D VECTOR bin packing skinned as a museum: each group is an item with a
two-component demand (people, docent-minutes); each tour is a bin with a two-
component capacity (C, T); "tours used" = bins, which we MINIMIZE.  Unlike 1-D
packing, a good layout must fill BOTH resources -- pairing a large-but-quick group
with a small-but-demanding one fills a tour on both axes at once.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "C": int, "T": int, "n": int,
             "people":  [p_0, ..., p_{n-1}],   # 1 <= p_i <= C
             "minutes": [f_0, ..., f_{n-1}]}    # 1 <= f_i <= T
  stdout: ONE JSON object:
            {"assign": [t_0, ..., t_{n-1}]}
          where t_i >= 0 is the tour index group i joins.  Tour indices need not be
          contiguous; a tour "exists" iff >=1 group joins it, and the number of
          DISTINCT non-empty tours is the objective we score.

  A layout is VALID iff `assign` is a list of exactly n non-negative integers and
  NO tour exceeds C people OR T docent-minutes.  Invalid output, wrong length, an
  over-capacity or over-time tour, a crash, a timeout, or non-JSON -> 0.0 on that
  instance.

SCORING (deterministic; no wall-time).  Per instance we compute three references:
    q_lb   = 2-D volume lower bound = max( ceil(sum(people)/C), ceil(sum(minutes)/T) )
    q_base = tours used by the internal 2-D NEXT-FIT operator   # weak baseline
    q_cand = tours used by the candidate layout
  and normalize with an affine anchor (weak baseline -> 0.1, volume ideal -> 1.0):
    r = clamp( 0.1 + 0.9 * (q_base - q_cand) / max(1e-9, q_base - q_lb), 0, 1 )
  A candidate matching next-fit scores ~0.1; a candidate reaching the (generally
  unreachable) volume bound scores 1.0; doing worse than next-fit scores < 0.1.

  Because the volume bound ignores the combinatorics of pairing two axes, even
  strong decreasing-order / best-fit packers stay strictly below 1.0 on most
  instances -> real headroom.

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  The references
(volume bound, next-fit baseline) are computed by THIS parent process, so a
frame-walking / introspecting candidate learns nothing useful.

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
def _build_groups(seed, n, C, T, dist):
    """Return (people[], minutes[]) of length n. Deterministic; 1<=p<=C, 1<=f<=T."""
    ni = _rng(seed)
    people, minutes = [], []
    for _ in range(n):
        if dist == "unif":                       # full-range on both axes
            p = ni(1, C); f = ni(1, T)
        elif dist == "small":                    # many small groups; packing matters
            p = ni(1, max(1, C // 3)); f = ni(1, max(1, T // 3))
        elif dist == "crowd_bind":               # crowd is the binding axis
            p = ni(max(1, C // 3), C // 2); f = ni(1, max(1, T // 4))
        elif dist == "time_bind":                # docent-minutes is the binding axis
            p = ni(1, max(1, C // 4)); f = ni(T // 3, T // 2)
        elif dist == "anti":                     # anti-correlated: big+quick vs small+demanding
            if ni(0, 99) < 50:
                p = ni(max(1, (3 * C) // 5), (9 * C) // 10); f = ni(1, max(1, T // 6))
            else:
                p = ni(1, max(1, C // 6)); f = ni((3 * T) // 5, (9 * T) // 10)
        else:
            p = ni(1, C); f = ni(1, T)
        p = min(C, max(1, p)); f = min(T, max(1, f))
        people.append(p); minutes.append(f)
    return people, minutes


def _build_instances():
    """Deterministic instance family. (seed, n, C, T, dist)."""
    specs = [
        (101, 40, 24, 100, "unif"),
        (102, 44, 24, 100, "small"),
        (103, 48, 22,  90, "crowd_bind"),
        (104, 42, 26, 110, "time_bind"),
        (105, 46, 24, 100, "anti"),
        (106, 50, 20,  90, "small"),
        (107, 40, 24, 100, "anti"),
        (108, 45, 22,  96, "unif"),
        # harder / larger held-out instances
        (211, 72, 24, 100, "anti"),
        (212, 80, 22,  90, "small"),
        (213, 68, 26, 110, "time_bind"),
        (214, 90, 24, 100, "unif"),
        (215, 84, 20,  88, "crowd_bind"),
    ]
    out = []
    for seed, n, C, T, dist in specs:
        people, minutes = _build_groups(seed, n, C, T, dist)
        out.append({"name": f"atrium{seed}", "C": C, "T": T, "n": n,
                    "people": people, "minutes": minutes, "dist": dist})
    return out


# ----------------------------- references ----------------------------------
def _volume_lb(people, minutes, C, T):
    lp = -(-sum(people) // C)     # ceil(sum(people) / C)
    lf = -(-sum(minutes) // T)    # ceil(sum(minutes) / T)
    return max(lp, lf)


def _next_fit(people, minutes, C, T):
    """Weak online operator: keep loading the current tour; open a new tour the
    moment the next group would violate EITHER the crowd cap or the time budget.
    Never looks back."""
    tours = 1
    rp, rf = C, T
    for p, f in zip(people, minutes):
        if p <= rp and f <= rf:
            rp -= p; rf -= f
        else:
            tours += 1
            rp = C - p; rf = T - f
    return tours


# ----------------------------- validation ----------------------------------
def _tours_used(inst, answer):
    """Validate answer against the instance. Return distinct-tour count or None."""
    if not isinstance(answer, dict):
        return None
    assign = answer.get("assign")
    if not isinstance(assign, list):
        return None
    people = inst["people"]; minutes = inst["minutes"]
    C = inst["C"]; T = inst["T"]; n = inst["n"]
    if len(assign) != n:
        return None
    load_p, load_f = {}, {}
    for i, t in enumerate(assign):
        if isinstance(t, bool) or not isinstance(t, int):
            return None
        if t < 0:
            return None
        load_p[t] = load_p.get(t, 0) + people[i]
        load_f[t] = load_f.get(t, 0) + minutes[i]
        if load_p[t] > C or load_f[t] > T:
            return None
    return len(load_p)


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        C = inst["C"]; T = inst["T"]
        people = inst["people"]; minutes = inst["minutes"]
        q_lb = _volume_lb(people, minutes, C, T)
        q_base = _next_fit(people, minutes, C, T)
        denom = q_base - q_lb
        if denom < 1e-9:
            denom = 1e-9
        public = {"name": inst["name"], "C": C, "T": T, "n": inst["n"],
                  "people": list(people), "minutes": list(minutes)}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            q_cand = _tours_used(inst, ans)
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
