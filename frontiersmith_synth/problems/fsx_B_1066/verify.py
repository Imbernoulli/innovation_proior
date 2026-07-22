#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for "Rainproof Season".

Feasibility: the artifact must be a full single round-robin schedule (every
unordered pair of teams exactly once) placed on (date,court) slots such that
no team plays twice on the same date, no two games share a (date,court) slot,
and all indices are in range. Any violation -> Ratio: 0.0.

Objective (minimize): simulate each weather scenario's FIXED makeup rule
(postponed game -> earliest later date, not itself cancelled, on which both
teams and a court are free) and score the worst-case scenario cost
(total makeup delay + LAMBDA * rest-fairness shortfall). Normalize against
an internal naive baseline schedule (first-fit lexicographic coloring).
"""
import sys
import math


def canonical_pairs(N):
    return [(i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1)]


def first_fit_schedule(N, D, C):
    busy = [set() for _ in range(D + 1)]
    ccount = [0] * (D + 1)
    sched = []
    for (i, j) in canonical_pairs(N):
        placed = False
        for d in range(1, D + 1):
            if ccount[d] < C and i not in busy[d] and j not in busy[d]:
                ccount[d] += 1
                busy[d].add(i)
                busy[d].add(j)
                sched.append((d, ccount[d]))
                placed = True
                break
        if not placed:
            return None
    return sched


def simulate(N, D, C, R, LAM, scenarios, sched, pairs):
    """Return the worst-case (max over scenarios) scenario cost of `sched`.
    sched: list[(date,court)] indexed by canonical game order."""
    worst = 0.0
    ngames = len(pairs)
    for scenario in scenarios:
        Sset = scenario
        team_busy = {}
        court_used = {}
        final_date = [0] * ngames
        hit = []
        for idx, (i, j) in enumerate(pairs):
            d, c = sched[idx]
            if d in Sset:
                hit.append(idx)
            else:
                final_date[idx] = d
                team_busy.setdefault(d, set()).update((i, j))
                court_used.setdefault(d, set()).add(c)
        hit.sort(key=lambda idx: (sched[idx][0], sched[idx][1], idx))
        cap = D + ngames + 5
        for idx in hit:
            i, j = pairs[idx]
            d0, c0 = sched[idx]
            d = d0 + 1
            placed = False
            while d <= cap:
                if d not in Sset:
                    tb = team_busy.get(d, ())
                    cu = court_used.get(d, ())
                    if i not in tb and j not in tb and len(cu) < C:
                        cchoice = None
                        for cc in range(1, C + 1):
                            if cc not in cu:
                                cchoice = cc
                                break
                        team_busy.setdefault(d, set()).update((i, j))
                        court_used.setdefault(d, set()).add(cchoice)
                        final_date[idx] = d
                        placed = True
                        break
                d += 1
            if not placed:
                final_date[idx] = cap  # safety fallback (should not trigger)
        delay_sum = sum(final_date[idx] - sched[idx][0] for idx in hit)

        team_dates = {t: [] for t in range(1, N + 1)}
        for idx, (i, j) in enumerate(pairs):
            fd = final_date[idx]
            team_dates[i].append(fd)
            team_dates[j].append(fd)
        rest_pen = 0
        for t in range(1, N + 1):
            ds = sorted(team_dates[t])
            for a, b in zip(ds, ds[1:]):
                gap = b - a
                if gap < R:
                    rest_pen += (R - gap)

        cost = delay_sum + LAM * rest_pen
        if cost > worst:
            worst = cost
    return worst


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    N = int(next(it)); D = int(next(it)); C = int(next(it))
    K = int(next(it)); R = int(next(it)); LAM = int(next(it))
    scenarios = []
    for _ in range(K):
        b = int(next(it))
        dates = frozenset(int(next(it)) for _ in range(b))
        scenarios.append(dates)
    return N, D, C, K, R, LAM, scenarios


def fail(msg):
    print("INVALID: %s Ratio: 0.0" % msg)
    sys.exit(0)


def main():
    if len(sys.argv) < 3:
        fail("bad args")
    in_path, out_path = sys.argv[1], sys.argv[2]
    N, D, C, K, R, LAM, scenarios = read_instance(in_path)
    pairs = canonical_pairs(N)
    ngames = len(pairs)

    with open(out_path) as f:
        raw = f.read().split()

    if len(raw) != 2 * ngames:
        fail(f"expected exactly {2 * ngames} tokens ({ngames} lines of 'date court'), got {len(raw)}")

    vals = []
    for tok in raw[:2 * ngames]:
        try:
            v = float(tok)
        except ValueError:
            fail("non-numeric token")
        if not math.isfinite(v):
            fail("non-finite token")
        iv = int(round(v))
        if abs(v - iv) > 1e-6:
            fail("non-integer token")
        vals.append(iv)

    sched = []
    used_slots = set()
    for g in range(ngames):
        d = vals[2 * g]
        c = vals[2 * g + 1]
        if not (1 <= d <= D):
            fail(f"game {g} date {d} out of range [1,{D}]")
        if not (1 <= c <= C):
            fail(f"game {g} court {c} out of range [1,{C}]")
        if (d, c) in used_slots:
            fail(f"duplicate slot (date={d},court={c})")
        used_slots.add((d, c))
        sched.append((d, c))

    date_teams = {}
    for idx, (i, j) in enumerate(pairs):
        d, c = sched[idx]
        s = date_teams.setdefault(d, set())
        if i in s or j in s:
            fail(f"team plays twice on date {d}")
        s.add(i); s.add(j)

    F = simulate(N, D, C, R, LAM, scenarios, sched, pairs)

    base_sched = first_fit_schedule(N, D, C)
    if base_sched is None:
        fail("internal: baseline infeasible for this instance")
    B = simulate(N, D, C, R, LAM, scenarios, base_sched, pairs)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("F=%.6f B=%.6f Ratio: %.6f" % (F, B, sc / 1000.0))
    sys.exit(0)


if __name__ == "__main__":
    main()
