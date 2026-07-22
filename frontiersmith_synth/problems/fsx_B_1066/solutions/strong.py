# TIER: strong
"""Genuine insight: this is a covering problem over scenario-correlated slot
sets, not a compaction problem. Start from the textbook compact round robin,
then run simulated annealing that directly minimizes the checker's own
worst-case-scenario objective, using the K weather scenarios given in the
input. This lets slack dates get planted exactly where jointly-cancelled
pairs of teams need a shared nearby free date, instead of being wasted (or
never used at all, as the compact/first-fit baselines do)."""
import sys
import random
import math


def canonical_pairs(N):
    return [(i, j) for i in range(1, N + 1) for j in range(i + 1, N + 1)]


def circle_rounds(N):
    teams = list(range(1, N + 1))
    fixed = teams[0]
    rot = teams[1:]
    rounds = []
    for _ in range(N - 1):
        arr = [fixed] + rot
        pairs = []
        for k in range(N // 2):
            a, b = arr[k], arr[N - 1 - k]
            pairs.append((min(a, b), max(a, b)))
        rounds.append(pairs)
        rot = [rot[-1]] + rot[:-1]
    return rounds


def simulate(N, D, C, R, LAM, scenarios, sched, pairs):
    worst = 0.0
    ngames = len(pairs)
    for Sset in scenarios:
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
                final_date[idx] = cap
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


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    N = int(next(it)); D = int(next(it)); C = int(next(it))
    K = int(next(it)); R = int(next(it)); LAM = int(next(it))
    scenarios = []
    for _ in range(K):
        b = int(next(it))
        scenarios.append(frozenset(int(next(it)) for _ in range(b)))

    pairs = canonical_pairs(N)
    ngames = len(pairs)

    rounds = circle_rounds(N)
    extra = D - (N - 1)
    base_dates = [1 + r + (r * extra) // (N - 1) for r in range(N - 1)]
    sched = [None] * ngames
    idx_of = {p: g for g, p in enumerate(pairs)}
    for r, rpairs in enumerate(rounds):
        d = base_dates[r]
        for k, (a, b) in enumerate(rpairs):
            sched[idx_of[(a, b)]] = (d, k + 1)

    date_teams = {}
    date_courts = {}
    for idx, (i, j) in enumerate(pairs):
        d, c = sched[idx]
        date_teams.setdefault(d, set()).update((i, j))
        date_courts.setdefault(d, set()).add(c)

    rng = random.Random(90192837465)
    cur_cost = simulate(N, D, C, R, LAM, scenarios, sched, pairs)
    best_sched = list(sched)
    best_cost = cur_cost

    ITERS = min(6000, max(1200, 55 * ngames))
    T0, T1 = 6.0, 0.05

    def remove(idx):
        i, j = pairs[idx]
        d, c = sched[idx]
        date_teams[d].discard(i); date_teams[d].discard(j)
        date_courts[d].discard(c)

    def place(idx, d, c):
        i, j = pairs[idx]
        sched[idx] = (d, c)
        date_teams.setdefault(d, set()).update((i, j))
        date_courts.setdefault(d, set()).add(c)

    def free_ok(idx, d, c):
        i, j = pairs[idx]
        return (c not in date_courts.get(d, ())) and \
               (i not in date_teams.get(d, ())) and (j not in date_teams.get(d, ()))

    for step in range(ITERS):
        T = T0 * ((T1 / T0) ** (step / max(1, ITERS - 1)))
        move = rng.random()
        if move < 0.6:
            idx = rng.randrange(ngames)
            d0, c0 = sched[idx]
            d1 = rng.randint(1, D)
            c1 = rng.randint(1, C)
            if (d1, c1) == (d0, c0):
                continue
            remove(idx)
            if free_ok(idx, d1, c1):
                place(idx, d1, c1)
                new_cost = simulate(N, D, C, R, LAM, scenarios, sched, pairs)
                delta = new_cost - cur_cost
                if delta <= 0 or rng.random() < math.exp(-delta / max(T, 1e-6)):
                    cur_cost = new_cost
                    if cur_cost < best_cost:
                        best_cost = cur_cost
                        best_sched = list(sched)
                else:
                    remove(idx)
                    place(idx, d0, c0)
            else:
                place(idx, d0, c0)
        else:
            idx1 = rng.randrange(ngames)
            idx2 = rng.randrange(ngames)
            if idx1 == idx2:
                continue
            d1, c1 = sched[idx1]
            d2, c2 = sched[idx2]
            if d1 == d2:
                continue
            remove(idx1); remove(idx2)
            ok1 = free_ok(idx1, d2, c2)
            ok2 = False
            if ok1:
                place(idx1, d2, c2)
                ok2 = free_ok(idx2, d1, c1)
                if ok2:
                    place(idx2, d1, c1)
            if ok1 and ok2:
                new_cost = simulate(N, D, C, R, LAM, scenarios, sched, pairs)
                delta = new_cost - cur_cost
                if delta <= 0 or rng.random() < math.exp(-delta / max(T, 1e-6)):
                    cur_cost = new_cost
                    if cur_cost < best_cost:
                        best_cost = cur_cost
                        best_sched = list(sched)
                else:
                    remove(idx1)
                    place(idx1, d1, c1)
                    place(idx2, d2, c2)
            else:
                if ok1:
                    remove(idx1)
                place(idx1, d1, c1)
                place(idx2, d2, c2)

    out = []
    for d, c in best_sched:
        out.append(f"{d} {c}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
