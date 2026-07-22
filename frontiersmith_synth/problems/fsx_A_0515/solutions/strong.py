# TIER: strong
# INSIGHT: the decision variable (the price path) is transformed by an embedded
# buyer best-response before scoring, so you must optimise the INDUCED behaviour,
# not the schedule shape.  Non-monotone paths -- an early low "sale" for the
# impatient crowd, a mid-window PRICE SPIKE that skims the high-value cluster that
# is only in the market mid-run, then a late markdown for the patient bargain
# crowd -- screen buyer types no monotone ladder can separate.
# We search the full (non-monotone) space: 3-segment hump templates + coordinate
# descent refinement, all scored by the exact simulator.
import sys

def read():
    tok = sys.stdin.read().split()
    it = iter(tok)
    T = int(next(it)); N = int(next(it)); K = int(next(it))
    s = int(next(it)); PMAX = int(next(it)); p0 = int(next(it))
    buyers = []
    for _ in range(N):
        v = int(next(it)); a = int(next(it)); h = int(next(it)); D = int(next(it))
        buyers.append((v, a, h, D))
    return T, N, K, s, PMAX, p0, buyers

def make_sim(T):
    def simulate(prices, buyers, K, s):
        sold = [0] * (T + 1)
        stock = K; rev = 0; powcache = {}
        for (v, a, h, D) in buyers:
            if stock <= 0:
                break
            pw = powcache.get(D)
            if pw is None:
                d = D / 1000.0; pw = [1.0] * (T + 1)
                for t in range(1, T + 1):
                    pw[t] = pw[t - 1] * d
                powcache[D] = pw
            best_t = -1; best_s = 1e-9
            for t in range(a, h + 1):
                if sold[t] >= s:
                    continue
                su = pw[t] * (v - prices[t - 1])
                if su > best_s:
                    best_s = su; best_t = t
            if best_t >= 0:
                rev += prices[best_t - 1]; sold[best_t] += 1; stock -= 1
        return rev
    return simulate

def main():
    T, N, K, s, PMAX, p0, buyers = read()
    sim = make_sim(T)
    step = max(1, PMAX // 12)
    grid = list(range(0, PMAX + 1, step))
    if grid[-1] != PMAX:
        grid.append(PMAX)

    best = [p0] * T
    best_rev = sim(best, buyers, K, s)

    # ---- (1) monotone family (so strong >= greedy) ----
    for hi in grid:
        for lo in grid:
            if lo > hi:
                continue
            sched = [int(round(hi + (lo - hi) * (t / (T - 1)))) for t in range(T)]
            r = sim(sched, buyers, K, s)
            if r > best_rev:
                best_rev = r; best = sched

    # ---- (2) 3-segment HUMP / spike templates: [x1]..[x2 spike]..[x3] ----
    b1_cands = [3, 5, 7, 9]
    b2_cands = [13, 15, 17, 20]
    lvl = grid[::2] if len(grid) > 8 else grid   # coarser level grid
    for b1 in b1_cands:
        for b2 in b2_cands:
            if b2 <= b1:
                continue
            for x1 in lvl:
                for x2 in lvl:
                    for x3 in lvl:
                        sched = [0] * T
                        for t in range(T):
                            day = t + 1
                            if day <= b1:
                                sched[t] = x1
                            elif day <= b2:
                                sched[t] = x2
                            else:
                                sched[t] = x3
                        r = sim(sched, buyers, K, s)
                        if r > best_rev:
                            best_rev = r; best = sched[:]

    # ---- (3) coordinate-descent refinement over ALL days (allows any shape) ----
    cur = best[:]
    cur_rev = best_rev
    fine = list(range(0, PMAX + 1, max(1, PMAX // 20)))
    if fine[-1] != PMAX:
        fine.append(PMAX)
    for _pass in range(3):
        improved = False
        for t in range(T):
            old = cur[t]
            bestp = old; bestr = cur_rev
            for cand in fine:
                if cand == old:
                    continue
                cur[t] = cand
                r = sim(cur, buyers, K, s)
                if r > bestr:
                    bestr = r; bestp = cand
            cur[t] = bestp
            if bestp != old:
                cur_rev = bestr; improved = True
        if cur_rev > best_rev:
            best_rev = cur_rev; best = cur[:]
        if not improved:
            break

    sys.stdout.write(" ".join(map(str, best)) + "\n")

if __name__ == "__main__":
    main()
