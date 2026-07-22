# TIER: greedy
# The obvious "markdown ladder": search only over MONOTONE non-increasing price
# schedules (start high, decline to a floor) and keep the best via exact simulation.
# This is the textbook clearance heuristic -- and it is exactly the trap: every
# monotone path lets patient buyers wait for the floor, so it cannot charge the
# mid-window high-value cluster more than the early crowd.
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
    step = max(1, PMAX // 16)
    grid = list(range(0, PMAX + 1, step))
    if grid[-1] != PMAX:
        grid.append(PMAX)

    best_prices = [p0] * T
    best_rev = sim(best_prices, buyers, K, s)

    # linear monotone declines from `hi` down to `lo` (hi >= lo), plus constants
    for hi in grid:
        for lo in grid:
            if lo > hi:
                continue
            if T == 1:
                sched = [hi]
            else:
                sched = [int(round(hi + (lo - hi) * (t / (T - 1)))) for t in range(T)]
            r = sim(sched, buyers, K, s)
            if r > best_rev:
                best_rev = r; best_prices = sched

    sys.stdout.write(" ".join(map(str, best_prices)) + "\n")

if __name__ == "__main__":
    main()
