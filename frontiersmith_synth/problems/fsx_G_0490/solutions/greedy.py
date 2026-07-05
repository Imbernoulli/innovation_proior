# TIER: greedy
"""Row-cycle trade local search from the cyclic baseline (single restart, no annealing).
A row-cycle move picks two rows of one square, decomposes the induced symbol permutation
into cycles, and swaps the columns of one cycle between the two rows -- a Latin-preserving
trade that generalizes the 2x2 intercalate. Deterministic (fixed seed). Beats trivial."""
import sys
from math import gcd, exp
import random

SEED = 101
RESTARTS = 1
ITERS = 5000
PERTURB = 0
USE_SA = False
T0 = 0.0
TEND = 0.0


def build_baseline(n, k):
    coprimes = [a for a in range(1, n) if gcd(a, n) == 1]
    L = len(coprimes)
    squares = []
    for m in range(k):
        a = coprimes[m % L]
        s = m // L
        squares.append([[(a * i + j + s) % n for j in range(n)] for i in range(n)])
    return squares


class Engine:
    def __init__(self, squares, n, k):
        self.sq = [[row[:] for row in s] for s in squares]
        self.n = n
        self.k = k
        self.cnt = {}
        self.dist = {}
        self.total = 0
        for i in range(k):
            for j in range(i + 1, k):
                c = {}
                A, B = self.sq[i], self.sq[j]
                for r in range(n):
                    for cc in range(n):
                        key = (A[r][cc], B[r][cc])
                        c[key] = c.get(key, 0) + 1
                self.cnt[(i, j)] = c
                self.dist[(i, j)] = len(c)
                self.total += len(c)

    def _update_cell(self, p, r, c, newv):
        sq = self.sq
        oldv = sq[p][r][c]
        if oldv == newv:
            return
        for q in range(self.k):
            if q == p:
                continue
            if p < q:
                i, j = p, q
                ok = (oldv, sq[j][r][c]); nk = (newv, sq[j][r][c])
            else:
                i, j = q, p
                ok = (sq[i][r][c], oldv); nk = (sq[i][r][c], newv)
            cnt = self.cnt[(i, j)]
            v = cnt[ok] - 1
            if v == 0:
                del cnt[ok]; self.dist[(i, j)] -= 1; self.total -= 1
            else:
                cnt[ok] = v
            if nk in cnt:
                cnt[nk] += 1
            else:
                cnt[nk] = 1; self.dist[(i, j)] += 1; self.total += 1
        sq[p][r][c] = newv

    def row_cycle(self, p, r1, r2, rng):
        """Columns of one random cycle of the symbol permutation induced by rows r1,r2."""
        sq = self.sq; n = self.n
        r1row = sq[p][r1]; r2row = sq[p][r2]
        inv1 = [0] * n
        for c in range(n):
            inv1[r1row[c]] = c
        start = rng.randrange(n)
        cols = []
        x = start
        seen = set()
        while x not in seen:
            seen.add(x)
            c = inv1[x]
            cols.append(c)
            x = r2row[c]
        if len(cols) < 2:
            return None
        return cols

    def do_cycle_swap(self, p, r1, r2, cols):
        sq = self.sq
        for c in cols:
            v1 = sq[p][r1][c]; v2 = sq[p][r2][c]
            self._update_cell(p, r1, c, v2)
            self._update_cell(p, r2, c, v1)


def solve(squares, n, k, seed, restarts, iters, perturb, use_sa, t0, tend):
    rng = random.Random(seed)
    best = [[row[:] for row in s] for s in squares]
    best_F = -1
    for rs in range(restarts):
        eng = Engine(squares, n, k)
        if rs > 0 and perturb > 0:
            for _ in range(perturb):
                p = rng.randrange(k)
                r1 = rng.randrange(n); r2 = rng.randrange(n)
                if r1 == r2:
                    continue
                cols = eng.row_cycle(p, r1, r2, rng)
                if cols:
                    eng.do_cycle_swap(p, r1, r2, cols)
        rb_F = eng.total
        rb = [[row[:] for row in s] for s in eng.sq]
        for it in range(iters):
            p = rng.randrange(k)
            r1 = rng.randrange(n); r2 = rng.randrange(n)
            if r1 == r2:
                continue
            cols = eng.row_cycle(p, r1, r2, rng)
            if not cols:
                continue
            before = eng.total
            eng.do_cycle_swap(p, r1, r2, cols)
            delta = eng.total - before
            accept = delta >= 0
            if not accept and use_sa:
                frac = it / max(1, iters - 1)
                T = t0 * ((tend / t0) ** frac) if t0 > 0 else 0.0
                if T > 0 and rng.random() < exp(delta / T):
                    accept = True
            if not accept:
                eng.do_cycle_swap(p, r1, r2, cols)  # revert (self-inverse)
            elif eng.total > rb_F:
                rb_F = eng.total
                rb = [[row[:] for row in s] for s in eng.sq]
        if rb_F > best_F:
            best_F = rb_F
            best = rb
    return best


def main():
    data = sys.stdin.read().split()
    n, k = int(data[0]), int(data[1])
    base = build_baseline(n, k)
    res = solve(base, n, k, SEED, RESTARTS, ITERS, PERTURB, USE_SA, T0, TEND)
    out = []
    for sq in res:
        for row in sq:
            out.append(" ".join(map(str, row)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
