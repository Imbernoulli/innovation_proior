# TIER: strong
import sys, random, time

def main():
    t0 = time.time()
    TL = 3.2  # seconds budget (config time limit is 5s; leave margin for sandbox/startup)
    d = sys.stdin.buffer.read().split()
    idx = 0
    n = int(d[idx]); idx += 1
    m = int(d[idx]); idx += 1

    W = [0] * m
    clause_lits = [None] * m           # list of literals (signed ints) per clause
    var_occ = [[] for _ in range(n + 1)]  # var -> list of (ci, sign)
    pos_w = [0] * (n + 1)
    neg_w = [0] * (n + 1)
    for ci in range(m):
        w = int(d[idx]); k = int(d[idx + 1]); idx += 2
        lits = []
        for _j in range(k):
            l = int(d[idx]); idx += 1
            lits.append(l)
            if l > 0:
                var_occ[l].append((ci, 1)); pos_w[l] += w
            else:
                var_occ[-l].append((ci, -1)); neg_w[-l] += w
        W[ci] = w
        clause_lits[ci] = lits

    def lit_true(l, x):
        return (x[l] == 1) if l > 0 else (x[-l] == 0)

    # true-literal count per clause under x
    def build_tcount(x):
        tc = [0] * m
        F = 0
        for ci in range(m):
            t = 0
            for l in clause_lits[ci]:
                if lit_true(l, x):
                    t += 1
            tc[ci] = t
            if t > 0:
                F += W[ci]
        return tc, F

    # gain in satisfied weight from flipping variable v (given x, tcount)
    def flip_gain(v, x, tc):
        g = 0
        xv = x[v]
        for (ci, sign) in var_occ[v]:
            # is this literal currently true?
            cur_true = (xv == 1) if sign > 0 else (xv == 0)
            t = tc[ci]
            if cur_true:
                # becomes false: t-1
                if t == 1:
                    g -= W[ci]     # clause was satisfied only by this literal
            else:
                # becomes true: t+1
                if t == 0:
                    g += W[ci]     # clause becomes satisfied
        return g

    def apply_flip(v, x, tc):
        xv = x[v]
        for (ci, sign) in var_occ[v]:
            cur_true = (xv == 1) if sign > 0 else (xv == 0)
            if cur_true:
                tc[ci] -= 1
            else:
                tc[ci] += 1
        x[v] = 1 - xv

    def score_of(x):
        _, F = build_tcount(x)
        return F

    # ---- start 1: weighted majority vote (same as greedy) ----
    x0 = [0] * (n + 1)
    for v in range(1, n + 1):
        x0[v] = 1 if pos_w[v] > neg_w[v] else 0

    best_x = x0[:]
    best_F = score_of(best_x)

    rng = random.Random(1234567)
    order = list(range(1, n + 1))

    def hill_climb(start):
        x = start[:]
        tc, F = build_tcount(x)
        while time.time() - t0 < TL:
            improved = False
            rng.shuffle(order)
            for v in order:
                g = flip_gain(v, x, tc)
                if g > 0:
                    apply_flip(v, x, tc)
                    F += g
                    improved = True
                if time.time() - t0 >= TL:
                    break
            if not improved:
                break
        return x, F

    # climb from the vote start
    x, F = hill_climb(x0)
    if F > best_F:
        best_F, best_x = F, x[:]

    # ---- restarts: perturbed / random starts, keep the best ----
    while time.time() - t0 < TL:
        start = [0] * (n + 1)
        r = rng.random()
        if r < 0.5:
            # perturb the current best: random 15% of cells flipped
            start = best_x[:]
            for v in range(1, n + 1):
                if rng.random() < 0.15:
                    start[v] = 1 - start[v]
        else:
            # fully random start
            for v in range(1, n + 1):
                start[v] = rng.randint(0, 1)
        x, F = hill_climb(start)
        if F > best_F:
            best_F, best_x = F, x[:]

    sys.stdout.write(" ".join(str(best_x[v]) for v in range(1, n + 1)) + "\n")

main()
