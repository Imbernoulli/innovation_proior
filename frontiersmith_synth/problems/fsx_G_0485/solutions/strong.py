# TIER: strong
# Multi-restart randomized greedy: run several independent greedy passes with
# larger candidate pools and keep the largest packing found.
import sys, random

def greedy_pass(n, w, tmax, seed, K):
    rnd = random.Random(seed)
    chosen = []
    cset = set()
    for _ in range(K):
        pts = rnd.sample(range(n), w)
        m = 0
        for pt in pts:
            m |= (1 << pt)
        if m in cset:
            continue
        ok = True
        for c in chosen:
            if bin(m & c).count("1") > tmax:
                ok = False
                break
        if ok:
            chosen.append(m)
            cset.add(m)
    return chosen

def main():
    p = sys.stdin.read().split()
    n, w, d = int(p[0]), int(p[1]), int(p[2])
    tmax = w - d // 2

    best = []
    base = 7000 + n * 977 + w * 53 + d
    for r in range(10):
        cand = greedy_pass(n, w, tmax, base + r * 101, 30000)
        if len(cand) > len(best):
            best = cand

    out = []
    for m in best:
        out.append(''.join('1' if (m >> i) & 1 else '0' for i in range(n)))
    sys.stdout.write('\n'.join(out) + ('\n' if out else ''))

if __name__ == "__main__":
    main()
