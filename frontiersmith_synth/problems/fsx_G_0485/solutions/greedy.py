# TIER: greedy
# Single-pass randomized greedy packing of weight-w words.
import sys, random

def main():
    p = sys.stdin.read().split()
    n, w, d = int(p[0]), int(p[1]), int(p[2])
    tmax = w - d // 2
    rnd = random.Random(1000 + n * 131 + w * 17 + d)

    chosen = []      # bit masks
    cset = set()
    K = 1200
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

    out = []
    for m in chosen:
        out.append(''.join('1' if (m >> i) & 1 else '0' for i in range(n)))
    sys.stdout.write('\n'.join(out) + ('\n' if out else ''))

if __name__ == "__main__":
    main()
