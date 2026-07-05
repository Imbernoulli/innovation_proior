# TIER: greedy
# Conservative planner: Mian-Chowla greedy, but only over the LOWER sub-band [1..n/2]
# (a common real practice of keeping a wide upper guard region). Beats the power-of-two
# baseline but leaves the upper half of the spectrum unused.
import sys

def build(order, forb):
    chosen = []
    sums = set()
    for c in order:
        if c in forb:
            continue
        cand = set()
        ok = True
        for a in chosen:
            s = a + c
            if s in sums or s in cand:
                ok = False
                break
            cand.add(s)
        if ok:
            s = 2 * c
            if s in sums or s in cand:
                ok = False
            else:
                cand.add(s)
        if ok:
            chosen.append(c)
            sums |= cand
    return chosen

def main():
    d = sys.stdin.read().split()
    n = int(d[0]); k = int(d[1])
    forb = set(int(x) for x in d[2:2 + k])
    order = range(1, n // 2 + 1)
    chosen = build(order, forb)
    print(" ".join(map(str, chosen)))

if __name__ == "__main__":
    main()
