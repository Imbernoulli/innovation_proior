# TIER: greedy
# Greedy Sidon prefix + marginal-gain fill.
# Grow a strict-Sidon (all pairwise differences distinct) prefix for as long as it
# fits in [0,M], then top up the remaining slots by picking whichever grid point
# adds the most new distinct sums+differences. A reasonable, purely constructive
# strategy that beats the arithmetic-run baseline but leaves collisions on the table.
import sys

def channels(A):
    s = set(); d = set()
    for a in A:
        for b in A:
            s.add(a + b); d.add(a - b)
    return len(s) + len(d)

def main():
    data = sys.stdin.read().split()
    n = int(data[0]); M = int(data[1])

    # --- strict-Sidon greedy prefix ---
    A = [0]
    diffs = {0}
    x = 1
    while len(A) < n and x <= M:
        dd = set(); ok = True
        for a in A:
            q = x - a
            if q in diffs or q in dd:
                ok = False; break
            dd.add(q); dd.add(-q)
        if ok:
            A.append(x)
            for a in list(A):
                diffs.add(x - a); diffs.add(a - x)
        x += 1

    # --- marginal-gain fill ---
    cur = set(A)
    while len(A) < n:
        best = None; bv = -1
        for x in range(M + 1):
            if x in cur:
                continue
            v = channels(A + [x])
            if v > bv:
                bv = v; best = x
        A.append(best); cur.add(best)

    A = sorted(A)
    out = [str(n)] + [str(v) for v in A]
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
