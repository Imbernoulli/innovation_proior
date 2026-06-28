import sys, random
# Max-scale performance input (NOT for brute comparison): n=q=1e5, large coords.
# Online encoding requires knowing answers; for perf we don't need correctness of
# encoding, only that the program does the full BIT-of-BIT work. We therefore emit
# raw (already-decoded) coordinates and set queries so lastAns stays 0 by making
# every query an empty-ish region? No -- to truly exercise work we let lastAns vary.
# Simpler: emit updates + queries with coordinates that DON'T depend on lastAns by
# using lastAns-independent ranges is impossible. Instead we just emit random encoded
# values; the program will decode to arbitrary (possibly empty) rectangles but still
# performs the same number of BIT operations (the heavy path is queryPrefix, run
# regardless of emptiness unless X1>X2). To guarantee heavy work, encode so decoded
# rectangles are large.
def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rnd = random.Random(seed)
    n = 100000
    q = 100000
    C = 10**9
    px = [rnd.randint(-C, C) for _ in range(n)]
    py = [rnd.randint(-C, C) for _ in range(n)]
    w  = [rnd.randint(-10**9, 10**9) for _ in range(n)]
    out = [f"{n} {q}"]
    for i in range(n):
        out.append(f"{px[i]} {py[i]} {w[i]}")
    # simulate to keep encoding valid AND rectangles large (full plane) so work is max
    last = 0
    total = sum(w)  # running full-plane sum, updated incrementally per update
    for j in range(q):
        if j % 5 == 0:
            idx = rnd.randint(0, n-1)
            d = rnd.randint(-10**9, 10**9)
            w[idx] += d
            total += d
            out.append(f"1 {idx} {d}")
        else:
            X1, Y1, X2, Y2 = -C, -C, C, C
            a = X1 ^ last; b = Y1 ^ last; c = X2 ^ last; e = Y2 ^ last
            out.append(f"2 {a} {b} {c} {e}")
            last = total  # full-plane sum equals running total
    sys.stdout.write("\n".join(out) + "\n")
main()
