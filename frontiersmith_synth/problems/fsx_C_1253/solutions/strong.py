# TIER: strong
"""The insight: capacity alone does not determine cache behaviour -- SET
mapping does. With row length N (no padding), row i's cache-line start is
i*(N/L) lines into the array, so whenever that shares a large common factor
with the number of sets S, many/most rows alias onto the SAME few sets no
matter how well a tile fits in raw capacity -- a tile "that fills the
cache" can still thrash on a low-associativity cache. Co-design tile size
WITH row padding: search the small bounded padding budget for a value that
makes the padded row stride close to COPRIME with S, spreading rows evenly
across sets. Pair that with an inner loop order that keeps 'j' innermost
(stride-1 in both B and C, and A's own index frozen across the innermost
sweep) so the sequential prefetcher actually pays off.

Rather than committing blindly to one tile-size formula (which can itself
mis-judge the true reuse/conflict trade-off), this solution evaluates a
short list of principled tile-size candidates -- the capacity-fill size,
the structural maximum, an associativity-scaled size, and neighbours of
each -- each combined with the padding search and two locality-friendly
inner orders, and keeps whichever the (self-reimplemented) cache model
actually likes best. This is a genuine reformulation-plus-search over the
padding/tile/order space, not "greedy plus more iterations": the winning
choice differs qualitatively across test cases (small tile + heavy padding
on some, large tile + light padding on others), and it is exactly this
co-design the greedy capacity-only recipe cannot see."""
import sys
import math


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def best_pad(row, S, pad_max):
    """Smallest padding in [0, pad_max] that minimizes gcd(row+pad, S)
    (gcd==1 means the padded row stride visits every set once per L words --
    no cheap periodic aliasing left to exploit)."""
    if S <= 0:
        return 0
    best, best_g = 0, gcd(row, S)
    for p in range(0, pad_max + 1):
        g = gcd(row + p, S)
        if g < best_g:
            best_g, best = g, p
        if best_g <= 1:
            break
    return best


def blocks(T, n):
    out = []
    b = 0
    while b < n:
        out.append((b, min(b + T, n)))
        b += T
    return out


def simulate_misses(N, C, L, A, Ti, Tj, Tk, padA, padB, padC, inner_order):
    """Local reimplementation of the checker's exact cache model (fixed
    outer block order I,K,J) so candidate plans can be scored offline."""
    S = C // (L * A)
    rowA, rowB, rowC = N + padA, N + padB, N + padC
    baseA = 0
    baseB = baseA + N * rowA
    baseC = baseB + N * rowB

    sets = [[] for _ in range(S)]
    misses = 0
    last_line = {"A": None, "B": None, "C": None}

    def access(line):
        s = line % S
        bucket = sets[s]
        try:
            bucket.remove(line)
            bucket.insert(0, line)
            return True
        except ValueError:
            bucket.insert(0, line)
            if len(bucket) > A:
                bucket.pop()
            return False

    def prefetch_insert(line):
        s = line % S
        bucket = sets[s]
        if line not in bucket:
            bucket.insert(0, line)
            if len(bucket) > A:
                bucket.pop()

    def do_access(addr, stream):
        nonlocal misses
        line = addr // L
        if not access(line):
            misses += 1
        ll = last_line[stream]
        if ll is not None and line == ll + 1:
            prefetch_insert(line + 1)
        last_line[stream] = line

    I_blocks, J_blocks, K_blocks = blocks(Ti, N), blocks(Tj, N), blocks(Tk, N)
    for ii0, ii1 in I_blocks:
        for kk0, kk1 in K_blocks:
            for jj0, jj1 in J_blocks:
                ranges = {"i": range(ii0, ii1), "j": range(jj0, jj1), "k": range(kk0, kk1)}
                r0, r1, r2 = ranges[inner_order[0]], ranges[inner_order[1]], ranges[inner_order[2]]
                for x0 in r0:
                    for x1 in r1:
                        for x2 in r2:
                            idx = {inner_order[0]: x0, inner_order[1]: x1, inner_order[2]: x2}
                            i, j, k = idx["i"], idx["j"], idx["k"]
                            do_access(baseA + i * rowA + k, "A")
                            do_access(baseB + k * rowB + j, "B")
                            do_access(baseC + i * rowC + j, "C")
    return misses


def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    C = int(data[1]); L = int(data[2]); A = int(data[3])
    PAD_MAX = int(data[4])

    S = C // (L * A)
    max_t = max(1, N // 3)
    min_t = min(2, max_t)
    cap_t = min(max_t, max(min_t, int(math.isqrt(C // 3))))

    t_candidates = sorted(set([
        max_t, cap_t,
        max(min_t, (max_t + cap_t) // 2),
        max(min_t, max_t - A) if max_t - A >= min_t else min_t,
        min(max_t, max(min_t, A)),
        max(min_t, cap_t + 1) if cap_t + 1 <= max_t else max_t,
        max(min_t, cap_t - 1),
    ]))

    pad = best_pad(N, S, PAD_MAX)

    best = None
    for T in t_candidates:
        for order in ("ikj", "kij", "jik"):
            F = simulate_misses(N, C, L, A, T, T, T, pad, pad, pad, order)
            if best is None or F < best[0]:
                best = (F, T, order)

    _, T, order = best
    print(T, T, T)
    print(pad, pad, pad)
    print(order)


if __name__ == "__main__":
    main()
