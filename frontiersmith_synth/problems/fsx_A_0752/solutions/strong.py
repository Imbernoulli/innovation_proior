# TIER: strong
# Insight: the run-legal codeword tree is NOT a plain binary tree -- a node that
# has just finished a run of length d has only ONE legal next bit, every other
# node has two. Because that branching pattern only ever depends on the current
# "run state" r (0 = fresh/root, 1..d = trailing run length), and NOT on where
# in the tree you are, the whole tree is built from a small, reusable family of
# subtree "types" T(0), T(1), ..., T(d): T(d) is forced (one child, of type
# T(1)); every T(r) with r<d branches into a "switch" child (type T(1)) and a
# "repeat" child (type T(r+1)) -- T(0), the root, branches into two T(1)s.
#
# Given that, the classic Huffman exchange argument (pair the biggest weight
# with the shallowest leaf) generalizes: sort the symbols by weight descending
# and, for a contiguous stretch of that sorted list handled by a subtree of a
# given type, exactly determine (by DP over which contiguous prefix goes to
# which child, and which child gets the higher-weight half) the split that
# minimizes total weighted depth -- exploiting the automaton's true shape,
# instead of pretending it is a plain binary tree and repairing afterwards.
import sys
sys.setrecursionlimit(10000)


def children_of(last_bit, run, d):
    if last_bit is None:
        return [(0, 1), (1, 1)]
    opts = [(1 - last_bit, 1)]
    if run < d:
        opts.append((last_bit, run + 1))
    return opts


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0]); d = int(toks[1])
    p = [int(x) for x in toks[2:2 + n]]

    order = sorted(range(n), key=lambda i: (-p[i], i))
    W = [p[i] for i in order]
    PS = [0] * (n + 1)
    for i in range(n):
        PS[i + 1] = PS[i] + W[i]

    def rangesum(i, m):
        return PS[i + m] - PS[i]

    memo = {}

    def solve(r, i, m):
        if m == 1:
            return 0
        key = (r, i, m)
        if key in memo:
            return memo[key]
        total = rangesum(i, m)
        if r == d:
            cost = total + solve(1, i, m)
        else:
            best = None
            for k in range(1, m):
                if r == 0:
                    cand = total + solve(1, i, k) + solve(1, i + k, m - k)
                    if best is None or cand < best:
                        best = cand
                else:
                    c1 = total + solve(1, i, k) + solve(r + 1, i + k, m - k)
                    c2 = total + solve(r + 1, i, k) + solve(1, i + k, m - k)
                    cand = c1 if c1 <= c2 else c2
                    if best is None or cand < best:
                        best = cand
            cost = best
        memo[key] = cost
        return cost

    def best_split(r, i, m):
        total = rangesum(i, m)
        best_k, best_mode, best_cost = None, None, None
        for k in range(1, m):
            if r == 0:
                cand = total + solve(1, i, k) + solve(1, i + k, m - k)
                mode = 'AA'
            else:
                c1 = total + solve(1, i, k) + solve(r + 1, i + k, m - k)
                c2 = total + solve(r + 1, i, k) + solve(1, i + k, m - k)
                if c1 <= c2:
                    cand, mode = c1, 'T1_first'
                else:
                    cand, mode = c2, 'Tr1_first'
            if best_cost is None or cand < best_cost:
                best_cost, best_k, best_mode = cand, k, mode
        return best_k, best_mode

    solve(0, 0, n)  # populate memo

    codeword_sorted = [None] * n

    def build(r, i, m, prefix, last_bit, run):
        if m == 1:
            codeword_sorted[i] = prefix if prefix else "0"
            return
        opts = children_of(last_bit, run, d)
        if r == d:
            bit, nrun = opts[0]
            build(1, i, m, prefix + str(bit), bit, nrun)
            return
        k, mode = best_split(r, i, m)
        if r == 0:
            (bA, rA), (bB, rB) = opts[0], opts[1]
            build(1, i, k, prefix + str(bA), bA, rA)
            build(1, i + k, m - k, prefix + str(bB), bB, rB)
        else:
            (b_sw, r_sw), (b_rp, r_rp) = opts[0], opts[1]
            if mode == 'T1_first':
                build(1, i, k, prefix + str(b_sw), b_sw, r_sw)
                build(r + 1, i + k, m - k, prefix + str(b_rp), b_rp, r_rp)
            else:
                build(r + 1, i, k, prefix + str(b_rp), b_rp, r_rp)
                build(1, i + k, m - k, prefix + str(b_sw), b_sw, r_sw)

    build(0, 0, n, "", None, 0)

    codeword = [None] * n
    for pos, orig in enumerate(order):
        codeword[orig] = codeword_sorted[pos]
    print(" ".join(codeword))


if __name__ == "__main__":
    main()
