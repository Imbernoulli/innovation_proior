#!/usr/bin/env python3
"""verify.py <in> <out> <ans>  -- deterministic scorer for the geothermal polarity grid.

Reads the instance (N, K, fixed cells) and the participant's 0/1 grid, validates
feasibility strictly, computes |det| EXACTLY via Bareiss fraction-free elimination,
and scores bit_length(|det|) against the block-grid baseline B = N//3 + 1.
Any violation -> 'Ratio: 0.0'.
"""
import sys


def read_ints(path):
    with open(path) as f:
        data = f.read().split()
    return data


def bareiss_det(mat, n):
    """Exact integer determinant via fraction-free Gaussian (Bareiss) elimination."""
    M = [row[:] for row in mat]
    sign = 1
    prev = 1
    for k in range(n - 1):
        if M[k][k] == 0:
            # find a pivot row to swap
            swap = -1
            for i in range(k + 1, n):
                if M[i][k] != 0:
                    swap = i
                    break
            if swap == -1:
                return 0
            M[k], M[swap] = M[swap], M[k]
            sign = -sign
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                M[i][j] = (M[i][j] * M[k][k] - M[i][k] * M[k][j]) // prev
            M[i][k] = 0
        prev = M[k][k]
    return sign * M[n - 1][n - 1]


def build_baseline(N):
    M = [[0] * N for _ in range(N)]
    pos = 0
    blk = [[1, 1, 0], [0, 1, 1], [1, 0, 1]]
    while pos + 3 <= N:
        for a in range(3):
            for b in range(3):
                M[pos + a][pos + b] = blk[a][b]
        pos += 3
    r = N - pos
    if r == 1:
        M[pos][pos] = 1
    elif r == 2:
        M[pos][pos] = 1
        M[pos][pos + 1] = 1
        M[pos + 1][pos] = 1
        M[pos + 1][pos + 1] = 0
    return M


def fail(msg):
    print("%s Ratio: 0.0" % msg)
    sys.exit(0)


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    tin = read_ints(in_path)
    idx = 0
    N = int(tin[idx]); idx += 1
    K = int(tin[idx]); idx += 1
    fixed = []
    for _ in range(K):
        r = int(tin[idx]); c = int(tin[idx + 1]); v = int(tin[idx + 2]); idx += 3
        fixed.append((r, c, v))

    tok = read_ints(out_path)
    if len(tok) != N * N:
        fail("wrong token count (got %d, need %d)." % (len(tok), N * N))
    M = [[0] * N for _ in range(N)]
    p = 0
    for i in range(N):
        for j in range(N):
            try:
                x = int(tok[p])
            except ValueError:
                fail("non-integer token.")
            if x not in (0, 1):
                fail("entry not in {0,1}.")
            M[i][j] = x
            p += 1

    for (r, c, v) in fixed:
        if not (0 <= r < N and 0 <= c < N):
            fail("fixed cell out of range.")
        if M[r][c] != v:
            fail("fixed cell (%d,%d) violated." % (r, c))

    det = bareiss_det(M, N)
    F = abs(det).bit_length()  # 0 if singular

    B = (N // 3) + 1  # bit_length(2^(N//3)) of the baseline block grid

    sc = min(1000.0, 100.0 * F / max(1e-9, float(B)))
    print("F=%d B=%d |det|_bits=%d Ratio: %.6f" % (F, B, F, sc / 1000.0))


if __name__ == "__main__":
    main()
