#!/usr/bin/env python3
"""Deterministic checker for fsx_C_0396 - Traffic Signal Phase Grid.

CLI: python3 verify.py <in> <out> <ans>   (<ans> is ignored)

Feasibility (hard, any violation -> Ratio: 0.0):
  * output must be exactly N lines, each with N integer tokens
  * every value must be an integer in {-1, 0, 1, ..., N-1} (nan/inf rejected)
  * every PREFILLED cell must keep its given phase

Objective F = number of ADDED cells (prefill == -1) that hold a phase which is
UNIQUE within its row and UNIQUE within its column (among all non-empty cells).

Internal baseline B = number of cells a weak deterministic "stripe-cyclic"
construction fills validly.  Normalisation (maximisation):
    sc = min(1000.0, 100.0 * F / max(1, B))
    Ratio = sc / 1000.0
"""
import sys

MOD_BASE = 3          # stripe sparsity of the internal baseline construction


def fail(msg):
    print("INVALID: %s  Ratio: 0.0" % msg)
    sys.exit(0)


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    if not toks:
        fail("empty instance")
    it = iter(toks)
    N = int(next(it))
    grid = [[-1] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            grid[i][j] = int(next(it))
    return N, grid


def read_output(path, N):
    try:
        with open(path) as f:
            raw = f.read()
    except Exception:
        fail("cannot read output")
    toks = raw.split()
    if len(toks) != N * N:
        fail("expected %d tokens, got %d" % (N * N, len(toks)))
    vals = []
    for tk in toks:
        # reject nan/inf/floats/garbage: must parse as a plain integer
        try:
            v = int(tk)
        except Exception:
            fail("non-integer token %r" % tk)
        vals.append(v)
    out = [[0] * N for _ in range(N)]
    k = 0
    for i in range(N):
        for j in range(N):
            out[i][j] = vals[k]
            k += 1
    return out


def baseline_count(N, prefill):
    """Weak deterministic construction: on stripe cells (i+j)%MOD_BASE==0 that
    are empty, try the cyclic phase (i+j)%N; place it iff it is currently unique
    in its row and column.  Return count of validly placed cells."""
    row_used = [0] * N
    col_used = [0] * N
    for i in range(N):
        for j in range(N):
            v = prefill[i][j]
            if v != -1:
                row_used[i] |= (1 << v)
                col_used[j] |= (1 << v)
    cnt = 0
    for i in range(N):
        for j in range(N):
            if prefill[i][j] != -1:
                continue
            if (i + j) % MOD_BASE != 0:
                continue
            s = (i + j) % N
            bit = 1 << s
            if (row_used[i] & bit) or (col_used[j] & bit):
                continue
            row_used[i] |= bit
            col_used[j] |= bit
            cnt += 1
    return cnt


def main():
    if len(sys.argv) < 3:
        print("usage: verify.py <in> <out> <ans>", file=sys.stderr)
        sys.exit(2)
    in_path, out_path = sys.argv[1], sys.argv[2]

    N, prefill = read_instance(in_path)
    out = read_output(out_path, N)

    # range + prefill-preservation checks
    for i in range(N):
        for j in range(N):
            v = out[i][j]
            if v < -1 or v > N - 1:
                fail("value %d out of range at (%d,%d)" % (v, i, j))
            if prefill[i][j] != -1 and v != prefill[i][j]:
                fail("prefilled cell (%d,%d) changed" % (i, j))

    # multiplicities of each phase per row / column among non-empty cells
    row_cnt = [dict() for _ in range(N)]
    col_cnt = [dict() for _ in range(N)]
    for i in range(N):
        for j in range(N):
            v = out[i][j]
            if v == -1:
                continue
            row_cnt[i][v] = row_cnt[i].get(v, 0) + 1
            col_cnt[j][v] = col_cnt[j].get(v, 0) + 1

    F = 0
    for i in range(N):
        for j in range(N):
            if prefill[i][j] != -1:
                continue                      # count ADDED cells only
            v = out[i][j]
            if v == -1:
                continue
            if row_cnt[i][v] == 1 and col_cnt[j][v] == 1:
                F += 1

    B = baseline_count(N, prefill)
    sc = min(1000.0, 100.0 * F / max(1, B))
    print("N=%d prefilled_cells added_valid=%d baseline=%d  Ratio: %.6f"
          % (N, F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
