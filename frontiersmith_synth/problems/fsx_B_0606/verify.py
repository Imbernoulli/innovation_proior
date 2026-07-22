import sys

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)

def read_instance(path):
    toks = open(path).read().split()
    it = iter(toks)
    N = int(next(it)); R = int(next(it)); C = int(next(it))
    r = [int(next(it)) for _ in range(N)]
    s = [int(next(it)) for _ in range(N)]
    w = [[int(next(it)) for _ in range(N)] for _ in range(N)]
    return N, R, C, r, s, w

def simulate(N, R, C, r, s, w, seq):
    """seq: list of tokens, each an int in [0,N-1] or the string 'C'. Returns (cost, err)."""
    CLEAN = "C"
    A = 0                # cumulative residue
    L = CLEAN            # machine starts clean
    cost = 0
    seen = [0] * N
    for t in seq:
        if t == CLEAN:
            cost += C; A = 0; L = CLEAN
        else:
            if t < 0 or t >= N:
                return None, "job id out of range: %d" % t
            seen[t] += 1
            cost += s[t] if L == CLEAN else w[L][t]
            A += r[t]
            if A > R:
                return None, "residue overflow (cap %d) before a clean" % R
            L = t
    for j in range(N):
        if seen[j] != 1:
            return None, "job %d produced %d times (must be exactly once)" % (j, seen[j])
    return cost, None

def main():
    N, R, C, r, s, w = read_instance(sys.argv[1])

    # ---- internal feasible baseline: clean before every job, id order ----
    #   cost_B = s[0] + sum_{j>=1} (C + s[j]) = sum(s) + (N-1)*C   (always feasible)
    B = sum(s) + (N - 1) * C
    B = max(1, B)

    # ---- parse participant artifact: whitespace tokens, each 'C' or an integer ----
    raw = open(sys.argv[2]).read().split()
    if not raw:
        fail("empty output")
    seq = []
    for tok in raw:
        if tok == "C" or tok == "c":
            seq.append("C")
        else:
            try:
                v = int(tok)
            except Exception:
                fail("bad token %r (expected an int job id or 'C')" % tok)
            seq.append(v)

    F, err = simulate(N, R, C, r, s, w, seq)
    if err is not None:
        fail(err)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("F=%d B=%d Ratio: %.6f" % (F, B, sc / 1000.0))

if __name__ == "__main__":
    main()
