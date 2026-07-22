import sys


def emit(r, why=""):
    if why:
        print("Ratio: %.6f (%s)" % (r, why))
    else:
        print("Ratio: %.6f" % r)
    sys.exit(0)


def read_instance(path):
    data = open(path).read().split()
    it = iter(data)
    N = int(next(it))
    T_r = int(next(it))
    W_max = int(next(it))
    jobs = []
    for _ in range(N):
        s = int(next(it))
        d = int(next(it))
        jobs.append((s, d))
    return N, T_r, W_max, jobs


def simulate(seq, T_r, W_max, jobs):
    """Total wall time of a schedule. seq is a list of tokens: 0 = re-dress,
    j in 1..N = process job j.  Cost is exact integer arithmetic. Wear is
    clamped to [0, W_max]."""
    w = 0        # integer wear, clamped to [0, W_max]
    cost = 0     # integer total time
    for tok in seq:
        if tok == 0:
            cost += T_r
            w = 0
        else:
            s, d = jobs[tok - 1]
            cost += s * (1 + w) * (1 + w)
            w = w + d
            if w < 0:
                w = 0
            elif w > W_max:
                w = W_max
    return cost


def main():
    inp, out = sys.argv[1], sys.argv[2]
    N, T_r, W_max, jobs = read_instance(inp)

    # ---- internal baseline B: process in the GIVEN input order, no re-dress ----
    B = simulate(list(range(1, N + 1)), T_r, W_max, jobs)
    B = max(1, B)

    # ---- parse participant artifact strictly ----
    try:
        toks = open(out).read().split()
    except Exception:
        emit(0.0, "no output")
    if len(toks) == 0:
        emit(0.0, "empty output")
    if len(toks) > 20 * N + 1000:
        emit(0.0, "too many tokens")

    seq = []
    seen = [False] * (N + 1)
    for tstr in toks:
        # tokens must be plain integers; this rejects nan/inf/floats/garbage
        try:
            v = int(tstr)
        except Exception:
            emit(0.0, "non-integer token %r" % tstr)
        if v == 0:
            seq.append(0)
            continue
        if v < 1 or v > N:
            emit(0.0, "job index out of range: %d" % v)
        if seen[v]:
            emit(0.0, "duplicate job %d" % v)
        seen[v] = True
        seq.append(v)

    for j in range(1, N + 1):
        if not seen[j]:
            emit(0.0, "missing job %d" % j)

    F = simulate(seq, T_r, W_max, jobs)
    if F <= 0:
        emit(0.0, "nonpositive cost")

    # minimization: smaller F is better. trivial (== baseline order) -> 0.1.
    sc = min(1000.0, 100.0 * B / max(1e-9, float(F)))
    emit(sc / 1000.0)


if __name__ == "__main__":
    main()
