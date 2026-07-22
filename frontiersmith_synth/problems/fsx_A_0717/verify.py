import sys


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def read_instance(path):
    try:
        toks = open(path).read().split()
    except Exception:
        fail("cannot read input")
    pos = [0]

    def nxt():
        v = int(toks[pos[0]])
        pos[0] += 1
        return v

    D = nxt()
    T = nxt()
    K = nxt()
    families = []
    for _ in range(K):
        n_f = nxt()
        fam = []
        for _ in range(n_f):
            lo = nxt()
            hi = nxt()
            fam.append((lo, hi))
        families.append(fam)
    return D, T, K, families


def kill_fracs(families, xs):
    fracs = []
    for fam in families:
        if not fam:
            fracs.append(1.0)
            continue
        killed = 0
        for lo, hi in fam:
            if any(lo <= x < hi for x in xs):
                killed += 1
        fracs.append(killed / len(fam))
    return fracs


def baseline_xs(T, families):
    """Checker's own trivial construction: split the budget round-robin
    across the K families (T // K each, remainder to the first families),
    and within a family aim at the midpoints of evenly-spaced mutants."""
    Kf = len(families)
    alloc = [T // Kf] * Kf
    rem = T - sum(alloc)
    for i in range(rem):
        alloc[i] += 1
    bxs = []
    for fi, fam in enumerate(families):
        n_f = len(fam)
        a_f = min(alloc[fi], n_f)
        if a_f <= 0:
            continue
        if a_f == 1:
            idxs = [0]
        else:
            idxs = sorted(set(round(i * (n_f - 1) / (a_f - 1)) for i in range(a_f)))
        for j in idxs:
            lo, hi = fam[j]
            bxs.append((lo + hi - 1) // 2)
    return bxs


def main():
    if len(sys.argv) < 3:
        fail("bad invocation")
    D, T, K, families = read_instance(sys.argv[1])

    try:
        out_toks = open(sys.argv[2]).read().split()
    except Exception:
        fail("cannot read output")
    if len(out_toks) == 0:
        fail("empty output")

    try:
        c = int(out_toks[0])
    except Exception:
        fail("first token (count) is not an integer")

    if c < 0 or c > T:
        fail("count out of [0, T] : %d" % c)
    if len(out_toks) != 1 + c:
        fail("expected %d tokens, got %d" % (1 + c, len(out_toks)))

    xs = []
    for t in out_toks[1:]:
        try:
            v = int(t)
        except Exception:
            fail("non-integer test point %r" % t)
        if not (0 <= v < D):
            fail("test point out of [0, D) : %r" % v)
        xs.append(v)

    F_fracs = kill_fracs(families, xs)
    F = min(F_fracs)

    B_fracs = kill_fracs(families, baseline_xs(T, families))
    B = max(min(B_fracs), 1e-9)

    sc = min(1000.0, 100.0 * F / B)
    print(
        "F=%.4f B=%.4f family_fracs=%s Ratio: %.6f"
        % (F, B, ["%.3f" % v for v in F_fracs], sc / 1000.0)
    )


if __name__ == "__main__":
    main()
