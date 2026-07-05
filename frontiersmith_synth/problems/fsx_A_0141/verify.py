import sys, itertools


def fail(msg):
    print("%s Ratio: 0.0" % msg)
    sys.exit(0)


def main():
    inf, outf = sys.argv[1], sys.argv[2]
    raw = open(inf).read().splitlines()
    ptr = 0
    n = int(raw[ptr].split()[0]); ptr += 1
    b = int(raw[ptr].split()[0]); ptr += 1
    blocked = set()
    for _ in range(b):
        blocked.add(raw[ptr].strip()); ptr += 1
    weights = list(map(int, raw[ptr].split())); ptr += 1

    strs = [''.join(map(str, v)) for v in itertools.product(range(3), repeat=n)]
    if len(weights) != len(strs):
        fail("(bad instance)")
    wt = {s: weights[i] for i, s in enumerate(strs)}

    # ---- parse participant artifact ----
    out_lines = [l.strip() for l in open(outf).read().split('\n')]
    towers = [l for l in out_lines if l != '']
    Sset = set()
    for line in towers:
        parts = line.split()
        if len(parts) != 1:
            fail("(malformed line)")
        s = parts[0]
        if len(s) != n or any(c not in '012' for c in s):
            fail("(bad watchtower code)")
        if s in blocked:
            fail("(placed on a blocked cliff)")
        if s in Sset:
            fail("(duplicate tower)")
        Sset.add(s)

    # ---- feasibility: no three collinear (cap-set condition over F_3^n) ----
    S = [tuple(int(c) for c in s) for s in Sset]
    Tset = set(S)
    m = len(S)
    for i in range(m):
        x = S[i]
        for j in range(i + 1, m):
            y = S[j]
            w = tuple((-(x[k] + y[k])) % 3 for k in range(n))
            if w in Tset and w != x and w != y:
                fail("(three collinear watchtowers)")

    # ---- objective: total surveillance value ----
    F = sum(wt[s] for s in Sset)

    # ---- checker's own baseline construction: the shoreline-ridge subcube ----
    B = 0
    for s in strs:
        if s[0] == '0' and all(c in '01' for c in s[1:]):
            B += wt[s]

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%d B=%d Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
