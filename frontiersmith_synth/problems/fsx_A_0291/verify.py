import sys

def fail(msg):
    print("Ratio: 0.0 " + msg)
    sys.exit(0)

def main():
    inp, out = sys.argv[1], sys.argv[2]

    # ---- parse instance ----
    itok = open(inp).read().split()
    idx = 0
    n = int(itok[idx]); idx += 1
    m = int(itok[idx]); idx += 1
    blocked = set()
    for _ in range(m):
        v = tuple(int(itok[idx + i]) for i in range(n)); idx += n
        blocked.add(v)
    space = 3 ** n

    # ---- internal baseline B: the half sub-cube {0}x{0,1}^(n-1) minus blocked ----
    #   (fix stage 1 to bin 0; a subset of the {0,1}^n cap set, so conflict-free)
    B = 0
    for msk in range(2 ** (n - 1)):
        v = (0,) + tuple((msk >> i) & 1 for i in range(n - 1))
        if v not in blocked:
            B += 1

    # ---- parse participant output (strict, finite, bounded) ----
    try:
        otok = open(out).read().split()
    except Exception:
        fail("noout")
    if not otok:
        fail("empty")
    # first token = k
    try:
        k = int(otok[0])
    except Exception:
        fail("badk")
    if k < 0 or k > space:
        fail("krange")
    if len(otok) != 1 + k * n:
        fail("tokcount")

    S = []
    Sset = set()
    idx = 1
    for _ in range(k):
        try:
            v = tuple(int(otok[idx + i]) for i in range(n))
        except Exception:
            fail("parse")
        idx += n
        for c in v:
            if c not in (0, 1, 2):
                fail("digit")
        if v in Sset:
            fail("dup")
        if v in blocked:
            fail("blocked")
        Sset.add(v)
        S.append(v)

    # ---- conflict-free (cap-set) check: no distinct triple summing to 0 mod 3 ----
    for a in range(k):
        pa = S[a]
        for b in range(a + 1, k):
            pb = S[b]
            z = tuple((3 - ((pa[i] + pb[i]) % 3)) % 3 for i in range(n))
            if z != pa and z != pb and z in Sset:
                fail("conflict")

    F = k
    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("Ratio: %.6f" % (sc / 1000.0))

main()
