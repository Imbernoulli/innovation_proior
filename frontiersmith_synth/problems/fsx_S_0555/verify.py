import sys

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)

SMAX = 5000
MMAX = 400000

def main():
    try:
        inp = open(sys.argv[1]).read().split()
        out = open(sys.argv[2]).read().split()
    except Exception:
        fail("io")

    # ---- parse instance ----
    try:
        it = iter(inp)
        alphabet = next(it)
        nP = int(next(it)); nN = int(next(it))
        P = [next(it) for _ in range(nP)]
        N = [next(it) for _ in range(nN)]
    except Exception:
        fail("bad instance")
    alpha = set(alphabet)

    # ---- internal baseline B = trie acceptor of P (accepts exactly P) ----
    pref = set()
    for s in P:
        for i in range(len(s) + 1):
            pref.add(s[:i])
    B = max(1, len(pref))

    # ---- parse participant NFA ----
    try:
        pit = iter(out)
        S = int(next(pit))
        ns = int(next(pit))
        start = [int(next(pit)) for _ in range(ns)]
        na = int(next(pit))
        accept = [int(next(pit)) for _ in range(na)]
        M = int(next(pit))
    except Exception:
        fail("parse header")
    if not (1 <= S <= SMAX):
        fail("state count out of range")
    if not (0 <= ns <= S) or not (0 <= na <= S):
        fail("bad start/accept count")
    if not (0 <= M <= MMAX):
        fail("bad transition count")
    for q in start:
        if not (0 <= q < S):
            fail("start index oob")
    for q in accept:
        if not (0 <= q < S):
            fail("accept index oob")

    # transition map: (from, sym) -> list of to
    trans = {}
    try:
        for _ in range(M):
            f = int(next(pit)); sym = next(pit); t = int(next(pit))
            if not (0 <= f < S) or not (0 <= t < S):
                fail("transition index oob")
            if sym not in alpha or len(sym) != 1:
                fail("transition symbol not in alphabet")
            trans.setdefault((f, sym), []).append(t)
    except SystemExit:
        raise
    except Exception:
        fail("parse transitions")

    start_set = frozenset(start)
    accept_set = frozenset(accept)

    def accepts(x):
        cur = set(start_set)
        for ch in x:
            if not cur:
                return False
            nxt = set()
            for q in cur:
                lst = trans.get((q, ch))
                if lst:
                    nxt.update(lst)
            cur = nxt
        return bool(cur & accept_set)

    # ---- strict feasibility: accept all P, reject all N ----
    for s in P:
        if not accepts(s):
            fail("positive string rejected")
    for s in N:
        if accepts(s):
            fail("negative string accepted")

    # ---- objective: minimize number of states ----
    F = float(S)
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("baseline(trie)=%d states=%d Ratio: %.6f" % (B, S, sc / 1000.0))

main()
