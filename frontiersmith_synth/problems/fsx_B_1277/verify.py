import sys, os
from fractions import Fraction

MAX_OUT_BYTES = 200_000  # generous for <=28 indices; guards huge-garbage adversarial outputs


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def main():
    inf, outf = sys.argv[1], sys.argv[2]

    try:
        sz = os.path.getsize(outf)
    except OSError:
        fail("no output")
    if sz > MAX_OUT_BYTES:
        fail("output too large")

    itoks = open(inf).read().split()
    it = iter(itoks)
    N = int(next(it)); C = int(next(it)); K = int(next(it)); OVER_MULT = int(next(it))
    storms = []
    for _ in range(K):
        cx = int(next(it)); cy = int(next(it)); R = int(next(it)); L = int(next(it)); sev = int(next(it))
        storms.append((cx, cy, R, L, sev))
    policies = []
    for _ in range(N):
        x = int(next(it)); y = int(next(it)); e = int(next(it)); p = int(next(it)); tech = int(next(it))
        policies.append((x, y, e, p, tech))

    # ---- bounded, strict parse of the participant artifact ----
    try:
        raw = open(outf).read()
    except OSError:
        fail("cannot read output")
    otoks = raw.split()
    if not otoks:
        fail("empty output")
    try:
        m = int(otoks[0])
    except (ValueError, OverflowError):
        fail("bad count token")
    if m < 0 or m > C:
        fail("count out of [0,C]: m=%d C=%d" % (m, C))
    if len(otoks) - 1 < m:
        fail("not enough index tokens")
    ids = []
    seen = set()
    for tok in otoks[1:1 + m]:
        try:
            idx = int(tok)
        except (ValueError, OverflowError):
            fail("bad index token %r" % tok[:20])
        if idx < 0 or idx >= N:
            fail("index out of range %d" % idx)
        if idx in seen:
            fail("duplicate index %d" % idx)
        seen.add(idx)
        ids.append(idx)
    # trailing tokens beyond the declared m are ignored (m is authoritative), but must
    # still be well-formed integers if present, else this is a malformed artifact.
    for tok in otoks[1 + m:]:
        try:
            int(tok)
        except (ValueError, OverflowError):
            fail("bad trailing token %r" % tok[:20])

    # ---- precompute footprint membership ----
    footprint = [[] for _ in range(N)]  # policy idx -> list of storm idx covering it
    for i, (x, y, e, p, tech) in enumerate(policies):
        for s, (cx, cy, R, L, sev) in enumerate(storms):
            if (x - cx) ** 2 + (y - cy) ** 2 <= R * R:
                footprint[i].append(s)

    def objective(book):
        Xs = [0] * K
        for i in book:
            for s in footprint[i]:
                Xs[s] += policies[i][2]
        prem_margin = sum(policies[i][3] - policies[i][4] for i in book)
        loss = Fraction(0)
        for s, (cx, cy, R, L, sev) in enumerate(storms):
            x = Xs[s]
            over = max(0, x - L)
            under = min(x, L)
            loss += Fraction(sev, 1000) * (under + OVER_MULT * over)
        return Fraction(prem_margin) - loss

    F = objective(ids)

    # ---- checker's own baseline B: a genuinely trivial, non-strategic construction --
    # write only the top ceil(C/3) candidates by isolated margin (a timid underwriter
    # who leaves most of the capacity idle out of excess caution), storms ignored
    # entirely. This is what solutions/trivial.py reproduces exactly. ----
    order = sorted(range(N), key=lambda i: (-(policies[i][3] - policies[i][4]), i))
    k = max(1, -(-C // 3))
    B = float(objective(order[:k]))
    if B <= 0:
        B = 1e-6

    Ff = float(F)
    sc = min(1000.0, 100.0 * Ff / max(1e-9, B))
    sc = max(0.0, sc)
    ratio = sc / 1000.0
    print("F=%.3f B=%.3f m=%d Ratio: %.6f" % (Ff, B, m, ratio))


if __name__ == "__main__":
    main()
