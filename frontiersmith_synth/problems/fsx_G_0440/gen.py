import sys, random

# gen.py <testId>  -- prints ONE GF(2) linear-map instance (a 0/1 matrix M) to stdout.
#
# Theme: crypto linear layer.  M is the binary matrix of a fixed GF(2) linear map
# y = M x (m output bits, n input bits).  The solver must emit a straight-line
# program of 2-input XOR gates computing every output bit exactly, using as few
# gates as possible (the classic "XOR-count" / linear straight-line program
# minimization problem -- NP-hard, so the true optimum is genuinely unknown).
#
# Each row is planted as a symmetric difference of a few shared "gadget" subsets
# plus a couple of raw inputs.  Re-used gadgets create abundant common
# subexpressions, so good common-subexpression elimination beats the naive
# per-row baseline -- yet the minimum XOR count stays open.  Difficulty grows
# with testId (more inputs/outputs, more gadgets).

def build(tid):
    rng = random.Random(770000 + 131 * tid)
    n = 10 + 2 * tid          # inputs
    m = 10 + 2 * tid          # outputs
    ngad = n // 2 + tid       # number of shared gadgets
    gadgets = []
    for _ in range(ngad):
        k = rng.randint(2, 4)
        gadgets.append(set(rng.sample(range(n), k)))
    rows = []
    seen = set()
    guard = 0
    while len(rows) < m:
        guard += 1
        if guard > 100000:
            break
        acc = set()
        for g in rng.sample(gadgets, rng.randint(2, 3)):
            acc ^= g
        for _ in range(rng.randint(0, 2)):
            acc ^= {rng.randrange(n)}
        if len(acc) < 3:          # guarantee popcount >= 3 (baseline stays > 0)
            continue
        fr = frozenset(acc)
        if fr in seen:
            continue
        seen.add(fr)
        rows.append(sorted(acc))
    M = [[1 if i in set(r) else 0 for i in range(n)] for r in rows]
    return n, m, M

def main():
    tid = int(sys.argv[1])
    n, m, M = build(tid)
    out = ["%d %d" % (m, n)]
    for r in M:
        out.append(" ".join(str(v) for v in r))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
