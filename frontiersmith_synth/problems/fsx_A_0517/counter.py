# counter.py -- Format D checker for hidden-factor-evaluation-dag.
#
# Input  (<in>):  p n k, then per polynomial:  M, then M lines "coeff e_0..e_{n-1}".
# Output (<out>): a straight-line program (SLP) over F_p computing all k targets.
#     L
#     L lines, each one of:
#         const <v>          # a field constant in [0,p)
#         add <i> <j>        # value[i] + value[j]
#         sub <i> <j>        # value[i] - value[j]
#         mul <i> <j>        # value[i] * value[j]
#     out <o_0> ... <o_{k-1}>
#   Nodes 0..n-1 are the input variables x_0..x_{n-1} (implicit).  Instruction t
#   (0-based) defines node n+t and may only reference earlier nodes (a DAG).
#
# COST MODEL (non-scalar / multiplicative complexity): a `mul` costs 1 ONLY when
# BOTH operands are non-constant (depend on the inputs).  Multiplication by a
# constant, addition and subtraction are FREE.  So a linear form costs 0 and the
# whole game is how few genuine products build all k targets.
#
# Scoring (minimize cost):
#   1) STRICT feasibility: parse + range check; reject non-integer / out-of-range.
#   2) EXACT equivalence at seeded random points in F_p^n (Schwartz-Zippel);
#      any mismatch -> Ratio 0.0.
#   3) Baseline B = op-count of the canonical shared-monomial evaluation the
#      checker builds itself (shared univariate powers + one product per distinct
#      monomial).  Ratio = min(1.0, 0.1 * B / cost).
import sys, random

def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)

def main():
    try:
        ind = open(sys.argv[1]).read().split()
        outd = open(sys.argv[2]).read().split()
    except Exception:
        fail("io")

    it = iter(ind)
    try:
        p = int(next(it)); n = int(next(it)); k = int(next(it))
    except Exception:
        fail("bad header")
    if not (2 <= p <= (1 << 62)):
        fail("bad p")
    if not (1 <= n <= 2000) or not (1 <= k <= 2000):
        fail("bad n/k")

    polys = []           # list of list of (coeff, exps-tuple)
    maxexp = [0] * n
    try:
        for _ in range(k):
            M = int(next(it))
            if M < 0 or M > 2_000_000:
                fail("bad M")
            mons = []
            for _ in range(M):
                c = int(next(it)) % p
                exps = tuple(int(next(it)) for _ in range(n))
                for j in range(n):
                    if exps[j] < 0:
                        fail("neg exp")
                    if exps[j] > maxexp[j]:
                        maxexp[j] = exps[j]
                mons.append((c, exps))
            polys.append(mons)
    except StopIteration:
        fail("truncated input")
    except Exception:
        fail("bad instance")

    # ---- checker baseline B: canonical shared-monomial evaluation ----
    distinct = set()
    for mons in polys:
        for c, exps in mons:
            distinct.add(exps)
    powers = sum(max(0, m - 1) for m in maxexp)
    combine = 0
    for exps in distinct:
        supp = sum(1 for e in exps if e >= 1)
        if supp >= 2:
            combine += supp - 1
    B = powers + combine
    if B <= 0:
        fail("degenerate instance")

    # ---- parse participant SLP ----
    if not outd:
        fail("empty output")
    oit = iter(outd)
    try:
        L = int(next(oit))
    except Exception:
        fail("bad L")
    if L < 0 or L > 6_000_000:
        fail("L out of range")

    ops = []                 # (kind, a, b)  kind in 0=const 1=add 2=sub 3=mul
    is_const = [False] * n   # inputs are non-constant
    cost = 0
    try:
        for t in range(L):
            op = next(oit)
            cur = n + t
            if op == "const":
                v = int(next(oit))
                if v < 0 or v >= p:
                    fail("const out of range")
                ops.append((0, v, 0))
                is_const.append(True)
            elif op == "add" or op == "sub" or op == "mul":
                a = int(next(oit)); b = int(next(oit))
                if not (0 <= a < cur and 0 <= b < cur):
                    fail("bad node reference")
                ca = is_const[a]; cb = is_const[b]
                if op == "mul":
                    if (not ca) and (not cb):
                        cost += 1
                    ops.append((3, a, b))
                    is_const.append(ca and cb)
                elif op == "add":
                    ops.append((1, a, b)); is_const.append(ca and cb)
                else:
                    ops.append((2, a, b)); is_const.append(ca and cb)
            else:
                fail("bad opcode")
        if next(oit) != "out":
            fail("missing out line")
        outs = [int(next(oit)) for _ in range(k)]
    except StopIteration:
        fail("truncated output")
    except SystemExit:
        raise
    except Exception:
        fail("parse error")
    for o in outs:
        if not (0 <= o < n + L):
            fail("bad out index")

    # ---- exact equivalence at seeded random points ----
    seedmix = 0x9E3779B97F4A7C15
    for mons in polys:
        for c, exps in mons:
            seedmix = (seedmix * 1000003 + c) & ((1 << 63) - 1)
    seedmix ^= (n << 17) ^ (k << 3) ^ B
    rng = random.Random(seedmix)
    T = 32
    for _ in range(T):
        pt = [rng.randrange(p) for _ in range(n)]
        val = list(pt)                      # nodes 0..n-1
        for (kind, a, b) in ops:
            if kind == 0:
                val.append(a % p)
            elif kind == 1:
                val.append((val[a] + val[b]) % p)
            elif kind == 2:
                val.append((val[a] - val[b]) % p)
            else:
                val.append((val[a] * val[b]) % p)
        for i in range(k):
            s = 0
            for c, exps in polys[i]:
                term = c
                for j in range(n):
                    e = exps[j]
                    if e == 1:
                        term = term * pt[j] % p
                    elif e:
                        term = term * pow(pt[j], e, p) % p
                s = (s + term) % p
            if val[outs[i]] % p != s:
                fail("reconstruction mismatch at target %d" % i)

    ratio = min(1.0, 0.1 * B / max(1, cost))
    print("B=%d cost=%d Ratio: %.6f" % (B, cost, ratio))

if __name__ == "__main__":
    main()
