import sys

# Format D checker -- shared addition-chain batch exponentiation (FLOPs = # of
# multiplication lines).  "one furnace schedule to forge many alloys".
#
#   1) Parse the batch of target exponents  e_1..e_k  from <in>.
#   2) Parse the participant's straight-line multiplication program from <out>:
#          L
#          a_1 b_1
#          ...
#          a_L b_L
#      Value index 0 is the formal symbol g (exponent 1).  Line n (1..L) defines
#      value index n whose EXPONENT = exponent[a_n] + exponent[b_n], with
#      0 <= a_n, b_n < n  (each line multiplies two earlier values).
#   3) FEASIBILITY (strict): valid indices, bounded L, bounded exponent size, and
#      every target exponent must appear as the exponent of some value.  Any
#      violation -> "Ratio: 0.0".
#   4) Objective (minimize) = L (multiplication count).
#      Baseline B = naive INDEPENDENT square-and-multiply per exponent
#                 = sum_i ( bitlen(e_i) - 1  +  popcount(e_i) - 1 ).
#      Ratio = min(1.0, 0.1 * B / L)   (fewer multiplies -> higher score).

LMAX = 100000


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def main():
    inp = open(sys.argv[1]).read().split()
    out = open(sys.argv[2]).read().split()

    # ---- parse targets ----
    it = iter(inp)
    try:
        k = int(next(it))
    except Exception:
        fail("bad k")
    if k < 1 or k > 1000:
        fail("bad k range")
    targets = []
    try:
        for _ in range(k):
            targets.append(int(next(it)))
    except Exception:
        fail("bad targets")
    if any(e < 1 for e in targets):
        fail("nonpositive target")

    maxt = max(targets)
    bitcap = maxt.bit_length() + 8

    # ---- baseline B (naive independent binary method) ----
    B = 0
    for e in set(targets):
        B += (e.bit_length() - 1) + (bin(e).count("1") - 1)
    if B <= 0:
        fail("degenerate baseline")

    # ---- parse participant program ----
    if not out:
        fail("empty output")
    try:
        L = int(out[0])
    except Exception:
        fail("bad L")
    if L < 0:
        fail("L < 0")
    if L > LMAX:
        fail("L too large")
    if len(out) != 1 + 2 * L:
        fail("wrong token count (got %d, need %d)" % (len(out), 1 + 2 * L))

    exps = [1]  # index 0 = g^1
    idx = 1
    try:
        for n in range(L):
            a = int(out[1 + 2 * n])
            b = int(out[2 + 2 * n])
            if a < 0 or b < 0 or a >= idx or b >= idx:
                fail("index out of range at line %d" % (n + 1))
            v = exps[a] + exps[b]
            if v.bit_length() > bitcap:
                fail("value exceeds bit cap at line %d" % (n + 1))
            exps.append(v)
            idx += 1
    except SystemExit:
        raise
    except Exception:
        fail("bad line")

    produced = set(exps)
    for e in targets:
        if e not in produced:
            fail("target %d not produced" % e)

    ratio = min(1.0, 0.1 * B / max(1, L))
    print("L=%d B=%d Ratio: %.6f" % (L, B, ratio))


if __name__ == "__main__":
    main()
