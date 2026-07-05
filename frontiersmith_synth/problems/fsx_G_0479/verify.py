import sys, math

# Deterministic checker for fsx_G_0479 (format C, maximization).
# Instance : n k  /  forbidden channels.
# Artifact : the chosen frequency channels (space/newline separated integers).
# Feasibility (ANY violation -> Ratio: 0.0):
#   - each channel is an integer in [1..n]
#   - channels are distinct
#   - no channel is forbidden
#   - the set is a Sidon (B2) set: all pairwise sums f_i+f_j (i<=j) are DISTINCT,
#     i.e. no two channel pairs share a second-order intermodulation product.
# Objective F = number of channels selected.
# Baseline  B = the power-of-two guard-band set {1,2,4,8,...}<=n (always feasible & Sidon).
# Score: sc = min(1000, 100*F/max(1e-9,B)); Ratio = sc/1000  (trivial=B -> 0.1).

def fail(msg):
    print("INVALID: " + msg)
    print("Ratio: 0.0")
    sys.exit(0)

def main():
    try:
        inp = open(sys.argv[1]).read().split()
    except Exception as e:
        fail("cannot read instance: %r" % e)
    if len(inp) < 2:
        fail("malformed instance")
    n = int(inp[0]); k = int(inp[1])
    forb = set(int(x) for x in inp[2:2 + k])

    # internal baseline: allowed powers of two
    B = 0
    v = 1
    while v <= n:
        if v not in forb:
            B += 1
        v *= 2
    if B < 1:
        B = 1

    try:
        raw = open(sys.argv[2]).read().split()
    except Exception:
        fail("cannot read output")

    # hard cap: a Sidon set in [1..n] has size < sqrt(n)+n^(1/4)+1; bound generously.
    hard_cap = 2 * math.isqrt(n) + 16
    if len(raw) > n + 5 or len(raw) > hard_cap:
        # too many tokens to ever be a valid Sidon set
        # (only reject-here if it exceeds the mathematical cap)
        if len(raw) > hard_cap:
            fail("output larger than any possible Sidon set")

    sel = []
    for tok in raw:
        try:
            x = int(tok)
        except Exception:
            fail("non-integer token: %r" % tok)
        # reject accidental floats / non-finite disguised as tokens
        if tok.lower() in ("nan", "inf", "-inf", "+inf"):
            fail("non-finite token")
        sel.append(x)

    seen = set()
    for x in sel:
        if x < 1 or x > n:
            fail("channel out of range: %d" % x)
        if x in forb:
            fail("forbidden channel used: %d" % x)
        if x in seen:
            fail("duplicate channel: %d" % x)
        seen.add(x)

    # Sidon check: all pairwise sums (i<=j) distinct
    sums = set()
    arr = sel
    L = len(arr)
    for i in range(L):
        ai = arr[i]
        for j in range(i, L):
            s = ai + arr[j]
            if s in sums:
                fail("intermodulation collision -> not a Sidon set")
            sums.add(s)

    F = L
    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    ratio = sc / 1000.0
    print("F=%d B=%d n=%d" % (F, B, n))
    print("Ratio: %.6f" % ratio)

if __name__ == "__main__":
    main()
