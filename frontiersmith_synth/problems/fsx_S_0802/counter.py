import sys, math

MAXBITS = 2_000_000     # guard against pathological (adversarial) intermediate blow-up
MAXCONST = 10 ** 6
MAXLINES = 3000


def fail(reason):
    print("# " + reason)
    print("Ratio: 0.0")
    sys.exit(0)


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path) as f:
        d = int(f.readline().split()[0])
        coeffs = list(map(int, f.readline().split()))
    if len(coeffs) != d + 1:
        fail("malformed instance")  # should never happen; defensive only

    try:
        with open(out_path) as f:
            raw_lines = f.readlines()
    except Exception:
        fail("cannot read output")

    lines = [ln.rstrip("\n") for ln in raw_lines]
    while lines and lines[-1].strip() == "":
        lines.pop()
    if not lines:
        fail("empty output")

    header = lines[0].split()
    if len(header) != 1:
        fail("bad header")
    try:
        L = int(header[0])
    except ValueError:
        fail("bad header (not an integer)")
    if L < 1 or L > MAXLINES:
        fail(f"instruction count {L} out of bounds [1,{MAXLINES}]")
    if len(lines) < 1 + L:
        fail("too few instruction lines")

    ops = []  # each: ("C", value) | ("A"|"S"|"M", a, b)
    for i in range(1, 1 + L):
        toks = lines[i].split()
        if not toks:
            fail(f"blank instruction line {i}")
        op = toks[0]
        cur_idx = i  # this instruction produces wire index i (wire 0 == x)
        if op == "C":
            if len(toks) != 2:
                fail(f"line {i}: C takes exactly 1 operand")
            try:
                c = int(toks[1])
            except ValueError:
                fail(f"line {i}: non-integer constant")
            if abs(c) > MAXCONST:
                fail(f"line {i}: constant out of bounds")
            ops.append(("C", c))
        elif op in ("A", "S", "M"):
            if len(toks) != 3:
                fail(f"line {i}: {op} takes exactly 2 operands")
            try:
                a = int(toks[1]); b = int(toks[2])
            except ValueError:
                fail(f"line {i}: non-integer wire index")
            if not (0 <= a < cur_idx) or not (0 <= b < cur_idx):
                fail(f"line {i}: wire index out of range (must reference a strictly earlier wire)")
            ops.append((op, a, b))
        else:
            fail(f"line {i}: unknown opcode '{op}'")

    mult_count = sum(1 for o in ops if o[0] == "M")
    if mult_count == 0:
        fail("no multiplications: cannot represent a degree>=2 polynomial with a_i not all 0")

    depth = [0] * (L + 1)  # depth[0] = 0 for wire 0 (== x)
    for i, o in enumerate(ops, start=1):
        if o[0] == "C":
            depth[i] = 0
        elif o[0] in ("A", "S"):
            depth[i] = max(depth[o[1]], depth[o[2]])
        else:
            depth[i] = max(depth[o[1]], depth[o[2]]) + 1
    mult_depth = depth[L]

    def evaluate(x):
        wires = [x]
        for o in ops:
            if o[0] == "C":
                v = o[1]
            elif o[0] == "A":
                v = wires[o[1]] + wires[o[2]]
            elif o[0] == "S":
                v = wires[o[1]] - wires[o[2]]
            else:
                v = wires[o[1]] * wires[o[2]]
            if v.bit_length() > MAXBITS:
                fail("intermediate value exceeds the size bound (circuit is degenerate/adversarial)")
            wires.append(v)
        return wires[L]

    def reference(x):
        v = coeffs[d]
        for i in range(d - 1, -1, -1):
            v = v * x + coeffs[i]
        return v

    # Exact-equivalence check: enough consecutive small points to force equality for
    # any degree-<=d candidate, plus a few large, far-apart points to catch a
    # higher-degree circuit that might spuriously agree only on the small range.
    small_pts = list(range(2, 2 + d + 6))
    big_pts = [10 ** 6 + 3, 10 ** 6 + 33, 2 * 10 ** 6 + 7]
    for x in small_pts + big_pts:
        got = evaluate(x)
        want = reference(x)
        if got != want:
            fail(f"circuit does not equal the target polynomial (mismatch at x={x})")

    # ---- scoring -------------------------------------------------------------
    # Cost F = mult_count * 2^mult_depth (an FHE-style leveled cost: additions are
    # free, every multiplication costs one level and the cost model is EXPONENTIAL
    # in the deepest multiplicative chain). We score in log-space because F itself
    # spans many orders of magnitude across strategies by design.
    #
    #   L(F)      = ln(mult_count) + mult_depth * ln(2)
    #   L_base    = the analytic cost of the naive "rebuild every power from
    #               scratch, no sharing between terms" construction:
    #               mult_count = d(d+1)/2, depth = d.
    #   L_floor   = an UNREACHABLE joint lower bound: the true minimum multiplication
    #               count for a generic degree-d polynomial is d (Motzkin-Belaga),
    #               and the information-theoretic minimum depth to reach degree d
    #               via repeated squaring is ceil(log2(d+1)); no single circuit
    #               attains both at once, so this floor is a normalizer, not a target.
    #   frac      = clip((L_base - L(F)) / (L_base - L_floor), 0, 1)
    #   Ratio     = 0.1 + 0.75 * frac        (trivial ~= 0.1; cap 0.85 leaves headroom)
    mc_base = d * (d + 1) // 2
    depth_base = d
    mc_floor = d
    depth_floor = max(1, math.ceil(math.log2(d + 1)))

    L_base = math.log(mc_base) + depth_base * math.log(2)
    L_floor = math.log(mc_floor) + depth_floor * math.log(2)
    L_part = math.log(mult_count) + mult_depth * math.log(2)

    denom = L_base - L_floor
    frac = 0.0 if denom <= 1e-9 else (L_base - L_part) / denom
    frac = max(0.0, min(1.0, frac))
    sc = 0.1 + 0.75 * frac
    sc = max(0.0, min(1.0, sc))

    print(f"# mult_count={mult_count} mult_depth={mult_depth} "
          f"L_base={L_base:.4f} L_floor={L_floor:.4f} L_part={L_part:.4f} frac={frac:.4f}")
    print("Ratio: %.6f" % sc)


if __name__ == "__main__":
    main()
