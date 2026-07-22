import sys


def fail(reason: str):
    print("INVALID: %s" % reason)
    print("Ratio: 0.0")
    sys.exit(0)


def main():
    if len(sys.argv) != 4:
        fail("bad checker invocation")
    in_path, out_path, _ans_path = sys.argv[1], sys.argv[2], sys.argv[3]

    with open(in_path, "r") as f:
        target = f.readline().rstrip("\n")
    n = len(target)
    if n == 0:
        fail("empty target instance")

    try:
        with open(out_path, "r") as f:
            raw_lines = f.readlines()
    except Exception:
        fail("cannot read participant output")

    lines = [ln.rstrip("\n") for ln in raw_lines]
    while lines and lines[-1].strip() == "":
        lines.pop()
    if not lines:
        fail("empty output")

    m_max = max(2000, 10 * n)
    if len(lines) > m_max:
        fail("too many rules (> %d)" % m_max)

    len_cap = 4 * n + 100  # any useful sub-expansion is a substring of target, so <= n;
                            # generous cap only guards against deliberate blow-up.

    expand = {}
    ops = 0

    for idx, line in enumerate(lines, start=1):
        parts = line.split()
        if not parts:
            fail("blank/malformed rule line at %d" % idx)
        tag = parts[0]

        if tag == "T":
            if len(parts) != 2 or len(parts[1]) != 1:
                fail("bad terminal rule at line %d" % idx)
            ch = parts[1]
            expand[idx] = ch
            ops += 1

        elif tag == "C":
            if len(parts) != 3:
                fail("bad concat rule at line %d" % idx)
            try:
                j, k = int(parts[1]), int(parts[2])
            except ValueError:
                fail("non-integer reference at line %d" % idx)
            if not (1 <= j < idx) or not (1 <= k < idx):
                fail("out-of-range/forward reference at line %d" % idx)
            s = expand[j] + expand[k]
            if len(s) > len_cap:
                fail("expansion too large at line %d" % idx)
            expand[idx] = s
            ops += 2

        elif tag == "R":
            if len(parts) != 2:
                fail("bad reverse rule at line %d" % idx)
            try:
                j = int(parts[1])
            except ValueError:
                fail("non-integer reference at line %d" % idx)
            if not (1 <= j < idx):
                fail("out-of-range/forward reference at line %d" % idx)
            s = expand[j][::-1]
            if len(s) > len_cap:
                fail("expansion too large at line %d" % idx)
            expand[idx] = s
            ops += 1

        else:
            fail("unknown rule tag '%s' at line %d" % (tag, idx))

    m = len(lines)
    final = expand[m]
    if len(final) != n or final != target:
        fail("final rule (start symbol) does not expand to the target string")

    f_ops = ops  # objective: total operation/symbol count across all right-hand sides
    baseline = 3 * n - 2  # trivial one-terminal-per-char + left chain of concatenations

    sc = min(1000.0, 100.0 * baseline / max(1e-9, f_ops))
    print("target_len=%d rules=%d ops=%d baseline=%d" % (n, m, f_ops, baseline))
    print("Ratio: %.6f" % (sc / 1000.0))
    sys.exit(0)


if __name__ == "__main__":
    main()
