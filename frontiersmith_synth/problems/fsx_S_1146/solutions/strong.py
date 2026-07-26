# TIER: strong
"""The insight: don't assume the multiplier, VERIFY it against the whole table --
that is exactly the 'find the law' half of the objective. Once the true (k,p)
pair is known exactly, the 'simplify the law' half is the SAME optimization as
the op-count objective: only when k genuinely equals 3 does the k*x*y term in
the denominator coincide with the 3*x*y term already needed for the numerator,
so only then can the two branches share one multiplication (6 ops). When k != 3
that sharing is algebraically unavailable, so the circuit correctly falls back
to computing k*x*y separately (7 ops) rather than emitting a circuit that merely
LOOKS compact but is wrong. Because the search covers the true (k,p) on every
instance, this stays exact everywhere -- including the 3 instances where the
'assume k=3' recipe silently breaks."""
import sys


def read_table():
    data = sys.stdin.read().split("\n")
    header = data[0].split()
    testId = int(header[0])
    M = int(header[1])
    table = []
    for x in range(M + 1):
        table.append([int(t) for t in data[1 + x].split()])
    return testId, M, table


def F(x, y, k, p):
    num = 3 * x * y * (x + y)
    den = k * x * y + p
    return num % den


def find_kp(M, table, kmax=40, pmax=40):
    for k in range(1, kmax + 1):
        for p in range(1, pmax + 1):
            ok = True
            for x in range(M + 1):
                for y in range(M + 1):
                    if F(x, y, k, p) != table[x][y]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return k, p
    return None, None


def main():
    testId, M, table = read_table()
    k, p = find_kp(M, table)
    if k is None:
        # Should never happen for a genuine instance of this family; fall back to
        # a safe (still-correct-if-lucky) guess rather than crashing.
        k, p = 1, 1

    lines = []
    if k == 3:
        # shared-subexpression circuit: 3*x*y feeds BOTH numerator and denominator
        lines.append("6")
        lines.append("MUL x y")      # R1 = x*y
        lines.append("ADD x y")      # R2 = x+y
        lines.append("MUL R1 3")     # R3 = 3*x*y
        lines.append("MUL R3 R2")    # R4 = numerator
        lines.append(f"ADD R3 {p}")  # R5 = denominator (k really is 3)
        lines.append("MOD R4 R5")    # R6 = F
        lines.append("OUT R6")
    else:
        # general circuit: numerator's 3*x*y and denominator's k*x*y cannot share
        # a register, so we pay one extra multiply -- but it is still exact.
        lines.append("7")
        lines.append("MUL x y")      # R1 = x*y
        lines.append("ADD x y")      # R2 = x+y
        lines.append("MUL R1 3")     # R3 = 3*x*y
        lines.append("MUL R3 R2")    # R4 = numerator
        lines.append(f"MUL R1 {k}")  # R5 = k*x*y
        lines.append(f"ADD R5 {p}")  # R6 = denominator
        lines.append("MOD R4 R6")    # R7 = F
        lines.append("OUT R7")
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
