# TIER: trivial
"""Reads the published table, brute-force-searches for constants (k,p) that
reproduce it exactly (so it is always *correct*, on every instance, including the
'trap' ones), but never looks for shared subexpressions: it emits the direct,
term-by-term textbook expansion of (x+y)^3 - x^3 - y^3, recomputing x*y a second
time for the denominator. Op count ~13 -- the 'obvious, unsimplified' baseline."""
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
    return 1, 1  # should not happen if the table is genuinely from this family


def main():
    testId, M, table = read_table()
    k, p = find_kp(M, table)

    lines = []
    lines.append("13")
    lines.append("ADD x y")          # R1 = x+y
    lines.append("MUL R1 R1")        # R2 = (x+y)^2
    lines.append("MUL R2 R1")        # R3 = (x+y)^3
    lines.append("MUL x x")          # R4 = x^2
    lines.append("MUL R4 x")         # R5 = x^3
    lines.append("MUL y y")          # R6 = y^2
    lines.append("MUL R6 y")         # R7 = y^3
    lines.append("SUB R3 R5")        # R8 = (x+y)^3 - x^3
    lines.append("SUB R8 R7")        # R9 = numerator = (x+y)^3 - x^3 - y^3
    lines.append("MUL x y")          # R10 = x*y (recomputed, not reused)
    lines.append(f"MUL R10 {k}")     # R11 = k*x*y
    lines.append(f"ADD R11 {p}")     # R12 = denom = k*x*y + p
    lines.append("MOD R9 R12")       # R13 = numerator mod denom
    lines.append("OUT R13")
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
