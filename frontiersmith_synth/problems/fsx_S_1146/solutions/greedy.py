# TIER: greedy
"""The 'obvious recipe': the coder notices the numerator looks like a scaled
xy(x+y) term and, having seen the coefficient 3 come out of the cube-difference
they expand on paper, ASSUMES the denominator uses the same coefficient (k=3)
for every instance instead of verifying it against the data. Under that
assumption they only search for the additive constant p, and -- because they
never re-derive k -- they also spot that 3*x*y can be reused for both the
numerator and the denominator, giving a nicely compact 6-op circuit. This
recipe is right most of the time (it IS a real algebraic simplification) but
is blind to instances whose true multiplier k != 3: there it silently locks
onto a best-effort p and ships a circuit that is wrong outside (and usually
even inside) the published table."""
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


def F_k3(x, y, p):
    num = 3 * x * y * (x + y)
    den = 3 * x * y + p
    return num % den


def find_p_assuming_k3(M, table, pmax=80):
    best_p, best_mismatches = 1, None
    for p in range(1, pmax + 1):
        mism = 0
        for x in range(M + 1):
            for y in range(M + 1):
                if F_k3(x, y, p) != table[x][y]:
                    mism += 1
        if mism == 0:
            return p, True
        if best_mismatches is None or mism < best_mismatches:
            best_mismatches, best_p = mism, p
    return best_p, False  # no p under the k=3 assumption reproduces the table


def main():
    testId, M, table = read_table()
    p, matched = find_p_assuming_k3(M, table)

    # Compact circuit that only works when the true multiplier really is 3:
    # R1=x*y, R2=x+y, R3=3*R1 (reused for BOTH numerator and denominator),
    # R4=R3*R2 (numerator), R5=R3+p (denominator), R6=R4 mod R5.
    lines = []
    lines.append("6")
    lines.append("MUL x y")     # R1 = x*y
    lines.append("ADD x y")     # R2 = x+y
    lines.append("MUL R1 3")    # R3 = 3*x*y  (assumed shared with denominator)
    lines.append("MUL R3 R2")   # R4 = numerator = 3*x*y*(x+y)
    lines.append(f"ADD R3 {p}")  # R5 = denominator, HARD-CODED k=3
    lines.append("MOD R4 R5")   # R6 = F
    lines.append("OUT R6")
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
