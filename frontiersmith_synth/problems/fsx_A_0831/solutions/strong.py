# TIER: strong
"""The insight: switching coordinates from x to s = digitsum_b(x) is exactly
what turns the cult's law into a plain quadratic -- it is invisible in x but
low-degree in the right coordinate. There are only 38 candidate radices, so
brute-force scan them; for each candidate, pick three logged tributes whose
digit sums in that base are pairwise distinct and solve the 3x3 Vandermonde
system EXACTLY (Fraction arithmetic, no rounding) for (a2, a1, a0); then
VERIFY the recovered quadratic reproduces EVERY other logged tribute's
blessing exactly. Only the true radix survives verification across dozens
of independent rows -- a wrong radix cannot accidentally satisfy that many
exact integer constraints. A handful of exact comparisons per candidate
base is enough to pin down the whole ritual."""
import sys
from fractions import Fraction

B_LO, B_HI = 3, 40


def digitsum_base(x, b):
    s = 0
    while x > 0:
        s += x % b
        x //= b
    return s


def det3(m):
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))


def solve3_exact(A, rhs):
    D = det3(A)
    if D == 0:
        return None
    out = []
    for i in range(3):
        Ai = [row[:] for row in A]
        for r in range(3):
            Ai[r][i] = rhs[r]
        out.append(Fraction(det3(Ai), D))
    return out


def try_base(b, xs, ys):
    ss = [digitsum_base(x, b) for x in xs]
    n = len(ss)
    picks = []
    seen = set()
    for i in range(n):
        if ss[i] not in seen:
            seen.add(ss[i])
            picks.append(i)
        if len(picks) == 3:
            break
    if len(picks) < 3:
        return None
    i0, i1, i2 = picks
    A = [[Fraction(ss[i0] * ss[i0]), Fraction(ss[i0]), Fraction(1)],
         [Fraction(ss[i1] * ss[i1]), Fraction(ss[i1]), Fraction(1)],
         [Fraction(ss[i2] * ss[i2]), Fraction(ss[i2]), Fraction(1)]]
    rhs = [Fraction(ys[i0]), Fraction(ys[i1]), Fraction(ys[i2])]
    sol = solve3_exact(A, rhs)
    if sol is None:
        return None
    a2, a1, a0 = sol
    if a2.denominator != 1 or a1.denominator != 1 or a0.denominator != 1:
        return None
    a2, a1, a0 = int(a2), int(a1), int(a0)
    for i in range(n):
        pred = a2 * ss[i] * ss[i] + a1 * ss[i] + a0
        if pred != ys[i]:
            return None
    if abs(a2) > 100 or abs(a1) > 1000 or abs(a0) > 100000:
        return None
    return (b, a2, a1, a0)


def main():
    header = sys.stdin.readline().split()
    K = int(header[1])
    xs = []
    ys = []
    for _ in range(K):
        parts = sys.stdin.readline().split()
        xs.append(int(parts[0]))
        ys.append(int(parts[1]))

    found = None
    for b in range(B_LO, B_HI + 1):
        res = try_base(b, xs, ys)
        if res is not None:
            found = res
            break

    if found is None:
        # Should not happen with >=54 diverse logged tributes; fall back to
        # the flat blind guess rather than emitting something infeasible.
        print(3, 0, 0, 0)
        return

    b, a2, a1, a0 = found
    print(b, a2, a1, a0)


if __name__ == "__main__":
    main()
