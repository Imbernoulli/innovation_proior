# TIER: greedy
"""The obvious first read of 'digit sum': assume the ambient, profane base
ten (the cult's secret radix is never even considered), then least-squares
fit a real-valued quadratic against the base-10 digit sum of each logged
tribute. This tracks the training rows loosely (the mod-q wraps just look
like scatter to a real-valued fit) but the recovered curve is anchored to
base-10 digit statistics that have nothing to do with the cult's actual
radix, so it fails badly once tributes grow to dozens/hundreds of digits."""
import sys


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


def solve3(A, rhs):
    D = det3(A)
    if abs(D) < 1e-9:
        return [0.0, 0.0, 0.0]
    out = []
    for i in range(3):
        Ai = [row[:] for row in A]
        for r in range(3):
            Ai[r][i] = rhs[r]
        out.append(det3(Ai) / D)
    return out


def main():
    header = sys.stdin.readline().split()
    K = int(header[1])
    xs = []
    ys = []
    for _ in range(K):
        parts = sys.stdin.readline().split()
        xs.append(int(parts[0]))
        ys.append(int(parts[1]))

    b = 10
    s = [digitsum_base(x, b) for x in xs]
    n = len(s)

    S0 = float(n)
    S1 = float(sum(s))
    S2 = float(sum(v * v for v in s))
    S3 = float(sum(v ** 3 for v in s))
    S4 = float(sum(v ** 4 for v in s))
    Y0 = float(sum(ys))
    Y1 = float(sum(v * yy for v, yy in zip(s, ys)))
    Y2 = float(sum((v * v) * yy for v, yy in zip(s, ys)))

    A = [[S4, S3, S2], [S3, S2, S1], [S2, S1, S0]]
    rhs = [Y2, Y1, Y0]
    a2, a1, a0 = solve3(A, rhs)

    a2 = max(-100, min(100, int(round(a2))))
    a1 = max(-1000, min(1000, int(round(a1))))
    a0 = max(-100000, min(100000, int(round(a0))))

    print(b, a2, a1, a0)


if __name__ == "__main__":
    main()
