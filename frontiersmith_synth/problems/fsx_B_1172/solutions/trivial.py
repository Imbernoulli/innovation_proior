# TIER: trivial
"""Constant-velocity, evenly-split-depth baseline. Uses only the very first
refraction pick (near-offset direct-wave slope) and the deepest reflection
time; ignores every other signal in the data (no segment detection, no use
of the interior tau values, no attempt to see the hidden layer)."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    test_id = int(next(it))
    N = int(next(it))
    M = int(next(it))
    xs, ts = [], []
    for _ in range(M):
        xs.append(float(next(it)))
        ts.append(float(next(it)))
    taus = [float(next(it)) for _ in range(N - 1)]
    # bounds V_MIN V_MAX H_MIN H_MAX unused by this tier

    v1 = xs[0] / ts[0] if ts[0] > 0 else 1000.0
    Dtot = taus[-1] * v1 / 2.0
    h_each = Dtot / (N - 1)

    out = []
    for _ in range(N - 1):
        out.append("%.6f %.6f" % (h_each, v1))
    out.append("%.6f" % v1)
    print("\n".join(out))


if __name__ == "__main__":
    main()
