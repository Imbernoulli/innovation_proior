import sys, math
import numpy as np


def die(msg):
    # any feasibility violation -> zero score
    print("reason: %s" % msg)
    print("Ratio: 0.0")
    sys.exit(0)


def read_ints_floats(path):
    with open(path) as f:
        return f.read().split()


def main():
    inf, outf = sys.argv[1], sys.argv[2]

    # ---------- parse instance ----------
    tk = read_ints_floats(inf)
    p = 0

    def nxt():
        nonlocal p
        v = tk[p]
        p += 1
        return v

    N = int(nxt())
    A = float(nxt())
    lam = float(nxt())
    K = int(nxt())
    filters = []
    for _ in range(K):
        Li = int(nxt())
        coef = [float(nxt()) for _ in range(Li)]
        filters.append(np.array(coef, dtype=np.float64))
    M = int(nxt())
    ystar = np.array([float(nxt()) for _ in range(M)], dtype=np.float64)

    # combined cascade filter
    h = np.array([1.0], dtype=np.float64)
    for f in filters:
        h = np.convolve(h, f)
    L = len(h)
    assert M == N + L - 1, "internal: M mismatch"

    # ---------- parse participant output ----------
    xt = read_ints_floats(outf)
    if len(xt) != N:
        die("expected %d numbers, got %d" % (N, len(xt)))
    try:
        x = np.array([float(v) for v in xt], dtype=np.float64)
    except ValueError:
        die("non-numeric token in output")
    if not np.all(np.isfinite(x)):
        die("non-finite value in output")
    if np.any(np.abs(x) > A + 1e-6):
        die("amplitude bound |x_i| <= %.10g violated" % A)

    # ---------- objective ----------
    y = np.convolve(x, h)                       # length M = N + L - 1
    resid = y - ystar
    F = float(np.dot(resid, resid) + lam * float(np.dot(x, x)))

    # internal baseline: the do-nothing input x = 0
    B = float(np.dot(ystar, ystar))

    if not math.isfinite(F):
        die("non-finite objective")

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("N=%d L=%d F=%.6g B=%.6g" % (N, L, F, B))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
