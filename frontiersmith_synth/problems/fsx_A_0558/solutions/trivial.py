# TIER: trivial
import sys, numpy as np

def main():
    d = sys.stdin.read().split()
    it = iter(d)
    N = int(next(it)); W = int(next(it))
    F_min = int(next(it)); F_max = int(next(it))
    k = int(next(it)); na = int(next(it))
    allowed = [float(next(it)) for _ in range(na)]
    E = np.array([float(next(it)) for _ in range(N)], dtype=float)

    # do-nothing baseline: one high tone -> flat envelope at ~amp/sqrt(2).
    target_amp = float(np.sqrt(2.0) * E.mean())
    a = min(allowed, key=lambda x: abs(x - target_amp))
    print("%d %.6f %.6f" % (F_min, a, 0.0))

main()
