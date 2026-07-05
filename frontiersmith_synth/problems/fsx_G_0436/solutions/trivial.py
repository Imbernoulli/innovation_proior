# TIER: trivial
"""Emit the constant train-mean of y -- matches the grader's baseline predictor,
so it scores ~0.1. No congestion law recovered at all."""
import sys


def main():
    vals = []
    for tk in sys.stdin.read().split():
        try:
            vals.append(float(tk))
        except ValueError:
            pass
    ys = [vals[i + 3] for i in range(0, len(vals) - (len(vals) % 4), 4)]
    m = sum(ys) / len(ys) if ys else 0.0
    print(repr(m))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
