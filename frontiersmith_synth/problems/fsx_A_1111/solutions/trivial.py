# TIER: trivial
# Do-nothing baseline: assume the lagoon is a single uniform medium with the
# SAME index as the entry water (n_1 = n0) -- i.e. the ray travels perfectly
# straight, no refraction at all. This reproduces the checker's own internal
# baseline construction exactly, so it scores ~0.1 by definition.
import sys


def main():
    data = sys.stdin.read().split()
    n0 = float(data[0])
    D = float(data[1])
    print(1)
    print("%.6f %.6f" % (D, n0))


if __name__ == "__main__":
    main()
