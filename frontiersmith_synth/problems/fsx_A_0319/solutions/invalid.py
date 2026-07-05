# TIER: invalid
# Emits M probes all sitting OUTSIDE the wafer square ([0,1]^2): coordinate 2.0.
# The checker rejects out-of-range coordinates -> Ratio 0.0.
import sys


def main():
    t = sys.stdin.read().split()
    d = int(t[0]); m = int(t[1])
    out = []
    for _ in range(m):
        out.append("2.0 2.0")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
