# TIER: invalid
# Emits stations OUTSIDE the triangular reserve (x + y > 1), so the containment
# check fails and the layout scores 0.
import sys


def main():
    N = int(sys.stdin.read().split()[0])
    out = []
    for k in range(N):
        out.append("5.0 5.0")  # far outside T
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
