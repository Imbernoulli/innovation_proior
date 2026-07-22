# TIER: invalid
# Deliberately infeasible: claims far more relays than the budget allows.
import sys


def main():
    t = sys.stdin.read().split()
    m = int(t[0]); R = int(t[1])
    out = []
    for _ in range(m):
        # blow the budget on every single pair (R+5 relays each), placed at
        # a fixed dummy coordinate -- checker must reject on budget alone
        k = R + 5
        parts = [str(k)] + ["1.0 1.0"] * k
        out.append(" ".join(parts))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
