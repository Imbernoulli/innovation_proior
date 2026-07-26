# TIER: invalid
"""Well-formed schema, but the identity map: fixes every point (S trivially
satisfied) yet the cycle containing 0 has length 1, far below any required L.
Must score 0.0 on every case via the cycle-length feasibility gate."""
import sys


def main():
    sys.stdin.read()  # ignore the instance
    out = ["1", "ADDC 0 0 0"]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
