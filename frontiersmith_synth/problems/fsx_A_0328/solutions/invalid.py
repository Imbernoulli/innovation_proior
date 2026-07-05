# TIER: invalid
# Well-formed schema but WRONG values: a single all-zero gadget reconstructs the zero
# tensor, which never equals a non-zero target -> checker must score 0.
import sys


def main():
    tok = sys.stdin.read().split()
    it = iter(tok)
    I = int(next(it)); J = int(next(it)); K = int(next(it))
    out = ["1"]
    out.append(" ".join(["0"] * I))
    out.append(" ".join(["0"] * J))
    out.append(" ".join(["0"] * K))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
