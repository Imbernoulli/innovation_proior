# TIER: invalid
# Well-formed schema but WRONG values: a single all-zero primitive reconstructs the zero
# tensor, which never matches a non-zero target -> the checker's exact gate must score 0.
import sys


def main():
    tok = sys.stdin.read().split()
    it = iter(tok)
    B = int(next(it)); L = int(next(it)); H = int(next(it))
    out = ["1"]
    out.append(" ".join(["0"] * B))
    out.append(" ".join(["0"] * L))
    out.append(" ".join(["0"] * H))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
