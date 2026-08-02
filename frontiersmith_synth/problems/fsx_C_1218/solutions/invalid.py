# TIER: invalid
# Emits a syntactically-plausible but out-of-range translation table (a new-type id far
# outside [0, M_NEW-1] on the very first entry) -> fails the schema/range check -> Ratio 0.
import sys


def main():
    toks = sys.stdin.read().split()
    M_OLD = int(toks[2])
    out = [str(M_OLD), "1 999999"]
    for _ in range(1, M_OLD):
        out.append("0")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
