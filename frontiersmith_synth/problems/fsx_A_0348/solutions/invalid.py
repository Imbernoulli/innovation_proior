# TIER: invalid
# Emits a well-formed-but-WRONG decomposition (two all-ones rank-1 terms) that does not
# reconstruct T.  Must score 0 -- exercises the checker's exact-reconstruction gate.
import sys


def main():
    toks = sys.stdin.read().split()
    I = int(toks[0]); J = int(toks[1]); K = int(toks[2])
    line = " ".join(["1"] * (I + J + K))
    sys.stdout.write("2\n" + line + "\n" + line + "\n")


if __name__ == "__main__":
    main()
