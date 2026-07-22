# TIER: trivial
# inject nothing: every inlet at zero concentration -> reproduces checker baseline.
import sys


def main():
    tok = sys.stdin.read().split()
    S = int(tok[0]); M = int(tok[3])
    lines = [" ".join(["0.0"] * M) for _ in range(S)]
    sys.stdout.write("\n".join(lines) + "\n")


main()
