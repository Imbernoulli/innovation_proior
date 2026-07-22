# TIER: invalid
# emits inlet concentrations far above the cap -> feasibility gate must reject.
import sys


def main():
    tok = sys.stdin.read().split()
    S = int(tok[0]); M = int(tok[3])
    lines = [" ".join(["9.0"] * M) for _ in range(S)]
    sys.stdout.write("\n".join(lines) + "\n")


main()
