# TIER: invalid
# Emits a non-permutation (everyone assigned to target 0) -> feasibility fail -> 0.
import sys

def main():
    d = sys.stdin.read().split()
    N = int(d[0])
    out = ["0 0" for _ in range(N)]
    sys.stdout.write("\n".join(out) + "\n")

main()
