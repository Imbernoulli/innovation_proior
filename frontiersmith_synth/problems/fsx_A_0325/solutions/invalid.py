# TIER: invalid
# Emits an infeasible schedule: every intensity exceeds the cap M.  Must score 0.
import sys

def main():
    d = sys.stdin.read().split()
    n = int(d[0]); M = int(d[1])
    f = [M + 7] * n
    sys.stdout.write(" ".join(map(str, f)) + "\n")

main()
