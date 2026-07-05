# TIER: invalid
# Emits n copies of the same position -> duplicate stages -> infeasible -> score 0.
import sys

def main():
    d = sys.stdin.read().split()
    n = int(d[0]); M = int(d[1])
    print(n)
    print("\n".join(["0"] * n))

main()
