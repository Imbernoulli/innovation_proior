# TIER: invalid
# Emits an infeasible datum tree (every feature claims the root as its datum,
# which violates the allowed-datum lists) and an over-budget precise list.
import sys

def main():
    tok = sys.stdin.read().split()
    it = iter(tok)
    n = int(next(it)); k = int(next(it))
    par = [-1] + [0] * (n - 1)   # illegal: 0 is not an allowed datum for leaves
    print(" ".join(map(str, par)))
    print(str(n) + " " + " ".join(map(str, range(n))))  # over budget + includes root

main()
