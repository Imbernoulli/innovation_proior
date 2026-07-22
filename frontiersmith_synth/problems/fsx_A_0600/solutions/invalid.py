# TIER: invalid
# Emits out-of-range garbage indices -> checker must score 0.0.
import sys

def main():
    d = sys.stdin.read().split()
    it = iter(d)
    n = int(next(it)); m = int(next(it)); k = int(next(it))
    # indices well beyond m, plus a duplicate -- infeasible on every gate
    print("%d %d %d" % (m + 5, m + 5, m + 99))

main()
