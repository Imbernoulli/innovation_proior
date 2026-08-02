# TIER: invalid
import sys

def main():
    d = sys.stdin.read().split()
    it = iter(d)
    n = int(next(it)); budget = int(next(it))
    # blow the budget: "buy" every project regardless of cost -- infeasible on
    # every non-trivial case (total cost vastly exceeds budget)
    idxs = list(range(1, n + 1))
    print(len(idxs))
    print(" ".join(map(str, idxs)))

main()
