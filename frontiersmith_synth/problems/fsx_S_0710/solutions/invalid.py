# TIER: invalid
import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); c = int(next(it))
    # blatantly infeasible: max out every bank's injection at once (sum >> C)
    print(" ".join(str(c) for _ in range(n)))

main()
