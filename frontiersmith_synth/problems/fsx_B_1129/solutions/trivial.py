# TIER: trivial
# Identity slot assignment: SKU i -> slot i (aisle = i//L+1, depth = i%L+1 in SKU-id
# order). This exactly reproduces the checker's internal baseline construction, so
# it scores ~0.1 by convention.
import sys

def main():
    d = sys.stdin.read().split()
    N = int(d[0])
    print(N)
    print("\n".join(str(i) for i in range(N)))

main()
