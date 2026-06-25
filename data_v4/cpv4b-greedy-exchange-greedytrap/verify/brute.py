import sys
from itertools import permutations

def cost_of_order(order, t, w):
    cur = 0
    total = 0
    for idx in order:
        cur += t[idx]
        total += w[idx] * cur
    return total

def main():
    data = sys.stdin.read().split()
    pos = 0
    n = int(data[pos]); pos += 1
    t = []
    w = []
    for i in range(n):
        ti = int(data[pos]); pos += 1
        wi = int(data[pos]); pos += 1
        t.append(ti)
        w.append(wi)
    # Brute force: try every ordering of the n jobs, take the minimum
    # weighted sum of completion times. Only correct for tiny n.
    best = None
    for order in permutations(range(n)):
        c = cost_of_order(order, t, w)
        if best is None or c < best:
            best = c
    if best is None:
        best = 0  # n == 0
    print(best)

main()
