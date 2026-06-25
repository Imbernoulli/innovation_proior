import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    C = int(data[idx]); idx += 1
    w = []
    b = []
    for _ in range(n):
        w.append(int(data[idx])); idx += 1
        b.append(int(data[idx])); idx += 1

    # Independent brute force: enumerate all 2^n subsets, keep best feasible brightness.
    best = 0
    for mask in range(1 << n):
        tot_w = 0
        tot_b = 0
        for i in range(n):
            if mask & (1 << i):
                tot_w += w[i]
                tot_b += b[i]
        if tot_w <= C and tot_b > best:
            best = tot_b
    print(best)

main()
