import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    m = int(data[idx]); idx += 1
    k = int(data[idx]); idx += 1
    cost = []
    mask = []
    for _ in range(k):
        c = int(data[idx]); idx += 1
        s = int(data[idx]); idx += 1
        cost.append(c)
        mask.append(s)

    FULL = (1 << m) - 1
    best = None
    # Exhaustive: try every subset of contractors, check it covers FULL, track min cost.
    for sub in range(1 << k):
        cov = 0
        tot = 0
        for i in range(k):
            if sub & (1 << i):
                cov |= mask[i]
                tot += cost[i]
        if cov == FULL:
            if best is None or tot < best:
                best = tot
    print(-1 if best is None else best)

if __name__ == "__main__":
    main()
