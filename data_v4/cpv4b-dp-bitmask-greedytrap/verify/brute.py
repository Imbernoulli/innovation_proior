import sys

def main():
    data = sys.stdin.read().split()
    idx = 0
    m = int(data[idx]); idx += 1
    k = int(data[idx]); idx += 1
    bursts = []
    for j in range(k):
        c = int(data[idx]); idx += 1
        t = int(data[idx]); idx += 1
        mk = 0
        for _ in range(t):
            ch = int(data[idx]); idx += 1
            mk |= (1 << ch)
        bursts.append((c, mk))

    full = (1 << m) - 1

    # Independent brute force: try EVERY subset of bursts, check if the union
    # of their masks covers all channels, track the minimum total cost.
    INF = float('inf')
    best = INF
    for sub in range(1 << k):
        cov = 0
        tot = 0
        s = sub
        while s:
            j = (s & -s).bit_length() - 1
            s &= s - 1
            cov |= bursts[j][1]
            tot += bursts[j][0]
        if cov == full:
            if tot < best:
                best = tot
    # empty subset covers nothing; only valid if full == 0
    if full == 0:
        best = min(best, 0)

    print(-1 if best == INF else best)

main()
