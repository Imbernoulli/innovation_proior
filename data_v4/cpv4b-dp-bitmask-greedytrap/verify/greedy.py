import sys

# The TEMPTING WRONG greedy: repeatedly fire the burst with the best
# cost-per-newly-covered-channel ratio until everything is covered.
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
    cov = 0
    tot = 0
    while cov != full:
        best_j = -1
        best_ratio = None
        for j in range(k):
            new = bin(bursts[j][1] & ~cov).count("1")
            if new == 0:
                continue
            ratio = bursts[j][0] / new
            if best_ratio is None or ratio < best_ratio:
                best_ratio = ratio
                best_j = j
        if best_j == -1:
            print(-1)
            return
        tot += bursts[best_j][0]
        cov |= bursts[best_j][1]
    print(tot)

main()
