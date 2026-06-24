import sys

def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    B = int(data[idx]); idx += 1
    c = []
    r = []
    for _ in range(n):
        c.append(int(data[idx])); idx += 1
        r.append(int(data[idx])); idx += 1

    # Exhaustive over all 2^n subsets. Obviously correct for tiny n.
    best = 0
    for mask in range(1 << n):
        cost = 0
        rew = 0
        m = mask
        i = 0
        while m:
            if m & 1:
                cost += c[i]
                rew += r[i]
            m >>= 1
            i += 1
        if cost <= B and rew > best:
            best = rew
    print(best)

main()
