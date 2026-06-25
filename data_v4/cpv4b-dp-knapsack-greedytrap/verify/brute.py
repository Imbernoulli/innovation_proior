import sys

def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    C = int(data[idx]); idx += 1
    w = []
    v = []
    for _ in range(n):
        w.append(int(data[idx])); idx += 1
        v.append(int(data[idx])); idx += 1

    # Independent brute force: enumerate every subset (n small), keep the best
    # total value among subsets whose total weight does not exceed C.
    best = 0  # empty subset
    for mask in range(1 << n):
        tw = 0
        tv = 0
        m = mask
        i = 0
        while m:
            if m & 1:
                tw += w[i]
                tv += v[i]
            m >>= 1
            i += 1
        if tw <= C and tv > best:
            best = tv
    print(best)

main()
