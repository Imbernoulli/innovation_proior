import sys

def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    C = int(data[idx]); idx += 1
    w = []
    v = []
    for i in range(n):
        w.append(int(data[idx])); idx += 1
        v.append(int(data[idx])); idx += 1

    best = None  # best total score over subsets with total weight exactly C
    # Enumerate every subset of the n items (n is tiny in tests).
    for mask in range(1 << n):
        tw = 0
        tv = 0
        for i in range(n):
            if mask & (1 << i):
                tw += w[i]
                tv += v[i]
        if tw == C:
            if best is None or tv > best:
                best = tv

    if best is None:
        print("IMPOSSIBLE")
    else:
        print(best)

if __name__ == "__main__":
    main()
