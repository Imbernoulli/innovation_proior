import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    W = int(data[idx]); idx += 1
    e = []
    v = []
    for _ in range(n):
        ei = int(data[idx]); idx += 1
        vi = int(data[idx]); idx += 1
        e.append(ei)
        v.append(vi)

    # Independent brute force: enumerate every subset (each experiment at most once),
    # keep those whose total energy <= W, maximize total value. Empty subset gives 0.
    best = 0
    for mask in range(1 << n):
        te = 0
        tv = 0
        for i in range(n):
            if mask & (1 << i):
                te += e[i]
                tv += v[i]
        if te <= W and tv > best:
            best = tv
    print(best)

main()
