import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        print(0)
        return
    it = iter(data)
    n = int(next(it))
    a = [int(next(it)) for _ in range(n)]
    if n == 0:
        print(0)
        return

    # Obviously-correct brute force: enumerate EVERY non-empty subset, keep the
    # ones that form a strictly increasing subsequence (values strictly up in
    # index order), and take the maximum sum. Exponential, for small n only.
    best = None
    for mask in range(1, 1 << n):
        idx = [i for i in range(n) if (mask >> i) & 1]
        ok = True
        for k in range(1, len(idx)):
            if not (a[idx[k - 1]] < a[idx[k]]):
                ok = False
                break
        if ok:
            s = sum(a[i] for i in idx)
            if best is None or s > best:
                best = s
    print(best)

if __name__ == "__main__":
    main()
