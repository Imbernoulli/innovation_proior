import sys

def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    m = int(data[idx]); idx += 1

    # Independent brute force: plain DSU, but for each range [l, r] we
    # explicitly union EVERY pair (l with l+1, l+1 with l+2, ..., r-1 with r).
    # This is O(sum of range lengths) and obviously correct: it directly
    # connects every star in [l, r]. No skip-pointer cleverness.
    par = list(range(n + 1))  # 1..n

    def find(x):
        root = x
        while par[root] != root:
            root = par[root]
        while par[x] != root:
            par[x], x = root, par[x]
        return root

    def unite(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            par[ra] = rb

    for _ in range(m):
        l = int(data[idx]); idx += 1
        r = int(data[idx]); idx += 1
        for k in range(l, r):      # links k -> k+1 for k = l .. r-1
            unite(k, k + 1)

    roots = set(find(v) for v in range(1, n + 1))
    print(len(roots))

main()
