# TIER: trivial
import sys


def group_elements(N, t):
    def idf(r, c):
        return (r, c)

    def flipH(r, c):
        return (r, N - 1 - c)

    def flipV(r, c):
        return (N - 1 - r, c)

    def rot180(r, c):
        return (N - 1 - r, N - 1 - c)

    def flipD(r, c):
        return (c, r)

    def flipAD(r, c):
        return (N - 1 - c, N - 1 - r)

    def rot90(r, c):
        return (c, N - 1 - r)

    def rot270(r, c):
        return (N - 1 - c, r)

    if t == 2:
        return [idf, flipH]
    if t == 4:
        return [idf, flipH, flipV, rot180]
    return [idf, flipH, flipV, rot180, flipD, flipAD, rot90, rot270]


def sheet_connected(N, removed):
    total = N * N
    if len(removed) >= total:
        return False
    start = None
    for r in range(N):
        for c in range(N):
            if (r, c) not in removed:
                start = (r, c)
                break
        if start is not None:
            break
    if start is None:
        return False
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= nr < N and 0 <= nc < N and (nr, nc) not in removed and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == total - len(removed)


def main():
    data = sys.stdin.read().split()
    N, t, V = int(data[0]), int(data[1]), int(data[2])
    r = max(1, N // 4)
    c0 = max(1, (3 * N) // 5)
    avail = max(0, (N - 2) - c0)
    K0 = max(1, min(V // 10, 1 + avail // 2))
    G = group_elements(N, t)
    marks = [(r, c0)]
    for k in range(K0, 0, -1):
        cand = [(r, c0 + 2 * i) for i in range(k)]
        removed = set()
        for (rr, cc) in cand:
            removed |= set(g(rr, cc) for g in G)
        if sheet_connected(N, removed):
            marks = cand
            break
    print(len(marks))
    for rr, cc in marks:
        print(rr, cc)


if __name__ == "__main__":
    main()
