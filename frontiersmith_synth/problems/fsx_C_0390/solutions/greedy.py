# TIER: greedy
# Single row-major pass: place the smallest reachable SKU that keeps the aisle,
# rack column and conveyor loop distinct.  Fast but dead-ends, leaving slots empty.
import sys


def read():
    data = [ln.split() for ln in sys.stdin.read().splitlines()]
    data = [t for t in data if t]
    N = int(data[0][0])
    P = [[int(x) for x in data[1 + i]] for i in range(N)]
    allowed = []
    idx = 1 + N
    for _ in range(N * N):
        row = data[idx]; idx += 1
        k = int(row[0])
        allowed.append([int(x) for x in row[1:1 + k]])
    return N, P, allowed


def main():
    N, P, allowed = read()
    S = [row[:] for row in P]
    rowset = [set() for _ in range(N)]
    colset = [set() for _ in range(N)]
    diaset = [set() for _ in range(N)]
    for r in range(N):
        for c in range(N):
            if P[r][c]:
                v = P[r][c]
                rowset[r].add(v); colset[c].add(v); diaset[(r + c) % N].add(v)
    for r in range(N):
        for c in range(N):
            if S[r][c]:
                continue
            g = (r + c) % N
            for s in sorted(allowed[r * N + c]):
                if s not in rowset[r] and s not in colset[c] and s not in diaset[g]:
                    S[r][c] = s
                    rowset[r].add(s); colset[c].add(s); diaset[g].add(s)
                    break
    out = "\n".join(" ".join(str(S[r][c]) for c in range(N)) for r in range(N))
    sys.stdout.write(out + "\n")


if __name__ == "__main__":
    main()
