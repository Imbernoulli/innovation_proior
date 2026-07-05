# TIER: greedy
# Row-major, smallest-available fill. Fast, but dead-ends: an early low-value
# choice can leave later slots with no legal model, so cells stay empty.
import sys


def main():
    tok = sys.stdin.read().split()
    n = int(tok[0])
    vals = tok[1:1 + n * n]
    G = [[-1] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(n):
            t = vals[idx]; idx += 1
            G[i][j] = -1 if t == '.' else int(t)

    rows = [set() for _ in range(n)]
    cols = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if G[i][j] != -1:
                rows[i].add(G[i][j])
                cols[j].add(G[i][j])

    S = [row[:] for row in G]
    for i in range(n):
        for j in range(n):
            if S[i][j] == -1:
                for v in range(n):
                    if v not in rows[i] and v not in cols[j]:
                        S[i][j] = v
                        rows[i].add(v)
                        cols[j].add(v)
                        break

    lines = []
    for i in range(n):
        lines.append(" ".join("." if S[i][j] == -1 else str(S[i][j])
                              for j in range(n)))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
