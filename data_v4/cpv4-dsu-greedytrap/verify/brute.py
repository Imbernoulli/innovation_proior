import sys

def main():
    data = sys.stdin.read().split()
    idx = 0
    def rd():
        nonlocal idx
        v = int(data[idx]); idx += 1
        return v

    n = rd(); m = rd(); q = rd()

    NEG = -1  # capacities are >= 1, so -1 means "no path known yet"
    # widest[i][j] = max over paths from i to j of the min edge capacity.
    widest = [[NEG] * n for _ in range(n)]
    for i in range(n):
        widest[i][i] = float('inf')

    for _ in range(m):
        u = rd(); v = rd(); c = rd()
        # keep the strongest direct edge between u and v
        if c > widest[u][v]:
            widest[u][v] = c
            widest[v][u] = c

    # Maximin Floyd-Warshall: widest[i][j] = max(widest[i][j],
    #   min(widest[i][k], widest[k][j])) over intermediate k.
    for k in range(n):
        wk = widest[k]
        for i in range(n):
            wik = widest[i][k]
            if wik == NEG:
                continue
            wi = widest[i]
            for j in range(n):
                cand = wik if wik < wk[j] else wk[j]
                if cand > wi[j]:
                    wi[j] = cand

    out = []
    for _ in range(q):
        s = rd(); t = rd()
        out.append(str(widest[s][t]))
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))

main()
