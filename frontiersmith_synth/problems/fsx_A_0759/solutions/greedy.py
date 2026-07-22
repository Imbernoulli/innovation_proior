# TIER: greedy
# The obvious first idea for "which junctions are connected": build a
# CORRELATION / similarity graph.  Pool every observed (source, time) snapshot
# across all 5 training impulses, z-score each node's series, and connect the
# pairs whose observed temperatures co-move most -- keep the top ~25 pairs by
# |correlation|, weight the edge by the (clipped, rescaled) correlation.
# This treats the data as "a family of related curves" rather than as one
# shared operator: it happily wires up nodes that are merely a few hops apart
# on the SAME diffusion front (their temperatures rise and fall together even
# with no direct pipe between them), and it has no way to tell a real edge
# from a strong indirect echo.  It fits the training impulses reasonably but
# regularly mis-wires the network, so it transfers only partially to impulses
# injected at unseen junctions.
import sys

N_EDGES_TARGET = 25


def read_input():
    data = sys.stdin.read().split()
    pos = 0
    t = int(data[pos]); pos += 1
    n = int(data[pos]); pos += 1
    S = int(data[pos]); pos += 1
    T = int(data[pos]); pos += 1
    src = [int(data[pos + i]) for i in range(S)]; pos += S
    blocks = []
    for _ in range(S):
        rows = []
        for _ in range(T + 1):
            row = [float(data[pos + k]) for k in range(n)]
            pos += n
            rows.append(row)
        blocks.append(rows)
    return n, S, T, src, blocks


def main():
    n, S, T, src, blocks = read_input()

    # pool ALL snapshots (every time row of every source impulse) as samples
    samples = []
    for rows in blocks:
        for row in rows:
            samples.append(row)
    m = len(samples)

    mean = [0.0] * n
    for row in samples:
        for j in range(n):
            mean[j] += row[j]
    for j in range(n):
        mean[j] /= m

    cent = [[row[j] - mean[j] for j in range(n)] for row in samples]
    var = [0.0] * n
    for row in cent:
        for j in range(n):
            var[j] += row[j] * row[j]
    std = [max((var[j] / m) ** 0.5, 1e-9) for j in range(n)]

    # correlation matrix (upper triangle only)
    cand = []
    for i in range(n):
        for j in range(i + 1, n):
            cov = 0.0
            for row in cent:
                cov += row[i] * row[j]
            cov /= m
            corr = cov / (std[i] * std[j])
            cand.append((abs(corr), i, j, corr))
    cand.sort(key=lambda x: -x[0])

    edges = {}
    for absc, i, j, corr in cand[:N_EDGES_TARGET]:
        w = max(corr, 0.01)
        edges[(i, j)] = w

    # feasibility repair: enforce per-node incident weight sum <= 1
    row_sum = [0.0] * n
    for (i, j), w in edges.items():
        row_sum[i] += w
        row_sum[j] += w
    mx = max(row_sum) if row_sum else 0.0
    if mx > 0.999:
        scale = 0.999 / mx
        edges = {k: v * scale for k, v in edges.items()}

    out = [str(len(edges))]
    for (i, j), w in edges.items():
        out.append("%d %d %.8f" % (i, j, w))
    print("\n".join(out))


if __name__ == "__main__":
    main()
