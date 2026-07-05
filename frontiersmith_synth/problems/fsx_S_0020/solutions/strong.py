# TIER: strong
# Multi-restart Kernighan-Lin style balanced local search. Runs balance-preserving best-gain
# swaps to convergence from several seeded starting bisections (the block split plus random
# balanced perturbations) and keeps the best cut found. Escapes the shallow local optima that
# trap the single-pass greedy on noisy instances.
import sys, random

def read_graph():
    data = sys.stdin.buffer.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    m = int(data[idx]); idx += 1
    edges = []
    wmap = {}
    for _ in range(m):
        u = int(data[idx]); v = int(data[idx + 1]); w = int(data[idx + 2]); idx += 3
        edges.append((u, v, w))
        a, b = (u, v) if u < v else (v, u)
        wmap[(a, b)] = wmap.get((a, b), 0) + w
    return n, edges, wmap

def cut_value(edges, side):
    F = 0
    for u, v, w in edges:
        if side[u] != side[v]:
            F += w
    return F

def local_search(n, edges, wmap, side, max_iters):
    for _ in range(max_iters):
        gain = [0] * (n + 1)
        for u, v, w in edges:
            if side[u] == side[v]:
                gain[u] += w; gain[v] += w
            else:
                gain[u] -= w; gain[v] -= w
        best_a = best_b = -1
        ga = gb = None
        for v in range(1, n + 1):
            if side[v] == 0:
                if ga is None or gain[v] > ga:
                    ga = gain[v]; best_a = v
            else:
                if gb is None or gain[v] > gb:
                    gb = gain[v]; best_b = v
        if best_a < 0 or best_b < 0:
            break
        a, b = (best_a, best_b) if best_a < best_b else (best_b, best_a)
        wab = wmap.get((a, b), 0)
        delta = gain[best_a] + gain[best_b] - 2 * wab
        if delta <= 0:
            break
        side[best_a] = 1
        side[best_b] = 0
    return side

def balanced_random(n, rng):
    labels = [0] * (n // 2) + [1] * (n - n // 2)
    rng.shuffle(labels)
    side = [0] * (n + 1)
    for v in range(1, n + 1):
        side[v] = labels[v - 1]
    return side

def main():
    n, edges, wmap = read_graph()
    rng = random.Random(20205 + n)
    max_iters = 2 * n + 10

    starts = []
    block = [0] * (n + 1)
    for v in range(1, n + 1):
        block[v] = 0 if v <= n // 2 else 1
    starts.append(block)
    for _ in range(4):
        starts.append(balanced_random(n, rng))

    best_side = None
    best_cut = -1
    for st in starts:
        s = local_search(n, edges, wmap, st[:], max_iters)
        c = cut_value(edges, s)
        if c > best_cut:
            best_cut = c
            best_side = s

    sys.stdout.write(" ".join(str(best_side[v]) for v in range(1, n + 1)) + "\n")

if __name__ == "__main__":
    main()
