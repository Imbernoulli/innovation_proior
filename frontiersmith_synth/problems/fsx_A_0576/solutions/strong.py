# TIER: strong
# INSIGHT: recover the hidden district partition (imprimitivity block recovery via the pair-orbit
# closure), identify the inter-district move as the generator with a non-trivial district-action,
# BFS the district graph to get a SHORT word per distinct district, and spend one word per district
# (a transversal) so coverage grows by ~t per word across districts. Leftover budget fills already
# reached districts with extra offsets (orbit-cover selection).
import sys
from collections import deque

def main():
    d = sys.stdin.read().split()
    idx = 0
    n = int(d[idx]); idx += 1
    m = int(d[idx]); idx += 1
    k = int(d[idx]); idx += 1
    L = int(d[idx]); idx += 1
    t = int(d[idx]); idx += 1
    Gens = []
    for _ in range(m):
        Gens.append([int(d[idx + i]) for i in range(n)]); idx += n
    S = [int(d[idx + i]) for i in range(t)]; idx += t

    Ginv = []
    for G in Gens:
        H = [0] * n
        for x in range(n):
            H[G[x]] = x
        Ginv.append(H)

    a = S[0]

    # ---- imprimitivity block recovery: minimal block system containing a pair {a, x} ----
    def closure(a0, b0):
        parent = list(range(n))
        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u
        def union(u, v):
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[ru] = rv
                return True
            return False
        q = deque()
        if union(a0, b0):
            q.append((a0, b0))
        while q:
            x, y = q.popleft()
            for G in Gens:
                nx, ny = G[x], G[y]
                if find(nx) != find(ny):
                    union(nx, ny)
                    q.append((nx, ny))
        return [find(i) for i in range(n)]

    def partition_of(labels):
        from collections import defaultdict
        groups = defaultdict(list)
        for i, l in enumerate(labels):
            groups[l].append(i)
        return list(groups.values())

    best_part = None
    for j in range(m):
        x = Gens[j][a]
        if x == a:
            continue
        labels = closure(a, x)
        parts = partition_of(labels)
        c = len(parts)
        sizes = set(len(p) for p in parts)
        if 1 < c < n and len(sizes) == 1 and next(iter(sizes)) >= 2:
            # prefer the FINEST valid block system (most classes)
            if best_part is None or c > len(best_part):
                best_part = parts

    words = []

    if best_part is not None:
        blk = [0] * n
        for bi, pts in enumerate(best_part):
            for p in pts:
                blk[p] = bi
        nb = len(best_part)
        start = blk[a]
        # district-action of each token (generator / inverse) as a permutation of districts
        def token_action(tok):
            G = Gens[tok - 1] if tok > 0 else Ginv[-tok - 1]
            act = [0] * nb
            for bi, pts in enumerate(best_part):
                act[bi] = blk[G[pts[0]]]
            return act
        toks = [x for x in range(-m, m + 1) if x != 0]
        acts = {tok: token_action(tok) for tok in toks}

        # BFS shortest word (length <= L) to each district
        wordto = {start: []}
        q = deque([start])
        while q:
            u = q.popleft()
            wu = wordto[u]
            if len(wu) >= L:
                continue
            for tok in toks:
                v = acts[tok][u]
                if v not in wordto:
                    wordto[v] = wu + [tok]
                    q.append(v)
        # transversal: one word per distinct district, shortest first
        reach = sorted(wordto.items(), key=lambda kv: len(kv[1]))
        for dist, w in reach:
            if len(words) >= k:
                break
            words.append(w)
        # fill leftover budget: extend reached districts with intra-district shuffles (move 1)
        if len(words) < k:
            base = list(words)
            shift = 1
            while len(words) < k:
                added = False
                for w in base:
                    if len(words) >= k:
                        break
                    if len(w) + shift <= L:
                        words.append(w + [1] * shift)
                        added = True
                shift += 1
                if not added:
                    break

    # fallback (recovery failed): just spam some moves
    if not words:
        for i in range(k):
            words.append([1] * min(i, L))

    out = []
    for w in words[:k]:
        out.append(" ".join(map(str, w)) if w else "0")
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
