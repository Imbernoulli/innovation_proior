# TIER: strong
"""Consensus-core / conflict-residual decomposition, placed by incremental
edge-weighted construction (not an independent per-symbol rank match).

Pooled raw counts (what `greedy` uses) let whichever language has the biggest
absolute corpus drown out the others. The graded objective normalizes each
language by its OWN random-layout baseline before taking the worst case, so
"important" must be judged per language, not by raw count.

1. Normalize each language's digraph table to a probability mass NF_k (divide
   by that language's own total off-diagonal count). This equalizes the three
   languages' "voice" regardless of corpus size.
2. Consensus edge weight: Con[i,j] = min_k NF_k[i,j] -- the mass every
   language agrees is at least this important for THIS digraph.
3. edge_score[i,j] = Con[i,j] + max_k NF_k[i,j]. This gives the shared,
   every-language-agrees part a DOUBLE vote (it appears once standalone and
   again inside the max), because satisfying it moves ALL K languages'
   normalized cost at once -- "optimize the consensus core hard". A digraph
   that only one language cares about (its private conflict/residual
   digraph) is counted just once, via the max, by whichever single language
   claims it most strongly -- never by summing claims across languages of
   wildly different corpus size (which is what `greedy` does, and what lets
   the biggest corpus silently drown the others out).
4. Process edges in decreasing edge_score. When both endpoints are still
   unplaced, seat them on the two cheapest still-open slots (central slots
   cluster together, so this keeps the pair close). When one endpoint is
   already seated, seat the other in whichever open slot is closest to its
   already-placed partner. This is an incremental chain-building
   construction: it correctly handles symbols that sit on TWO high-value
   digraphs (a shared one and a private one) by keeping the whole chain
   compact, which an independent per-symbol importance ranking (as `greedy`
   uses) cannot do.
"""
import sys
import math


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); k = int(next(it))
    coords = []
    for _ in range(n):
        x = float(next(it)); y = float(next(it))
        coords.append((x, y))
    freq = []
    for _lang in range(k):
        mat = []
        for _i in range(n):
            row = [int(next(it)) for _ in range(n)]
            mat.append(row)
        freq.append(mat)

    T = [[0.0] * n for _ in range(n)]
    for u in range(n):
        xu, yu = coords[u]
        for v in range(n):
            if u != v:
                xv, yv = coords[v]
                T[u][v] = math.hypot(xu - xv, yu - yv)

    # combined (undirected) per-language weight for each unordered pair
    comb = [[[0.0] * n for _ in range(n)] for _ in range(k)]
    S = [0.0] * k
    for lang in range(k):
        fk = freq[lang]
        for i in range(n):
            for j in range(i + 1, n):
                v = fk[i][j] + fk[j][i]
                comb[lang][i][j] = v
                comb[lang][j][i] = v
                S[lang] += v

    # normalized frequency (probability mass) per language
    NF = [[[0.0] * n for _ in range(n)] for _ in range(k)]
    for lang in range(k):
        s = S[lang] if S[lang] > 1e-9 else 1.0
        for i in range(n):
            for j in range(n):
                if i != j:
                    NF[lang][i][j] = comb[lang][i][j] / s

    # edge score: consensus counted TWICE (helps all K languages at once, so
    # it is worth optimizing hard) plus the single strongest per-language
    # claim on this edge (protects whichever language cares most, instead of
    # summing raw claims across languages of very different corpus size).
    edge_score = {}
    for i in range(n):
        for j in range(i + 1, n):
            con = min(NF[lang][i][j] for lang in range(k))
            best_claim = max(NF[lang][i][j] for lang in range(k))
            edge_score[(i, j)] = con + best_claim

    edges = sorted(edge_score.keys(), key=lambda e: (-edge_score[e], e))

    # slots sorted by centrality (ascending = cheapest first); central slots
    # cluster together on a physically laid-out keyboard grid.
    slot_centrality = [sum(T[u][v] for v in range(n) if v != u) for u in range(n)]
    open_slots = sorted(range(n), key=lambda u: (slot_centrality[u], u))
    open_ptr = 0  # next never-yet-considered cheap slot

    perm = [None] * n
    placed = [False] * n
    open_set = set(range(n))

    def take_cheapest_open():
        nonlocal open_ptr
        while open_ptr < len(open_slots) and open_slots[open_ptr] not in open_set:
            open_ptr += 1
        s = open_slots[open_ptr]
        open_set.discard(s)
        return s

    def take_closest_open(to_slot):
        best_s, best_d = None, None
        for s in open_set:
            d = T[to_slot][s]
            if best_d is None or d < best_d or (d == best_d and s < best_s):
                best_s, best_d = s, d
        open_set.discard(best_s)
        return best_s

    for (i, j) in edges:
        if placed[i] and placed[j]:
            continue
        if not placed[i] and not placed[j]:
            s1 = take_cheapest_open()
            s2 = take_closest_open(s1) if open_set else take_cheapest_open()
            perm[i] = s1; placed[i] = True
            perm[j] = s2; placed[j] = True
        elif placed[i]:
            s = take_closest_open(perm[i])
            perm[j] = s; placed[j] = True
        else:
            s = take_closest_open(perm[j])
            perm[i] = s; placed[i] = True

    for i in range(n):
        if not placed[i]:
            s = take_cheapest_open()
            perm[i] = s; placed[i] = True

    print(" ".join(str(perm[i]) for i in range(n)))


if __name__ == "__main__":
    main()
