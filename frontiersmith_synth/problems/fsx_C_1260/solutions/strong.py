# TIER: strong
"""The insight: a code that corrects one bit per word is useless against a burst that
hits several PHYSICALLY ADJACENT cells inside the same contiguous word. Instead of only
trying contiguous ("logical order") blocks, also try physically INTERLEAVING the layout
-- codeword b owns every cell congruent to b modulo the interleave distance B = N/w --
so that a contiguous burst of Len physically-adjacent cells is spread (pigeonhole) across
Len distinct codewords whenever Len <= B, landing at most ceil(Len/B) cells in any single
codeword instead of all Len. That drops the required correction strength t from
min(LMAX, w) (contiguous) down to ceil(LMAX / B) (interleaved), which can be MUCH cheaper
to decode. strong exhaustively compares both layouts, for every offered word length, and
also the whole-row fallback, and keeps the globally cheapest valid combination."""
import sys
import math


def main():
    toks = sys.stdin.read().split()
    p = 0
    N, M, LMAX = int(toks[p]), int(toks[p + 1]), int(toks[p + 2])
    p += 3
    catalog = []
    for i in range(M):
        w, t, c = int(toks[p]), int(toks[p + 1]), int(toks[p + 2])
        p += 3
        catalog.append((w, t, c))

    all_w = sorted(set(w for (w, t, c) in catalog))

    def cheapest(w, need_t):
        best = None
        for idx, (ww, tt, cc) in enumerate(catalog):
            if ww == w and tt >= need_t:
                if best is None or cc < best[1]:
                    best = (idx, cc)
        return best

    best = None  # (total, layout, w, idx)
    for w in all_w:
        if N % w != 0:
            continue
        B = N // w
        # contiguous ("logical order") layout
        req_c = min(LMAX, w)
        rc = cheapest(w, req_c)
        if rc is not None:
            idx, cost = rc
            total = B * cost
            if best is None or total < best[0]:
                best = (total, "contig", w, idx)
        # physically interleaved layout, interleave distance B
        req_i = math.ceil(LMAX / B) if B > 0 else LMAX
        ri = cheapest(w, req_i)
        if ri is not None:
            idx, cost = ri
            total = B * cost
            if best is None or total < best[0]:
                best = (total, "inter", w, idx)

    _, layout, w, idx = best
    B = N // w
    out = [str(B)]
    if layout == "contig":
        for b in range(B):
            cells = range(b * w, (b + 1) * w)
            out.append(f"{idx} " + " ".join(map(str, cells)))
    else:
        buckets = [[] for _ in range(B)]
        for c in range(N):
            buckets[c % B].append(c)
        for b in range(B):
            out.append(f"{idx} " + " ".join(map(str, buckets[b])))
    print("\n".join(out))


if __name__ == "__main__":
    main()
