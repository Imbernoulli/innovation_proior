# TIER: greedy
"""The obvious first-draft engineer's approach: keep the physical layout in logical
(contiguous) order -- codeword b just owns cells [b*w, (b+1)*w) -- and search over word
length w for the CHEAPEST code that is strong enough to survive a burst landing fully
inside one contiguous block. This is a real, honest optimization (it does pick the best
w), it just never considers spreading physically-adjacent cells across DIFFERENT
codewords, so it always pays for correction capability t = min(LMAX, w) instead of the
much weaker t an interleaved layout would need."""
import sys


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

    menu_w = sorted(set(w for (w, t, c) in catalog if w < N))

    def cheapest(w, need_t):
        best = None
        for idx, (ww, tt, cc) in enumerate(catalog):
            if ww == w and tt >= need_t:
                if best is None or cc < best[1]:
                    best = (idx, cc)
        return best

    best_total, best_w, best_idx = None, None, None
    for w in menu_w:
        req_t = min(LMAX, w)
        res = cheapest(w, req_t)
        if res is None:
            continue
        idx, cost = res
        total = (N // w) * cost
        if best_total is None or total < best_total:
            best_total, best_w, best_idx = total, w, idx

    if best_w is None:
        # fall back to the whole-row option (always present, w == N)
        for idx, (w, t, c) in enumerate(catalog):
            if w == N:
                best_w, best_idx, best_total = w, idx, c
                break

    B = N // best_w
    out = [str(B)]
    for b in range(B):
        cells = range(b * best_w, (b + 1) * best_w)
        out.append(f"{best_idx} " + " ".join(map(str, cells)))
    print("\n".join(out))


if __name__ == "__main__":
    main()
