# TIER: trivial
"""Reproduces the checker's own baseline exactly: contiguous ("logical order") blocking
using the single largest sub-N word length offered, with whatever correction strength
plain blocking forces for that block size. No search, no interleaving."""
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
    w_top = menu_w[-1]
    req_t = min(LMAX, w_top)
    best_idx, best_cost = None, None
    for idx, (w, t, c) in enumerate(catalog):
        if w == w_top and t >= req_t:
            if best_cost is None or c < best_cost:
                best_cost, best_idx = c, idx

    B = N // w_top
    out = [str(B)]
    for b in range(B):
        cells = range(b * w_top, (b + 1) * w_top)
        out.append(f"{best_idx} " + " ".join(map(str, cells)))
    print("\n".join(out))


if __name__ == "__main__":
    main()
