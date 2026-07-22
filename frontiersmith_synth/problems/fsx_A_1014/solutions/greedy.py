# TIER: greedy
# The "obvious" first attempt: pretend every pipe is a LINEAR resistor
# (Q ~ deltaH / r, Ohm's law) instead of the stated quadratic law, and pretend
# each outlet is fed by a single SERIES chain of its own trunk pipes plus its
# own outlet pipe -- ignoring the parallel bypass at every stage, ignoring
# cross-ties, and never re-solving the network jointly across outlets. For
# each outlet it throttles only its own immediate trunk pipe, solving the
# 1-D linear equation "Hsrc / (upstream_series_R + x) = target" for x. This
# both mis-sizes the valve (the true law is quadratic, not linear) and
# under-shoots whenever the bypass/cross-ties give the flow somewhere else
# to go around the throttled pipe.
import sys


def main():
    tok = sys.stdin.read().split()
    p = 0
    n_hub = int(tok[p]); p += 1
    K = int(tok[p]); p += 1
    n_edges = int(tok[p]); p += 1
    Hsrc = float(tok[p]); p += 1
    lam = float(tok[p]); p += 1
    outlets = [int(tok[p + i]) for i in range(K)]; p += K
    targets = {}
    for i in range(K):
        targets[outlets[i]] = float(tok[p + i])
    p += K
    edges = []
    cap = []
    for _ in range(n_edges):
        u = int(tok[p]); v = int(tok[p + 1]); r = float(tok[p + 2]); c = int(tok[p + 3])
        p += 4
        edges.append((u, v, r))
        cap.append(c)

    r_T = [edges[i][2] for i in range(n_hub)]          # trunk pipes, index 0..n_hub-1
    r_O = {}
    for idx, (u, v, r) in enumerate(edges):
        if cap[idx] == 0:
            r_O[u] = r

    X_MAX = 1000.0
    x = [0.0] * n_edges
    for j in outlets:
        series_R = sum(r_T[0:j - 1]) + r_O[j]           # everything upstream of the last trunk stage
        t = targets[j]
        if t <= 1e-9:
            continue
        need_total_R = Hsrc / t                          # naive linear Ohm's-law inversion
        new_last = need_total_R - series_R
        old_last = r_T[j - 1]
        xj = new_last - old_last
        xj = max(0.0, min(X_MAX, xj))
        x[j - 1] = xj                                     # trunk pipe T_j sits at edge index j-1

    print(" ".join(f"{v:.6f}" for v in x))


if __name__ == "__main__":
    main()
