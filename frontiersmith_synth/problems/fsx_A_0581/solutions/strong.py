# TIER: strong
# Insight: F = sum_edges len(e) * ( Wm*flow(e)^(2/3) + Wd*flow(e) ), flow(e) = demand
# of the subtree below e.  flow^(2/3) is CONCAVE, so pooling many demands onto one
# trunk over the long near-source haul is much cheaper than paying that haul per
# organ or per cluster.  The right shape is a HIERARCHY that SPLITS LATE: one thick
# trunk leaves the source, pools the corridor flow, and bifurcates into per-cluster
# trunks, then terminals.  We enumerate hierarchical topologies (flat hubs,
# two-level trunk, and a radial spine over several directional partitions), then --
# the key step -- for each FIXED topology we place all junctions at the weighted
# GEOMETRIC MEDIAN of their tree-neighbours (edge weight = Wm*flow^(2/3)+Wd*flow) by
# joint Weiszfeld relaxation, which is exactly the optimum split-point placement for
# that topology.  Obstacles are repaired by dissolving any tube that would cross.
# We also keep the star and the nearest-reach tree, and output the cheapest feasible
# network.
import sys, math

A = 2.0 / 3.0
GEOM_EPS = 1e-7


def seg_hits_rect(ax, ay, bx, by, rect):
    x0, y0, x1, y1 = rect
    x0 += GEOM_EPS; y0 += GEOM_EPS; x1 -= GEOM_EPS; y1 -= GEOM_EPS
    if x1 <= x0 or y1 <= y0:
        return False
    dx = bx - ax; dy = by - ay
    p = [-dx, dx, -dy, dy]
    q = [ax - x0, x1 - ax, ay - y0, y1 - ay]
    t0, t1 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < 1e-15:
            if qi < 0:
                return False
        else:
            r = qi / pi
            if pi < 0:
                if r > t1:
                    return False
                if r > t0:
                    t0 = r
            else:
                if r < t0:
                    return False
                if r < t1:
                    t1 = r
    return t1 - t0 > 1e-12


def main():
    t = sys.stdin.read().split()
    it = iter(t)
    K = int(next(it)); M = int(next(it))
    Wm = float(next(it)); Wd = float(next(it))
    sx = float(next(it)); sy = float(next(it))
    org = [(sx, sy, 0.0)]
    for _ in range(K):
        x = float(next(it)); y = float(next(it)); d = float(next(it))
        org.append((x, y, d))
    rects = []
    for _ in range(M):
        x0 = float(next(it)); y0 = float(next(it))
        x1 = float(next(it)); y1 = float(next(it))
        rects.append((x0, y0, x1, y1))

    def w_of(f):
        return Wm * (f ** A) + Wd * f

    def clear(ax, ay, bx, by):
        for r in rects:
            if seg_hits_rect(ax, ay, bx, by, r):
                return False
        return True

    def flows_of(par, pos, n):
        flow = [0.0] * n
        for i in range(1, K + 1):
            flow[i] = org[i][2]
        depth = [0] * n
        for i in range(1, n):
            j = i; d = 0
            while j != 0:
                j = par[j]; d += 1
                if d > n:
                    break
            depth[i] = d
        for i in sorted(range(1, n), key=lambda k: depth[k], reverse=True):
            flow[par[i]] += flow[i]
        return flow

    def relax(par, pos, n, iters=60):
        # pos: list of [x,y] for all nodes; 0..K fixed, junctions (>K) movable.
        flow = flows_of(par, pos, n)
        children = [[] for _ in range(n)]
        for i in range(1, n):
            children[par[i]].append(i)
        for _ in range(iters):
            for j in range(K + 1, n):
                nb = []  # (x, y, weight)
                p = par[j]
                nb.append((pos[p][0], pos[p][1], w_of(flow[j]) + 1e-12))
                for c in children[j]:
                    nb.append((pos[c][0], pos[c][1], w_of(flow[c]) + 1e-12))
                if not nb:
                    continue
                nx = ny = den = 0.0
                hit = False
                for (px, py, w) in nb:
                    dd = math.hypot(pos[j][0] - px, pos[j][1] - py)
                    if dd < 1e-9:
                        hit = True
                        break
                    wd = w / dd
                    nx += px * wd; ny += py * wd; den += wd
                if hit or den == 0.0:
                    continue
                pos[j][0] = nx / den; pos[j][1] = ny / den
        return flow

    def finalize_and_cost(par, pos, n):
        """Dissolve any tube crossing an obstacle (organs fall back to the always-
        clear source link), compact surviving junctions, and return (F, steiner,
        parents) with parents over nodes 1..K+P'. F=inf if something is still bad."""
        par = par[:]
        alive = set(range(K + 1, n))
        changed = True; guard = 0
        while changed and guard < n + 5:
            changed = False; guard += 1
            for i in range(1, n):
                if i > K and i not in alive:
                    continue
                p = par[i]
                if p > K and p not in alive:
                    par[i] = 0; changed = True; continue
                ax, ay = pos[i]; bx, by = pos[p]
                if not clear(ax, ay, bx, by):
                    if i > K:
                        alive.discard(i); changed = True
                    elif p != 0:
                        par[i] = 0; changed = True
        # drop childless junctions
        while True:
            childcount = {}
            for i in range(1, n):
                if i > K and i not in alive:
                    continue
                childcount[par[i]] = childcount.get(par[i], 0) + 1
            dead = [j for j in alive if childcount.get(j, 0) == 0]
            if not dead:
                break
            for j in dead:
                alive.discard(j)
            for i in range(1, n):
                if i <= K or i in alive:
                    if par[i] in dead:
                        par[i] = 0
        # relabel
        js = sorted(alive)
        newid = {0: 0}
        for i in range(1, K + 1):
            newid[i] = i
        steiner = []
        for j in js:
            newid[j] = K + 1 + len(steiner)
            steiner.append((pos[j][0], pos[j][1]))
        nn = K + 1 + len(steiner)
        parents = [0] * (nn)   # index by new node id
        for i in range(1, K + 1):
            parents[i] = newid[par[i]]
        for j in js:
            parents[newid[j]] = newid[par[j]]
        # cost with final positions
        allc = [(sx, sy)] + [(org[i][0], org[i][1]) for i in range(1, K + 1)] + steiner
        pl = [0] * nn
        for i in range(1, nn):
            pl[i] = parents[i]
        # feasibility: no crossing (should hold), acyclic
        for i in range(1, nn):
            ax, ay = allc[i]; bx, by = allc[pl[i]]
            if not clear(ax, ay, bx, by):
                return float("inf"), None, None
        flow = [0.0] * nn
        for i in range(1, K + 1):
            flow[i] = org[i][2]
        depth = [0] * nn
        for i in range(1, nn):
            jj = i; d = 0
            while jj != 0:
                jj = pl[jj]; d += 1
                if d > nn:
                    return float("inf"), None, None
            depth[i] = d
        for i in sorted(range(1, nn), key=lambda k: depth[k], reverse=True):
            flow[pl[i]] += flow[i]
        F = 0.0
        for i in range(1, nn):
            f = flow[i]
            if f <= 0.0:
                continue
            ax, ay = allc[i]; bx, by = allc[pl[i]]
            F += math.hypot(ax - bx, ay - by) * w_of(f)
        return F, steiner, [pl[i] for i in range(1, nn)]

    # ---------- candidate topologies ----------
    results = []  # (F, steiner, parents)

    # star
    results.append(finalize_and_cost([0] * (K + 1), [[sx, sy]] + [[org[i][0], org[i][1]] for i in range(1, K + 1)], K + 1))

    # nearest-reach
    par = [0] * (K + 1)
    tree = [0]; rem = set(range(1, K + 1))
    while rem:
        best = None
        for o in rem:
            ox, oy, _ = org[o]
            for nd in tree:
                nx, ny = (sx, sy) if nd == 0 else (org[nd][0], org[nd][1])
                if not clear(ox, oy, nx, ny):
                    continue
                dd = (ox - nx) ** 2 + (oy - ny) ** 2
                if best is None or (dd, o, nd) < best:
                    best = (dd, o, nd)
        if best is None:
            o = min(rem); par[o] = 0
        else:
            _, o, nd = best; par[o] = nd
        tree.append(o); rem.discard(o)
    results.append(finalize_and_cost(par, [[sx, sy]] + [[org[i][0], org[i][1]] for i in range(1, K + 1)], K + 1))

    ang = [0.0] + [math.atan2(org[o][1] - sy, org[o][0] - sx) % (2 * math.pi) for o in range(1, K + 1)]

    def buckets_of(nb, off):
        width = 2 * math.pi / nb
        b = [[] for _ in range(nb)]
        for o in range(1, K + 1):
            b[int(((ang[o] - off) % (2 * math.pi)) / width) % nb].append(o)
        return [x for x in b if x]

    def wcentroid(members):
        tw = sum(org[o][2] for o in members) or 1.0
        cx = sum(org[o][0] * org[o][2] for o in members) / tw
        cy = sum(org[o][1] * org[o][2] for o in members) / tw
        return cx, cy

    maxb = min(7, K)
    for nb in range(1, maxb + 1):
        for off in (0.0, math.pi / nb):
            bs = buckets_of(nb, off)
            if not bs:
                continue

            # --- flat hubs (one level) ---
            par = [0] * (K + 1)
            pos = [[sx, sy]] + [[org[i][0], org[i][1]] for i in range(1, K + 1)]
            jids = []
            for bk in bs:
                jid = len(pos)
                cx, cy = wcentroid(bk)
                pos.append([cx, cy]); jids.append((jid, bk))
                par.append(0)
            for (jid, bk) in jids:
                for o in bk:
                    par[o] = jid
            n = len(pos)
            relax(par, pos, n)
            results.append(finalize_and_cost(par, pos, n))

            # --- two-level trunk (global J0 -> cluster hubs -> organs) ---
            if len(bs) >= 2:
                par = [0] * (K + 1)
                pos = [[sx, sy]] + [[org[i][0], org[i][1]] for i in range(1, K + 1)]
                gx, gy = wcentroid([o for bk in bs for o in bk])
                j0 = len(pos); pos.append([0.5 * gx, 0.5 * gy]); par.append(0)
                for bk in bs:
                    jid = len(pos)
                    cx, cy = wcentroid(bk)
                    pos.append([cx, cy]); par.append(j0)
                    for o in bk:
                        par[o] = jid
                n = len(pos)
                relax(par, pos, n)
                results.append(finalize_and_cost(par, pos, n))

            # --- radial spine (cluster hubs chained inner->outer to source) ---
            if len(bs) >= 2:
                par = [0] * (K + 1)
                pos = [[sx, sy]] + [[org[i][0], org[i][1]] for i in range(1, K + 1)]
                order = sorted(range(len(bs)), key=lambda ci: sum(
                    math.hypot(org[o][0] - sx, org[o][1] - sy) for o in bs[ci]) / len(bs[ci]))
                prev = 0
                for ci in order:
                    bk = bs[ci]
                    jid = len(pos)
                    cx, cy = wcentroid(bk)
                    pos.append([cx, cy]); par.append(prev)
                    for o in bk:
                        par[o] = jid
                    prev = jid
                n = len(pos)
                relax(par, pos, n)
                results.append(finalize_and_cost(par, pos, n))

    bestF = float("inf"); best = ([], [0] * K)
    for (F, st, pr) in results:
        if st is not None and F < bestF:
            bestF = F; best = (st, pr)

    st, pr = best
    out = [str(len(st))]
    for (jx, jy) in st:
        out.append("%.6f %.6f" % (jx, jy))
    for p in pr:
        out.append(str(p))
    sys.stdout.write("\n".join(out) + "\n")


main()
