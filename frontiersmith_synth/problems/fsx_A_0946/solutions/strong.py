# TIER: strong
"""Insight: route and depth belong to ONE warped cost space, not two separate
decisions. Compute required_depth(type) honestly from the frost physics (the
worst RECORDED winter per cover type -- correctly noticing that the worst
winter for snow-holding ground is a low-snow winter, not necessarily the
coldest one), then treat cell (r,c)'s cost to enter as required_depth(type)^2
and build the pipe network with a nearest-terminal Dijkstra Steiner
heuristic IN THAT WEIGHTED GRAPH (not Euclidean/Manhattan space): repeatedly
grow the tree towards whichever unconnected building is cheapest to reach,
where re-entering an already-built cell is free. Finally bury each cell at
EXACTLY its own required depth -- no uniform over-provisioning. A longer
route through peat/snow country can and does beat the geometric shortest
path once cost is quadratic in depth and depth is soil-dependent."""
import sys, json, math, heapq


def required_depth(t, kappa, depth_scale, winters):
    best = 0.0
    for w in winters:
        if t == 0:
            k = kappa["bare"]
        elif t == 1:
            k = kappa["pavement"]
        elif t == 2:
            k = kappa["snow_base"] / (1.0 + kappa["snow_sensitivity"] * w["snow"])
        else:
            k = kappa["peat"]
        d = depth_scale * math.sqrt(2.0 * k * w["fdd"])
        if d > best:
            best = d
    return best


def main():
    inst = json.load(sys.stdin)
    R, C = inst["rows"], inst["cols"]
    grid = inst["grid"]
    buildings = [tuple(b) for b in inst["buildings"]]
    kappa = inst["kappa"]
    depth_scale = inst["depth_scale"]
    winters = inst["winters"]
    rd = {t: required_depth(t, kappa, depth_scale, winters) for t in (0, 1, 2, 3)}

    w = [[rd[grid[r][c]] ** 2 for c in range(C)] for r in range(R)]

    n = len(buildings)
    tree = {buildings[0]}
    remaining = set(range(1, n))
    while remaining:
        dist = [[math.inf] * C for _ in range(R)]
        prev = {}
        pq = []
        for (r, c) in tree:
            dist[r][c] = 0.0
            heapq.heappush(pq, (0.0, r, c))
        while pq:
            d, r, c = heapq.heappop(pq)
            if d > dist[r][c]:
                continue
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < R and 0 <= nc < C:
                    step = 0.0 if (nr, nc) in tree else w[nr][nc]
                    nd = d + step
                    if nd < dist[nr][nc] - 1e-12:
                        dist[nr][nc] = nd
                        prev[(nr, nc)] = (r, c)
                        heapq.heappush(pq, (nd, nr, nc))
        best_ri, best_d = None, math.inf
        for ri in remaining:
            b = buildings[ri]
            if dist[b[0]][b[1]] < best_d:
                best_d = dist[b[0]][b[1]]
                best_ri = ri
        cur = buildings[best_ri]
        path = [cur]
        while cur in prev:
            cur = prev[cur]
            path.append(cur)
        for cell in path:
            tree.add(cell)
        remaining.discard(best_ri)

    route = [{"r": r, "c": c, "depth": rd[grid[r][c]]} for (r, c) in tree]
    print(json.dumps({"route": route}))


if __name__ == "__main__":
    main()
