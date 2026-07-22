# TIER: strong
"""The insight: empty area is not fungible once cuts are irrevocable, so manage
the SHAPE of remaining space, not just its total -- AND treat every accepted
low-value order as an option forfeited against the space a future value-dense
order will need, not merely "it happened to fit."

1. Look at the full known arrival mixture (its statistics, not the future
   itself -- the geometric packing problem is still hard) and derive TWO
   adaptive value-density quantiles from THIS instance's own distribution
   (not a hard-coded category boundary, so it generalizes to held-out
   mixtures): a high cut for the value-dense class, and a separate, lower
   cut below which an order is not even worth the kerf it would burn.
2. Reserve a CORRIDOR (a column-range) sized, via a quick shelf-pack
   estimate, to the total kerf-inflated footprint of the value-dense class.
   That corridor is the shape-grammar "option" the policy is buying.
3. Replay the stream causally: value-dense orders try the corridor first
   (falling back to the general zone only if the corridor has no room);
   mid-value orders (above the low cut) may only use the general zone;
   anything below the low cut is DISCARDED outright -- accepting it would
   spend real kerf-locked space on an order not worth defending. Nothing
   below the value-dense class may ever intrude on the reserved corridor.
"""
import sys, json, math


def legal(grid, W, H, k, x, y, dw, dh):
    if x < 0 or y < 0 or dw <= 0 or dh <= 0 or x + dw > W or y + dh > H:
        return False
    kx0 = max(0, x - k); kx1 = min(W, x + dw + k)
    ky0 = max(0, y - k); ky1 = min(H, y + dh + k)
    for yy in range(ky0, ky1):
        row = grid[yy]
        for xx in range(kx0, kx1):
            if row[xx]:
                return False
    return True


def commit(grid, W, H, k, x, y, dw, dh):
    kx0 = max(0, x - k); kx1 = min(W, x + dw + k)
    ky0 = max(0, y - k); ky1 = min(H, y + dh + k)
    for yy in range(ky0, ky1):
        row = grid[yy]
        for xx in range(kx0, kx1):
            row[xx] = True


def try_zone(grid, W, H, k, x_lo, x_hi, dw, dh):
    x_hi = min(x_hi, W)
    for y in range(0, H - dh + 1):
        for x in range(x_lo, x_hi - dw + 1):
            if legal(grid, W, H, k, x, y, dw, dh):
                return (x, y)
    return None


def main():
    inst = json.load(sys.stdin)
    W, H, k = inst["W"], inst["H"], inst["k"]
    items = inst["items"]
    n = len(items)

    # -- 1. two adaptive value-density quantiles computed from THIS instance's
    #    own distribution (no hard-coded category cut): a high cut for the
    #    value-dense class that earns a reserved corridor, and a lower cut
    #    below which an order is not even worth the kerf it would burn.
    densities = sorted(it["value"] / max(1, it["w"] * it["h"]) for it in items)
    hi_idx = min(n - 1, max(0, int(0.85 * (n - 1))))
    lo_idx = min(n - 1, max(0, int(0.70 * (n - 1))))
    hi_thresh, lo_thresh = densities[hi_idx], densities[lo_idx]
    high_idx = [i for i in range(n)
                if items[i]["value"] / max(1, items[i]["w"] * items[i]["h"]) >= hi_thresh]
    high_set = set(high_idx)

    # -- 2. corridor sizing via a rough kerf-inflated area estimate, with slack.
    high_area = sum((items[i]["w"] + 2 * k) * (items[i]["h"] + 2 * k) for i in high_idx)
    corridor_w = min(W - 2, max(3, math.ceil(1.2 * high_area / max(1, H))))

    grid = [[False] * W for _ in range(H)]
    out = [None] * n
    for i, it in enumerate(items):
        dens = it["value"] / max(1, it["w"] * it["h"])
        placed, rot_used = None, 0
        if i in high_set:
            for rot in (0, 1):
                dw, dh = (it["w"], it["h"]) if rot == 0 else (it["h"], it["w"])
                p = try_zone(grid, W, H, k, 0, corridor_w, dw, dh)
                if p:
                    placed, rot_used = p, rot
                    break
            if not placed:
                for rot in (0, 1):
                    dw, dh = (it["w"], it["h"]) if rot == 0 else (it["h"], it["w"])
                    p = try_zone(grid, W, H, k, corridor_w, W, dw, dh)
                    if p:
                        placed, rot_used = p, rot
                        break
        elif dens >= lo_thresh:
            for rot in (0, 1):
                dw, dh = (it["w"], it["h"]) if rot == 0 else (it["h"], it["w"])
                p = try_zone(grid, W, H, k, corridor_w, W, dw, dh)
                if p:
                    placed, rot_used = p, rot
                    break
            # deliberately never intrudes on the reserved corridor -- discard instead
        # below the low cut: not worth the kerf it would burn -- straight discard
        if placed:
            x, y = placed
            dw, dh = (it["w"], it["h"]) if rot_used == 0 else (it["h"], it["w"])
            commit(grid, W, H, k, x, y, dw, dh)
            out[i] = {"action": "place", "x": x, "y": y, "rot": rot_used}
        else:
            out[i] = {"action": "discard"}

    print(json.dumps({"decisions": out}))


if __name__ == "__main__":
    main()
