# TIER: strong
# Hottest-first greedy construction (same as the greedy tier) + local search:
#   1) Sort jobs by descending load; drop each onto the free slot that adds the least
#      squared over-temperature given deposits so far (overlap- and cooling-aware).
#   2) Local search over the FULL objective:
#        - RELOCATE: move a job to any free slot if it lowers total penalty.
#        - SWAP: exchange the slots of two jobs (different loads) if it lowers penalty.
#      Deterministic scan order and a fixed pass cap keep it reproducible within the
#      operation budget.  Relocation/swap undo the greedy's myopia, but because total
#      deposited heat exceeds total cooling on every hall, penalty 0 is unreachable, so
#      the normalized score stays well below 1.0.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]; J = inst["j"]; loads = inst["loads"]; cool = inst["cool"]
kernel = inst["kernel"]

dep = [[0] * n for _ in range(n)]


def delta_place(r, c, w):
    d = 0
    for dr in (-1, 0, 1):
        rr = r + dr
        if rr < 0 or rr >= n:
            continue
        krow = kernel[dr + 1]
        for dc in (-1, 0, 1):
            cc = c + dc
            if cc < 0 or cc >= n:
                continue
            add = w * krow[dc + 1]
            old = dep[rr][cc] - cool[rr][cc]
            oldp = old * old if old > 0 else 0
            new = old + add
            newp = new * new if new > 0 else 0
            d += newp - oldp
    return d


def apply(r, c, w, sign):
    for dr in (-1, 0, 1):
        rr = r + dr
        if rr < 0 or rr >= n:
            continue
        krow = kernel[dr + 1]
        for dc in (-1, 0, 1):
            cc = c + dc
            if cc < 0 or cc >= n:
                continue
            dep[rr][cc] += sign * w * krow[dc + 1]


# ---- 1) greedy build ----
order = sorted(range(J), key=lambda jj: (-loads[jj], jj))
used = [[False] * n for _ in range(n)]
place = [None] * J
for jj in order:
    w = loads[jj]
    best = None; best_d = None
    for r in range(n):
        for c in range(n):
            if used[r][c]:
                continue
            d = delta_place(r, c, w)
            if best_d is None or d < best_d or (d == best_d and (r, c) < best):
                best_d = d; best = (r, c)
    r, c = best
    used[r][c] = True
    place[jj] = [r, c]
    apply(r, c, w, +1)


def total_penalty():
    tot = 0
    for r in range(n):
        cr = cool[r]; dr_ = dep[r]
        for c in range(n):
            ov = dr_[c] - cr[c]
            if ov > 0:
                tot += ov * ov
    return tot


# ---- 2) local search ----
cur = total_penalty()
for _pass in range(6):
    improved = False

    # RELOCATE each job to a free slot
    for jj in range(J):
        w = loads[jj]
        r0, c0 = place[jj]
        apply(r0, c0, w, -1)          # temporarily remove job jj
        best = (r0, c0); best_d = delta_place(r0, c0, w)
        for r in range(n):
            for c in range(n):
                if used[r][c] and (r, c) != (r0, c0):
                    continue
                d = delta_place(r, c, w)
                if d < best_d or (d == best_d and (r, c) < best):
                    best_d = d; best = (r, c)
        r1, c1 = best
        apply(r1, c1, w, +1)          # re-add at chosen slot
        if (r1, c1) != (r0, c0):
            used[r0][c0] = False
            used[r1][c1] = True
            place[jj] = [r1, c1]
            improved = True

    # SWAP slots of two jobs with different loads
    for a in range(J):
        for b in range(a + 1, J):
            wa = loads[a]; wb = loads[b]
            if wa == wb:
                continue
            ra, ca = place[a]; rb, cb = place[b]
            # remove both
            apply(ra, ca, wa, -1)
            apply(rb, cb, wb, -1)
            before = delta_place(ra, ca, wa)
            apply(ra, ca, wa, +1)
            before += delta_place(rb, cb, wb)
            apply(ra, ca, wa, -1)
            # swapped: a -> (rb,cb), b -> (ra,ca)
            after = delta_place(rb, cb, wa)
            apply(rb, cb, wa, +1)
            after += delta_place(ra, ca, wb)
            apply(rb, cb, wa, -1)
            if after < before:
                apply(rb, cb, wa, +1)
                apply(ra, ca, wb, +1)
                place[a] = [rb, cb]; place[b] = [ra, ca]
                improved = True
            else:
                apply(ra, ca, wa, +1)
                apply(rb, cb, wb, +1)

    if not improved:
        break

print(json.dumps({"place": place}))
