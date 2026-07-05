# TIER: strong
# Temperature-aware best-fit-decreasing + seeded multi-restart + relocation local
# search.  For each of several deterministic well orderings we run a best-fit pass
# (drop each well into the compatible loop that leaves the LEAST slack, breaking
# ties toward the closest temperature), then repeatedly try to disband the
# smallest loop by relocating its wells elsewhere.  We keep the layout with the
# fewest active loops.  Fully deterministic (own seeded PRNG).
import sys, json

inst = json.load(sys.stdin)
flow = inst["flow"]
temp = inst["temp"]
C = inst["capacity"]
band = inst["band"]
N = inst["n"]


def _rng(seed):
    st = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def rnd():
        nonlocal st
        st = (st * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return (st >> 11) / float(1 << 53)
    return rnd


def compatible(loop, f, t):
    if f > loop[0]:
        return False
    nlo = t if t < loop[1] else loop[1]
    nhi = t if t > loop[2] else loop[2]
    return nhi - nlo <= band


def best_fit(order):
    """loops: list of [rem, tmin, tmax, [members]]. Returns list of loops."""
    loops = []
    for i in order:
        f = flow[i]
        t = temp[i]
        best = -1
        best_slack = None
        best_tspread = None
        for j, lp in enumerate(loops):
            if not compatible(lp, f, t):
                continue
            slack = lp[0] - f
            nlo = t if t < lp[1] else lp[1]
            nhi = t if t > lp[2] else lp[2]
            tspread = nhi - nlo
            key = (slack, tspread)
            if best < 0 or key < (best_slack, best_tspread):
                best = j
                best_slack = slack
                best_tspread = tspread
        if best < 0:
            loops.append([C - f, t, t, [i]])
        else:
            lp = loops[best]
            lp[0] -= f
            lp[1] = min(lp[1], t)
            lp[2] = max(lp[2], t)
            lp[3].append(i)
    return loops


def recompute(members):
    rem = C - sum(flow[i] for i in members)
    tlo = min(temp[i] for i in members)
    thi = max(temp[i] for i in members)
    return [rem, tlo, thi, list(members)]


def relocate(loops):
    """Try to disband small loops by moving their wells into others."""
    improved = True
    while improved:
        improved = False
        # try smallest (fewest members) loops first
        order = sorted(range(len(loops)), key=lambda j: len(loops[j][3]))
        for j in order:
            members = list(loops[j][3])
            targets = {}
            ok = True
            for i in members:
                f = flow[i]
                t = temp[i]
                dest = -1
                best_slack = None
                for k in range(len(loops)):
                    if k == j:
                        continue
                    lp = loops[k]
                    # account for wells already tentatively moved into k
                    extra_f = sum(flow[m] for m in targets.get(k, []))
                    extra_lo = min([temp[m] for m in targets.get(k, [])] + [lp[1]])
                    extra_hi = max([temp[m] for m in targets.get(k, [])] + [lp[2]])
                    if f > lp[0] - extra_f:
                        continue
                    nlo = min(extra_lo, t)
                    nhi = max(extra_hi, t)
                    if nhi - nlo > band:
                        continue
                    slack = (lp[0] - extra_f) - f
                    if dest < 0 or slack < best_slack:
                        dest = k
                        best_slack = slack
                if dest < 0:
                    ok = False
                    break
                targets.setdefault(dest, []).append(i)
            if ok:
                # commit: remove loop j, add wells to targets
                for k, ws in targets.items():
                    loops[k][3].extend(ws)
                new_loops = []
                for idx, lp in enumerate(loops):
                    if idx == j:
                        continue
                    new_loops.append(recompute(lp[3]))
                loops[:] = new_loops
                improved = True
                break
    return loops


orders = []
orders.append(sorted(range(N), key=lambda i: -flow[i]))
orders.append(sorted(range(N), key=lambda i: (temp[i], -flow[i])))
orders.append(sorted(range(N), key=lambda i: (-flow[i], temp[i])))
for s in range(1, 30):
    rnd = _rng(s * 2654435761 + N)
    perm = sorted(range(N), key=lambda i: rnd())
    orders.append(perm)

best_loops = None
for order in orders:
    loops = best_fit(order)
    loops = relocate(loops)
    if best_loops is None or len(loops) < len(best_loops):
        best_loops = loops

assign = [0] * N
for j, lp in enumerate(best_loops):
    for i in lp[3]:
        assign[i] = j

print(json.dumps({"assign": assign}))
