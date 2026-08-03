# TIER: strong
# Cardinality-aware decreasing-order packing, keep whichever layout rolls out
# fewer stages:
#   * first-fit-decreasing (FFD): slot the largest acts first onto the lowest-index
#     stage that has both resource room AND a free changeover window (< K acts);
#   * best-fit-decreasing (BFD): slot the largest acts first onto the TIGHTEST
#     stage that still fits both constraints (leaves the roomiest gaps open for
#     later big rigs).
# Then a short deterministic local-search pass tries to relocate single acts off
# the least-loaded stage to eliminate it entirely.  Sorting big-first lets small
# acts top off partly-filled stages under the K cap, so waste drops well below the
# online rules -- but the loose combined L1 bound keeps the score below 1.0 on most
# instances.
import sys, json

inst = json.load(sys.stdin)
C = inst["capacity"]
K = inst["max_acts"]
acts = inst["acts"]
N = len(acts)

order = sorted(range(N), key=lambda i: acts[i], reverse=True)


def ffd():
    rem = []
    cnt = []
    gof = [0] * N
    for i in order:
        s = acts[i]
        placed = -1
        for b in range(len(rem)):
            if rem[b] >= s and cnt[b] < K:
                placed = b
                break
        if placed < 0:
            rem.append(C - s)
            cnt.append(1)
            gof[i] = len(rem) - 1
        else:
            rem[placed] -= s
            cnt[placed] += 1
            gof[i] = placed
    return gof, len(rem)


def bfd():
    rem = []
    cnt = []
    gof = [0] * N
    for i in order:
        s = acts[i]
        best = -1
        best_rem = C + 1
        for b in range(len(rem)):
            if rem[b] >= s and cnt[b] < K and rem[b] < best_rem:
                best_rem = rem[b]
                best = b
        if best < 0:
            rem.append(C - s)
            cnt.append(1)
            gof[i] = len(rem) - 1
        else:
            rem[best] -= s
            cnt[best] += 1
            gof[i] = best
    return gof, len(rem)


def local_improve(gof, nbins):
    # Deterministic relocation: repeatedly try to empty the stage holding the
    # fewest acts by moving each of its acts onto some other feasible stage.
    for _ in range(3):
        # recompute stage loads/counts/members
        rem = [C] * nbins
        cnt = [0] * nbins
        members = [[] for _ in range(nbins)]
        for i in range(N):
            b = gof[i]
            rem[b] -= acts[i]
            cnt[b] += 1
            members[b].append(i)
        # target = non-empty stage with the fewest acts
        target = -1
        best_cnt = 1 << 30
        for b in range(nbins):
            if cnt[b] > 0 and cnt[b] < best_cnt:
                best_cnt = cnt[b]
                target = b
        if target < 0:
            break
        moves = {}
        ok = True
        # try to relocate every act of target elsewhere
        rem_w = rem[:]
        cnt_w = cnt[:]
        for i in sorted(members[target], key=lambda j: acts[j], reverse=True):
            s = acts[i]
            dest = -1
            best_rem = C + 1
            for b in range(nbins):
                if b == target:
                    continue
                if rem_w[b] >= s and cnt_w[b] < K and rem_w[b] < best_rem:
                    best_rem = rem_w[b]
                    dest = b
            if dest < 0:
                ok = False
                break
            rem_w[dest] -= s
            cnt_w[dest] += 1
            moves[i] = dest
        if ok and moves:
            for i, d in moves.items():
                gof[i] = d
            # relabel to keep indices dense (not required, but tidy)
            used = sorted(set(gof))
            remap = {u: k for k, u in enumerate(used)}
            gof = [remap[g] for g in gof]
            nbins = len(used)
        else:
            break
    return gof, nbins


fa, fb = ffd()
ba, bb = bfd()
if fb <= bb:
    assign, nb = fa, fb
else:
    assign, nb = ba, bb

assign, nb = local_improve(assign, nb)

print(json.dumps({"assign": assign}))
