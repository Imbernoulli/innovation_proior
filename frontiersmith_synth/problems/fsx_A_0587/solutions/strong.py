# TIER: strong
# INSIGHT: the empty berths are a positionable resource, not leftover space.
# An absent lookup at berth g pays the forward distance to the FIRST empty berth,
# so a berth with high absent-weight that sits far (in probe order) from any empty
# is catastrophic -- and greedy packing leaves every empty bunched at the tail of a
# run.  Relocating one empty backward into a hot zone shifts a whole block of boats
# forward by one (a globally coupled move): we pay a little present cost but slash
# the absent cost of every high-weight berth that now reaches the empty quickly.
# We start from the frequency-aware layout, then apply breakwater relocations while
# they lower the TOTAL cost.
import sys

def main():
    toks = open(sys.argv[1]).read().split() if len(sys.argv) > 1 else sys.stdin.read().split()
    it = iter(toks)
    m = int(next(it)); n = int(next(it)); R = int(next(it))
    home = [0]*n; fp = [0]*n
    for i in range(n):
        home[i] = int(next(it)); fp[i] = int(next(it))
    a = [0]*m
    for g in range(m):
        a[g] = int(next(it))

    # ---- frequency-aware first-fit base layout ----
    owner = [-1]*m
    berth_of = [0]*n
    for i in sorted(range(n), key=lambda i: (-fp[i], home[i], i)):
        h = home[i]
        for d in range(R+1):
            s = (h+d) % m
            if owner[s] == -1:
                owner[s] = i; berth_of[i] = s; break

    # prefix sums of the (fixed) absent weights
    Sa = [0]*(m+1); Sag = [0]*(m+1)
    for g in range(m):
        Sa[g+1] = Sa[g] + a[g]
        Sag[g+1] = Sag[g] + a[g]*g

    def sumA(l, r):      # sum a[g], g in [l,r]
        if r < l: return 0
        return Sa[r+1] - Sa[l]
    def sumAg(l, r):
        if r < l: return 0
        return Sag[r+1] - Sag[l]

    def disp(x):
        o = owner[x]
        return (x - home[o]) % m

    # ---- breakwater relocation passes ----
    for _pass in range(6):
        changed = False
        # empties in increasing order
        empties = [x for x in range(m) if owner[x] == -1]
        eset = set(empties)
        for e in empties:
            if e not in eset:          # already moved this pass
                continue
            # run to the left: p0 = nearest empty < e (occupied strictly between)
            p0 = e - 1
            while p0 >= 0 and owner[p0] != -1:
                p0 -= 1
            if p0 < 0:                 # no left boundary (edge); skip
                continue
            # nearest empty to the right of e (for berths that lose the empty at e)
            e2 = e + 1
            while e2 < m and owner[e2] != -1:
                e2 += 1
            if e2 >= m:
                continue
            # before-cost over window [p0+1, e]
            l = p0 + 1
            before = (e*sumA(l, e-1) - sumAg(l, e-1)) + sumA(l, e-1) + a[e]*1
            best_delta = 0
            best_q = -1
            best_pd = 0
            pd = 0                     # present delta accumulated as q descends
            q = e - 1
            while q >= l:
                if disp(q) >= R:       # this boat cannot shift forward -> stop
                    break
                pd += fp[owner[q]]      # boats [q..e-1] each move +1 berth
                # after-cost with empty at q instead of e
                after = ( (q*sumA(l, q-1) - sumAg(l, q-1)) + sumA(l, q-1)
                          + a[q]*1
                          + (e2*sumA(q+1, e) - sumAg(q+1, e)) + sumA(q+1, e) )
                delta = pd + (after - before)
                if delta < best_delta:
                    best_delta = delta; best_q = q; best_pd = pd
                q -= 1
            if best_q >= 0:
                # apply: shift boats [best_q .. e-1] forward by one, empty best_q
                for x in range(e, best_q, -1):
                    b = owner[x-1]
                    owner[x] = b; berth_of[b] = x
                owner[best_q] = -1
                eset.discard(e); eset.add(best_q)
                changed = True
        if not changed:
            break

    sys.stdout.write(" ".join(str(berth_of[i]) for i in range(n)) + "\n")

if __name__ == "__main__":
    main()
