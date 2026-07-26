# TIER: strong
import sys

CLASSES = "HML"


def main():
    data = sys.stdin.read().split()
    idx = [0]

    def nxt():
        v = data[idx[0]]
        idx[0] += 1
        return v

    N = int(nxt())
    D = int(nxt())
    FIXSEP = int(nxt())
    S = [[0, 0, 0] for _ in range(3)]
    for a in range(3):
        for b in range(3):
            S[a][b] = int(nxt())
    fw = [int(nxt()) for _ in range(3)]
    r = []
    cls = []
    for i in range(N):
        r.append(int(nxt()))
        cls.append(CLASSES.index(nxt()))

    def simulate(order):
        """Given a crossing ORDER (permutation of aircraft indices), replay the
        shared-fix pipeline (FIFO through the choke point, so fix_time is fully
        determined by the order) and then list-schedule the two runways. Returns
        (F, fixt, landing_of, runway_of)."""
        fixt = {}
        clock = None
        for j in order:
            ft = r[j] if clock is None else max(r[j], clock + FIXSEP)
            fixt[j] = ft
            clock = ft
        runway_land = [None, None]
        runway_cls = [None, None]
        landing_of = {}
        runway_of = {}
        for j in order:
            ft = fixt[j]
            best_rw, best_lt = None, None
            for rw in (0, 1):
                lt = ft + D
                if runway_land[rw] is not None:
                    lt = max(lt, runway_land[rw] + S[runway_cls[rw]][cls[j]])
                if best_lt is None or lt < best_lt:
                    best_lt, best_rw = lt, rw
            runway_land[best_rw] = best_lt
            runway_cls[best_rw] = cls[j]
            landing_of[j] = best_lt
            runway_of[j] = best_rw
        F = sum(fw[cls[j]] * (landing_of[j] - r[j]) for j in order)
        return F, fixt, landing_of, runway_of

    # ---- Phase 1: build an initial batch structure with a nearest-neighbour
    # dispatch rule on the class-transition graph S (a TSP-like ordering
    # problem: the object that matters is the run-length composition, not the
    # raw arrival order), corrected by a fuel-weighted "staleness" term so a
    # class that has been sitting past ready time eventually forces a switch
    # -- otherwise one class could hog the queue forever.
    queues = [[], [], []]
    for i in sorted(range(N), key=lambda i: (r[i], i)):
        queues[cls[i]].append(i)
    qi = [0, 0, 0]

    def front_r(c):
        return r[queues[c][qi[c]]] if qi[c] < len(queues[c]) else None

    order = []
    clock = None
    last_class = None
    for _ in range(N):
        fronts = [(c, front_r(c)) for c in range(3) if front_r(c) is not None]
        # Never idle the fix waiting for a "nicer" class to arrive: only choose
        # among aircraft that are ALREADY ready (front_r <= clock). If NONE are
        # ready yet there is no real choice -- the fix must sit idle until the
        # next arrival regardless of class, so take whichever front is earliest;
        # picking a later one "for the score" would be pure wasted delay for
        # everybody behind.
        ready = [(c, fr) for c, fr in fronts if clock is not None and fr <= clock]
        if ready:
            pool = ready
        else:
            min_fr = min(fr for _, fr in fronts)
            pool = [(c, fr) for c, fr in fronts if fr == min_fr]
        best_c, best_score, best_fr = None, None, None
        for c, fr in pool:
            trans = 0 if last_class is None else S[last_class][c]
            stale = 0 if clock is None else fw[c] * max(0, clock - fr)
            score = trans - stale
            if (best_score is None or score < best_score - 1e-9 or
                    (abs(score - best_score) <= 1e-9 and fr < best_fr)):
                best_c, best_score, best_fr = c, score, fr
        j = queues[best_c][qi[best_c]]
        qi[best_c] += 1
        clock = r[j] if clock is None else max(r[j], clock + FIXSEP)
        last_class = best_c
        order.append(j)

    # ---- Phase 2: local-search refinement. The dispatch rule is a one-step
    # greedy over the batch structure; a bounded adjacent-swap hill-climb on
    # the crossing order (re-evaluated against the REAL two-runway objective,
    # not the 1-step proxy) cleans up the boundaries between runs and the
    # runway split itself. Cheap (O(passes * N) simulations) and deterministic.
    bestF, _, _, _ = simulate(order)
    for _ in range(25):
        improved = False
        for k in range(N - 1):
            order[k], order[k + 1] = order[k + 1], order[k]
            f2, _, _, _ = simulate(order)
            if f2 < bestF - 1e-9:
                bestF = f2
                improved = True
            else:
                order[k], order[k + 1] = order[k + 1], order[k]
        if not improved:
            break
    # Or-opt pass: relocate one aircraft at a time to its best slot anywhere in
    # the order (adjacent swaps alone can be slow to migrate an element far
    # across a long run; direct relocation captures that in one move).
    for _ in range(3):
        improved = False
        for i in range(N):
            j_aircraft = order[i]
            without = order[:i] + order[i + 1:]
            best_pos, best_val = i, bestF
            for pos in range(len(without) + 1):
                cand = without[:pos] + [j_aircraft] + without[pos:]
                f2, _, _, _ = simulate(cand)
                if f2 < best_val - 1e-9:
                    best_val, best_pos = f2, pos
            if best_pos != i:
                order = without[:best_pos] + [j_aircraft] + without[best_pos:]
                bestF = best_val
                improved = True
        if not improved:
            break

    F, fixt, landing_of, runway_of = simulate(order)
    out = [None] * N
    for j in order:
        out[j] = (runway_of[j] + 1, fixt[j], landing_of[j])

    lines = ["%d %d %d" % t for t in out]
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
