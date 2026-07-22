import sys

# Deterministic scorer for the dirty-loop-writeback-cache problem.
# CLI:  python3 verify.py <in> <out> <ans>   (ans ignored)
# Prints a single final line "... Ratio: <r in [0,1]>".  Objective = MINIMIZE
# total charged cost, so ratio = min(1, 0.1 * B / F_participant) where B is an
# internal LRU-no-cleaning baseline the checker builds itself.

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)

def read_instance(path):
    tok = open(path).read().split()
    it = iter(tok)
    k  = int(next(it))
    F  = int(next(it))
    Ce = int(next(it))
    De = int(next(it))
    Pc = int(next(it))
    M  = int(next(it))
    isw = [0] * M
    pg  = [0] * M
    for i in range(M):
        t = next(it)
        p = int(next(it))
        isw[i] = 1 if t == 'W' else 0
        pg[i]  = p
    return k, F, Ce, De, Pc, M, isw, pg

def lru_baseline(k, F, Ce, De, Pc, M, isw, pg):
    # LRU eviction, NO proactive cleans.  Same cost model + end flush.
    resident = {}          # page -> dirty(bool)
    last = {}              # page -> last access op index (recency)
    cost = 0
    for i in range(M):
        p = pg[i]
        if p in resident:
            if isw[i]:
                resident[p] = True
        else:
            if len(resident) >= k:
                # evict least-recently-used
                victim = min(resident, key=lambda x: last[x])
                cost += De if resident[victim] else Ce
                del resident[victim]
                del last[victim]
            cost += F
            resident[p] = bool(isw[i])
        last[p] = i
    for p, d in resident.items():
        if d:
            cost += (De - Ce)   # final dirty writeback surcharge
    return cost

def main():
    if len(sys.argv) < 3:
        fail("usage")
    try:
        k, F, Ce, De, Pc, M, isw, pg = read_instance(sys.argv[1])
    except Exception:
        fail("bad input")

    # ---------- internal baseline ----------
    try:
        B = lru_baseline(k, F, Ce, De, Pc, M, isw, pg)
    except Exception:
        fail("baseline error")
    B = max(1, B)

    # ---------- parse participant script ----------
    try:
        raw = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")
    if not raw:
        fail("empty output")
    try:
        A = int(raw[0])
    except Exception:
        fail("bad action count")
    if A < 0 or A > 6 * M + 16:
        fail("action count out of range")
    if len(raw) < 1 + 3 * A:
        fail("truncated actions")

    cleans_at = {}     # t -> list of pages to clean before op t
    evict_at  = {}     # t -> victim page for the miss at op t
    try:
        idx = 1
        for _ in range(A):
            kind = raw[idx]
            t = int(raw[idx + 1])
            p = int(raw[idx + 2])
            idx += 3
            if t < 0 or t >= M:
                fail("action time out of range")
            if kind == "CLEAN":
                cleans_at.setdefault(t, []).append(p)
            elif kind == "EVICT":
                if t in evict_at:
                    fail("two evictions at same op")
                evict_at[t] = p
            else:
                fail("unknown action kind")
    except SystemExit:
        raise
    except Exception:
        fail("bad action token")

    # ---------- replay the participant script under the exact cost model ----------
    resident = {}      # page -> dirty(bool)
    cost = 0
    consumed = set()   # op indices where an EVICT was actually used
    for i in range(M):
        # 1) apply scheduled proactive cleans
        if i in cleans_at:
            for p in cleans_at[i]:
                if p not in resident:
                    fail("clean of non-resident page at op %d" % i)
                # cleaning a clean page is wasteful but legal (still charged)
                resident[p] = False
                cost += Pc
        # 2) process the access
        p = pg[i]
        if p in resident:
            if isw[i]:
                resident[p] = True
        else:
            if len(resident) < k:
                pass  # free slot, no eviction directive expected
            else:
                if i not in evict_at:
                    fail("no eviction directive at forced miss op %d" % i)
                v = evict_at[i]
                if v not in resident:
                    fail("victim %d not resident at op %d" % (v, i))
                cost += De if resident[v] else Ce
                del resident[v]
                consumed.add(i)
            cost += F
            resident[p] = bool(isw[i])

    # every EVICT directive must have been used at a genuine forced miss
    if consumed != set(evict_at.keys()):
        fail("stray eviction directive at a non-miss op")

    # end-of-trace: resident dirty pages need a final writeback
    for p, d in resident.items():
        if d:
            cost += (De - Ce)

    Fp = max(1, cost)
    sc = min(1000.0, 100.0 * B / Fp)
    print("B=%d cost=%d Ratio: %.6f" % (B, cost, sc / 1000.0))

if __name__ == "__main__":
    main()
