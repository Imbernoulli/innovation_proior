# TIER: greedy
# The obvious offline approach: Belady's farthest-in-future eviction, which
# minimizes the number of fetches -- but ignores ink.  It happily evicts wet
# (dirty) pages at the full writeback charge, and issues no proactive cleans.
import sys

INF = 1 << 60

def main():
    tok = sys.stdin.buffer.read().split()
    it = iter(tok)
    k = int(next(it)); F = int(next(it)); Ce = int(next(it))
    De = int(next(it)); Pc = int(next(it)); M = int(next(it))
    isw = [0]*M; pg = [0]*M
    for i in range(M):
        t = next(it); pg[i] = int(next(it))
        isw[i] = 1 if t == b'W' else 0

    # next occurrence of the same page after each position
    next_at = [INF]*M
    seen = {}
    for i in range(M-1, -1, -1):
        p = pg[i]
        next_at[i] = seen.get(p, INF)
        seen[p] = i

    resident = {}     # page -> dirty
    last = {}         # page -> last op index
    actions = []
    for i in range(M):
        p = pg[i]
        if p in resident:
            if isw[i]:
                resident[p] = True
        else:
            if len(resident) >= k:
                # evict page whose next use is farthest in the future
                victim = max(resident, key=lambda x: next_at[last[x]])
                actions.append("EVICT %d %d" % (i, victim))
                del resident[victim]; del last[victim]
            resident[p] = bool(isw[i])
        last[p] = i

    out = [str(len(actions))]
    out.extend(actions)
    sys.stdout.write("\n".join(out) + "\n")

main()
