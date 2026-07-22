# TIER: strong
# Insight: fault-count and dollar-cost diverge.  Keep Belady's fetch-optimal
# eviction ORDER, but treat the dirty-writeback asymmetry as a second timescale:
# every page that Belady is about to evict wet, or that would be left wet at the
# end, is proactively cleaned in its safe window (after its last write, before it
# leaves).  A clean costs Pc; a wet writeback costs De-Ce > Pc, so every planted
# wet eviction is converted from De to Pc+Ce.  No single-metric eviction rule
# (LRU / plain Belady) finds this schedule.
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

    next_at = [INF]*M
    seen = {}
    for i in range(M-1, -1, -1):
        p = pg[i]
        next_at[i] = seen.get(p, INF)
        seen[p] = i

    resident = {}         # page -> dirty
    last = {}             # page -> last op index
    last_write = {}       # page -> last op index that WROTE it (else absent)
    cleans = []           # (t, page)
    evicts = []           # (t, page)
    for i in range(M):
        p = pg[i]
        if p in resident:
            if isw[i]:
                resident[p] = True
                last_write[p] = i
        else:
            if len(resident) >= k:
                victim = max(resident, key=lambda x: next_at[last[x]])
                # cheaper to clean-then-clean-evict than to pay a wet writeback
                if resident[victim] and Pc + Ce < De:
                    cleans.append((i, victim))
                evicts.append((i, victim))
                del resident[victim]; del last[victim]
                last_write.pop(victim, None)
            resident[p] = bool(isw[i])
            if isw[i]:
                last_write[p] = i
        last[p] = i

    # pages left resident and wet at the end: clean them in their safe window
    if Pc < De - Ce:
        for p, d in resident.items():
            if d:
                lw = last_write.get(p, -1)
                t = lw + 1
                if 0 <= t <= M - 1:
                    cleans.append((t, p))

    actions = []
    for t, p in cleans:
        actions.append("CLEAN %d %d" % (t, p))
    for t, p in evicts:
        actions.append("EVICT %d %d" % (t, p))

    out = [str(len(actions))]
    out.extend(actions)
    sys.stdout.write("\n".join(out) + "\n")

main()
