# TIER: trivial
# LRU eviction, no proactive cleans -- reproduces the checker's internal
# baseline exactly, so it scores the calibrated ~0.1.
import sys

def main():
    tok = sys.stdin.buffer.read().split()
    it = iter(tok)
    k = int(next(it)); F = int(next(it)); Ce = int(next(it))
    De = int(next(it)); Pc = int(next(it)); M = int(next(it))
    isw = [0]*M; pg = [0]*M
    for i in range(M):
        t = next(it); pg[i] = int(next(it))
        isw[i] = 1 if t == b'W' else 0

    resident = {}    # page -> dirty
    last = {}
    actions = []
    for i in range(M):
        p = pg[i]
        if p in resident:
            if isw[i]:
                resident[p] = True
        else:
            if len(resident) >= k:
                victim = min(resident, key=lambda x: last[x])
                actions.append("EVICT %d %d" % (i, victim))
                del resident[victim]; del last[victim]
            resident[p] = bool(isw[i])
        last[p] = i

    out = [str(len(actions))]
    out.extend(actions)
    sys.stdout.write("\n".join(out) + "\n")

main()
