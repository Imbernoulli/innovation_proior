# TIER: strong
import sys

def overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]

def fits(iv, occ):
    for o in occ:
        if overlaps(iv, o):
            return False
    return True

def insert(iv, occ):
    occ.append(iv)
    occ.sort()

def clip_valid(iv, actual):
    s, e = iv
    as_, ae = actual
    return s >= as_ and e <= ae and s < e

def main():
    data = sys.stdin.read().split()
    pos = 0
    def nxt():
        nonlocal pos
        v = data[pos]; pos += 1
        return v
    S = int(nxt()); K = int(nxt()); P = int(nxt()); R = int(nxt()); T = int(nxt())
    sats = []
    for _ in range(S):
        acc = int(nxt()); cap = int(nxt()); ph = int(nxt())
        sats.append((acc, cap, ph))
    stations = []
    for _ in range(K):
        drain = int(nxt()); off = int(nxt()); dur = int(nxt())
        stations.append((drain, off, dur))

    # Local (cycle-0) visibility window for every (satellite, station) pair.
    win = {}
    for i in range(S):
        ph = sats[i][2]
        for k in range(K):
            drain, off, dur = stations[k]
            base = ph + off
            win[(i, k)] = (base, base + dur)

    # Insight: windows recur identically every cycle (period P), so the schedule
    # only has to be solved ONCE on the periodic quotient (a single representative
    # cycle) and then replicated -- this is a conflict-free phase-coloring of
    # (satellite, station) pairs, not a per-cycle re-decision.
    #
    # Step 1: bucket satellites whose phases put them in near-simultaneous contact
    # with the stations (this is where naive scheduling would collide).
    order = sorted(range(S), key=lambda i: sats[i][2])
    dur0 = stations[0][2]
    thresh = max(1, dur0 // 2)
    buckets = []
    cur = []
    last_ph = None
    for i in order:
        ph = sats[i][2]
        if cur and ph - last_ph > thresh:
            buckets.append(cur)
            cur = []
        cur.append(i)
        last_ph = ph
    if cur:
        buckets.append(cur)

    occ = [[] for _ in range(K)]      # occupied local intervals per station
    primary = {}                       # i -> list of (k, start, end)

    for bucket in buckets:
        # Round-robin the contending satellites across DIFFERENT stations
        # instead of piling them all onto the best one.
        groups = [[] for _ in range(K)]
        for idx, i in enumerate(bucket):
            groups[idx % K].append(i)

        for k in range(K):
            members = groups[k]
            if not members:
                continue
            if len(members) == 1:
                i = members[0]
                iv = win[(i, k)]
                if fits(iv, occ[k]):
                    primary.setdefault(i, []).append((k, iv[0], iv[1]))
                    insert(iv, occ[k])
                continue

            # Multiple satellites still contend for the same station: time-slice
            # their common overlap into non-overlapping full-rate sub-slots
            # instead of transmitting simultaneously at quartered rate.
            spans = [win[(i, k)] for i in members]
            lo = max(s for s, e in spans)
            hi = min(e for s, e in spans)
            dur_total = hi - lo
            m = len(members)
            if dur_total >= m:
                chunk = dur_total // m
                for j, i in enumerate(members):
                    s = lo + j * chunk
                    e = s + chunk if j < m - 1 else hi
                    iv = (s, e)
                    actual = win[(i, k)]
                    if clip_valid(iv, actual) and fits(iv, occ[k]):
                        primary.setdefault(i, []).append((k, s, e))
                        insert(iv, occ[k])
            else:
                for i in members:
                    iv = win[(i, k)]
                    if fits(iv, occ[k]):
                        primary.setdefault(i, []).append((k, iv[0], iv[1]))
                        insert(iv, occ[k])

    # Step 2: mop up any additional non-conflicting opportunity a satellite has
    # at OTHER stations (its own windows never self-overlap by construction).
    for i in range(S):
        for k in range(K):
            iv = win[(i, k)]
            if fits(iv, occ[k]):
                primary.setdefault(i, []).append((k, iv[0], iv[1]))
                insert(iv, occ[k])

    # Replicate the cycle-0 phase assignment across all R cycles.
    out = []
    cnt = 0
    for i in range(S):
        for (k, s, e) in primary.get(i, []):
            for c in range(R):
                cs = c * P + s
                ce = c * P + e
                out.append(f"{i} {k} {cs} {ce}")
                cnt += 1
    print(cnt)
    if out:
        sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
