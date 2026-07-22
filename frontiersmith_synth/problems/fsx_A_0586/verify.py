import sys

# ------------------------------------------------------------------
# Checker for downwash-formation-morph (format C, minimization).
#
# Instance:
#   line 1:  N L P W K
#   line 2:  Rbase Rslope_num Rslope_den Hmax Wpen Wmake
#   next N lines: F0[i] = xi yi zi         (drone i start position)
#   next N lines: F1[j] = xj yj zj         (target point j, shuffled)
#
# Participant output: N lines "t_i w_i"
#   t   = permutation of 0..N-1  (drone i is assigned target point t_i)
#   w_i = wave slot in 0..W-1
#
# Cost of a (assignment, schedule) via deterministic replay:
#   * energy      = sum_i |F0[i] - F1[t_i]|^2                (straight moves)
#   * downwash    = # ordered (a,b) cone-ticks over the timeline
#   * makespan    = number of DISTINCT wave slots used
#   cost = energy + Wpen*downwash + Wmake*makespan
#
# Waves used are compacted to sorted order 0..u-1; wave w owns global
# ticks [w*K, (w+1)*K).  During its own wave a drone slides in K equal
# integer sub-steps from start to target; before its wave it waits at
# the start, after its wave it rests at the target.  Coordinates are
# scaled by K during replay so every sampled point is an exact integer.
#
# Downwash predicate (truncated cone below a higher drone), all integer:
#   drone a is above b  (z_a > z_b);  dz = z_a - z_b <= Hmax*K;
#   horizontal_dist^2 <= radius^2  with
#   radius = Rbase*K + (Rslope_num*dz)//Rslope_den.
#
# Score:  B = cost of identity matching (drone i -> target i, all wave 0);
#         Ratio = min(1000, 100*B/F) / 1000.
# ------------------------------------------------------------------

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)

def read_instance(path):
    d = open(path).read().split()
    it = iter(d)
    N = int(next(it)); L = int(next(it)); P = int(next(it))
    W = int(next(it)); K = int(next(it))
    Rbase = int(next(it)); Rsn = int(next(it)); Rsd = int(next(it))
    Hmax = int(next(it)); Wpen = int(next(it)); Wmake = int(next(it))
    F0 = [(int(next(it)), int(next(it)), int(next(it))) for _ in range(N)]
    F1 = [(int(next(it)), int(next(it)), int(next(it))) for _ in range(N)]
    return dict(N=N, L=L, P=P, W=W, K=K, Rbase=Rbase, Rsn=Rsn, Rsd=Rsd,
                Hmax=Hmax, Wpen=Wpen, Wmake=Wmake, F0=F0, F1=F1)

def replay_cost(inst, assign, waves):
    """Deterministic integer replay -> total cost."""
    N = inst["N"]; K = inst["K"]
    F0 = inst["F0"]; F1 = inst["F1"]
    Rbase = inst["Rbase"]; Rsn = inst["Rsn"]; Rsd = inst["Rsd"]
    Hmax = inst["Hmax"]; Wpen = inst["Wpen"]; Wmake = inst["Wmake"]

    p = F0
    q = [F1[assign[i]] for i in range(N)]

    # energy (schedule-independent)
    energy = 0
    for i in range(N):
        dx = p[i][0] - q[i][0]; dy = p[i][1] - q[i][1]; dz = p[i][2] - q[i][2]
        energy += dx * dx + dy * dy + dz * dz

    # compact used wave slots to 0..u-1 in ascending order
    used = sorted(set(waves))
    rank = {w: r for r, w in enumerate(used)}
    u = len(used)
    eo = [rank[waves[i]] for i in range(N)]

    HmaxK = Hmax * K
    RbaseK = Rbase * K

    downwash = 0
    T = u * K
    # precompute scaled start/target
    pS = [(p[i][0] * K, p[i][1] * K, p[i][2] * K) for i in range(N)]
    qS = [(q[i][0] * K, q[i][1] * K, q[i][2] * K) for i in range(N)]
    dvec = [(q[i][0] - p[i][0], q[i][1] - p[i][1], q[i][2] - p[i][2]) for i in range(N)]

    for t in range(T):
        cur = t // K
        sub = (t % K) + 1                 # 1..K
        pos = [None] * N
        for i in range(N):
            if eo[i] < cur:
                pos[i] = qS[i]
            elif eo[i] > cur:
                pos[i] = pS[i]
            else:
                px, py, pz = pS[i]
                dx, dy, dz = dvec[i]
                pos[i] = (px + dx * sub, py + dy * sub, pz + dz * sub)
        # ordered cone check
        for a in range(N):
            xa, ya, za = pos[a]
            for b in range(N):
                if a == b:
                    continue
                zb = pos[b][2]
                if za <= zb:
                    continue
                ddz = za - zb
                if ddz > HmaxK:
                    continue
                dxh = xa - pos[b][0]; dyh = ya - pos[b][1]
                rad = RbaseK + (Rsn * ddz) // Rsd
                if dxh * dxh + dyh * dyh <= rad * rad:
                    downwash += 1

    makespan = u
    return energy + Wpen * downwash + Wmake * makespan

def main():
    inst = read_instance(sys.argv[1])
    N = inst["N"]; W = inst["W"]

    # ---- internal baseline B: identity matching, all wave 0 ----
    B = replay_cost(inst, list(range(N)), [0] * N)
    B = max(1, B)

    # ---- parse participant output strictly ----
    toks = open(sys.argv[2]).read().split()
    if len(toks) < 2 * N:
        fail("too few tokens")
    assign = [0] * N
    waves = [0] * N
    try:
        for i in range(N):
            ts = toks[2 * i]; ws = toks[2 * i + 1]
            # reject non-finite / non-integer explicitly
            if any(c in ts.lower() for c in ("n", "i", ".")) or \
               any(c in ws.lower() for c in ("n", "i", ".")):
                fail("non-integer token")
            t = int(ts); w = int(ws)
            assign[i] = t; waves[i] = w
    except Exception:
        fail("parse error")

    if sorted(assign) != list(range(N)):
        fail("assignment is not a permutation")
    for w in waves:
        if w < 0 or w >= W:
            fail("wave out of range")

    F = replay_cost(inst, assign, waves)
    F = max(1, F)

    sc = min(1000.0, 100.0 * B / F)
    print("B=%d F=%d Ratio: %.6f" % (B, F, sc / 1000.0))

if __name__ == "__main__":
    main()
