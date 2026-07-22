import sys, random

# ---- tunable constants (balanced so the ladder hits acceptance numbers) ----
C_COLL   = 6      # colliders per hot cluster (all share one home berth)
G_GAP    = 3      # empty berths at the tail of each period
R_REACH  = 14     # reach window: a boat moors within [h, h+R] (circular)
F_BIG    = 220    # frequency of the hot collider (rest of a cluster are light)
W_CLUST  = 2      # absent-query weight on the hot cluster berths (far from the gap)
W_EARLY  = 1      # absent-query weight on the early single berths
W_BASE   = 1      # baseline absent-query weight everywhere else

def build(testId):
    rnd = random.Random(97 * testId + 12345)
    U = 8 + 10 * testId          # single (light) boats per period
    K = 3 + 3 * testId           # number of periods
    Hs = max(2, U // 3)          # early-single high-weight zone length

    period = C_COLL + U + G_GAP
    m = K * period

    homes = []
    fps = []
    a = [W_BASE] * m

    for p in range(K):
        s = p * period
        # hot cluster: C_COLL boats, all home == s. Light first, HEAVY last
        # (so home-order first-fit displaces the heavy one -> present penalty).
        light = [rnd.randint(1, 3) for _ in range(C_COLL - 1)]
        for w in light:
            homes.append(s); fps.append(w)
        homes.append(s); fps.append(F_BIG)   # heavy collider = highest local index
        # single light boats, one per berth just after the cluster
        for j in range(U):
            homes.append(s + C_COLL + j)
            fps.append(1)
        # absent-query weights: heavy on the cluster (which sits FAR, in probe
        # order, from this period's only empties at the tail) and on early singles
        for j in range(C_COLL):
            a[s + j] = W_CLUST
        for j in range(Hs):
            a[s + C_COLL + j] = W_EARLY
        # the tail G_GAP berths get no home -> they are the empties of this period

    n = len(homes)
    out = []
    out.append("%d %d %d" % (m, n, R_REACH))
    for i in range(n):
        out.append("%d %d" % (homes[i], fps[i]))
    out.append(" ".join(str(x) for x in a))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    build(int(sys.argv[1]))
