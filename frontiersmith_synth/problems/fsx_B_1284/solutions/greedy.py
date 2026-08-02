# TIER: greedy
# The obvious professional first move: drop excluded names and scale the SURVIVORS'
# original benchmark weights up proportionally so they sum to 1 again. This preserves
# each survivor's *relative* weight -- it looks principled -- but it is blind to factor
# exposure: when the exclusion set is concentrated in one sector (the trap cases), the
# proportional rescale still shifts the portfolio's sector exposure away from the
# benchmark, because it never looks at sector/size loadings, only weights.
import sys


def main():
    toks = sys.stdin.read().split()
    p = 0

    def nxt():
        nonlocal p
        v = toks[p]
        p += 1
        return v

    N = int(nxt())
    S = int(nxt())
    K = S + 1
    T = float(nxt())
    for _ in range(K * K):
        nxt()

    w = []
    esg = []
    cap = []
    for _ in range(N):
        nxt()          # sector
        nxt()          # size
        e = float(nxt())
        wi = float(nxt())
        ci = float(nxt())
        nxt()          # d
        esg.append(e)
        w.append(wi)
        cap.append(ci)

    eligible = [e >= T for e in esg]
    survivor_w = sum(w[i] for i in range(N) if eligible[i])

    x = [0.0] * N
    if survivor_w > 1e-15:
        scale = 1.0 / survivor_w
        for i in range(N):
            if eligible[i]:
                xi = w[i] * scale
                if xi > cap[i]:
                    xi = cap[i]   # respect the capacity cap if the naive rescale overshoots
                x[i] = xi

    # any shortfall from capacity clipping is dumped pro-rata onto the remaining
    # eligible names that still have headroom (still purely weight-driven, no factor logic)
    shortfall = 1.0 - sum(x)
    guard = 0
    while shortfall > 1e-9 and guard < 50:
        room = [(i, cap[i] - x[i]) for i in range(N) if eligible[i] and cap[i] - x[i] > 1e-12]
        if not room:
            break
        room_w = sum(w[i] for i, _ in room)
        if room_w <= 1e-15:
            break
        add_total = 0.0
        for i, r in room:
            add = min(r, shortfall * (w[i] / room_w))
            x[i] += add
            add_total += add
        shortfall -= add_total
        guard += 1

    out = ["%.10f" % v for v in x]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
