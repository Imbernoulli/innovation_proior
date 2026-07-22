import sys, math

# Deterministic scorer for the boutique-clearance markdown-wait game.
#   in  : T N K s PMAX p0 ; then N lines "v a h D"
#   out : exactly T integers (prices day 1..T), each in [0,PMAX]
# Objective: revenue produced by the forward-looking buyer best-response replay.
# Baseline B: revenue of the constant-price schedule p_t = p0 (the checker's own
# trivial construction).  Ratio = min(1000,100*F/B)/1000.

BASELINE_SCALE = 0.40  # knob to keep score headroom (trivial stays in [0.03,0.35])
EPS = 1e-9

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)

def simulate(prices, buyers, K, s, T):
    """Replay buyers in order; return revenue. prices is 1-indexed conceptually via
    prices[t-1]. buyers: list of (v,a,h,D)."""
    sold = [0] * (T + 1)
    stock = K
    rev = 0
    powcache = {}
    for (v, a, h, D) in buyers:
        if stock <= 0:
            break
        pw = powcache.get(D)
        if pw is None:
            d = D / 1000.0
            pw = [1.0] * (T + 1)
            for t in range(1, T + 1):
                pw[t] = pw[t - 1] * d
            powcache[D] = pw
        best_t = -1
        best_s = EPS            # strictly-positive surplus required to buy
        for t in range(a, h + 1):
            if sold[t] >= s:
                continue
            surplus = pw[t] * (v - prices[t - 1])
            if surplus > best_s:      # strict '>' keeps the earliest day on ties
                best_s = surplus
                best_t = t
        if best_t >= 0:
            rev += prices[best_t - 1]
            sold[best_t] += 1
            stock -= 1
    return rev

def main():
    try:
        inp = open(sys.argv[1]).read().split()
        it = iter(inp)
        T = int(next(it)); N = int(next(it)); K = int(next(it))
        s = int(next(it)); PMAX = int(next(it)); p0 = int(next(it))
        buyers = []
        for _ in range(N):
            v = int(next(it)); a = int(next(it)); h = int(next(it)); D = int(next(it))
            buyers.append((v, a, h, D))
    except Exception:
        fail("bad input")

    # ---- parse participant output: exactly T finite integers in [0,PMAX] ----
    raw = open(sys.argv[2]).read().split()
    if len(raw) != T:
        fail("need exactly %d prices, got %d" % (T, len(raw)))
    prices = []
    for tok in raw:
        try:
            f = float(tok)
        except ValueError:
            fail("non-numeric price %r" % tok)
        if not math.isfinite(f):
            fail("non-finite price")
        # require integer value
        if abs(f - round(f)) > 1e-9:
            fail("price not integer %r" % tok)
        p = int(round(f))
        if p < 0 or p > PMAX:
            fail("price %d out of [0,%d]" % (p, PMAX))
        prices.append(p)

    F = simulate(prices, buyers, K, s, T)

    base_sched = [p0] * T
    B_raw = simulate(base_sched, buyers, K, s, T)
    B = max(1e-9, BASELINE_SCALE * B_raw)

    sc = min(1000.0, 100.0 * F / B)
    print("F=%d B=%.3f Ratio: %.6f" % (F, B, sc / 1000.0))

if __name__ == "__main__":
    main()
