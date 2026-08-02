import sys, math


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def parse_input(path):
    toks = open(path).read().split()
    it = iter(toks)

    def nx():
        return next(it)

    T = int(nx())
    n0 = int(nx()); a0 = float(nx()); w0 = int(nx()); K = int(nx())
    ts = [int(nx()) for _ in range(K)]
    m = int(nx()); d_lo = int(nx()); d_hi = int(nx())
    cost_split = float(nx()); cost_delay = float(nx()); value_frac = float(nx())
    Lc = int(nx())
    clusters = []
    for _ in range(Lc):
        nl = int(nx()); tl = int(nx()); wl = int(nx()); al = float(nx())
        clusters.append((nl, tl, wl, al))
    c1 = float(nx()); c2 = float(nx()); p = float(nx())
    Vmax = int(nx()); Wmax = int(nx()); Amax = float(nx())
    return dict(T=T, n0=n0, a0=a0, w0=w0, K=K, ts=ts, m=m, d_lo=d_lo, d_hi=d_hi,
                cost_split=cost_split, cost_delay=cost_delay, value_frac=value_frac,
                clusters=clusters, c1=c1, c2=c2, p=p, Vmax=Vmax, Wmax=Wmax, Amax=Amax)


def events_evenly(count, start, span, amount):
    """`count` transactions of `amount` each, spread evenly over [start, start+span)."""
    if count <= 0:
        return []
    span = max(1, span)
    if count == 1:
        return [(start, amount)]
    return [(start + (j * span) // count, amount) for j in range(count)]


def blocked_flags(events, V, W, A):
    """events: list of (t, amount). A transaction is blocked if its amount exceeds A,
    OR if it is one of MORE than V transactions of THIS SAME entity-local stream that
    fall in the trailing window (t-W, t]. Returns (sorted_events, flags)."""
    ev = sorted(events, key=lambda x: x[0])
    n = len(ev)
    times = [e[0] for e in ev]
    flags = [False] * n
    j = 0
    for idx in range(n):
        t, amt = ev[idx]
        lo = t - W + 1
        while j < idx and times[j] < lo:
            j += 1
        cnt = idx - j + 1
        flags[idx] = (amt > A + 1e-9) or (cnt > V)
    return ev, flags


RECIPES = [
    # name, split_factor, delay, is_none
    ("NONE", 1, 0, True),
    ("BASE", 1, 0, False),
    ("SPLIT", "m", 0, False),
    ("DELAY_LO", 1, "d_lo", False),
    ("DELAY_HI", 1, "d_hi", False),
    ("SPLIT_DELAY_LO", "m", "d_lo", False),
    ("SPLIT_DELAY_HI", "m", "d_hi", False),
]


def wave_best_response(n0, a0, w0, ts_k, m, d_lo, d_hi, cost_split, cost_delay,
                        value_frac, V, W, A):
    """The attacker deterministically enumerates its 7 candidate recipes for this wave
    and adopts whichever maximizes ITS OWN net (value captured minus adaptation cost)
    against the exact submitted rules. Returns (recipe_name, amount_through, cost)."""
    subst = {"m": m, "d_lo": d_lo, "d_hi": d_hi}
    best = None
    best_net = None
    for name, split_k, delay_k, is_none in RECIPES:
        if is_none:
            amount_through, cost, net = 0.0, 0.0, 0.0
        else:
            split = subst.get(split_k, split_k)
            delay = subst.get(delay_k, delay_k)
            C = n0 * split
            amt = a0 / split
            span = w0 + delay
            evs = events_evenly(C, ts_k, span, amt)
            _, flags = blocked_flags(evs, V, W, A)
            amount_through = sum(a for (_, a), f in zip(evs, flags) if not f)
            cost = cost_split * n0 * (split - 1) + cost_delay * delay
            net = value_frac * amount_through - cost
        if best_net is None or net > best_net + 1e-9:
            best_net = net
            best = (name, amount_through, cost)
    return best


def compute_F(V, W, A, data):
    total_loss = 0.0
    total_max = 0.0
    for ts_k in data["ts"]:
        _, amount_through, _ = wave_best_response(
            data["n0"], data["a0"], data["w0"], ts_k, data["m"], data["d_lo"], data["d_hi"],
            data["cost_split"], data["cost_delay"], data["value_frac"], V, W, A)
        total_loss += data["value_frac"] * amount_through
        total_max += data["value_frac"] * data["n0"] * data["a0"]
    fraud_prevented = total_max - total_loss

    blocked_amount = 0.0
    blocked_count = 0
    for (nl, tl, wl, al) in data["clusters"]:
        evs = events_evenly(nl, tl, wl, al)
        _, flags = blocked_flags(evs, V, W, A)
        for (_, a2), f in zip(evs, flags):
            if f:
                blocked_amount += a2
                blocked_count += 1
    friction = data["c1"] * blocked_amount + data["c2"] * (blocked_count ** data["p"])
    return fraud_prevented - friction


def main():
    try:
        data = parse_input(sys.argv[1])
    except Exception as e:
        fail("bad input: %r" % (e,))

    # internal trivial baseline: "block every transaction" (V=0,W=1,A=0) -- maximal,
    # cost-unaware defense that needs no awareness of the attacker's adaptation
    # economics or of which legitimate clusters resemble the attack. gen.py
    # calibrates c1/c2 so this always costs a substantial, bounded fraction of the
    # fraud value at risk, keeping B positive and comfortably below what a
    # selective, cost-aware rule set can achieve.
    B = compute_F(0, 1, 0.0, data)
    B = max(B, 1e-6)

    try:
        toks = open(sys.argv[2]).read().split()
    except Exception as e:
        fail("cannot read output: %r" % (e,))
    if len(toks) != 3:
        fail("expected exactly 3 tokens 'V W A', got %d" % len(toks))
    try:
        V = int(toks[0])
        Wv = int(toks[1])
        A = float(toks[2])
    except Exception:
        fail("V and W must be integers, A a number")

    if not (math.isfinite(V) and math.isfinite(Wv) and math.isfinite(A)):
        fail("non-finite value")
    if not (0 <= V <= data["Vmax"]):
        fail("V=%d out of range [0,%d]" % (V, data["Vmax"]))
    if not (1 <= Wv <= data["Wmax"]):
        fail("W=%d out of range [1,%d]" % (Wv, data["Wmax"]))
    if not (0 <= A <= data["Amax"]):
        fail("A=%.4f out of range [0,%.4f]" % (A, data["Amax"]))

    F = compute_F(V, Wv, A, data)
    Fc = max(0.0, F)
    sc = min(1000.0, 100.0 * Fc / max(1e-9, B))
    ratio = sc / 1000.0
    print("F=%.6f B=%.6f Ratio: %.6f" % (F, B, ratio))


if __name__ == "__main__":
    main()
