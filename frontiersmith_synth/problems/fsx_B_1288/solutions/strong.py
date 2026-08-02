# TIER: strong
import sys


def events_evenly(count, start, span, amount):
    if count <= 0:
        return []
    span = max(1, span)
    if count == 1:
        return [(start, amount)]
    return [(start + (j * span) // count, amount) for j in range(count)]


def blocked_flags(events, V, W, A):
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
    ("NONE", 1, 0, True),
    ("BASE", 1, 0, False),
    ("SPLIT", "m", 0, False),
    ("DELAY_LO", 1, "d_lo", False),
    ("DELAY_HI", 1, "d_hi", False),
    ("SPLIT_DELAY_LO", "m", "d_lo", False),
    ("SPLIT_DELAY_HI", "m", "d_hi", False),
]


def wave_amount_through(n0, a0, w0, ts_k, m, d_lo, d_hi, cost_split, cost_delay,
                         value_frac, V, W, A):
    subst = {"m": m, "d_lo": d_lo, "d_hi": d_hi}
    best_net = None
    best_amt = 0.0
    for name, split_k, delay_k, is_none in RECIPES:
        if is_none:
            amount_through, net = 0.0, 0.0
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
            best_amt = amount_through
    return best_amt


def compute_F(V, W, A, data):
    total_loss = 0.0
    total_max = 0.0
    for ts_k in data["ts"]:
        amount_through = wave_amount_through(
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
    toks = sys.stdin.read().split()
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
    data = dict(T=T, n0=n0, a0=a0, w0=w0, K=K, ts=ts, m=m, d_lo=d_lo, d_hi=d_hi,
                cost_split=cost_split, cost_delay=cost_delay, value_frac=value_frac,
                clusters=clusters, c1=c1, c2=c2, p=p, Vmax=Vmax, Wmax=Wmax, Amax=Amax)

    # --- The insight: don't threshold-fit to the observed pattern. Instead pick
    # (V,W,A) so the attacker's CHEAPEST available adaptation costs more than the
    # value it would recover -- while staying as LOOSE as possible to hold down
    # customer friction. We search a small candidate grid whose points are derived
    # from the economics in the input (split/delay knobs, legit-amount percentiles),
    # scoring each candidate with the exact same simulation the checker uses.
    w_candidates = sorted(set([
        w0, w0 + d_lo, w0 + d_hi, max(1, w0 + d_lo - 1), max(1, w0 + d_hi - 1),
        w0 + 1, w0 + 2, min(Wmax, w0 + d_hi + 2), min(Wmax, 2 * (w0 + d_hi)),
    ]))
    w_candidates = [w for w in w_candidates if 1 <= w <= Wmax]

    v_candidates = sorted(set([
        max(0, n0 - 1), n0, n0 + 1, max(0, n0 - 2),
        max(0, n0 * m - 1), n0 * m, max(0, n0 * m - 2), max(0, (n0 * m) // 2),
        max(0, (n0 * m * 3) // 4), Vmax,
    ]))
    v_candidates = [v for v in v_candidates if 0 <= v <= Vmax]

    legit_amounts = sorted(a for (_, _, _, a) in clusters) or [1.0]

    def pct(q):
        idx = min(len(legit_amounts) - 1, max(0, int(round(q * (len(legit_amounts) - 1)))))
        return legit_amounts[idx]

    a_candidates = sorted(set([
        round(a0 - 0.01, 4), round(a0 / m - 0.01, 4), round(a0 * 0.6, 4),
        round(pct(0.5), 4), round(pct(0.75), 4), round(pct(0.9), 4),
        round(pct(0.95) * 1.2, 4), round(Amax, 4),
    ]))
    a_candidates = [a for a in a_candidates if 0 <= a <= Amax]

    best = None
    best_F = None
    for W in w_candidates:
        for V in v_candidates:
            for A in a_candidates:
                F = compute_F(V, W, A, data)
                if best_F is None or F > best_F:
                    best_F = F
                    best = (V, W, A)

    V, W, A = best
    print("%d %d %.6f" % (V, W, A))


if __name__ == "__main__":
    main()
