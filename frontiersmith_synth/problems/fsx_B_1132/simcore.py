"""simcore.py -- shared deterministic simulation core for the
envelope-mix-inhibition-feeding / "biogas digester fed from spoiling
stockpiles" problem. Imported by BOTH gen.py and verify.py (never shipped
into solutions/, which run sandboxed and must stay self-contained).

Model (all three spec mechanisms compose into ONE per-day objective):
  - substrate-inhibition-thresholds: quality(x) = [c.x + synergy cross
    terms], multiplicatively crushed once any component's fraction
    exceeds its own threshold.
  - adaptation-memory-state: an EWMA memory M drifts toward whatever you
    actually fed; today's realized value is quality(x) MINUS an explicit
    switch cost proportional to how far x sits from M (L1 distance),
    clipped at 0 (a reckless jump earns nothing, never negative total).
  - perishable-feedstock-queue: a FIFO per-type inventory with a
    baseline shelf life PLUS optional separate fast-spoiling "spike"
    consignments carrying their own (shorter) shelf life, so an
    everyday delivery is not made fragile just because a rare bulk
    glut of the same type is.

All state is plain Python floats/ints; no randomness, no wall-clock, no
dict/set-order dependence (every loop is over range(K) / range(T)).
"""
from collections import deque

EPS = 1e-4  # tolerance vs 6-decimal-formatted output summed over up to K=6 terms


def quality(K, c, s, thr, pen, x):
    """rate(x) * inhibition(x): the pure chemistry value of mix x (a
    length-K list of fractions summing to ~1), IGNORING the adaptation
    state entirely."""
    rate = 0.0
    for k in range(K):
        rate += c[k] * x[k]
    for i in range(K):
        xi = x[i]
        if xi <= 0.0:
            continue
        for j in range(i + 1, K):
            rate += s[i][j] * xi * x[j]
    if rate < 0.0:
        rate = 0.0
    inh = 1.0
    for k in range(K):
        thr_frac = thr[k] / 1000.0
        if x[k] > thr_frac:
            factor = 1.0 - pen[k] * (x[k] - thr_frac)
            if factor < 0.0:
                factor = 0.0
            inh *= factor
    return rate * inh


def day_net(K, c, s, thr, pen, switch_cost, x, M):
    """Per-unit-mass realized value: chemistry quality minus an explicit
    switching cost for deviating from the current adaptation state M
    (L1 distance), floored at 0."""
    q = quality(K, c, s, thr, pen, x)
    l1 = 0.0
    for k in range(K):
        d = x[k] - M[k]
        l1 += d if d >= 0.0 else -d
    net = q - switch_cost * l1
    return net if net > 0.0 else 0.0


def new_inventory(K):
    return [deque() for _ in range(K)]


def expire_and_arrive(K, inv, shelf, arr_row, day, spike_row=None):
    """Mutates inv in place: drop batches whose usable window has ended,
    then append today's arrivals. Regular arrivals (arr_row) use the
    type's baseline shelf[k]; an optional spike_row (list of (amount,
    shelf_life) or None per type) appends a SEPARATE batch with its own
    (typically shorter) shelf life -- a fast-spoiling bulk consignment,
    without shortening the type's everyday shelf life. Batches carry
    their own shelf life as [arrival_day, remaining_amount, shelf_life].
    Returns per-type available totals."""
    for k in range(K):
        dq = inv[k]
        while dq and (day - dq[0][0]) >= dq[0][2]:
            dq.popleft()
        if arr_row[k] > 0:
            dq.append([day, float(arr_row[k]), shelf[k]])
        if spike_row is not None and spike_row[k]:
            amt, sh = spike_row[k]
            if amt > 0:
                dq.append([day, float(amt), sh])
    return [sum(b[1] for b in inv[k]) for k in range(K)]


def consume_fifo(K, inv, feed_row):
    """Mutates inv in place, consuming oldest batches first. Caller must
    have already verified feed_row[k] <= available[k] (+EPS)."""
    for k in range(K):
        need = feed_row[k]
        dq = inv[k]
        while need > EPS and dq:
            b = dq[0]
            take = need if need < b[1] else b[1]
            b[1] -= take
            need -= take
            if b[1] <= EPS:
                dq.popleft()


def days_left(K, inv, day):
    """For each type, how many more days (including today) the OLDEST
    surviving batch remains usable (using THAT BATCH's own shelf life);
    a large sentinel if the queue is empty (no pressure)."""
    out = [10 ** 9] * K
    for k in range(K):
        dq = inv[k]
        if dq:
            out[k] = dq[0][2] - (day - dq[0][0])
    return out


def simulate(K, c, s, thr, pen, shelf, alpha_milli, switch_cost, cap, M0, arr, feed, spike=None):
    """Full deterministic simulation of a T-day feed plan.
    arr, feed: T x K (lists of lists), arr entries are ints/floats >=0.
    spike (optional): T x K, each entry None or (amount, shelf_life) for
    a separate fast-spoiling consignment delivered that day.
    Returns (total_gas, ok, reason). ok=False -> caller must score 0."""
    T = len(arr)
    inv = new_inventory(K)
    M = list(M0)
    total_gas = 0.0
    for t in range(T):
        spike_row = spike[t] if spike is not None else None
        avail = expire_and_arrive(K, inv, shelf, arr[t], t, spike_row)
        row = feed[t]
        total_mass = 0.0
        for k in range(K):
            f = row[k]
            if f != f or f in (float("inf"), float("-inf")):
                return 0.0, False, f"non-finite feed day {t} type {k}"
            if f < -EPS:
                return 0.0, False, f"negative feed day {t} type {k}"
            if f > avail[k] + EPS:
                return 0.0, False, (
                    f"feed exceeds available inventory day {t} type {k}: "
                    f"{f:.6f} > {avail[k]:.6f}")
            total_mass += f
        if total_mass > cap + EPS:
            return 0.0, False, f"total feed day {t} exceeds capacity: {total_mass:.6f} > {cap:.6f}"
        consume_fifo(K, inv, row)
        if total_mass > EPS:
            x = [row[k] / total_mass for k in range(K)]
            net = day_net(K, c, s, thr, pen, switch_cost, x, M)
            total_gas += total_mass * net
            a = alpha_milli / 1000.0
            M = [(1.0 - a) * M[k] + a * x[k] for k in range(K)]
        # else: no feed today -> gas 0, M unchanged
    return total_gas, True, "ok"


def baseline_feed(K, T, shelf, arr, spike, cap):
    """The checker's own reference construction: feed today's freshly
    AVAILABLE stock in ITS OWN arrival ratio (no planning, no synergy
    optimization, oblivious to inhibition/adaptation), scaled down only
    as far as needed to respect the digester's fixed daily capacity.
    Always feasible by construction."""
    inv = new_inventory(K)
    feed = []
    for t in range(T):
        spike_row = spike[t] if spike is not None else None
        avail = expire_and_arrive(K, inv, shelf, arr[t], t, spike_row)
        tot = sum(avail)
        if tot <= EPS:
            row = [0.0] * K
        else:
            f = min(1.0, cap / tot)
            row = [v * f for v in avail]
        feed.append(row)
        consume_fifo(K, inv, row)
    return feed


def waterfill(K, target, avail, cap):
    """Allocate up to `cap` total mass across K types, following the
    relative proportions in `target` (a nonneg length-K weight vector) as
    closely as possible, but never exceeding `avail[k]` for any type:
    whichever type's availability binds gets saturated at avail[k], and
    the freed-up remaining capacity is reallocated to the other types in
    proportion to their target weight (classic water-filling). Returns a
    length-K feed vector with sum <= cap and feed[k] <= avail[k]."""
    active = [k for k in range(K) if target[k] > 1e-12 and avail[k] > 1e-12]
    feed = [0.0] * K
    remaining = cap
    while active and remaining > 1e-9:
        wsum = sum(target[k] for k in active)
        if wsum <= 1e-12:
            break
        saturate = []
        for k in active:
            proposed = remaining * (target[k] / wsum)
            if proposed >= avail[k] - feed[k] - 1e-9:
                saturate.append(k)
        if saturate:
            for k in saturate:
                take = avail[k] - feed[k]
                if take < 0:
                    take = 0.0
                feed[k] += take
                remaining -= take
            active = [k for k in active if k not in saturate]
        else:
            for k in active:
                feed[k] += remaining * (target[k] / wsum)
            remaining = 0.0
    return feed


def best_static_mix(K, score_fn, extra_starts=None):
    """Deterministic (no RNG) coordinate hill-climb over the simplex,
    from several fixed starting points, decreasing step size. Returns the
    best mix found as a length-K list summing to 1.0."""
    starts = [[1.0 / K] * K]
    for k in range(K):
        v = [0.15 / (K - 1) if K > 1 else 0.0] * K
        v[k] = 0.85 if K > 1 else 1.0
        ssum = sum(v)
        starts.append([e / ssum for e in v])
    if extra_starts:
        starts.extend(extra_starts)
    best_x, best_v = None, None
    for x0 in starts:
        x = list(x0)
        cur = score_fn(x)
        step = 0.16
        while step > 1e-4:
            improved = True
            while improved:
                improved = False
                for i in range(K):
                    for j in range(K):
                        if i == j:
                            continue
                        if x[i] < step - 1e-12:
                            break  # x[i] now too small to give away `step`; try next i
                        nx = list(x)
                        nx[i] -= step
                        nx[j] += step
                        v = score_fn(nx)
                        if v > cur + 1e-12:
                            x, cur = nx, v
                            improved = True
            step *= 0.5
        if best_v is None or cur > best_v:
            best_v, best_x = cur, x
    return best_x
