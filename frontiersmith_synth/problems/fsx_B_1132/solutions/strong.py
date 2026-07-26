# TIER: strong
"""Insight: the true decision variable is not "today's best mix" but a
whole trajectory, because (a) every deviation from the recent adaptation
state M is explicitly taxed (the switch-cost term), and (b) perishable
consignments must be pre-planned for, not reacted to at the last second.

Strategy: search over a SINGLE committed target recipe x* for the whole
horizon (full-horizon foresight -- score a candidate x* by literally
REPLAYING the entire T-day schedule with it, since the whole schedule is
known up front), then follow x* every day EXCEPT for a bounded, proactive
top-up whenever a batch is about to spoil (a planned deviation, capped so
it never wrecks the inhibition threshold). This reliably beats a per-day
reactive optimizer, which pays the switch-cost tax every time the day's
locally-best mix drifts, and which does not defend against spoilage
until it is already the very last moment.

Self-contained (no imports beyond stdlib)."""
import sys
from collections import deque


def read_instance():
    data = sys.stdin.read().split()
    pos = [0]

    def nxt():
        v = data[pos[0]]
        pos[0] += 1
        return v

    T = int(nxt()); K = int(nxt())
    alpha_milli = int(nxt())
    switch_cost_i100 = int(nxt())
    cap = int(nxt())
    c = [int(nxt()) for _ in range(K)]
    s = [[0] * K for _ in range(K)]
    for i in range(K):
        for j in range(i + 1, K):
            v = int(nxt())
            s[i][j] = v; s[j][i] = v
    thr = [int(nxt()) for _ in range(K)]
    pen = [int(nxt()) for _ in range(K)]
    shelf = [int(nxt()) for _ in range(K)]
    M0 = [int(nxt()) / 1000.0 for _ in range(K)]
    arr = [[int(nxt()) for _ in range(K)] for _ in range(T)]
    n_spikes = int(nxt())
    spike = [[None] * K for _ in range(T)]
    for _ in range(n_spikes):
        d = int(nxt()); typ = int(nxt()); amt = int(nxt()); sh = int(nxt())
        if 0 <= d < T and 0 <= typ < K:
            spike[d][typ] = (amt, sh)
    switch_cost = switch_cost_i100 / 100.0
    return dict(T=T, K=K, alpha_milli=alpha_milli, switch_cost=switch_cost, cap=cap,
                c=c, s=s, thr=thr, pen=pen, shelf=shelf, M0=M0, arr=arr, spike=spike)


def quality(K, c, s, thr, pen, x):
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
    q = quality(K, c, s, thr, pen, x)
    l1 = 0.0
    for k in range(K):
        d = x[k] - M[k]
        l1 += d if d >= 0.0 else -d
    net = q - switch_cost * l1
    return net if net > 0.0 else 0.0


def waterfill(K, target, avail, cap):
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


CAP_MARGIN = 0.15
WINDOW = 1
URGENT_MIN = 1.0


def replay(inst, xstar):
    K = inst["K"]; thr = inst["thr"]; shelf = inst["shelf"]; arr = inst["arr"]
    spike = inst["spike"]; cap = inst["cap"]; alpha_milli = inst["alpha_milli"]
    inv = [deque() for _ in range(K)]
    M = list(inst["M0"])
    feed = []
    for t in range(inst["T"]):
        for k in range(K):
            dq = inv[k]
            while dq and (t - dq[0][0]) >= dq[0][2]:
                dq.popleft()
            if arr[t][k] > 0:
                dq.append([t, float(arr[t][k]), shelf[k]])
            if spike[t][k]:
                amt, sh = spike[t][k]
                if amt > 0:
                    dq.append([t, float(amt), sh])
        avail = [sum(b[1] for b in inv[k]) for k in range(K)]

        dleft = [10 ** 9] * K
        for k in range(K):
            if inv[k]:
                dleft[k] = inv[k][0][2] - (t - inv[k][0][0])
        bumped = [k for k in range(K)
                  if dleft[k] <= WINDOW and inv[k] and inv[k][0][1] > URGENT_MIN]

        x = list(xstar)
        if bumped:
            for k in bumped:
                capfrac = thr[k] / 1000.0 + CAP_MARGIN
                if x[k] < capfrac:
                    x[k] = capfrac
            ssum = sum(x)
            x = [v / ssum for v in x]

        row = waterfill(K, x, avail, cap)
        feed.append(row)
        for k in range(K):
            need = row[k]
            dq = inv[k]
            while need > 1e-6 and dq:
                b = dq[0]
                take = need if need < b[1] else b[1]
                b[1] -= take
                need -= take
                if b[1] <= 1e-6:
                    dq.popleft()
        tm = sum(row)
        if tm > 1e-6:
            xr = [row[k] / tm for k in range(K)]
            a = alpha_milli / 1000.0
            M = [(1 - a) * M[k] + a * xr[k] for k in range(K)]
    return feed


def full_score(inst, xstar):
    K = inst["K"]
    feed = replay(inst, xstar)
    M = list(inst["M0"])
    inv = [deque() for _ in range(K)]
    shelf = inst["shelf"]; arr = inst["arr"]; spike = inst["spike"]
    alpha_milli = inst["alpha_milli"]; switch_cost = inst["switch_cost"]
    c = inst["c"]; s = inst["s"]; thr = inst["thr"]; pen = inst["pen"]
    total = 0.0
    for t in range(inst["T"]):
        for k in range(K):
            dq = inv[k]
            while dq and (t - dq[0][0]) >= dq[0][2]:
                dq.popleft()
            if arr[t][k] > 0:
                dq.append([t, float(arr[t][k]), shelf[k]])
            if spike[t][k]:
                amt, sh = spike[t][k]
                if amt > 0:
                    dq.append([t, float(amt), sh])
        row = feed[t]
        tm = sum(row)
        for k in range(K):
            need = row[k]
            dq = inv[k]
            while need > 1e-6 and dq:
                b = dq[0]
                take = need if need < b[1] else b[1]
                b[1] -= take
                need -= take
                if b[1] <= 1e-6:
                    dq.popleft()
        if tm > 1e-9:
            x = [row[k] / tm for k in range(K)]
            net = day_net(K, c, s, thr, pen, switch_cost, x, M)
            total += tm * net
            a = alpha_milli / 1000.0
            M = [(1 - a) * M[k] + a * x[k] for k in range(K)]
    return total


def best_mix(K, score_fn, extra_starts=None):
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
                            break
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


def main():
    inst = read_instance()
    K = inst["K"]; c = inst["c"]
    csum = sum(c)

    def score(x):
        return full_score(inst, x)

    starts = [[c[k] / csum for k in range(K)], [1.0 / K] * K]
    xstar = best_mix(K, score, extra_starts=starts)
    feed = replay(inst, xstar)
    out_lines = [" ".join("%.6f" % v for v in row) for row in feed]
    sys.stdout.write("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main()
