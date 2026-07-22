# TIER: strong
# The insight: under a MIN-over-scenarios objective with void/shortfall
# penalties, the battery's real product is OPTIONALITY.  Stage 1 commits only
# in scenario-consensus hours (zero scenarios void them) so the committed
# trajectory executes identically in every scenario -- zero penalty exposure
# -- and keeps state-of-charge buffers (stored options) on both sides.
# Stage 2 evaluates the TRUE worst-scenario profit and hedges it: drop
# commitments the worst scenario voids, then selectively add exposure in
# penalty-adjusted hours only where the worst-scenario math (not the mean)
# says the trade pays.
import sys


def read_instance():
    tok = sys.stdin.read().split()
    pos = 0
    T = int(tok[pos]); S = int(tok[pos + 1]); pos += 2
    Emax = float(tok[pos]); Pmax = float(tok[pos + 1]); eta = float(tok[pos + 2])
    E0 = float(tok[pos + 3]); lam = float(tok[pos + 4]); mu = float(tok[pos + 5])
    pos += 6
    rho, prices, outages = [], [], []
    for _ in range(S):
        rho.append(float(tok[pos])); pos += 1
        prices.append([float(v) for v in tok[pos:pos + T]]); pos += T
        outages.append([int(v) for v in tok[pos:pos + T]]); pos += T
    return T, S, Emax, Pmax, eta, E0, lam, mu, rho, prices, outages


T, S, Emax, Pmax, eta, E0, lam, mu, rho, prices, outages = read_instance()
void_n = [sum(outages[s][t] for s in range(S)) for t in range(T)]
adj_dis = [sum((-lam if outages[s][t] else prices[s][t]) for s in range(S)) / S
           for t in range(T)]
adj_ch = [sum((lam if outages[s][t] else prices[s][t]) for s in range(S)) / S
          for t in range(T)]


def simulate(q):
    """(worst profit, committed feasible) exactly as the checker scores it."""
    E = E0
    for t in range(T):
        v = q[t]
        E = E - v if v > 0 else E + eta * (-v)
        if E < -1e-9 or E > Emax + 1e-9:
            return None, False
    worst = None
    for s in range(S):
        p, o = prices[s], outages[s]
        E = E0
        prof = 0.0
        for t in range(T):
            v = q[t]
            if o[t]:
                prof -= lam * (v if v >= 0 else -v)
            elif v > 0:
                d = v if v <= E else E
                if d < 0:
                    d = 0.0
                prof += p[t] * d - mu * (v - d)
                E -= d
            elif v < 0:
                head = (Emax - E) / eta
                if head < 0:
                    head = 0.0
                a = -v if -v <= head else head
                prof -= p[t] * a + mu * ((-v) - a)
                E += eta * a
        prof += rho[s] * E
        if worst is None or prof < worst:
            worst = prof
    return worst, True


def committed_soc(q):
    E = E0
    out = []
    for t in range(T):
        v = q[t]
        E = E - v if v > 0 else E + eta * (-v)
        out.append(E)
    return out


# ---------- stage 1: consensus-only dispatch with SoC buffers ----------
free_hours = [t for t in range(T) if void_n[t] == 0]
if len(free_hours) < 10:
    free_hours = [t for t in range(T) if void_n[t] <= 1]
loB, hiB = 0.05 * Emax, 0.95 * Emax
q = [0.0] * T
E = E0
dis_vals = sorted((adj_dis[t] for t in free_hours))
ch_vals = sorted((adj_ch[t] for t in free_hours))
if dis_vals and ch_vals:
    dth = dis_vals[int(0.55 * (len(dis_vals) - 1))]
    cth = ch_vals[int(0.45 * (len(ch_vals) - 1))]
    for t in free_hours:
        if adj_dis[t] >= dth and adj_dis[t] > 0 and E > loB:
            v = min(Pmax, E - loB)
            if v > 1e-9:
                q[t] = v
                E -= v
        elif adj_ch[t] <= cth and E < hiB:
            v = min(Pmax, (hiB - E) / eta)
            if v > 1e-9:
                q[t] = -v
                E += eta * v

best_q = q[:]
best_F, feas = simulate(best_q)
if not feas:
    best_q = [0.0] * T
    best_F, _ = simulate(best_q)
zero_F, _ = simulate([0.0] * T)
if zero_F > best_F:
    best_q, best_F = [0.0] * T, zero_F


def try_edit(q2):
    global best_q, best_F
    F2, ok = simulate(q2)
    if ok and F2 is not None and F2 > best_F + 1e-9:
        best_q, best_F = q2, F2
        return True
    return False


# ---------- stage 2: hedge the worst scenario, then re-open exposure ----------
for _round in range(3):
    improved = False
    soc = committed_soc(best_q)
    # suffix min / prefix max of committed SoC for feasibility-preserving edits
    suf_min = [0.0] * T
    m = float("inf")
    for t in range(T - 1, -1, -1):
        m = min(m, soc[t])
        suf_min[t] = m
    pre_max = [0.0] * T
    m = -float("inf")
    for t in range(T):
        m = max(m, soc[t])
        pre_max[t] = m

    cands = []  # (heuristic gain, hour, kind)
    for t in range(T):
        v = best_q[t]
        if v != 0.0 and void_n[t] > 0:
            # worst-scenario hedge: shrinking this commitment saves void penalties
            gain = lam * abs(v) * (void_n[t] / S) - max(0.0, adj_dis[t]) * max(0.0, v)
            cands.append((gain + 1e-6, t, "shrink"))
        if v == 0.0:
            cap = min(Pmax, suf_min[t])
            if cap > 1e-9 and adj_dis[t] > 0:
                cands.append((adj_dis[t] * cap, t, "dis"))
            room = min(Pmax, (Emax - pre_max[t]) / eta)
            if room > 1e-9:
                cands.append((-adj_ch[t] * room + 1e-9, t, "ch"))
    cands.sort(reverse=True)
    for gain, t, kind in cands[:48]:
        q2 = best_q[:]
        if kind == "shrink":
            q2[t] = 0.0
            if try_edit(q2):
                improved = True
                continue
            q2[t] = best_q[t] * 0.5
            if try_edit(q2):
                improved = True
        elif kind == "dis":
            q2[t] = min(Pmax, suf_min[t])
            if try_edit(q2):
                improved = True
        else:
            q2[t] = -min(Pmax, (Emax - pre_max[t]) / eta)
            if try_edit(q2):
                improved = True
    if not improved:
        break

sys.stdout.write(" ".join(repr(v) for v in best_q) + "\n")
