# TIER: strong
"""The insight: separate nucleation from growth IN TIME (burst nucleation),
instead of tuning a single constant heat level and hoping for the best.

1. Fire exactly ONE hot pulse sized to the hottest level's own kinetic
   capacity (threshold + cap*monomer-cost-per-nucleus). Any smaller wastes
   burst capacity; any bigger just leaves excess monomer sitting above
   threshold, which would re-trigger nucleation on the very next step -- so
   the pulse is SIZED, not maximized. This produces every particle at (almost)
   exactly the same birth time -- a single cohort, which by construction is a
   fixed point of the Ostwald-ripening redistribution rule (mean == every
   particle), so ripening can never broaden it.
2. After the burst, drop to a colder "hold" level whose threshold safely
   exceeds the leftover monomer (no accidental re-nucleation) while keeping
   growth as fast as that safety constraint allows.
3. Growth self-throttles via surfactant coverage, so once the population is
   already this close to monodisperse, the only remaining decisions are WHICH
   surfactant best matches the reachable growth window, and WHEN to fall back
   to the coldest (near-zero-growth) level to stop before overshooting the
   target. Both are small, enumerable choices (<= L holds x T stop-times x S
   surfactants) -- a light combinatorial search over the natural decision
   space the phase-separated schedule creates, re-using the exact scoring
   simulation so the choice is picked by its own true payoff, not a heuristic
   proxy. This is fundamentally different from greedy's single constant-heat
   guess: it exploits the SAME single-burst structural insight across every
   instance, then optimizes only the few genuinely free parameters it leaves
   behind.

Any level whose kinetic pulse cannot even reach its own threshold (a cold
level with an unreachably high bar) is simply skipped by the search.
"""
import math
import sys


def simulate(inst, temp_sched, inject_sched, surf_idx):
    thr = inst['thr']; cap = inst['cap']; gcoef = inst['gcoef']
    v0 = inst['v0']; r0 = inst['r0']
    theta_ripen = inst['theta_ripen']; ripening_rate = inst['ripening_rate']
    bind_rate, p = inst['surf'][surf_idx]

    M = 0.0
    cohorts = []
    for t in range(inst['T']):
        lvl = temp_sched[t]
        M += inject_sched[t]
        if M > thr[lvl]:
            max_possible = math.floor((M - thr[lvl]) / v0)
            n_new = min(max_possible, cap[lvl])
            if n_new > 0:
                M -= n_new * v0
                cohorts.append([n_new, r0, 0.0])
        for c in cohorts:
            throttle = (1.0 - c[2]) ** p
            dr = gcoef[lvl] * throttle
            c[1] += dr
            c[2] = min(1.0, c[2] + bind_rate * (1.0 - c[2]) * dr)
        if M < theta_ripen and cohorts:
            total_count = sum(c[0] for c in cohorts)
            r_mean = sum(c[0] * c[1] for c in cohorts) / total_count
            for c in cohorts:
                delta = ripening_rate * (c[1] - r_mean)
                c[1] = max(0.0, c[1] + delta)

    total = sum(c[0] for c in cohorts)
    if total <= 0:
        return 0.0
    target = inst['target']; disp = inst['disp_limit']
    num = 0.0
    for c in cohorts:
        q = max(0.0, 1.0 - abs(c[1] - target) / disp)
        num += c[0] * q
    return num / total


def solve(inst):
    T = inst['T']; L = inst['L']; S = inst['S']
    thr = inst['thr']; cap = inst['cap']; gcoef = inst['gcoef']; v0 = inst['v0']
    C0 = inst['C0']; max_inject = inst['max_inject']

    best = None  # (F, temp, inject, surf)
    for burst_level in range(L - 1, -1, -1):
        need = thr[burst_level] + cap[burst_level] * v0
        pulse = min(need, max_inject, C0 * 0.9)
        if pulse <= thr[burst_level] + 1e-9:
            continue
        max_possible = math.floor((pulse - thr[burst_level]) / v0)
        n_new = min(max_possible, cap[burst_level])
        if n_new <= 0:
            continue
        M_after = pulse - n_new * v0
        for hold_level in range(L):
            if thr[hold_level] <= M_after + 1e-9:
                continue  # would immediately re-nucleate -- unsafe hold level
            for stop_step in range(1, T + 1):
                # grow at hold_level for steps [1, stop_step), then fall back
                # to the coldest level (near-zero growth) for the remainder --
                # this is the "when to stop" control.
                temp = ([burst_level] + [hold_level] * (stop_step - 1)
                        + [0] * (T - stop_step))
                inject = [pulse] + [0.0] * (T - 1)
                for s in range(S):
                    f = simulate(inst, temp, inject, s)
                    if best is None or f > best[0]:
                        best = (f, temp, inject, s)

    if best is None:
        # No level can even afford its own kinetic pulse (degenerate instance):
        # fall back to holding at the coldest level with no injection at all.
        temp = [0] * T
        inject = [0.0] * T
        best_s, best_f = 0, -1.0
        for s in range(S):
            f = simulate(inst, temp, inject, s)
            if f > best_f:
                best_f, best_s = f, s
        return temp, inject, best_s

    return best[1], best[2], best[3]


def main():
    data = sys.stdin.read().split()
    pos = 0

    def nxt():
        nonlocal pos
        v = data[pos]; pos += 1
        return v

    T = int(nxt()); L = int(nxt()); S = int(nxt())
    r0 = float(nxt()); v0 = float(nxt())
    theta_ripen = float(nxt()); ripening_rate = float(nxt())
    thr, cap, gcoef = [], [], []
    for _ in range(L):
        thr.append(float(nxt())); cap.append(int(nxt())); gcoef.append(float(nxt()))
    surf = []
    for _ in range(S):
        b = float(nxt()); p = float(nxt())
        surf.append((b, p))
    C0 = float(nxt()); max_inject = float(nxt())
    target = float(nxt()); disp_limit = float(nxt())

    inst = dict(T=T, L=L, S=S, r0=r0, v0=v0, theta_ripen=theta_ripen,
                ripening_rate=ripening_rate, thr=thr, cap=cap, gcoef=gcoef,
                surf=surf, C0=C0, max_inject=max_inject, target=target,
                disp_limit=disp_limit)

    temp, inject, surf_idx = solve(inst)

    out = [" ".join(map(str, temp)), " ".join(f"{x:.6f}" for x in inject), str(surf_idx)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
