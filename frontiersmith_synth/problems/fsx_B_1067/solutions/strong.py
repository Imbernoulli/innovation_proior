# TIER: strong
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    U = int(next(it)); T = int(next(it))
    PEN = float(next(it)); BUDGET = float(next(it))  # PEN unused: bonus stops at BUDGET, not priced
    C = []; R = []; r = []; D = []
    for _ in range(U):
        C.append(float(next(it))); R.append(float(next(it)))
        r.append(float(next(it))); D.append(int(next(it)))
    demand = [float(next(it)) for _ in range(T)]

    total_cap = sum(C)

    # ---- Step 1: schedule retrofits by DISPATCHING the "who goes dark when"
    # decision against the demand calendar, instead of a static dirty-first
    # rank. Process units in order of cleanup payoff (R-r)*C, but for EACH
    # unit choose the EARLIEST offline window that causes zero projected
    # coverage shortfall against everyone already committed offline (falling
    # back to the least-bad option if no safe window exists) -- interleaving
    # outages with demand troughs, and converting units in PARALLEL whenever
    # a trough is wide/deep enough to absorb it, instead of packing them one
    # at a time blindly from t=1.
    order = sorted(range(U), key=lambda i: -(R[i] - r[i]) * C[i])
    schedule = [0] * U
    offline_cap = [0.0] * (T + 2)  # 1-indexed; capacity already committed dark at t

    for i in order:
        best = None  # (shortfall, start)
        for s in range(1, T - D[i] + 2):
            shortfall = 0.0
            for t in range(s, s + D[i]):
                avail = total_cap - C[i] - offline_cap[t]
                if avail < demand[t - 1]:
                    shortfall += demand[t - 1] - avail
            key = (shortfall, s)
            if best is None or key < best:
                best = key
        s_best = best[1]
        schedule[i] = s_best
        for t in range(s_best, s_best + D[i]):
            offline_cap[t] += C[i]

    def cap_rate(i, t):
        s = schedule[i]
        if s == 0:
            return C[i], R[i]
        if s <= t <= s + D[i] - 1:
            return 0.0, 0.0
        if t < s:
            return C[i], R[i]
        return C[i], r[i]

    # ---- Step 2: dispatch. Cover mandatory demand via cheapest-rate-first
    # merit order (minimizes forced emissions), then treat the BUDGET as a
    # depleting resource: spend the remaining headroom on bonus dispatch
    # from the cheapest still-idle capacity first, permanently stopping the
    # instant the budget line is reached (a store that only depletes, never
    # refills -- so it must be rationed in the order that pays off most).
    cum_em = 0.0
    out_lines = [" ".join(str(s) for s in schedule)]
    for t in range(1, T + 1):
        specs = []
        for i in range(U):
            cap, rate = cap_rate(i, t)
            if cap > 1e-9:
                specs.append((rate, i, cap))
        specs.sort()

        row = [0.0] * U
        need = demand[t - 1]
        leftover = []
        for rate, i, cap in specs:
            if need <= 1e-9:
                leftover.append((rate, i, cap))
                continue
            use = min(cap, need)
            row[i] = use
            cum_em += use * rate
            need -= use
            if cap - use > 1e-9:
                leftover.append((rate, i, cap - use))

        for rate, i, lc in leftover:
            if cum_em >= BUDGET:
                break
            room = (BUDGET - cum_em) / rate if rate > 1e-9 else lc
            take = min(lc, room)
            if take > 0:
                row[i] += take
                cum_em += take * rate

        out_lines.append(" ".join("%.4f" % x for x in row))

    sys.stdout.write("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main()
