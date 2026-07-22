# TIER: greedy
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    U = int(next(it)); T = int(next(it))
    next(it); next(it)  # PEN, BUDGET -- unused by this strategy
    C = []; R = []; r = []; D = []
    for _ in range(U):
        C.append(float(next(it))); R.append(float(next(it)))
        r.append(float(next(it))); D.append(int(next(it)))
    demand = [float(next(it)) for _ in range(T)]

    # The obvious textbook move: retrofit the DIRTIEST unit first (it "does
    # the most good"), then the next dirtiest, packed back-to-back starting
    # immediately at t=1 -- purely ranked by emission rate, with no regard
    # for the demand calendar. Clip to the horizon if a unit does not fit.
    order = sorted(range(U), key=lambda i: -R[i])
    schedule = [0] * U
    cursor = 1
    for i in order:
        latest = T - D[i] + 1
        if latest < 1:
            continue  # does not fit at all -- skip retrofitting it
        s = min(cursor, latest)
        s = max(1, s)
        schedule[i] = s
        cursor = s + D[i]

    def cap_rate(i, t):
        s = schedule[i]
        if s == 0:
            return C[i], R[i]
        if s <= t <= s + D[i] - 1:
            return 0.0, 0.0
        if t < s:
            return C[i], R[i]
        return C[i], r[i]

    # Dispatch: merit order (cheapest AVAILABLE rate first) to meet demand
    # EXACTLY every step -- "do the job, nothing fancy". No bonus dispatch:
    # this strategy never notices that surplus clean energy is worth selling.
    out_lines = [" ".join(str(s) for s in schedule)]
    for t in range(1, T + 1):
        avail = []
        for i in range(U):
            cap, rate = cap_rate(i, t)
            if cap > 1e-9:
                avail.append((rate, i, cap))
        avail.sort()
        need = demand[t - 1]
        row = [0.0] * U
        for rate, i, cap in avail:
            if need <= 1e-9:
                break
            use = min(cap, need)
            row[i] = use
            need -= use
        out_lines.append(" ".join("%.4f" % x for x in row))

    sys.stdout.write("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main()
