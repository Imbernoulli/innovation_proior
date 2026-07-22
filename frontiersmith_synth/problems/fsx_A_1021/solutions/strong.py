# TIER: strong
# Insight: degradation is c * max(0,theta-THETA_SAFE)^2 * soc, and soc is exactly 0
# until a bus starts charging -- so waiting is FREE cooling, and once charging starts
# a GENTLER power keeps the steady-state temperature low. Both levers (how long to
# wait, how gently to charge) are individually free for one bus in isolation -- but a
# gentler power means a longer stay on a shared berth, and only C berths exist, so
# gentleness is a scarce resource that must be RATIONED, not handed out uniformly
# (the "greedy" tier's flat fixed power).
#
# We run a BEST-IMPROVEMENT EXCHANGE local search: start every bus at the always-
# feasible full-power-immediate fallback, then repeatedly find the SINGLE (bus,
# one-power-step-gentler) move with the largest degradation payoff that is still
# feasible against the shared berth timeline right now, and apply only that one move
# -- re-scanning from scratch afterwards. This is an exchange argument over a shared
# resource (each step is a real, capacity-checked trade of berth-time for coolness,
# always taking the globally best trade available), not a fixed priority order or a
# single aggregate-budget approximation: no bus can hoard gentleness the moment a
# GLOBALLY better use of that same berth-time exists elsewhere. Every accepted move
# strictly lowers total degradation and never breaks feasibility, so the result is
# always at least as good as the fallback floor and always feasible.
import sys, math

AMBIENT = 25
THETA_SAFE = 35
RHO = 0.82
HEAT = 2.0
# Beyond this many idle ticks, Newton cooling has already erased ~95% of the excess
# heat (RHO**K < 0.05) -- further delay is nearly worthless, so we cap how far into
# the night any single bus reaches for "free cooling" when picking a target slot.
K_COOL = max(1, math.ceil(math.log(0.05) / math.log(RHO)))


def fcfs_schedule(buses, D, C, power_level):
    n = len(buses)
    occ = [0] * (D + 1)
    starts = [None] * n
    order = sorted(range(n), key=lambda i: (buses[i][0], i))
    for i in order:
        a_i, e_i = buses[i]
        L = -(-e_i // power_level) if e_i > 0 else 0
        if L == 0:
            starts[i] = a_i
            continue
        s = a_i
        placed = False
        while s + L <= D:
            if all(occ[t] < C for t in range(s, s + L)):
                placed = True
                break
            s += 1
        if not placed:
            s = max(a_i, D - L)
            placed = True
        starts[i] = s
        for t in range(s, min(D, s + L)):
            occ[t] += 1
    return starts


def L_of(e_i, p):
    return -(-e_i // p) if e_i > 0 else 0


def degrade_isolated(a_i, theta0, e_i, c10, D, start, power):
    """Degradation of a bus that idles (power=0, cooling for free since soc=0) from
    `a_i` to `start`, then charges continuously at `power` from `start` to D-1 (no
    other buses considered). `start` must satisfy a_i <= start <= D."""
    c = c10 / 10.0
    theta = float(theta0)
    energy_cum = 0.0
    deg = 0.0
    for t in range(a_i, D):
        soc = min(1.0, energy_cum / e_i) if e_i > 0 else 1.0
        excess = theta - THETA_SAFE
        if excess > 0:
            deg += c * excess * excess * soc
        p = power if (t >= start and energy_cum < e_i) else 0
        theta = AMBIENT + (theta - AMBIENT) * RHO + HEAT * p
        energy_cum += p
    return deg


def best_slot(occ, C, D, a_i, L, s_star):
    """Latest-preferred feasible start in [a_i, D-L] closest to s_star, or None."""
    if a_i + L > D:
        return None
    for s in sorted(range(a_i, D - L + 1), key=lambda s: (abs(s - s_star), -s)):
        if all(occ[t] < C for t in range(s, s + L)):
            return s
    return None


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); D = int(next(it)); C = int(next(it)); pmax = int(next(it))
    buses = []
    for _ in range(n):
        a_i = int(next(it)); theta0 = int(next(it)); e_i = int(next(it)); c10 = int(next(it))
        buses.append((a_i, theta0, e_i, c10))

    # Always-feasible starting point: immediate full-power FCFS (identical to the
    # "greedy" tier's own construction).
    pairs = [(buses[i][0], buses[i][2]) for i in range(n)]
    fb_starts = fcfs_schedule(pairs, D, C, pmax)

    occ = [0] * D
    chosen_p = [pmax] * n
    chosen_s = list(fb_starts)
    chosen_L = [L_of(buses[i][2], pmax) for i in range(n)]
    chosen_deg = [0.0] * n
    for i, (a_i, theta0, e_i, c10) in enumerate(buses):
        s, l = chosen_s[i], chosen_L[i]
        for t in range(s, min(D, s + l)):
            occ[t] += 1
        chosen_deg[i] = degrade_isolated(a_i, theta0, e_i, c10, D, s, pmax)

    max_iters = n * max(1, pmax - 1)
    for _ in range(max_iters):
        best = None  # (benefit, i, new_p, new_s, new_L, new_deg)
        for i, (a_i, theta0, e_i, c10) in enumerate(buses):
            p = chosen_p[i]
            if p <= 1:
                continue
            new_p = p - 1
            new_L = L_of(e_i, new_p)
            s_star = min(D - new_L, a_i + K_COOL) if a_i + new_L <= D else None
            if s_star is None:
                continue
            # free this bus's current reservation while probing
            s0, l0 = chosen_s[i], chosen_L[i]
            for t in range(s0, min(D, s0 + l0)):
                occ[t] -= 1
            new_s = best_slot(occ, C, D, a_i, new_L, s_star)
            if new_s is not None:
                new_deg = degrade_isolated(a_i, theta0, e_i, c10, D, new_s, new_p)
                benefit = chosen_deg[i] - new_deg
                if benefit > 1e-9 and (best is None or benefit > best[0]):
                    best = (benefit, i, new_p, new_s, new_L, new_deg)
            # restore (we only commit the single globally-best move per round)
            for t in range(s0, min(D, s0 + l0)):
                occ[t] += 1

        if best is None:
            break
        _, i, new_p, new_s, new_L, new_deg = best
        s0, l0 = chosen_s[i], chosen_L[i]
        for t in range(s0, min(D, s0 + l0)):
            occ[t] -= 1
        for t in range(new_s, min(D, new_s + new_L)):
            occ[t] += 1
        chosen_p[i], chosen_s[i], chosen_L[i], chosen_deg[i] = new_p, new_s, new_L, new_deg

    out_lines = []
    for i, (a_i, theta0, e_i, c10) in enumerate(buses):
        power = [0] * D
        t = chosen_s[i]
        remaining = e_i
        p = chosen_p[i]
        while remaining > 0 and t < D:
            use = min(p, remaining)
            power[t] = use
            remaining -= use
            t += 1
        out_lines.append(" ".join(map(str, power)))
    sys.stdout.write("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main()
