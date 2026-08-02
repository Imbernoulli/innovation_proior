# TIER: greedy
"""The obvious first attempt: rate constants are treated as a proxy for importance.
Only reactions in the SLOWER half (by rate constant) are ever considered prunable --
"the slowest reactions are the obvious thing to remove" -- and a candidate removal is
accepted whenever it still reproduces the target trajectory under a single reference
condition (the first one given). It never re-checks the other held-out conditions, and
it never even considers pruning a fast reaction, however irrelevant that reaction is to
the target. Both blind spots are exploited by the generator's trap cases.
"""
import sys


def simulate_target(n, reacts, c0, T_horizon, N_steps, target):
    dt = T_horizon / N_steps
    c = list(c0)

    def deriv(state):
        d = [0.0] * n
        for (r, p, rate) in reacts:
            f = rate * state[r]
            d[r] -= f
            d[p] += f
        return d

    for _ in range(N_steps):
        k1 = deriv(c)
        c2 = [c[i] + 0.5 * dt * k1[i] for i in range(n)]
        k2 = deriv(c2)
        c3 = [c[i] + 0.5 * dt * k2[i] for i in range(n)]
        k3 = deriv(c3)
        c4 = [c[i] + dt * k3[i] for i in range(n)]
        k4 = deriv(c4)
        c = [c[i] + (dt / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(n)]
    return c[target]


def main():
    tokens = sys.stdin.read().split()
    ip = 0
    n = int(tokens[ip]); ip += 1
    m = int(tokens[ip]); ip += 1
    target = int(tokens[ip]); ip += 1
    P = int(tokens[ip]); ip += 1
    T_horizon = float(tokens[ip]); ip += 1
    N_steps = int(tokens[ip]); ip += 1
    eps = float(tokens[ip]); ip += 1

    reactions = []
    for _ in range(m):
        r = int(tokens[ip]); ip += 1
        p = int(tokens[ip]); ip += 1
        rate = float(tokens[ip]); ip += 1
        reactions.append((r, p, rate))

    conditions = []
    for _ in range(P):
        c = [float(tokens[ip + j]) for j in range(n)]
        ip += n
        conditions.append(c)

    ref_cond = conditions[0]  # only ever validated against this one condition

    def full_target(idx_set):
        return simulate_target(n, [reactions[i] for i in idx_set], ref_cond, T_horizon, N_steps, target)

    kept = set(range(m))
    baseline_val = full_target(kept)

    order_by_rate = sorted(range(m), key=lambda i: reactions[i][2])
    candidates = order_by_rate[: m // 2]  # only the slower half is ever tried

    for r in candidates:
        trial = kept - {r}
        val = full_target(trial)
        if abs(val - baseline_val) <= eps:
            kept = trial

    kept_list = sorted(kept)
    print(len(kept_list))
    print(" ".join(str(i) for i in kept_list))


if __name__ == "__main__":
    main()
