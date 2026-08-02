# TIER: strong
"""Insight: rate constant is not importance. Rank each reaction by its simulated
IMPACT on the target trajectory (removing it alone from the full network and
re-integrating every held-out condition -- a one-shot local sensitivity analysis),
not by its raw rate constant. A reaction that is dynamically irrelevant (no simulated
impact under ANY condition) is safe to drop no matter how fast it is -- this is what
lets a stiff/fast decoy sub-network be eliminated wholesale. A reaction that is the
sole path feeding the target under even ONE held-out condition shows up with large
impact and is kept, no matter how slow it is. Removals are then applied in ascending
impact order and validated against the accuracy-preservation bound over ALL held-out
conditions (not a single reference condition) before being committed, so the final
mechanism is feasible by construction.
"""
import sys


def simulate_target_traj(n, reacts, c0, T_horizon, N_steps, target):
    dt = T_horizon / N_steps
    c = list(c0)

    def deriv(state):
        d = [0.0] * n
        for (r, p, rate) in reacts:
            f = rate * state[r]
            d[r] -= f
            d[p] += f
        return d

    traj = [c[target]]
    for _ in range(N_steps):
        k1 = deriv(c)
        c2 = [c[i] + 0.5 * dt * k1[i] for i in range(n)]
        k2 = deriv(c2)
        c3 = [c[i] + 0.5 * dt * k2[i] for i in range(n)]
        k3 = deriv(c3)
        c4 = [c[i] + dt * k3[i] for i in range(n)]
        k4 = deriv(c4)
        c = [c[i] + (dt / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(n)]
        traj.append(c[target])
    return traj


def max_dev(traj_a, traj_b):
    return max(abs(x - y) for x, y in zip(traj_a, traj_b))


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

    all_idx = list(range(m))

    def sim_all(idx_set):
        reacts = [reactions[i] for i in idx_set]
        return [simulate_target_traj(n, reacts, cond, T_horizon, N_steps, target) for cond in conditions]

    full_baseline = sim_all(all_idx)  # trajectories of the true full network, per condition

    def max_dev_vs_baseline(idx_set):
        trajs = sim_all(idx_set)
        return max(max_dev(trajs[p], full_baseline[p]) for p in range(P))

    # phase 1: one-shot sensitivity -- impact of removing each reaction alone from the FULL network
    impact = {}
    for i in all_idx:
        trial = set(all_idx) - {i}
        impact[i] = max_dev_vs_baseline(trial)

    # phase 2: commit removals in ascending impact (least damaging first), each re-validated
    # against the FULL held-out condition set and the true accuracy-preservation bound
    order = sorted(all_idx, key=lambda i: impact[i])
    kept = set(all_idx)
    for r in order:
        trial = kept - {r}
        if max_dev_vs_baseline(trial) <= eps:
            kept = trial

    kept_list = sorted(kept)
    print(len(kept_list))
    print(" ".join(str(i) for i in kept_list))


if __name__ == "__main__":
    main()
