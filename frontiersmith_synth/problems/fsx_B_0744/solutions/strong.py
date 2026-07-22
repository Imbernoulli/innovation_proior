# TIER: strong
"""
The insight: don't trust the stale reference concentration x_ref. Instead,
ask what concentration EACH metabolite would have to sit at for the
TARGET flux vector itself to be self-consistent, via the same
production/consumption balance the network actually obeys:

    x_target_i = tau_i * max(yield_i * v_target_i - sum_{k consumes i} v_target_k, 0)

This is computable directly from v_target and the tree structure (no
simulation needed): metabolite i's producing reaction is unique (reaction
i itself), so its implied concentration only needs the target fluxes of
reaction i and its children -- all given.

Then, for EACH reaction, classify it against Km at that derived
concentration (deeply saturated: x_target >> Km, rate is capped and
almost enzyme-only; deeply limiting: x_target << Km, rate is
concentration-starved and needs much more enzyme to compensate) and invert
the EXACT rate law there:

    e_i = v_target_i * (Km_i + x_target_{parent_i}) / (kcat_i * x_target_{parent_i})

rather than at the reference point. When x_target is small (limiting
regime) this yields a substantially LARGER e_i than the greedy guess;
when x_target is large (saturated) it converges toward v_target_i/kcat_i.
Either way it reacts to what the target itself implies, not to a fixed
nominal snapshot.
"""
import sys

X0 = 20.0
EPS = 1e-9


def main():
    toks = sys.stdin.read().split()
    ptr = 0
    R = int(toks[ptr]); ptr += 1
    ptr += 1  # X0
    parent = [0] * R
    yield_ = [0] * R
    kcat = [0.0] * R
    Km = [0.0] * R
    tau = [0.0] * R
    e_max = [0.0] * R
    for i in range(R):
        parent[i] = int(toks[ptr]); ptr += 1
        yield_[i] = int(toks[ptr]); ptr += 1
        kcat[i] = float(toks[ptr]); ptr += 1
        Km[i] = float(toks[ptr]); ptr += 1
        tau[i] = float(toks[ptr]); ptr += 1
        e_max[i] = float(toks[ptr]); ptr += 1
        ptr += 1  # cost
    ptr += R  # x_ref -- deliberately NOT used by strong
    v_target = [float(toks[ptr + k]) for k in range(R)]
    ptr += R

    children = [[] for _ in range(R + 1)]
    for j in range(R):
        children[parent[j]].append(j)

    # x_target_i: the concentration metabolite i would need for the target
    # flux vector to be self-consistent at that node.
    x_target = [0.0] * R
    for i in range(R):
        cons = sum(v_target[k] for k in children[i + 1])
        net = yield_[i] * v_target[i] - cons
        x_target[i] = tau[i] * max(net, 0.0)

    e = [0.0] * R
    for i in range(R):
        xp = X0 if parent[i] == 0 else x_target[parent[i] - 1]
        val = v_target[i] * (Km[i] + xp) / (kcat[i] * xp + EPS)
        e[i] = min(max(val, 0.0), e_max[i])

    print(" ".join(str(v) for v in e))


if __name__ == "__main__":
    main()
