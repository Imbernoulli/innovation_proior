#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for garbage-collector-schedule.
Prints "... Ratio: <float in [0,1]>" on its own final line and exits 0.
"""
import sys, math


def simulate(T, Y, K, Bpause, c0_minor, c0_major, k_young, k_old, f_rate, penalty,
             alloc, lifetime, actions):
    if len(actions) != T:
        return False, "wrong action count", 0.0, 0
    young = []   # [remaining, death_time, survive_count]
    old = []     # [remaining, death_time]
    total = 0.0
    violations = 0
    for t in range(1, T + 1):
        a = actions[t - 1]
        if a not in (0, 1, 2):
            return False, f"invalid action token at step {t}", 0.0, 0
        if a != 0:
            pre_y = sum(c[0] for c in young)
            nxt = []
            for rem, death, surv in young:
                if t >= death:
                    continue
                surv += 1
                if surv >= K:
                    old.append([rem, death])
                else:
                    nxt.append([rem, death, surv])
            young = nxt
            cost = c0_minor + k_young * pre_y
            if a == 2:
                pre_o = sum(c[0] for c in old)
                nxt_o = []
                for rem, death in old:
                    if t >= death:
                        continue
                    nxt_o.append([rem, death])
                old = nxt_o
                cost = c0_major + k_young * pre_y + k_old * pre_o
            if cost > Bpause + 1e-9:
                violations += 1
                total += penalty * (cost - Bpause)
            total += cost
        if alloc[t - 1] > 0:
            young.append([alloc[t - 1], t + lifetime[t - 1], 0])
        yres = sum(c[0] for c in young)
        if yres > Y:
            return False, f"young heap overflow at step {t}: {yres} > {Y}", 0.0, 0
        ores = sum(c[0] for c in old)
        total += f_rate * (yres + ores)
    return True, "", total, violations


def fail(reason):
    print(f"INFEASIBLE: {reason}")
    print("Ratio: 0.0")
    sys.exit(0)


def parse_ints(tok_list, n, name):
    if len(tok_list) != n:
        fail(f"expected {n} tokens for {name}, got {len(tok_list)}")
    out = []
    for tok in tok_list:
        try:
            v = int(tok)
        except ValueError:
            fail(f"non-integer token in {name}: {tok!r}")
        if not math.isfinite(v):
            fail(f"non-finite token in {name}: {tok!r}")
        out.append(v)
    return out


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path) as f:
        header = f.readline().split()
        T = int(header[0]); Y = int(header[1]); K = int(header[2])
        Bpause = int(header[3]); c0_minor = int(header[4]); c0_major = int(header[5])
        k_young = int(header[6]); k_old = int(header[7]); f_rate = float(header[8])
        penalty = int(header[9])
        alloc = [int(x) for x in f.readline().split()]
        lifetime = [int(x) for x in f.readline().split()]
    if len(alloc) != T or len(lifetime) != T:
        fail("corrupt instance (harness bug)")

    try:
        with open(out_path) as f:
            raw_tokens = f.read().split()
    except FileNotFoundError:
        fail("no output produced")

    if len(raw_tokens) == 0:
        fail("empty output")
    if len(raw_tokens) != T:
        fail(f"expected {T} action tokens, got {len(raw_tokens)}")

    actions = []
    for tok in raw_tokens:
        try:
            v = int(tok)
        except ValueError:
            fail(f"non-integer action token: {tok!r}")
        if v not in (0, 1, 2):
            fail(f"action token out of range {{0,1,2}}: {v}")
        actions.append(v)

    feas, reason, F, violations = simulate(T, Y, K, Bpause, c0_minor, c0_major,
                                            k_young, k_old, f_rate, penalty,
                                            alloc, lifetime, actions)
    if not feas:
        fail(reason)

    # internal baseline: the trivial feasible construction "collect (minor) every step"
    trivial_actions = [1] * T
    bfeas, breason, Bref, _ = simulate(T, Y, K, Bpause, c0_minor, c0_major,
                                        k_young, k_old, f_rate, penalty,
                                        alloc, lifetime, trivial_actions)
    if not bfeas:
        fail(f"internal baseline infeasible (harness bug): {breason}")

    sc = min(1000.0, 100.0 * Bref / max(1e-9, F))
    print(f"total_cost={F:.4f} baseline={Bref:.4f} violations={violations}")
    print("Ratio: %.6f" % (sc / 1000.0))
    sys.exit(0)


if __name__ == "__main__":
    main()
