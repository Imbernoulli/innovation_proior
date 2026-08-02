#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic scorer for the diffusion-fed
catalytic-strip site-placement problem. Prints 'Ratio: <float in [0,1]>' on
its own final line. Maximization objective: higher total conversion is
better, normalized against the checker's own reference (centered block)
placement.
"""
import sys
import math

T = 12       # reaction cycles
R = 25       # diffusion micro-steps per cycle
DT = 0.1     # micro-step size


def fail(msg):
    print("INFEASIBLE: %s" % msg)
    print("Ratio: 0.0")
    sys.exit(0)


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    pos = 0

    def nxt():
        nonlocal pos
        v = toks[pos]
        pos += 1
        return v

    L = int(nxt())
    B = int(nxt())
    D = float(nxt())
    v_max = float(nxt())
    gamma = float(nxt())
    r_screen = int(nxt())
    poison_rate = float(nxt())
    C0 = float(nxt())
    return L, B, D, v_max, gamma, r_screen, poison_rate, C0


def simulate(L, active, D, v_max, gamma, r_screen, poison_rate, C0):
    """Runs the exact T-cycle / R-micro-step reaction-diffusion protocol
    described in statement.md and returns the total conversion F."""
    poison = [0.0] * L
    c = [C0] * L
    total = 0.0
    for _t in range(T):
        # crowding multiplier: 1/(1+gamma*n_i), n_i = active neighbors within r_screen
        mult = [1.0] * L
        for i in range(L):
            if not active[i]:
                continue
            cnt = 0
            for dd in range(1, r_screen + 1):
                if i - dd >= 0 and active[i - dd]:
                    cnt += 1
                if i + dd < L and active[i + dd]:
                    cnt += 1
            mult[i] = 1.0 / (1.0 + gamma * cnt)
        k = [0.0] * L
        for i in range(L):
            if active[i]:
                k[i] = v_max * mult[i] * (1.0 - poison[i])
        turnover = [0.0] * L
        for _m in range(R):
            new_c = [0.0] * L
            for i in range(L):
                left = c[i - 1] if i - 1 >= 0 else C0
                right = c[i + 1] if i + 1 < L else C0
                produced = k[i] * c[i] * DT
                turnover[i] += produced
                val = c[i] + DT * D * (left + right - 2.0 * c[i]) - produced
                if val < 0.0:
                    val = 0.0
                elif val > C0:
                    val = C0
                new_c[i] = val
            c = new_c
        for i in range(L):
            if active[i]:
                total += turnover[i]
                poison[i] = min(1.0, poison[i] + poison_rate * turnover[i])
    return total


def centered_block(L, B):
    active = [0] * L
    start = (L - B) // 2
    for i in range(start, start + B):
        active[i] = 1
    return active


def main():
    inp, outp = sys.argv[1], sys.argv[2]
    L, B, D, v_max, gamma, r_screen, poison_rate, C0 = read_instance(inp)

    try:
        with open(outp) as f:
            toks = f.read().split()
    except Exception as e:
        fail("cannot read output: %s" % e)
        return

    if len(toks) != L:
        fail("expected exactly %d tokens, got %d" % (L, len(toks)))
        return

    active = []
    for i, tok in enumerate(toks):
        try:
            v = int(tok)
        except ValueError:
            fail("token %d (%r) is not an integer" % (i, tok))
            return
        if v not in (0, 1):
            fail("position %d value %d not in {0,1}" % (i, v))
            return
        active.append(v)

    n_active = sum(active)
    if n_active > B:
        fail("budget exceeded: %d active sites > budget %d" % (n_active, B))
        return

    F = simulate(L, active, D, v_max, gamma, r_screen, poison_rate, C0)
    if not math.isfinite(F) or F < 0:
        fail("non-finite or negative conversion")
        return

    ref_active = centered_block(L, B)
    F_ref = simulate(L, ref_active, D, v_max, gamma, r_screen, poison_rate, C0)

    sc = min(1000.0, 100.0 * F / max(1e-9, F_ref))
    print("n_active=%d/%d F=%.6f F_ref=%.6f" % (n_active, B, F, F_ref))
    print("Ratio: %.6f" % (sc / 1000.0))
    sys.exit(0)


if __name__ == "__main__":
    main()
