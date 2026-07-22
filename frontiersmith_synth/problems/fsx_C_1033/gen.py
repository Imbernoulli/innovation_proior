import sys, random

# Electric ferry overnight charging bill: T 15-minute steps, price p_t, a
# per-step quadratic loss coefficient alpha_t, a hard rate cap Rmax, a
# hardware minimum operating rate r_min (the charger cannot idle at a
# trickle -- it is either off or drawing at least r_min), a fixed fee per
# contiguous charging session, a demand charge on the peak rate used, and an
# energy target E_target that must be delivered by morning.

T_BY_TEST = [10, 12, 16, 20, 24, 30, 36, 42, 50, 60]

# Cases where the obvious price-sorted greedy is deliberately trapped: a
# narrow, ultra-cheap, high-loss band inside a DELIBERATELY undersized
# off-peak block, plus a scattered decoy slot outside it that greedy is
# forced to reach once the block runs dry. >=3 of the 10 cases (spec floor).
TRAP_IDS = {3, 4, 7, 10}


def gen(test_id):
    rng = random.Random(20260 + 97 * test_id)
    T = T_BY_TEST[test_id - 1]
    dt = 0.25          # hours per step (15 minutes)
    Rmax = 40.0         # kW cap on the charger
    r_min = round(rng.uniform(0.15, 0.4), 4)   # minimum nonzero operating rate

    Fee = round(rng.uniform(6.0, 12.0), 2)
    D = round(rng.uniform(0.015, 0.045), 4)
    alpha_base = round(rng.uniform(0.0004, 0.0009), 6)

    # Flat two-tariff overnight structure: one contiguous off-peak block at a
    # low price, on-peak elsewhere at a higher price. No noise inside a
    # tariff, so price ties break by index (Python's sort is stable) -- a
    # price-sorted greedy therefore always fills a genuinely CONTIGUOUS
    # prefix of the off-peak block, never fragmenting "by accident".
    p_cheap = rng.uniform(0.015, 0.04)
    p_pricey = rng.uniform(0.45, 0.65)
    is_trap = test_id in TRAP_IDS
    if is_trap:
        # deliberately undersized off-peak block: its own capacity (even
        # including the ultra-cheap valley below) cannot cover E_target, so
        # greedy is forced out to the decoy slot -- this is enforced by
        # construction below, not left to chance.
        off_frac = rng.uniform(0.10, 0.16)
    else:
        off_frac = rng.uniform(0.45, 0.65)
    off_len = max(2, int(round(off_frac * T)))
    off_start = rng.randint(0, T - off_len)
    prices = [p_pricey] * T
    for t in range(off_start, off_start + off_len):
        prices[t] = p_cheap
    alpha = [alpha_base * rng.uniform(0.85, 1.15) for _ in range(T)]

    if is_trap:
        # A narrow band strictly cheaper than the rest of the off-peak block,
        # but with a hugely boosted loss coefficient: raw price makes it look
        # like the best slot in the whole night; the true marginal cost
        # (price + 2*alpha*rate) says otherwise once the rate climbs.
        vlen = max(1, (off_len * 2) // 5)
        vstart = rng.randint(off_start, off_start + off_len - vlen)
        vprice = rng.uniform(0.004, 0.012)
        boost = rng.uniform(120.0, 250.0)
        for t in range(vstart, vstart + vlen):
            prices[t] = vprice
            alpha[t] = alpha_base * boost * rng.uniform(0.9, 1.1)

        # A decoy: one isolated slot OUTSIDE the off-peak block, cheaper than
        # on-peak but pricier than off-peak -- fragments a naive greedy into
        # a second (fee-paying) session once the off-peak block runs dry.
        # It must be separated from the block by at least one genuine
        # on-peak step, or a greedy fill that reaches it would just extend
        # the SAME session instead of paying a second fee.
        outside = [t for t in range(T)
                   if t < off_start - 1 or t > off_start + off_len]
        if not outside:
            outside = [t for t in range(T) if t < off_start or t >= off_start + off_len]
        if outside:
            idx = rng.choice(outside)
            prices[idx] = rng.uniform(0.20, 0.28)

    prices = [round(max(0.001, p), 5) for p in prices]
    alpha = [round(max(1e-6, a), 6) for a in alpha]

    max_capacity = Rmax * dt * T
    if is_trap:
        # E_target's slot-need is guaranteed (by construction) to exceed the
        # off-peak block's own capacity by LESS than one slot's worth, so the
        # single decoy slot is exactly enough to cover the shortfall -- this
        # forces greedy out to the decoy without also dragging it deep into
        # the expensive on-peak pool (which would just be a bigger version
        # of the same trap and wash out the fee-fragmentation effect).
        off_capacity = Rmax * dt * off_len
        E_target = off_capacity + rng.uniform(0.3, 0.85) * Rmax * dt
        E_target = min(E_target, 0.85 * max_capacity)
    else:
        frac = rng.uniform(0.16, 0.26)
        E_target = frac * max_capacity
    E_target = round(E_target, 3)

    out = []
    out.append(f"{T} {dt} {Rmax} {Fee} {D} {r_min}")
    out.append(" ".join(f"{p:.5f}" for p in prices))
    out.append(" ".join(f"{a:.6f}" for a in alpha))
    out.append(f"{E_target:.3f}")
    return "\n".join(out) + "\n"


def main():
    if len(sys.argv) != 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(1)
    test_id = int(sys.argv[1])
    sys.stdout.write(gen(test_id))


if __name__ == "__main__":
    main()
