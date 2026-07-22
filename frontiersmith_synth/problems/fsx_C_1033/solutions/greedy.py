# TIER: greedy
# The obvious "textbook" recipe: sort time steps by price ascending and bin-fill
# each cheap slot at the maximum allowed rate until the energy target is met.
# It never equalizes the marginal quadratic loss against price (so it blasts full
# rate into whichever slots are cheapest -- huge quadratic loss and a maxed-out
# demand charge), and it never accounts for session fees when choosing which
# slots to use (so genuinely cheap but scattered slots each open their own
# session). Both effects are the traps this family plants.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    T = int(next(it))
    dt = float(next(it))
    Rmax = float(next(it))
    Fee = float(next(it))
    D = float(next(it))
    r_min = float(next(it))
    prices = [float(next(it)) for _ in range(T)]
    alpha = [float(next(it)) for _ in range(T)]
    E_target = float(next(it))

    order = sorted(range(T), key=lambda t: prices[t])
    rates = [0.0] * T
    remaining = E_target
    for t in order:
        if remaining <= 1e-9:
            break
        cap_energy = Rmax * dt
        if remaining >= cap_energy:
            rates[t] = Rmax
            remaining -= cap_energy
        else:
            # a charger cannot idle below r_min: the last partial slot rounds
            # up to the minimum operating rate if the true remainder is finer
            r = max(remaining / dt, r_min)
            rates[t] = min(r, Rmax)
            remaining = 0.0

    print(" ".join(f"{r:.6f}" for r in rates))


if __name__ == "__main__":
    main()
