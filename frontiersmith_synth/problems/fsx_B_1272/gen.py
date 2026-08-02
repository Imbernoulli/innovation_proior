import sys, random

K = 5  # number of typologies, fixed
# testIds whose regulator-mandated typology carries a BINDING coverage floor (the trap set);
# the remaining testIds carry a zero floor on the mandated typology (no trap that round).
TRAP_IDS = {3, 5, 8, 10}


def reg_typology(test_id):
    """The typology the regulator designates this round (must match verify.py exactly)."""
    return ((test_id - 1) % K) + 1


def main():
    i = int(sys.argv[1])
    rng = random.Random(900001 + 97 * i)

    N = 24 + 7 * i                      # 31 .. 94 alerts
    R = reg_typology(i)
    trap = i in TRAP_IDS

    # per-typology visible score distribution: on a trap round the regulator-designated
    # typology R is deliberately scored LOW by the model (hard-to-score typology), other
    # typologies score higher on average -- this is the visible signal a precision-greedy
    # solver will chase. On a non-trap round every typology uses the same ordinary
    # distribution (nothing to trap -- the mandate for R is simply waived this shift).
    mean = {}
    std = {}
    for t in range(1, K + 1):
        if trap and t == R:
            mean[t], std[t] = 16.0, 8.0
        else:
            mean[t], std[t] = float(rng.randint(48, 74)), float(rng.randint(10, 16))

    alerts = []  # (id, typology, cost, score)
    for aid in range(1, N + 1):
        t = rng.randint(1, K)
        # a wider cost spread on the mandated typology this trap round means "cheapest"
        # and "best value-per-minute" mandatory picks can genuinely diverge in quality.
        if trap and t == R:
            cost = rng.randint(5, 70)
        else:
            cost = rng.randint(5, 45)
        raw = rng.gauss(mean[t], std[t])
        score = max(0, min(100, int(round(raw))))
        alerts.append((aid, t, cost, score))

    by_t = {t: [a for a in alerts if a[1] == t] for t in range(1, K + 1)}
    cnt = {t: len(by_t[t]) for t in range(1, K + 1)}

    # coverage floors: only the regulator-mandated typology R ever carries a floor this
    # shift, and only on trap rounds -- a binding ~50% floor. Other typologies: 0 (keeps
    # the mandate crisp and avoids accidental floor misses on unrelated typologies).
    import math
    min_cover = {}
    for t in range(1, K + 1):
        frac = 0.50 if (t == R and trap) else 0.0
        min_cover[t] = min(cnt[t], int(math.ceil(frac * cnt[t])))

    # capacity: must comfortably admit BOTH a cheapest-first mandatory selection and a
    # highest-score-first mandatory selection (two different reasonable strategies), plus
    # genuine slack for a discretionary phase -- this is what makes the budget a real
    # trade-off rather than an accident of construction.
    def subset_cost(pick_key, reverse):
        total = 0
        for t in range(1, K + 1):
            need = min_cover[t]
            if need <= 0:
                continue
            ordered = sorted(by_t[t], key=pick_key, reverse=reverse)
            total += sum(a[2] for a in ordered[:need])
        return total

    mincost_total = subset_cost(lambda a: a[2], False)                  # cheapest-first per typology
    hicost_total = subset_cost(lambda a: (a[3] / a[2], -a[2]), True)    # best score/cost-density first
    base_needed = max(mincost_total, hicost_total)
    totalcost = sum(a[2] for a in alerts)
    slack = int(round(0.06 * max(0, totalcost - base_needed)))
    C = base_needed + slack
    C = max(C, 1)

    out = [f"{i} {N} {K} {C}"]
    for aid, t, cost, score in alerts:
        out.append(f"{aid} {t} {cost} {score}")
    for t in range(1, K + 1):
        out.append(f"{t} {min_cover[t]}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
