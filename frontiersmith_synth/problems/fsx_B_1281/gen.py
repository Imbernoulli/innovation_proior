import sys, random

# Fields per project (all integers, space-separated on one line):
#   price_per_tonne  claimed_tonnes  reported_baseline_bps  reference_baseline_bps
#   irr_without_carbon_bps  irr_threshold_bps  reversal_risk_bps  permanence_years  buffer_pct
#
# Three archetypes are mixed into every case:
#   scam    : cheap $/tonne, huge claimed_tonnes, inflated reported baseline, would-happen-anyway
#             IRR, weak permanence -> low true effective_tonnes despite big cheap nominal supply.
#   quality : pricier $/tonne, modest claimed_tonnes, honest baseline, carbon-revenue-dependent
#             IRR, strong permanence -> high true effective_tonnes.
#   mid     : fully randomized mix (including a few *expensive* scams) so no simple price-only
#             or tonnage-only or "always pick expensive" rule works either.


def clampi(v, lo, hi):
    return max(lo, min(hi, v))


def make_scam(rng):
    price = rng.randint(2, 6)
    tonnes = rng.randint(300, 1600)
    ref_base = rng.randint(80, 350)
    # baseline-gaming is the dominant, near-always-present flaw
    reported = clampi(int(ref_base * rng.uniform(2.0, 4.2)), ref_base + 1, 20000)
    threshold = rng.randint(800, 1500)
    # financially borderline rather than uniformly "would happen anyway" -- keeps
    # a portfolio of scams worth LITTLE per dollar, but not a coin-flip zero.
    irr = threshold - rng.randint(30, 400)
    reversal = rng.randint(300, 800)
    perm_years = rng.randint(7, 18)
    buffer = rng.randint(0, 25)
    return [price, tonnes, reported, ref_base, irr, threshold, reversal, perm_years, buffer]


def make_quality(rng):
    price = rng.randint(11, 30)
    tonnes = rng.randint(60, 320)
    ref_base = rng.randint(250, 900)
    reported = clampi(int(ref_base * rng.uniform(1.0, 1.22)), ref_base, ref_base * 2)
    threshold = rng.randint(800, 1500)
    irr = threshold - rng.randint(1000, 2600)       # carbon-finance-dependent
    reversal = rng.randint(5, 140)
    perm_years = rng.randint(20, 50)
    buffer = rng.randint(45, 95)
    return [price, tonnes, reported, ref_base, irr, threshold, reversal, perm_years, buffer]


def make_anchor(rng):
    # a cheap, always-first, genuinely-additional project: guarantees the naive
    # "buy in listed order" baseline construction is never stuck at zero true value.
    price = rng.randint(11, 15)
    tonnes = rng.randint(60, 100)
    ref_base = rng.randint(400, 700)
    reported = ref_base + rng.randint(0, int(0.1 * ref_base))
    threshold = rng.randint(1000, 1400)
    irr = threshold - rng.randint(1800, 2400)
    reversal = rng.randint(5, 40)
    perm_years = rng.randint(30, 50)
    buffer = rng.randint(70, 95)
    return [price, tonnes, reported, ref_base, irr, threshold, reversal, perm_years, buffer]


def make_mid(rng):
    price = rng.randint(2, 35)
    tonnes = rng.randint(40, 900)
    ref_base = rng.randint(80, 900)
    reported = clampi(int(ref_base * rng.uniform(1.0, 4.5)), ref_base, ref_base * 6)
    threshold = rng.randint(800, 1500)
    irr = threshold + rng.randint(-2600, 1400)
    reversal = rng.randint(5, 1300)
    perm_years = rng.randint(3, 50)
    buffer = rng.randint(0, 95)
    return [price, tonnes, reported, ref_base, irr, threshold, reversal, perm_years, buffer]


def cost_of(p):
    return p[0] * p[1]


def eff_tonnes(p):
    price, tonnes, reported, ref_base, irr, threshold, reversal, perm_years, buffer = p
    inflation_penalty = min(1.0, ref_base / reported)
    fin_add = (threshold - irr) / 1000.0
    fin_add = 0.0 if fin_add < 0.0 else (1.0 if fin_add > 1.0 else fin_add)
    additionality = min(inflation_penalty, fin_add)
    survival = (1.0 - reversal / 10000.0) ** perm_years
    permanence = 1.0 - (1.0 - buffer / 100.0) * (1.0 - survival)
    return tonnes * additionality * permanence


def main():
    i = int(sys.argv[1])
    rng = random.Random(48173 + 97 * i)

    n = 12 + 3 * (i - 1)                      # 12 .. 39, difficulty ladder
    # trap severity: cases with index in this set get an aggressive scam-heavy mix
    # engineered so the price-ascending greedy lands far from the informed optimum.
    trap_case = i in (3, 5, 7, 9)

    if trap_case:
        n_scam = max(2, round(n * 0.28))
        n_quality = max(2, round(n * 0.40))
    else:
        n_scam = max(2, round(n * 0.11))
        n_quality = max(2, round(n * 0.46))
    n_mid = max(0, n - n_scam - n_quality)

    projects = []
    for _ in range(n_scam):
        projects.append(make_scam(rng))
    for _ in range(n_quality):
        projects.append(make_quality(rng))
    for _ in range(n_mid):
        # a slice of "mid" are deliberately expensive-but-fake, to punish
        # any "just buy the pricier ones" heuristic too
        if rng.random() < 0.15:
            p = make_scam(rng)
            p[0] = rng.randint(20, 40)          # re-price expensive
            projects.append(p)
        else:
            projects.append(make_mid(rng))

    rng.shuffle(projects)
    n_anchor = 1
    anchors = [make_anchor(rng) for _ in range(n_anchor)]
    projects = anchors + projects           # anchors always occupy input positions 1..n_anchor
    n = len(projects)
    anchor_cost = sum(cost_of(a) for a in anchors)

    quality_cost = sum(cost_of(p) for p in projects if eff_tonnes(p) >= 0.7 * p[1])
    quality_cost = max(quality_cost, anchor_cost)
    frac = 0.50 if trap_case else 0.62
    budget = max(anchor_cost * 2, int(frac * quality_cost))

    out = [str(n), str(budget)]
    for p in projects:
        out.append(" ".join(str(x) for x in p))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
