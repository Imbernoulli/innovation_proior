"""ring_truth.py -- shared deterministic construction for the "Claim Ring Audit"
problem (fsx_B_1267, format C). Imported by BOTH gen.py (prints the visible
instance) and verify.py (re-derives the hidden fraud/ring ground truth that
gen.py's stdout never reveals). Nothing here reads argv or does I/O.

Model (bipartite-collusion-graph + per-claim-plausibility + ring-detection
budget):
  Every claim links exactly one claimant, one provider and one adjuster
  (its three "parties"; a claim is a triangle over the claimant/provider/
  adjuster bipartite-incidence graph). Legitimate claims and lone "sloppy"
  fraud draw their three parties independently from a WIDE id pool, so no
  pair of parties recurs together more than by pure chance. A planted
  COLLUSION RING instead draws every one of its claims' parties from its
  own tiny reserved sub-pool (as few as 1-3 ids per role) -- so the same
  claimant/provider/adjuster PAIRS recur again and again across that
  ring's claims. That recurrence is the only signature of a ring: no
  single claim in it looks structurally different from an ordinary one.

  Each claim also carries a public "plausibility" p in [0,1] (higher =
  looks more routine). Lone sloppy fraud is deliberately clumsy and drawn
  from a low, clearly-separated plausibility band. Ring fraud is drawn
  from EXACTLY the same plausibility band as ordinary legitimate claims --
  by design, per-claim plausibility carries zero information about ring
  membership. The only honest signal is the recurring-party structure.
"""
import random

# ladder: bigger id pools / more claims / more rings as testId grows.
# rings: list of dicts, each a planted collusion ring with its own tiny
# reserved (claimant, provider, adjuster) sub-pool sizes and claim count.
LADDER = {
    1: dict(n_legit=34, n_sloppy=3, wide_c=42, wide_p=26, wide_a=16,
            budget_frac=0.24,
            rings=[dict(nc=2, np=2, na=1, n_claims=7)]),
    2: dict(n_legit=44, n_sloppy=3, wide_c=52, wide_p=30, wide_a=18,
            budget_frac=0.22,
            rings=[dict(nc=2, np=2, na=1, n_claims=8),
                   dict(nc=2, np=1, na=2, n_claims=6)]),
    3: dict(n_legit=54, n_sloppy=4, wide_c=60, wide_p=36, wide_a=22,
            budget_frac=0.20,
            rings=[dict(nc=2, np=2, na=2, n_claims=10),
                   dict(nc=3, np=2, na=1, n_claims=8)]),
    4: dict(n_legit=64, n_sloppy=4, wide_c=70, wide_p=42, wide_a=26,
            budget_frac=0.19,
            rings=[dict(nc=2, np=2, na=1, n_claims=9),
                   dict(nc=2, np=1, na=2, n_claims=7),
                   dict(nc=3, np=2, na=2, n_claims=10)]),
    5: dict(n_legit=76, n_sloppy=5, wide_c=82, wide_p=48, wide_a=30,
            budget_frac=0.18,
            rings=[dict(nc=2, np=2, na=1, n_claims=10),
                   dict(nc=2, np=1, na=2, n_claims=8),
                   dict(nc=3, np=2, na=1, n_claims=9),
                   dict(nc=2, np=2, na=2, n_claims=7)]),
    6: dict(n_legit=90, n_sloppy=5, wide_c=96, wide_p=56, wide_a=34,
            budget_frac=0.17,
            rings=[dict(nc=2, np=2, na=1, n_claims=11),
                   dict(nc=2, np=1, na=2, n_claims=9),
                   dict(nc=3, np=2, na=1, n_claims=10),
                   dict(nc=2, np=2, na=2, n_claims=8),
                   dict(nc=3, np=1, na=2, n_claims=7)]),
    7: dict(n_legit=104, n_sloppy=6, wide_c=110, wide_p=64, wide_a=38,
            budget_frac=0.165,
            rings=[dict(nc=2, np=2, na=1, n_claims=12),
                   dict(nc=2, np=1, na=2, n_claims=10),
                   dict(nc=3, np=2, na=1, n_claims=11),
                   dict(nc=2, np=2, na=2, n_claims=9),
                   dict(nc=3, np=1, na=2, n_claims=8),
                   dict(nc=2, np=3, na=1, n_claims=8)]),
    8: dict(n_legit=118, n_sloppy=6, wide_c=124, wide_p=72, wide_a=44,
            budget_frac=0.155,
            rings=[dict(nc=2, np=2, na=1, n_claims=12),
                   dict(nc=2, np=1, na=2, n_claims=10),
                   dict(nc=3, np=2, na=1, n_claims=11),
                   dict(nc=2, np=2, na=2, n_claims=9),
                   dict(nc=3, np=1, na=2, n_claims=9),
                   dict(nc=2, np=3, na=1, n_claims=8),
                   dict(nc=3, np=2, na=2, n_claims=10)]),
    9: dict(n_legit=136, n_sloppy=7, wide_c=142, wide_p=82, wide_a=50,
            budget_frac=0.145,
            rings=[dict(nc=2, np=2, na=1, n_claims=13),
                   dict(nc=2, np=1, na=2, n_claims=11),
                   dict(nc=3, np=2, na=1, n_claims=12),
                   dict(nc=2, np=2, na=2, n_claims=10),
                   dict(nc=3, np=1, na=2, n_claims=10),
                   dict(nc=2, np=3, na=1, n_claims=9),
                   dict(nc=3, np=2, na=2, n_claims=11),
                   dict(nc=2, np=2, na=1, n_claims=8)]),
    10: dict(n_legit=156, n_sloppy=7, wide_c=162, wide_p=94, wide_a=58,
             budget_frac=0.135,
             rings=[dict(nc=2, np=2, na=1, n_claims=14),
                    dict(nc=2, np=1, na=2, n_claims=12),
                    dict(nc=3, np=2, na=1, n_claims=13),
                    dict(nc=2, np=2, na=2, n_claims=11),
                    dict(nc=3, np=1, na=2, n_claims=11),
                    dict(nc=2, np=3, na=1, n_claims=10),
                    dict(nc=3, np=2, na=2, n_claims=12),
                    dict(nc=2, np=2, na=1, n_claims=9)]),
}

AMOUNT_LO, AMOUNT_HI = 50.0, 500.0
SLOPPY_AMOUNT_LO, SLOPPY_AMOUNT_HI = 300.0, 650.0  # clumsy fraud pays out big, on average
LEGIT_P_LO, LEGIT_P_HI = 0.55, 1.00      # ordinary claims: high plausibility
RING_P_LO, RING_P_HI = 0.55, 1.00        # ring fraud: IDENTICAL band (indistinguishable)
SLOPPY_P_LO, SLOPPY_P_HI = 0.00, 0.25    # lone fraud: clearly separated, low band
COST_CHOICES = (1, 2, 3)


def build(test_id):
    """Deterministic full ground truth for testId (1..10, clamped). Returns
    a dict with N, budget, claims=[{claimant,provider,adjuster,amount,
    plausibility,cost,fraud,ring}] (order already shuffled -- index carries
    no information), and pool sizes NC/NP/NA."""
    tid = min(max(int(test_id), 1), 10)
    cfg = LADDER[tid]
    rng = random.Random(9_176_233 * tid + 20260726)

    wide_c, wide_p, wide_a = cfg["wide_c"], cfg["wide_p"], cfg["wide_a"]
    next_c, next_p, next_a = wide_c, wide_p, wide_a

    claims = []

    def draw_amount():
        return round(rng.uniform(AMOUNT_LO, AMOUNT_HI), 2)

    def draw_cost():
        return rng.choice(COST_CHOICES)

    # ---- planted collusion rings: each gets its OWN tiny reserved id
    # sub-pool per role, disjoint from the wide pool and from every other
    # ring, so recurring (claimant,provider)/(claimant,adjuster)/
    # (provider,adjuster) pairs are a structural fact, not luck. ----
    for ridx, rspec in enumerate(cfg["rings"]):
        c_pool = list(range(next_c, next_c + rspec["nc"])); next_c += rspec["nc"]
        p_pool = list(range(next_p, next_p + rspec["np"])); next_p += rspec["np"]
        a_pool = list(range(next_a, next_a + rspec["na"])); next_a += rspec["na"]
        for _ in range(rspec["n_claims"]):
            claims.append(dict(
                claimant=rng.choice(c_pool), provider=rng.choice(p_pool),
                adjuster=rng.choice(a_pool), amount=draw_amount(),
                plausibility=round(rng.uniform(RING_P_LO, RING_P_HI), 4),
                cost=draw_cost(), fraud=True, ring=ridx))

    NC, NP, NA = next_c, next_p, next_a  # total id-space sizes (wide + all rings)

    # ---- lone "sloppy" fraud: unique-ish parties from the WIDE pool
    # (structurally invisible, no recurring pair) but a clumsy, clearly
    # low plausibility score. ----
    for _ in range(cfg["n_sloppy"]):
        claims.append(dict(
            claimant=rng.randrange(wide_c), provider=rng.randrange(wide_p),
            adjuster=rng.randrange(wide_a),
            amount=round(rng.uniform(SLOPPY_AMOUNT_LO, SLOPPY_AMOUNT_HI), 2),
            plausibility=round(rng.uniform(SLOPPY_P_LO, SLOPPY_P_HI), 4),
            cost=draw_cost(), fraud=True, ring=None))

    # ---- ordinary legitimate claims: WIDE pool parties, high plausibility,
    # not fraud. ----
    for _ in range(cfg["n_legit"]):
        claims.append(dict(
            claimant=rng.randrange(wide_c), provider=rng.randrange(wide_p),
            adjuster=rng.randrange(wide_a), amount=draw_amount(),
            plausibility=round(rng.uniform(LEGIT_P_LO, LEGIT_P_HI), 4),
            cost=draw_cost(), fraud=False, ring=None))

    rng.shuffle(claims)  # claim index carries no ring/order information
    N = len(claims)
    total_cost = sum(c["cost"] for c in claims)
    budget = max(1, round(cfg["budget_frac"] * total_cost))

    return dict(N=N, NC=NC, NP=NP, NA=NA, budget=budget, claims=claims,
                test_id=tid)
