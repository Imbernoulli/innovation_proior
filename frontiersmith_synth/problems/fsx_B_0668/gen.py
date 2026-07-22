#!/usr/bin/env python3
"""gen.py <testId> -- prints one estate-auction instance to stdout.

Each instance blends two planted structures over a shared filler-bidder floor:

  (1) GENERIC region hubs: each has a home region of lots it values highly, plus a
      fraction of lots also co-valued by one random other generic hub (contested).
      Ample hub budgets here mean the real lever is RESERVE TUNING: the natural
      second-price floor on an uncontested lot is set by a cheap filler bidder, far
      below the hub's true value, so a reserve close to that true value captures far
      more revenue -- reserve=0 (or a crude fixed-fraction reserve) leaves most of
      it on the table.

  (2) CHAIN hubs C_0 - C_1 - ... - C_{k-1}: consecutive hubs duel over a block of
      lots, block values strictly increasing along the chain, and within a duel the
      LOWER-indexed hub is the usual winner while the HIGHER-indexed hub is the
      valuable underbidder that sets the clearing price. Each interior hub C_i is
      therefore both a cheap "protector" (underbidder) for the lower-value block
      (i-1,i) and an expensive "spender" (winner) for the higher-value block
      (i,i+1). A schedule that sells lots by descending sticker value processes the
      priciest (highest-index) block first, so C_i blows its budget winning that
      block BEFORE the lower block (i-1,i) is sold -- once broke, C_i can no longer
      underbid there, and that block crashes to the filler floor. Selling blocks in
      ASCENDING chain order (or otherwise keeping each hub solvent while it is still
      needed as an underbidder) avoids the crash and captures much more revenue.

Hub/filler identities and lot rows are randomly permuted before printing so no
structure leaks through index order.
"""
import random
import sys


def build(test_id: int):
    rng = random.Random(900001 * test_id + 137)

    configs = [
        dict(n=10,  g_hubs=2, g_fillers=4,  c_hubs=3),
        dict(n=16,  g_hubs=2, g_fillers=6,  c_hubs=3),
        dict(n=24,  g_hubs=3, g_fillers=8,  c_hubs=3),
        dict(n=32,  g_hubs=3, g_fillers=10, c_hubs=4),
        dict(n=50,  g_hubs=3, g_fillers=14, c_hubs=4),
        dict(n=70,  g_hubs=3, g_fillers=18, c_hubs=5),
        dict(n=100, g_hubs=4, g_fillers=22, c_hubs=5),
        dict(n=140, g_hubs=4, g_fillers=28, c_hubs=6),
        dict(n=180, g_hubs=5, g_fillers=34, c_hubs=6),
        dict(n=200, g_hubs=5, g_fillers=38, c_hubs=7),
    ]
    sizecfg = configs[(test_id - 1) % len(configs)]
    n = sizecfg["n"]
    g_hubs, g_fillers, c_hubs = sizecfg["g_hubs"], sizecfg["g_fillers"], sizecfg["c_hubs"]

    alpha_g, alpha_c = 0.60, 0.25          # generic hubs flush; chain hubs tight
    bfrac = 0.25                            # fraction of generic lots that are bridges
    chain_frac = 0.30                       # fraction of n that is chain-structured
    vlo, vhi = 300, 2000
    chain_vlo, chain_step, chain_margin = 200, 200, 100
    flo, fhi = 15, 80

    m = g_hubs + c_hubs + g_fillers
    # bidder id layout: [0..g_hubs) generic hubs, [g_hubs..g_hubs+c_hubs) chain hubs
    # (chain index order = C_0..C_{k-1}), [g_hubs+c_hubs..m) fillers.

    n_chain = max(c_hubs - 1, int(n * chain_frac))
    n_chain = min(n_chain, n - g_hubs)      # leave room for >=1 lot per generic hub
    n_chain = max(n_chain, c_hubs - 1)
    n_generic = n - n_chain

    values = []

    # ---- generic portion ----
    base = n_generic // g_hubs
    extra = n_generic % g_hubs
    region_of = []
    for h in range(g_hubs):
        size = base + (1 if h < extra else 0)
        region_of += [h] * size
    while len(region_of) < n_generic:
        region_of.append(rng.randrange(g_hubs))
    rng.shuffle(region_of)

    for lot in range(n_generic):
        row = [0] * m
        h = region_of[lot]
        home_val = rng.randint(vlo, vhi)
        row[h] = home_val
        if rng.random() < bfrac and g_hubs > 1:
            other = rng.choice([x for x in range(g_hubs) if x != h])
            partner_val = max(50, min(3 * vhi, home_val + rng.randint(-150, 150)))
            row[other] = partner_val
        values.append(row)

    # ---- chain portion: c_hubs-1 duel blocks, values increasing along the chain ----
    n_duels = max(1, c_hubs - 1)
    per_duel = n_chain // n_duels
    chain_rows = []
    for g in range(n_duels):
        vlo_g = chain_vlo + g * chain_step
        vhi_g = vlo_g + chain_step
        cnt = per_duel if g < n_duels - 1 else (n_chain - per_duel * (n_duels - 1))
        for _ in range(cnt):
            row = [0] * m
            lo_val = rng.randint(vlo_g, vhi_g)
            hi_val = lo_val + rng.randint(0, chain_margin)
            row[g_hubs + g] = hi_val
            row[g_hubs + g + 1] = lo_val
            chain_rows.append(row)
    while len(chain_rows) < n_chain:
        row = [0] * m
        h = rng.randrange(c_hubs)
        row[g_hubs + h] = rng.randint(chain_vlo, chain_vlo + chain_step * n_duels)
        chain_rows.append(row)

    values += chain_rows
    assert len(values) == n

    # ---- filler bidders: cheap floor on everything ----
    for f in range(g_fillers):
        j = g_hubs + c_hubs + f
        k = rng.randint(1, 3)
        lots = rng.sample(range(n), min(k, n))
        for lot in lots:
            values[lot][j] = rng.randint(flo, fhi)

    # safety net: every lot needs >=1 positive-value bidder so the baseline sells it.
    for lot in range(n):
        if all(values[lot][j] == 0 for j in range(m)):
            j = rng.randrange(g_hubs + c_hubs, m) if g_fillers > 0 else rng.randrange(m)
            values[lot][j] = rng.randint(flo, fhi)

    # ---- budgets ----
    budgets = [0] * m
    for h in range(g_hubs):
        total = sum(values[lot][h] for lot in range(n))
        budgets[h] = max(1, round(alpha_g * total))
    for h in range(c_hubs):
        j = g_hubs + h
        total = sum(values[lot][j] for lot in range(n))
        budgets[j] = max(1, round(alpha_c * total))
    for f in range(g_fillers):
        j = g_hubs + c_hubs + f
        mx = max((values[lot][j] for lot in range(n)), default=0)
        budgets[j] = max(1, (mx * rng.randint(11, 15)) // 10)

    # shuffle lot rows and bidder columns so index order carries no structure.
    lot_perm = list(range(n))
    rng.shuffle(lot_perm)
    values = [values[lot_perm[i]] for i in range(n)]

    col_perm = list(range(m))
    rng.shuffle(col_perm)
    values = [[row[col_perm[j]] for j in range(m)] for row in values]
    budgets = [budgets[col_perm[j]] for j in range(m)]

    return n, m, values, budgets


def main():
    test_id = int(sys.argv[1])
    n, m, values, budgets = build(test_id)
    out = [f"{n} {m}"]
    for row in values:
        out.append(" ".join(map(str, row)))
    out.append(" ".join(map(str, budgets)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
