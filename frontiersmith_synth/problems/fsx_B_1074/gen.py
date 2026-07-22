#!/usr/bin/env python3
"""
gen.py <testId> -- Guild Clearinghouse settlement instance.

Builds a hidden per-party net-balance vector with a planted structure:
  - a reserved fraction of parties already net to exactly zero,
  - the rest are packed into small disjoint groups of size 2 or 3 whose values
    sum to exactly zero (the "true" zero-sum partition an insightful solver
    should recover),
  - ~10% of those groups are DECOYS: perturbed by a small nonzero delta so
    they do NOT sum to zero (the delta is parked on a separate corrector
    party so the grand total still nets to zero).

The balances are never printed directly. Instead the generator realizes them
as a season of raw pairwise IOUs (obligations): a chain construction over a
random party order reproduces the exact target balances using few "real"
edges, and a pile of self-cancelling noise pairs (x owes y w, y owes x w)
pads the obligation count and hides all structure -- a solver MUST net every
obligation per party to recover the balance vector; the raw edges alone
never reveal group membership.

Deterministic: all randomness seeded from testId only.
"""
import sys, random

N_LIST = [24, 40, 60, 90, 130, 180, 240, 310, 390, 480]


def build_balances(rnd, n):
    pool = list(range(n))
    rnd.shuffle(pool)

    balances = [0] * n
    zero_reserve = max(1, int(round(n * 0.56)))
    zero_reserve = min(zero_reserve, n - 2)  # leave room for at least one real group
    zero_parties = pool[:zero_reserve]
    rest = pool[zero_reserve:]

    decoy_deltas = []
    i = 0
    m = len(rest)
    while i < m:
        remaining = m - i
        if remaining == 1:
            # stray singleton -> fold into the zero-reserve pool
            zero_parties.append(rest[i])
            i += 1
            continue
        if remaining == 3:
            size = 3
        elif remaining >= 4:
            # heavily bias toward triples: an exact 3-way zero-sum match is
            # invisible to a pairwise max-debtor/max-creditor greedy, which can
            # only ever stumble onto 2-way exact cancellations.
            size = 3 if rnd.random() < 0.85 else 2
            if remaining - size == 1:
                size = 3  # avoid leaving exactly one straggler
        else:
            size = 2
        group = rest[i:i + size]
        i += size

        is_decoy = rnd.random() < 0.10
        if size == 2:
            v = rnd.randint(5, 240)
            vals = [v, -v]
        else:
            # two small same-signed "helper" values plus one big offsetting value --
            # the big value's magnitude lands squarely in the SAME range pair-cluster
            # magnitudes use, and the helper values overlap the low end of that range
            # too, so a magnitude-sorted greedy constantly finds a plausible (but
            # wrong) cross-cluster partner instead of this triple's true mates.
            sign = rnd.choice((-1, 1))
            h1 = rnd.randint(15, 150)
            h2 = rnd.randint(15, 150)
            v1 = -sign * h1
            v2 = -sign * h2
            v3 = sign * (h1 + h2)
            vals = [v1, v2, v3]

        if is_decoy:
            delta = rnd.choice((1, 2, 3, -1, -2, -3))
            vals[-1] += delta
            decoy_deltas.append(delta)

        for pid, val in zip(group, vals):
            balances[pid] = val

    for d in decoy_deltas:
        if zero_parties:
            p = zero_parties.pop()
            balances[p] = -d
        else:
            balances[rest[0]] -= d

    assert sum(balances) == 0, "internal generator invariant violated"
    return balances


def realize_edges(rnd, n, final_balance):
    """final_balance: dict/list indexed by party id 1..n. Returns list of (debtor, creditor, amount)."""
    chain_order = list(range(1, n + 1))
    rnd.shuffle(chain_order)

    edges = []
    carry = 0
    for t in range(n - 1):
        carry += final_balance[chain_order[t]]
        if carry != 0:
            a, b = chain_order[t], chain_order[t + 1]
            # obligation (d, c, amt): d owes c amt -> contributes -amt to d, +amt to c.
            # prefix(t) = sum of balances[order[0..t]]; to keep every prefix party's net
            # correct we need: prefix>0 -> b owes a; prefix<0 -> a owes b.
            if carry > 0:
                edges.append((b, a, carry))
            else:
                edges.append((a, b, -carry))

    noise_count = max(20, 5 * n)
    for _ in range(noise_count):
        x = rnd.randint(1, n)
        y = rnd.randint(1, n)
        if x == y:
            continue
        w = rnd.randint(1, 97)
        edges.append((x, y, w))
        edges.append((y, x, w))

    rnd.shuffle(edges)
    return edges


def main():
    if len(sys.argv) != 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(1)
    test_id = int(sys.argv[1])
    n = N_LIST[(test_id - 1) % len(N_LIST)]

    rnd = random.Random(20260721 + 104729 * test_id)

    balances0 = build_balances(rnd, n)

    id_perm = list(range(1, n + 1))
    rnd.shuffle(id_perm)
    # internal index i -> final party id id_perm[i]
    final_balance = [0] * (n + 1)
    for internal_idx, bal in enumerate(balances0):
        final_balance[id_perm[internal_idx]] = bal

    edges = realize_edges(rnd, n, final_balance)

    out = [f"{n} {len(edges)}"]
    for d, c, a in edges:
        out.append(f"{d} {c} {a}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
