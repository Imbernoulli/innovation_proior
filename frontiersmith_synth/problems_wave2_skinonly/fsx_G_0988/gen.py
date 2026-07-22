#!/usr/bin/env python3
import random
import sys

SALT = 501094890


def item_mask(i, m, salt, profile, rng):
    bits = set()
    span = 3 + ((i + profile) % 3)
    anchor = (i * 7 + salt + profile * 11) % m
    for k in range(span):
        bits.add((anchor + k * (profile + 3) + i * (k + 1)) % m)
    if (i + salt) % 9 == 0:
        bits.add((anchor + m // 2 + profile) % m)
    if rng.randrange(5) == 0:
        bits.add(rng.randrange(m))
    mask = 0
    for b in bits:
        mask |= 1 << b
    return mask


def main():
    t = 1
    if len(sys.argv) > 1:
        try:
            t = int(sys.argv[1])
        except ValueError:
            t = 1
    t = max(1, min(10, t))
    rng = random.Random(SALT * 1000003 + t * 9176)
    profile = SALT % 11
    n = 38 + 4 * t + (SALT % 13) + 2 * (profile % 4)
    m = 18 + (SALT % 8) + (t % 3)
    groups = 5 + (SALT % 5)
    group_cap = 3 + ((SALT + t) % 3)

    rows = []
    for i in range(n):
        group = (i * 37 + SALT + rng.randrange(groups)) % groups
        cost = 5 + ((i * 19 + SALT * 3 + t * 5) % 18) + (group % 4)
        if i < n // 5:
            value = 24 + ((i * 13 + t + SALT) % 28)
        else:
            value = 38 + ((i * 31 + SALT + t * 7) % 90)
        if (i + profile) % (7 + profile % 4) == 0:
            value += 36 + 4 * t
        if (i * 5 + SALT) % 17 == 0:
            cost = max(3, cost - 5)
            value += 18
        x = (i * 29 + SALT * 7 + rng.randrange(97)) % 211
        y = (i * 43 + SALT * 11 + rng.randrange(103)) % 211
        mask = item_mask(i, m, SALT + t * 13, profile, rng)
        rows.append((cost, group, value, x, y, mask))

    target = 10 + t + (profile % 5)
    cheapest = sorted(r[0] for r in rows)
    budget = sum(cheapest[:min(target, len(cheapest))]) + 6 + 2 * t

    print(n, m, groups, budget, group_cap, SALT + t * 13, profile)
    for cost, group, value, x, y, mask in rows:
        print(cost, group, value, x, y, mask)


if __name__ == "__main__":
    main()
