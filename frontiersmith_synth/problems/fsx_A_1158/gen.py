import sys, random

# Difficulty/adversarial ladder for the icebreaker-channel-lease family.
# Each entry: (L, M, B, c, r, speed_palette, trap)
#   speed_palette: list of distinct per-cell paces to draw ships' speeds from.
#   trap=True  -> speeds and destinations are drawn INDEPENDENTLY (so destination-sorted
#                 batching mixes speeds inside convoys) AND at least one speed value has
#                 more than c ships (forcing either multiple escort trips or a lease wave).
LADDER = [
    # 1: tiny sanity, almost uniform speed
    (8,   3,  1, 2, 6,  [1, 1, 2],                 False),
    # 2: small, mild speed variety, still gentle
    (14,  6,  1, 2, 6,  [1, 2, 2, 3],               False),
    # 3: small trap -- one speed bucket already exceeds capacity
    (20,  10, 1, 2, 7,  [1, 1, 1, 1, 4, 4],         True),
    # 4: medium, two breakers, moderate mixing
    (30,  16, 2, 3, 8,  [1, 2, 3],                  False),
    # 5: medium trap -- generous refreeze window, wide speed spread
    (42,  24, 2, 3, 11, [1, 1, 1, 5, 5, 9],         True),
    # 6: medium trap -- tighter capacity relative to fleet, three buckets
    (55,  30, 2, 4, 9,  [2, 2, 6, 6, 6, 10],        True),
    # 7: large, gentler ladder step (sanity that scale alone isn't the trap)
    (80,  40, 3, 4, 14, [1, 2, 3, 4],                False),
    # 8: large trap -- heavy mixing, several mid-size buckets above capacity
    (95,  50, 2, 4, 10, [1, 1, 2, 2, 3, 3, 7, 7, 7, 12], True),
    # 9: adversarial -- tight-ish refreeze window stresses lease-timing precision
    (130, 60, 3, 5, 8,  [1, 1, 1, 3, 4, 5, 6, 9, 9, 12], True),
    # 10: adversarial-large -- biggest scale, many buckets so no single lease sweeps everyone
    (160, 70, 3, 5, 14, [1, 1, 2, 2, 3, 4, 5, 9, 9, 9, 12], True),
]

i = int(sys.argv[1])
idx = min(max(i, 1), len(LADDER)) - 1
L, M, B, c, r, palette, trap = LADDER[idx]

rng = random.Random(20260726 + 97 * i)

lines = ["%d %d %d %d %d" % (L, M, B, c, r)]
if trap:
    # Speeds and destinations chosen independently -> sorting by destination
    # (the obvious greedy batching order) interleaves speeds inside convoys.
    for _ in range(M):
        s = rng.choice(palette)
        d = rng.randint(1, L)
        w = rng.randint(1, 10)
        lines.append("%d %d %d" % (s, d, w))
else:
    # Gentler instances: destinations correlate loosely with speed bucket so the
    # obvious greedy batching happens to align reasonably often (still not perfect).
    for _ in range(M):
        s = rng.choice(palette)
        base = 1 + (s - 1) * (L // (max(palette) + 1))
        d = min(L, max(1, base + rng.randint(-L // 6 - 1, L // 6 + 1)))
        w = rng.randint(1, 10)
        lines.append("%d %d %d" % (s, d, w))

print("\n".join(lines))
