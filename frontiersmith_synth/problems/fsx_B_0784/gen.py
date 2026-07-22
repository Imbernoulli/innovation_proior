import sys, random

# Deterministic ladder: testId 1..10 -> (rail length L, color pattern).
# All instances share H=3 heads, K=4 colors, throttle distance d=3, setup cost S=5.
# N = L+1 strokes: one stroke at EVERY integer rail position 0..L (dense), listed
# in ASCENDING position order (this fixed order is also what the checker's internal
# single-head baseline sweeps through).
LADDER = [
    (10, "block"),
    (14, "mixed"),
    (18, "interleave"),
    (22, "block"),
    (26, "interleave"),
    (30, "interleave3"),
    (34, "mixed"),
    (40, "interleave"),
    (48, "shuffled"),
    (56, "interleave"),
]

H = 3
K = 4
D = 3
S = 5


def block_color(p, span, k):
    idx = int(p * k / (span + 1))
    if idx >= k:
        idx = k - 1
    return idx + 1


def color_for(pattern, p, L, rng_colors):
    if pattern == "block":
        return block_color(p, L, K)
    if pattern == "interleave":
        return (p % K) + 1
    if pattern == "interleave3":
        return ((p * 3) % K) + 1
    if pattern == "mixed":
        half = L // 2
        if p <= half:
            return block_color(p, half, K)
        return (p % K) + 1
    if pattern == "shuffled":
        return rng_colors[p]
    raise ValueError("bad pattern")


def main():
    i = int(sys.argv[1])
    idx = min(max(i, 1), len(LADDER)) - 1
    L, pattern = LADDER[idx]
    N = L + 1

    rng_colors = None
    if pattern == "shuffled":
        rng = random.Random(900000 + i)
        rng_colors = [(p % K) + 1 for p in range(N)]
        rng.shuffle(rng_colors)

    starts = []
    for h in range(H):
        s = round((h + 0.5) * L / H)
        s = min(max(s, 0), L)
        starts.append(s)

    lines = []
    lines.append("%d %d %d %d %d %d" % (L, H, D, S, K, N))
    lines.append(" ".join(str(x) for x in starts))
    for p in range(N):
        c = color_for(pattern, p, L, rng_colors)
        lines.append("%d %d" % (p, c))

    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
