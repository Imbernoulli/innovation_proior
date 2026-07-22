import sys, random, heapq

WALL = 90  # ticks-to-ignite sentinel so large it never fires within any test's horizon

DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def dijkstra(R, C, delay, origins, lakeset):
    dist = [[None] * C for _ in range(R)]
    heap = []
    for (r, c) in origins:
        if (r, c) in lakeset:
            continue
        heapq.heappush(heap, (0, r, c))
    while heap:
        t, r, c = heapq.heappop(heap)
        if dist[r][c] is not None:
            continue
        dist[r][c] = t
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C and (nr, nc) not in lakeset and dist[nr][nc] is None:
                heapq.heappush(heap, (t + delay[nr][nc], nr, nc))
    return dist


def build_instance(testId):
    rng = random.Random(9000 + testId * 17)

    R = 12 + testId * 2
    C = 12 + testId * 2
    corr_w = min(5 + testId // 2, C - 4)
    gap_w = 1 if testId % 3 == 1 else 2
    K = 2 + testId // 3
    tank = 4 + testId // 3
    Rfill = 3 + (testId % 3)
    move = 1
    band_thick = 2
    origin_delay = 2
    town_delay = 1

    delay = [[WALL] * C for _ in range(R)]
    left = (C - corr_w) // 2
    right = left + corr_w
    band_row = int(R * 0.5)
    gap_col = left + (corr_w - gap_w) // 2 + rng.choice([-1, 0, 1])
    gap_col = max(left, min(right - gap_w, gap_col))

    for r in range(0, band_row):
        for c in range(left, right):
            delay[r][c] = origin_delay
    for r in range(band_row, band_row + band_thick):
        for c in range(left, right):
            delay[r][c] = origin_delay if gap_col <= c < gap_col + gap_w else WALL
    for r in range(band_row + band_thick, R):
        for c in range(0, C):
            delay[r][c] = town_delay

    origins = [(0, c) for c in range(left, right)]

    lake_col = left - 1 if left >= 1 else right
    lakes = [(0, lake_col)]

    base0 = dijkstra(R, C, delay, origins, set(lakes))
    gap_cells = [(r, c) for r in range(band_row, band_row + band_thick)
                 for c in range(gap_col, gap_col + gap_w)]
    gap_dists = [base0[r][c] for (r, c) in gap_cells if base0[r][c] is not None]
    gap_dist = min(gap_dists) if gap_dists else band_row * origin_delay
    town_margin = origin_delay * band_thick + town_delay * (3 + testId // 2)
    Tmax = gap_dist + town_margin

    planes = [(lakes[0][0], lakes[0][1], tank) for _ in range(K)]

    return dict(R=R, C=C, Tmax=Tmax, move=move, Rfill=Rfill, origins=origins,
                lakes=lakes, planes=planes, delay=delay)


def main():
    testId = int(sys.argv[1])
    inst = build_instance(testId)
    R, C = inst["R"], inst["C"]
    out = []
    out.append(f"{R} {C} {inst['Tmax']}")
    out.append(f"{inst['move']} {inst['Rfill']}")
    out.append(str(len(inst["origins"])))
    for (r, c) in inst["origins"]:
        out.append(f"{r} {c}")
    out.append(str(len(inst["lakes"])))
    for (r, c) in inst["lakes"]:
        out.append(f"{r} {c}")
    out.append(str(len(inst["planes"])))
    for (r, c, tank) in inst["planes"]:
        out.append(f"{r} {c} {tank}")
    for r in range(R):
        out.append(" ".join(str(inst["delay"][r][c]) for c in range(C)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
