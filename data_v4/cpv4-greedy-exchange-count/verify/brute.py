import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    iv = []
    for _ in range(n):
        l = int(data[idx]); r = int(data[idx+1]); idx += 2
        iv.append((l, r))

    # Reproduce the canonical greedy stabbing set (sort by right endpoint).
    order = sorted(iv, key=lambda t: (t[1], t[0]))
    pts = []
    last = None
    for (l, r) in order:
        if last is None or last < l:
            last = r
            pts.append(last)

    num_points = len(pts)

    # Independent counting: for EACH interval, brute-count how many placed points
    # fall inside it (O(n * #points) double loop). "multi" = intervals with >= 2.
    multi = 0
    for (l, r) in iv:
        cnt = 0
        for p in pts:
            if l <= p <= r:
                cnt += 1
        if cnt >= 2:
            multi += 1

    print(num_points, multi)

main()
