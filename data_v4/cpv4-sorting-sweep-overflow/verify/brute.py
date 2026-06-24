import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    streams = []
    coords = set()
    for _ in range(n):
        s = int(data[idx]); e = int(data[idx+1]); w = int(data[idx+2]); idx += 3
        if s < e:
            streams.append((s, e, w))
            coords.add(s)
    # The load is a step function that changes only at start coordinates of streams.
    # On the half-open intervals [s, e), the peak (if any positive weight exists) is
    # attained at some stream's start time s. Evaluate the load at every distinct s.
    best = 0
    for t in sorted(coords):
        load = 0
        for (s, e, w) in streams:
            if s <= t < e:
                load += w
        if load > best:
            best = load
    print(best)

main()
