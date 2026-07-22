# TIER: greedy
"""The obvious first recipe: you have F bookmarks, so spend all F of them,
evenly, across chronological time. Split the M touches into F contiguous
equal-sized windows and drop one bookmark at each window's MEDIAN position,
right at the start of the window. This never looks at where the actual
epoch boundaries or revisited hubs are -- it blindly assumes bursts are
evenly sized and evenly spaced in time, which the trap cases violate hard:
one giant burst gets needlessly split across several windows/bookmarks
while a cluster of many tiny, widely-scattered bursts gets crammed into a
single window whose "median" is a meaningless point nowhere near any of
them.
"""
import sys


def main():
    data = sys.stdin.buffer.read().split()
    N = int(data[0]); M = int(data[1]); F = int(data[2])
    pts = [int(x) for x in data[3:3 + M]]

    chunk = (M + F - 1) // F
    events = []
    finger = 0
    start = 0
    while start < M and finger < F:
        end = min(M, start + chunk)
        window = sorted(pts[start:end])
        med = window[len(window) // 2]
        finger += 1
        events.append((start + 1, finger, med))
        start = end

    out = [str(len(events))]
    for (t, i, q) in events:
        out.append(f"{t} {i} {q}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
