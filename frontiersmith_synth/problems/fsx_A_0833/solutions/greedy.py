# TIER: greedy
"""Textbook recipe: find the amplitude peaks (the echo resonances), number
them 1..k in ascending frequency, and fit a single straight line through the
origin in the natural wavenumber variable j/(N+1) -- the standard 'linear
dispersion' (Debye) approximation. It fits the training band (all low-order,
small-angle modes) almost perfectly, then is extrapolated as-is to every
(j, N). It has no way to know the true relation saturates near the zone
edge, so it overshoots badly there."""
import sys


def read_input():
    data = sys.stdin.read().split("\n")
    idx = 0
    int(data[idx].strip()); idx += 1
    n_train = int(data[idx].strip()); idx += 1
    train = []
    for _ in range(n_train):
        parts = data[idx].split(); idx += 1
        N = int(parts[0]); npts = int(parts[1])
        omegas = []; amps = []
        for _k in range(npts):
            a, b = data[idx].split(); idx += 1
            omegas.append(float(a)); amps.append(float(b))
        train.append((N, omegas, amps))
    return train


def find_peaks(omegas, amps):
    n = len(omegas)
    if n < 5:
        return []
    med = sorted(amps)[n // 2]
    thresh = med * 1.15
    cand = []
    for i in range(2, n - 2):
        window = amps[i - 2:i + 3]
        if amps[i] >= max(window) and amps[i] > thresh:
            cand.append((omegas[i], amps[i]))
    cand.sort()
    merged = []
    span = (omegas[-1] - omegas[0]) if n > 1 else 1.0
    min_gap = span / 60.0
    for om, am in cand:
        if merged and om - merged[-1][0] < min_gap:
            if am > merged[-1][1]:
                merged[-1] = (om, am)
        else:
            merged.append((om, am))
    return [om for om, _am in merged]


def main():
    train = read_input()
    xs = []; ys = []
    for N, omegas, amps in train:
        peaks = find_peaks(omegas, amps)
        for rank, om in enumerate(peaks, start=1):
            xs.append(rank / (N + 1.0))
            ys.append(om)
    if not xs:
        print("1.0 * j / (N + 1)")
        return
    num = sum(x * y for x, y in zip(xs, ys))
    den = sum(x * x for x in xs)
    A = num / den if den > 1e-9 else 1.0
    print("%.6f * j / (N + 1)" % A)


if __name__ == "__main__":
    main()
