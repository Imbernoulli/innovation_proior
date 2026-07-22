# TIER: strong
"""Insight: the amplitude curve is secondary -- the echo PEAK POSITIONS are
poles of a driven finite chain, which for a uniform 1-D lattice with fixed
ends must be BOUNDED (a normal mode cannot exceed 2*sqrt(K/m)) and periodic
in the wavenumber, i.e. of the sine-dispersion shape omega_j(N) =
A * sin(j*pi / (2*(N+1))), not a raw line in j/(N+1). Fit the single
proportionality constant A directly against sin(j*pi/(2*(N+1))) (not against
j/(N+1)) using every detected peak across every training chain size jointly
-- this is a one-parameter regression in the CORRECT nonlinear feature, and
it recovers the pole law itself rather than the shape of one curve."""
import sys, math


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
            x = math.sin(rank * math.pi / (2.0 * (N + 1)))
            xs.append(x); ys.append(om)
    if not xs:
        print("2.0 * sin(pi * j / (2 * (N + 1)))")
        return
    num = sum(x * y for x, y in zip(xs, ys))
    den = sum(x * x for x in xs)
    A = num / den if den > 1e-9 else 2.0
    print("%.6f * sin(pi * j / (2 * (N + 1)))" % A)


if __name__ == "__main__":
    main()
