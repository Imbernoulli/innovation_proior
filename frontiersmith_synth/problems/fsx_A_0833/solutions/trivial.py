# TIER: trivial
"""Ignore j and N entirely: read off the single loudest echo across all the
training sweeps and predict that one constant frequency for everything."""
import sys


def read_input():
    data = sys.stdin.read().split("\n")
    idx = 0
    int(data[idx].strip()); idx += 1  # test id (unused)
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


def main():
    train = read_input()
    peak_freqs = []
    for _N, omegas, amps in train:
        best_i = max(range(len(amps)), key=lambda i: amps[i])
        peak_freqs.append(omegas[best_i])
    const = sum(peak_freqs) / len(peak_freqs) if peak_freqs else 1.0
    print("%.6f" % const)


if __name__ == "__main__":
    main()
