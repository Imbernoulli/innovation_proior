#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic scorer for the four-hour
fragrance-blend-trajectory problem.

Reads the instance from <in> (K ingredients with descriptor/decay-rate/IFRA
cap, a KxK perceptual-masking table, and a 5-checkpoint target profile
spanning 0..4 hours) and the participant blend from <out>. Validates
strictly: token count, integer ranges, no duplicate ingredient, each
concentration finite and in (0, cap_i], and the total blend concentration
<= a fixed TOTAL_CAP=1.0. On ANY violation prints `Ratio: 0.0` and exits 0.

Otherwise simulates the deterministic evaporation model:
  raw_i(t)       = c_i(0) * exp(-k_i * t)                (volatility-layered release)
  perceived_i(t) = raw_i(t) / (1 + sum_{j!=i} mask[i][j]*raw_j(t))   (perceptual masking)
  profile_a(t)   = sum_i desc_i[a] * perceived_i(t)
at all 5 checkpoints, and scores the MEAN squared error `E` against the
target trajectory (averaged over the 5 checkpoints and 4 axes).
F = 1 / (E + EPS) (higher is better -- a genuine inverse-error goodness
measure; EPS keeps it finite and bounds how far a near-zero error can run
the ratio up, preserving headroom). The checker also builds its own
baseline: use ONLY the first provided ingredient, at its own IFRA cap,
evaluated the SAME way (no search, no trajectory reasoning) -> B.
sc = min(1000, 100*F/max(1e-9,B)); Ratio = sc/1000.

Pure function of (in,out): no randomness, no wall-time. O(K^2 * T) per
case with K<=8, T=5 -> a few hundred flops, far under the time limit.
"""
import sys
import math

TOTAL_CAP = 1.0
EPS = 0.01


def fail(reason):
    print("reason:", reason)
    print("Ratio: 0.0")
    sys.exit(0)


def parse_instance(path):
    with open(path) as f:
        toks = f.read().split()
    p = 0
    K = int(toks[p]); p += 1
    D = int(toks[p]); p += 1
    T = int(toks[p]); p += 1
    descs, ks, caps = [], [], []
    for _ in range(K):
        desc = [float(toks[p + a]) for a in range(D)]
        p += D
        k = float(toks[p]); p += 1
        cap = float(toks[p]); p += 1
        descs.append(desc); ks.append(k); caps.append(cap)
    mask = []
    for _ in range(K):
        row = [float(toks[p + j]) for j in range(K)]
        p += K
        mask.append(row)
    times = [float(toks[p + t]) for t in range(T)]
    p += T
    targets = []
    for _ in range(T):
        g = [float(toks[p + a]) for a in range(D)]
        p += D
        targets.append(g)
    return K, D, T, descs, ks, caps, mask, times, targets


def parse_output(path, K, caps):
    try:
        with open(path) as f:
            toks = f.read().split()
    except Exception:
        return None
    if len(toks) < 1:
        return None
    try:
        Mtok = toks[0]
        M = int(Mtok)
        if str(M) != Mtok:
            return None
    except ValueError:
        return None
    if M < 1 or M > K:
        return None
    if len(toks) != 1 + 2 * M:
        return None
    entries = []
    seen = set()
    p = 1
    total = 0.0
    for _ in range(M):
        itok = toks[p]; p += 1
        ctok = toks[p]; p += 1
        try:
            idx = int(itok)
            if str(idx) != itok:
                return None
        except ValueError:
            return None
        if idx < 1 or idx > K:
            return None
        if idx in seen:
            return None
        seen.add(idx)
        try:
            c = float(ctok)
        except ValueError:
            return None
        if not math.isfinite(c):
            return None
        cap = caps[idx - 1]
        if c <= 0.0 or c > cap + 1e-9:
            return None
        entries.append((idx - 1, c))
        total += c
    if total > TOTAL_CAP + 1e-9:
        return None
    return entries


def simulate_error(c0, descs, ks, mask, times, targets, K, D, T):
    tot = 0.0
    for ti in range(T):
        t = times[ti]
        raw = [c0[i] * math.exp(-ks[i] * t) for i in range(K)]
        perceived = [0.0] * K
        for i in range(K):
            denom = 1.0
            row = mask[i]
            for j in range(K):
                if j == i:
                    continue
                denom += row[j] * raw[j]
            perceived[i] = raw[i] / denom
        g = targets[ti]
        e = 0.0
        for a in range(D):
            s = 0.0
            for i in range(K):
                s += descs[i][a] * perceived[i]
            d = s - g[a]
            e += d * d
        tot += e
    return tot / T


def score_goodness(mean_sq_err):
    return 1.0 / (max(mean_sq_err, 0.0) + EPS)


def main():
    if len(sys.argv) < 3:
        fail("usage")
    in_path, out_path = sys.argv[1], sys.argv[2]
    try:
        K, D, T, descs, ks, caps, mask, times, targets = parse_instance(in_path)
    except Exception as e:
        fail("bad instance: %r" % (e,))

    entries = parse_output(out_path, K, caps)
    if entries is None:
        fail("malformed output (token count / range / duplicate / cap / total budget)")

    c0 = [0.0] * K
    for idx, c in entries:
        c0[idx] = c

    mse = simulate_error(c0, descs, ks, mask, times, targets, K, D, T)
    F = score_goodness(mse)

    # Baseline: only the first provided ingredient, at its own IFRA cap.
    c0_base = [0.0] * K
    c0_base[0] = caps[0]
    mse_base = simulate_error(c0_base, descs, ks, mask, times, targets, K, D, T)
    B = score_goodness(mse_base)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("mse=%.6f F=%.6f mse_base=%.6f B=%.6f" % (mse, F, mse_base, B))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
