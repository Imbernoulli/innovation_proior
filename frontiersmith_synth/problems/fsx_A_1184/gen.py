#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE chromatogram instance to stdout.

Family: chromatogram-peak-unmix.  A reference retention-index table lists M
candidate compound slots at fixed nominal retention times r_1 < ... < r_M
(spacing D).  A hidden subset of the slots is truly present, each with a
hidden area.  Every peak in THIS run is shaped by the SAME physical column,
so it is drawn from an exponentially-modified-Gaussian (EMG) profile sharing
ONE asymmetry (tailing) time constant tau across ALL compounds -- tau is a
column property, not a per-compound one.  tau itself is NOT printed; the
solver only ever sees the summed, noisy intensity trace.

The hidden truth (which slots are present, their areas, and tau) is
regenerated identically -- and ONLY -- inside verify.py from the testId; it
is never written to stdout here.
"""
import sys, math, random

D = 20.0
SIGMA = 3.0
BASE_AREA = 40.0
THR = 0.25 * BASE_AREA
LAM = 0.15


def emg(t, mu, sigma, tau):
    """Exponentially-modified-Gaussian pdf (unit area), tau=0 -> plain Gaussian."""
    if tau < 1e-9:
        z = (t - mu) / sigma
        return math.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))
    z = t - mu
    a = (sigma * sigma) / (2.0 * tau * tau) - z / tau
    a = max(-700.0, min(700.0, a))
    arg = sigma / (tau * math.sqrt(2.0)) - z / (sigma * math.sqrt(2.0))
    return math.exp(a) * math.erfc(arg) / (2.0 * tau)


def instance_spec(testId):
    """Deterministic ground truth + layout for this testId.  Duplicated verbatim
    in verify.py (never derived from anything printed to stdout)."""
    rng = random.Random(90210 + testId * 7919)
    M = 8 + (testId - 1)
    if testId == 1:
        tau_true = 0.0
    else:
        tau_true = min(10.0, 2.0 + 1.0 * (testId - 2))
    r = [int(D * (i + 1)) for i in range(M)]
    T = int(M * D + 8 * SIGMA + 4 * tau_true + 20)

    K = max(2, int(round(M * 0.5)))
    order = list(range(M))
    rng.shuffle(order)
    present = sorted(order[:K])
    areas = {i: rng.uniform(0.6, 1.6) * BASE_AREA for i in present}

    # planted trap: a heavily-tailed real peak immediately followed by an EMPTY
    # slot.  A symmetric (tau=0) fit spills the peak's own tail mass into that
    # empty neighbour and reports a phantom compound there.
    if testId >= 4 and tau_true > 0:
        candidates = [i for i in present if i + 1 < M]
        rng.shuffle(candidates)
        placed = False
        for j in candidates:
            if (j + 1) not in present:
                areas[j] = rng.uniform(1.8, 2.6) * BASE_AREA
                placed = True
                break
        if not placed and candidates:
            j = candidates[0]
            present = sorted(set(present) - {j + 1})
            areas.pop(j + 1, None)
            areas[j] = rng.uniform(1.8, 2.6) * BASE_AREA

    return dict(T=T, M=M, r=r, sigma=SIGMA, thr=THR, lam=LAM,
                tau_true=tau_true, present=set(present), areas=areas)


def build_trace(spec, testId):
    T, M, r = spec['T'], spec['M'], spec['r']
    sigma, tau_true = spec['sigma'], spec['tau_true']
    present, areas = spec['present'], spec['areas']

    trace = [0.0] * T
    for i in present:
        a = areas[i]
        mu = r[i]
        for t in range(T):
            trace[t] += a * emg(float(t), float(mu), sigma, tau_true)

    peak_h = BASE_AREA / (sigma * math.sqrt(2.0 * math.pi))
    noise_sigma = 0.01 * peak_h
    nrng = random.Random(555 + testId * 31)
    for t in range(T):
        trace[t] = max(0.0, trace[t] + nrng.gauss(0.0, noise_sigma))
    return trace


def main():
    testId = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    spec = instance_spec(testId)
    trace = build_trace(spec, testId)

    out = []
    out.append("%d %d %d" % (testId, spec['T'], spec['M']))
    out.append("%.6f %.6f %.6f" % (spec['sigma'], spec['thr'], spec['lam']))
    out.append(" ".join(str(v) for v in spec['r']))
    out.append(" ".join("%.4f" % v for v in trace))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
