#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy TRAIN sample to stdout.

Family: gpu-batch-throughput-forecast.  A fictitious accelerator serves one op
(a fixed kernel) at growing batch size.  Two hardware ceilings compete:

    compute ceiling  Qc = C / F   (peak FLOPs/s) / (FLOPs per sample)
    bandwidth ceiling Qw = W / D  (peak bytes/s)  / (bytes per sample)

and the attainable steady-state throughput at batch size x follows a
saturating ramp toward the tighter (smaller) ceiling:

    P   = min(Qc, Qw)
    T(x) = P * x / (x + K)

K (a per-instance "knee scale", hidden) is NEVER printed -- it lives only in
gen.py and verify.py.  The measured TRAIN batches are all well below the knee
(x << K), where T(x) is still climbing roughly with x (the "compute-bound"-
looking ramp).  The HELD-OUT grading batches (regenerated only inside
verify.py) are far past the knee, deep in the bandwidth-saturated plateau --
a different regime the solver never observes directly.

STDOUT prints ONLY:
    line 1:  "<n_train> <test_id>"
    line 2:  "<C> <W> <F> <D>"            (peak compute, peak bandwidth,
                                            flops/sample, bytes/sample -- all
                                            visible hardware/op numbers)
    n_train lines: "<batch> <measured_throughput>"
The hidden knee K and the seed are NEVER printed -- only data + the four
visible constants.
"""
import sys, random

TRAIN_X = [4, 8, 12, 16, 24, 32, 48, 64, 80, 100]


def params(t):
    """Hidden/visible instance parameters for test id t (lives in gen AND verify,
    must stay byte-identical between the two files)."""
    rng = random.Random(5_312_009 + t * 7919)
    Qc = rng.uniform(80.0, 1500.0)
    Qw = rng.uniform(80.0, 1500.0)
    F = rng.uniform(50.0, 500.0)     # FLOPs per sample
    D = rng.uniform(10.0, 200.0)     # bytes per sample
    C = Qc * F                        # peak compute, FLOPs/s
    W = Qw * D                        # peak bandwidth, bytes/s
    K = rng.uniform(60.0, 180.0)      # hidden knee scale (batch units)
    return C, W, F, D, K


def true_throughput(x, C, W, F, D, K):
    P = min(C / F, W / D)
    return P * x / (x + K)


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    C, W, F, D, K = params(t)

    sigma = 0.04 + 0.003 * (t - 1)     # mild difficulty ladder: noisier at later ids
    rng = random.Random(9_004_207 + t * 104729)

    rows = []
    for x in TRAIN_X:
        mu = true_throughput(x, C, W, F, D, K)
        y = mu * (1.0 + rng.gauss(0.0, sigma))
        y = max(y, 1e-6)
        rows.append((x, y))

    out = [f"{len(TRAIN_X)} {t}", f"{C:.6f} {W:.6f} {F:.6f} {D:.6f}"]
    for x, y in rows:
        out.append(f"{x} {y:.6f}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
