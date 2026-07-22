#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy TRAIN sample to stdout.

Theme: deduce a buried crystal's spring law from a muffled echo sweep.

Hidden physics (lives here AND in verify.py, never printed): a uniform 1-D
mass-spring chain of N atoms, fixed at both ends, mass m and spring constant K
identical everywhere, driven harmonically at one end with damping gamma and
probed at the far end. The steady-state response amplitude vs driving
frequency shows resonance peaks at the chain's normal-mode frequencies. Each
testId fixes a DIFFERENT hidden omega0=sqrt(K/m) and gamma. On top of the
ideal sine pole law there is a small higher-harmonic anharmonic correction
whose strength RAMPS UP with chain size (real small crystals behave almost
ideally; the imperfection only becomes material in much bigger specimens) --
it is exactly zero for every training-sized chain here, so it leaves no
trace in what the solver sees, but the held-out grading (large chains, inside
verify.py) is not perfectly pure sine.

The solver only ever sees amplitude-vs-frequency sweeps for several SMALL
chains (N), and the driving-frequency sweep is confined to a PARTIAL band (the
echo equipment is muffled / band-limited) -- many resonances, especially the
higher modes and anything on much larger chains, are never swept at all. The
held-out grading (inside verify.py) asks for resonance frequencies of MUCH
LARGER chains and at mode indices whose true frequency lies outside the swept
band -- genuine size + frequency extrapolation, regenerated only in the
checker. STDOUT here prints ONLY test id + amplitude/frequency rows -- no
hidden constants, no mode labels beyond ascending order, no law.
"""
import sys, math, random

N_POINTS = 130
RAMP_LO, RAMP_HI = 12.0, 50.0   # correction is 0 up to the largest possible
                                  # training chain (pool max 12), full by N=50
CORR_STRENGTH = 0.15


def build_instance(test_id):
    rng = random.Random(7000003 * test_id + 911)
    omega0 = rng.uniform(0.8, 2.2)
    gamma = rng.uniform(0.04, 0.10) * omega0
    if test_id <= 3:
        frac_band = rng.uniform(0.62, 0.78)
        n_sizes = rng.randint(3, 4)
        pool = list(range(3, 9))
    elif test_id <= 7:
        frac_band = rng.uniform(0.42, 0.58)
        n_sizes = rng.randint(3, 5)
        pool = list(range(3, 11))
    else:
        frac_band = rng.uniform(0.28, 0.38)
        n_sizes = rng.randint(4, 5)
        pool = list(range(4, 13))
    rng.shuffle(pool)
    N_train = sorted(pool[:n_sizes])
    noise_sigma = rng.uniform(0.03, 0.07)
    ho_pool = [n for n in range(25, 161) if n not in N_train]
    rng.shuffle(ho_pool)
    N_test = sorted(ho_pool[:4])
    return dict(omega0=omega0, gamma=gamma,
                frac_band=frac_band, N_train=N_train,
                noise_sigma=noise_sigma, N_test=N_test)


def true_omega(j, N, omega0):
    x = j * math.pi / (2.0 * (N + 1))
    base = 2.0 * omega0 * math.sin(x)
    ramp = max(0.0, min(1.0, (N - RAMP_LO) / (RAMP_HI - RAMP_LO)))
    corr = CORR_STRENGTH * ramp * 2.0 * omega0 * math.sin(3.0 * x)
    return base + corr


def amplitude(omega, N, inst):
    omega0, gamma = inst["omega0"], inst["gamma"]
    s = 0.0
    for j in range(1, N + 1):
        wj = math.sin(j * math.pi / (N + 1)) ** 2 + 0.08
        oj = true_omega(j, N, omega0)
        denom = (oj * oj - omega * omega) ** 2 + (gamma * omega) ** 2
        s += wj / math.sqrt(max(denom, 1e-12))
    return s


def main():
    test_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    inst = build_instance(test_id)
    omega0, frac_band = inst["omega0"], inst["frac_band"]
    band_max = frac_band * 2.0 * omega0
    noise_rng = random.Random(31337 + test_id * 104729)

    lines = [str(test_id), str(len(inst["N_train"]))]
    for N in inst["N_train"]:
        lines.append("%d %d" % (N, N_POINTS))
        domega = band_max / N_POINTS
        for k in range(1, N_POINTS + 1):
            omega = domega * k
            A = amplitude(omega, N, inst)
            A_noisy = A * (1.0 + inst["noise_sigma"] * noise_rng.gauss(0.0, 1.0))
            if A_noisy < 1e-6:
                A_noisy = 1e-6
            lines.append("%.6f %.6f" % (omega, A_noisy))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
