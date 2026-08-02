#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE calm-period TRAIN trace to stdout.

Family: tide-plus-surge-forecast.  A coastal water-level station has a tide
gauge.  Two physical drivers are modelled separately for each timestamp:
  - T(t): the astronomic tide -- a fixed sum of harmonic constituents
    (frequencies, amplitudes, phases hidden inside `instance_params`).
  - S(t): a surge-forcing proxy (wind/pressure driven), also deterministic.

The TRUE water level is NOT the plain sum T+S.  In shallow water the surge
and tide interact nonlinearly: y = T + S - kappa*T*S + noise.  kappa is an
independently-surveyed shallow-water interaction coefficient for this
station (bathymetry-derived, not something you can refit from a gentle
week).  It is handed to you directly in the header because the historical
log below was recorded during CALM weather, where S stays small and the
kappa*T*S correction is far smaller than the sensor noise -- statistically
invisible in this very series.  The held-out grading event is a storm
surge, generated later and NEVER shown here, timed to land near a HIGH
tide (a deliberately adversarial "peak-on-peak" case): there the
correction is no longer negligible.

STDOUT format:
  line 1: "<testId> <n_train>"
  line 2: "<kappa>"                (float, 6 decimals)
  then n_train lines: "<T> <S> <y>" (floats, 6 decimals)

Nothing else is printed: no seed, no harmonic constituents, no storm
parameters.  Only the calm observational rows and the survey constant.
"""
import sys, math, random

FREQS = [1.0, 1.9322, 0.1341]          # fixed harmonic frequencies (cycles / time unit)
N_TRAIN = 150
T_SPAN = 30.0                            # calm training window: t in [0, T_SPAN)


def instance_params(testId):
    """Hidden per-station parameters (lives in gen.py AND verify.py, never printed
    beyond kappa itself). Deterministic function of testId only."""
    rng = random.Random(20000 + testId * 7919)
    amps = [0.7 + 0.3 * rng.random() for _ in FREQS]
    phases = [rng.uniform(0.0, 2 * math.pi) for _ in FREQS]
    T0 = 0.0
    surge_freq = 0.31 + 0.01 * rng.random()
    surge_phase = rng.uniform(0.0, 2 * math.pi)
    calm_surge_amp = 0.08 + 0.06 * rng.random()
    kappa = 0.30 + 0.02 * testId + 0.003 * rng.random()
    storm_mult = 2.5 + 0.3 * testId + 0.09 * rng.random()
    storm_width = 1.2 + 0.1 * rng.random()
    train_noise_std = 0.02
    storm_noise_std = 0.10
    return dict(amps=amps, phases=phases, T0=T0, surge_freq=surge_freq,
                surge_phase=surge_phase, calm_surge_amp=calm_surge_amp,
                kappa=kappa, storm_mult=storm_mult, storm_width=storm_width,
                train_noise_std=train_noise_std, storm_noise_std=storm_noise_std)


def tide(t, p):
    return p['T0'] + sum(A * math.cos(2 * math.pi * f * t + ph)
                          for A, f, ph in zip(p['amps'], FREQS, p['phases']))


def calm_surge(t, p):
    return p['calm_surge_amp'] * math.sin(2 * math.pi * p['surge_freq'] * t + p['surge_phase'])


def main():
    testId = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    p = instance_params(testId)
    rng = random.Random(30000 + testId * 104729)
    dt = T_SPAN / N_TRAIN

    lines = ["%d %d" % (testId, N_TRAIN), "%.6f" % p['kappa']]
    for i in range(N_TRAIN):
        t = i * dt
        T = tide(t, p)
        S = calm_surge(t, p)
        noise = rng.gauss(0.0, p['train_noise_std'])
        y = T + S - p['kappa'] * T * S + noise
        lines.append("%.6f %.6f %.6f" % (T, S, y))

    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
