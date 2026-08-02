#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE historical-record TRAIN sample to stdout.

Family: crop-yield-stress-forecast.  A breeding station keeps a season-by-
season yield log for one field.  For every recorded season it derives two
weather-summary numbers from that season's own daily maximum-temperature
trace:

  - G: total growing-degree-days (GDD) accumulated over the whole season,
    the standard capped heat-accumulation sum used to predict crop growth.
  - H: heat EXCEEDANCE accumulated only across that season's own FLOWERING
    window (a data-dependent stretch of days located by that season's
    thermal history, not a fixed calendar date -- a cool spring pushes
    flowering later, a warm spring pulls it earlier).

The true agronomic law behind the logged yield is

    yield = Y0 + a*G - BETA*H**2 + (measurement noise)

BETA is a physiological heat-sensitivity coefficient for this cultivar,
independently measured in a growth-chamber assay -- it is handed to you
directly in the header, because the historical log below was recorded
during ordinary seasons.  In an ordinary season the flowering window
simply never sees an extreme-heat excursion, so H stays tiny (H**2 is
utterly swamped by the noise): nothing in these residuals will ever
statistically reveal BETA, no matter how carefully you regress this log.

You will be graded on a DIFFERENT, held-out season: a heat wave, timed to
land on top of that season's OWN flowering window -- exactly where the
squared penalty stops being negligible. That event, and its yield, are
never shown to you.

STDOUT format:
  line 1: "<testId> <n_train>"
  line 2: "<BETA>"                       (float, 6 decimals)
  then n_train lines: "<G> <H> <y>"      (floats, 6 decimals)

Nothing else is printed: no seed, no daily temperatures, no flowering
dates, no Y0/a. Only the two season-summary numbers, the observed yield,
and the survey constant.
"""
import sys, math, random

L = 100                     # season length in days
TBASE = 10.0                 # GDD base temperature
TCAP = 30.0                  # GDD growth-response cap temperature
THEAT = 32.3                 # extreme-heat exceedance threshold
FLOWER_DURATION = 12
BASE_FLOWER_START = 30.0
WARMTH_SHIFT_COEF = 14.0
N_TRAIN = 60


def instance_params(testId):
    """Hidden per-field parameters. Deterministic function of testId only.
    Lives in gen.py AND verify.py, never printed beyond BETA itself."""
    rng = random.Random(40000 + testId * 7919)
    Y0 = 40.0 + 6.0 * rng.random()
    a_gdd = 0.018 + 0.004 * rng.random()
    BETA = 0.024 + 0.0044 * testId + 0.0007 * rng.random()
    Tmid = 21.0 + 2.0 * rng.random()
    Tamp = 8.0 + 2.0 * rng.random()
    Wscale = 3.0 + 0.5 * rng.random()
    day_noise_std = 0.8 + 0.2 * rng.random()
    calm_bump_amp = 1.0 + 0.7 * rng.random()
    calm_bump_width = 4.0 + 2.0 * rng.random()
    storm_bump_amp = 3.8 + 0.24 * testId + 0.3 * rng.random()
    storm_bump_width = 3.0 + 1.0 * rng.random()
    train_noise_std = 1.0
    holdout_noise_std = 1.4
    return dict(Y0=Y0, a_gdd=a_gdd, BETA=BETA, Tmid=Tmid, Tamp=Tamp,
                Wscale=Wscale, day_noise_std=day_noise_std,
                calm_bump_amp=calm_bump_amp, calm_bump_width=calm_bump_width,
                storm_bump_amp=storm_bump_amp, storm_bump_width=storm_bump_width,
                train_noise_std=train_noise_std, holdout_noise_std=holdout_noise_std)


def flower_window(w, p):
    fs = BASE_FLOWER_START - WARMTH_SHIFT_COEF * w
    fs = int(round(fs))
    fs = max(1, min(L - FLOWER_DURATION + 1, fs))
    fe = fs + FLOWER_DURATION - 1
    return fs, fe


def gdd_and_flower_exceed(tmax_of_day, fs, fe):
    """tmax_of_day: callable d -> Tmax(d) for d in 1..L (already includes
    any bumps/noise). Returns (gdd_total, flower_exceed)."""
    gdd = 0.0
    flower_exc = 0.0
    for d in range(1, L + 1):
        Tm = tmax_of_day(d)
        gdd += max(0.0, min(Tm, TCAP) - TBASE)
        if fs <= d <= fe:
            flower_exc += max(0.0, Tm - THEAT)
    return gdd, flower_exc


def train_row(rng, p):
    w = rng.uniform(-1.0, 1.0)
    bump_center = rng.uniform(1.0, L)
    day_noise = [rng.gauss(0.0, p['day_noise_std']) for _ in range(L)]

    def tmax(d):
        base = (p['Tmid'] + p['Tamp'] * math.sin(math.pi * d / L)
                + w * p['Wscale'] + day_noise[d - 1])
        bump = p['calm_bump_amp'] * math.exp(-((d - bump_center) ** 2) / (2.0 * p['calm_bump_width'] ** 2))
        return base + bump

    fs, fe = flower_window(w, p)
    gdd, flower_exc = gdd_and_flower_exceed(tmax, fs, fe)
    y = p['Y0'] + p['a_gdd'] * gdd - p['BETA'] * (flower_exc ** 2) + rng.gauss(0.0, p['train_noise_std'])
    return gdd, flower_exc, y


def main():
    testId = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    p = instance_params(testId)
    rng = random.Random(50000 + testId * 104729)

    lines = ["%d %d" % (testId, N_TRAIN), "%.6f" % p['BETA']]
    for _ in range(N_TRAIN):
        gdd, flower_exc, y = train_row(rng, p)
        lines.append("%.6f %.6f %.6f" % (gdd, flower_exc, y))

    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
