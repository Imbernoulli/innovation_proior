#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy WINTER (visible-window) telemetry trace
for a rooftop/field PV array to stdout.

Hidden per-site physics (lives in gen.py AND verify.py, never printed):
  raw(G,T)  = eff * N * (G / 1000.0) * (1 - 0.004 * (T - 25.0))     # STC-normalised
  P_true    = min(raw(G,T), rho * N)   + sensor/microclimate noise   # inverter clip
  eff  in [0.92, 1.03]   -- per-array mismatch/soiling factor (never printed)
  rho  in [0.58, 0.88]   -- inverter clipping level as a FRACTION of nameplate
                            (never printed; the statement only gives the *range*)

The TRAIN trace handed to the solver is drawn from a SLOW WINTER regime: low
sun elevation keeps irradiance G safely below the level that would ever reach
the clip (this is enforced exactly, using the coldest-temperature -- i.e.
worst-case -- breakeven irradiance for THIS site, so no training row ever
clips). In that regime raw(G,T) < rho*N always, so the data looks like a
clean (noisy) product of G and a mild temperature factor -- there is no
visible "knee".

The HELD-OUT grading trace (regenerated only inside verify.py, never printed
here) is a FAST SUMMER regime: irradiance regularly exceeds the clip level,
so a large fraction of held-out rows are flat at the ceiling. A predictor
that only ever saw the winter branch has no direct evidence of the ceiling's
existence or level -- inferring it is the whole point.

STDOUT prints ONLY:
  line 1: "<n_train> <test_id> <N>"     N = nameplate DC capacity (kW)
  n_train lines: "<G> <T> <P>"          irradiance W/m^2, temp C, power kW
No seed, no eff, no rho, no formula is ever printed.
"""
import sys, random


def site_params(t):
    """Hidden per-site physics for this test id (duplicated verbatim in verify.py)."""
    rng = random.Random(90210 + t * 7919)
    N = rng.uniform(50.0, 150.0)      # nameplate DC capacity, kW
    eff = rng.uniform(0.92, 1.03)     # array mismatch/soiling factor
    rho = rng.uniform(0.58, 0.88)     # inverter clip level as a fraction of N
    return N, eff, rho


SIGMA_FRAC = 0.11  # sensor + microclimate noise, as a fraction of N


def train_rows(t, n):
    rng = random.Random(11000 + t * 104729)
    N, eff, rho = site_params(t)
    Pcap = rho * N
    # worst-case (coldest -> highest-efficiency-factor) breakeven irradiance,
    # so we can guarantee NO training row ever reaches the clip.
    Tcold = -5.0
    factor_cold = 1.0 - 0.004 * (Tcold - 25.0)
    Gbreak_cold = 1000.0 * rho / (eff * factor_cold)
    Gtrain_max = 0.80 * Gbreak_cold
    rows = []
    for _ in range(n):
        T = rng.uniform(-5.0, 12.0)
        G = rng.uniform(0.0, Gtrain_max)
        factor = 1.0 - 0.004 * (T - 25.0)
        raw = eff * N * (G / 1000.0) * factor
        P = raw + rng.gauss(0.0, SIGMA_FRAC * N)
        P = max(0.0, P)
        rows.append((G, T, P))
    return rows, N


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    n = 220 + 18 * (t - 1)
    rows, N = train_rows(t, n)
    out = ["%d %d %.6f" % (n, t, N)]
    for G, T, P in rows:
        out.append("%.6f %.6f %.6f" % (G, T, P))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
