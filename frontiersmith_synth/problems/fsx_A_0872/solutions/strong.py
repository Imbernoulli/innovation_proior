# TIER: strong
"""The insight: don't inject a big seed, engineer the dispersion relation.

The kernel's Fourier transform is
    lambda(m) = w_0 + 2 * sum_{j=1}^{L} w_j * cos(2*pi*j*m/N)
which is exactly the growth-rate curve of the linear stage. Shaping the
kernel itself as a windowless cosine tuned to the TARGET wavenumber,
    w_j = A * cos(2*pi*k*j/N),   j = 0..L_max
makes lambda(m) peak (a Dirichlet-kernel resonance) at m = k, so mode k
reliably outgrows every competitor before the cubic term saturates it --
regardless of which particular noise draw seeded the run. No bias is needed:
the ensemble of noise draws already contains a nonzero k-component; shaping
growth is what makes THAT component win robustly, not amplitude injected by
hand. Amplitude A is pushed to (a safe fraction of) the allowed cap so the
growth-rate gap over neighboring wavenumbers is as large as the budget
allows, without risking numerical overshoot from an unwindowed kernel that
uses every last allowed tap at extreme amplitude."""
import sys, json, math


def main():
    inst = json.load(sys.stdin)
    n = inst["N"]
    k = inst["k"]
    Lmax = inst["L_max"]
    wmax = inst["W_max"]

    amp = min(0.5, wmax)
    kernel = [amp * math.cos(2 * math.pi * k * j / n) for j in range(Lmax + 1)]
    bias = [0.0] * n

    print(json.dumps({"kernel": kernel, "bias": bias}))


main()
