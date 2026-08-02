# TIER: strong
"""
The insight: throughput can never exceed EITHER hardware ceiling converted
to a per-sample rate. Compute the roofline ceiling

    P = min(C/F, W/D)

EXACTLY from the four given constants -- no fitting needed for the ceiling
itself, only recognising which of the two ratios (compute-bound vs
bandwidth-bound) is the binding one for this kernel.

The remaining unknown is how fast the ramp approaches that ceiling. Model
throughput as the saturating ramp  T(x) = P*x/(x+K)  and note its reciprocal
is LINEAR in 1/x with a KNOWN intercept:

    1/T(x) = 1/P + (K/P) * (1/x)

Since 1/P is already known exactly, fit only the one remaining slope
(K/P) by fixed-intercept least squares against the noisy training rows --
this uses the sub-knee training ramp to recover the knee scale without ever
having to see the plateau directly. Emit P*x/(x + K_hat).
"""
import sys


def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    idx += 1  # test id
    C = float(data[idx]); idx += 1
    W = float(data[idx]); idx += 1
    F = float(data[idx]); idx += 1
    D = float(data[idx]); idx += 1
    xs, ys = [], []
    for _ in range(n):
        x = float(data[idx]); idx += 1
        y = float(data[idx]); idx += 1
        xs.append(x); ys.append(y)

    P = min(C / F, W / D)
    a0 = 1.0 / P

    num = 0.0
    den = 0.0
    for x, y in zip(xs, ys):
        if y <= 0:
            continue
        u = 1.0 / x
        z = 1.0 / y
        num += u * (z - a0)
        den += u * u
    m_slope = (num / den) if den > 1e-12 else 0.0
    K_hat = m_slope * P
    if K_hat <= 1e-6:
        K_hat = 1.0

    print(f"{P!r} * x / ( x + {K_hat!r} )")


if __name__ == "__main__":
    main()
