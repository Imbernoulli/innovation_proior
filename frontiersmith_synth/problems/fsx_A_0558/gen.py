import sys, numpy as np

# ---- fixed instrument constants (shared with verify.py) ----
N       = 1024                 # time samples over one analysis window
W       = 41                   # sliding-RMS window length (odd)
F_MIN   = 96
F_MAX   = 500
AMPS    = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]

def main():
    i = int(sys.argv[1])
    rng = np.random.default_rng(50558 + i)

    # difficulty ladder ---------------------------------------------------
    if i <= 3:
        H, k, depth = 3, 6, 0.20 + 0.03 * (i - 1)          # shallow, budget covers all
    elif i <= 6:
        H, k, depth = 5, 8, 0.40 + 0.05 * (i - 4)          # sparse budget begins (4 pairs < 5)
    elif i <= 8:
        H, k, depth = 7, 8, 0.58 + 0.05 * (i - 7)          # deep + sparse (trap)
    else:
        H, k, depth = 9, 10, 0.66 + 0.06 * (i - 9)         # deepest + sparse (trap)

    # planted slow harmonics of the TARGET ENVELOPE (integer beat frequencies)
    gs   = rng.choice(np.arange(1, 15), size=H, replace=False)
    gs   = np.sort(gs)
    bmag = (0.05 + rng.random(H)) ** 3                      # heavy-tailed: few dominant, many weak
    bmag = bmag / bmag.sum() * depth                        # sum |b_h| = depth < 1  => E>0
    psi  = rng.uniform(0.0, 2 * np.pi, size=H)

    n = np.arange(N)
    E = np.ones(N)
    for g, b, p in zip(gs, bmag, psi):
        E += b * np.cos(2 * np.pi * g * n / N + p)          # slow shimmer, strictly positive

    out = []
    out.append("%d %d %d %d %d %d" % (N, W, F_MIN, F_MAX, k, len(AMPS)))
    out.append(" ".join("%.6f" % a for a in AMPS))
    out.append(" ".join("%.8f" % v for v in E))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
