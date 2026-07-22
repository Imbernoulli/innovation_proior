# TIER: greedy
# The "obvious" first instinct: treat each component slot as if it alone sets a local
# corner frequency in its own patch of the spectrum (impedance-match Z0 there), choosing
# every component independently and always populating all N slots. This ignores that
# every component actually shifts the WHOLE cascaded response at once (global coupling),
# so it gets the correct taper across slots wrong.
import sys, math


def main():
    data = sys.stdin.read().split("\n")
    header = data[0].split()
    N = int(header[0]); Z0 = float(header[1]); M = int(header[2])

    l_tok = data[1].split()
    K_L = int(l_tok[0]); L_grid = [float(x) for x in l_tok[1:1 + K_L]]

    c_tok = data[2].split()
    K_C = int(c_tok[0]); C_grid = [float(x) for x in c_tok[1:1 + K_C]]

    freqs = [float(x) for x in data[3].split()]

    f_lo, f_hi = freqs[0], freqs[-1]
    out = []
    for i in range(1, N + 1):
        # geometric center of this slot's equal-log band
        band_lo = f_lo * ((f_hi / f_lo) ** ((i - 1) / N))
        band_hi = f_lo * ((f_hi / f_lo) ** (i / N))
        f_i = math.sqrt(band_lo * band_hi)
        w_i = 2.0 * math.pi * f_i
        if i % 2 == 1:  # inductor: match reactance to Z0 at f_i
            target = Z0 / w_i
            grid = L_grid
        else:  # capacitor: match reactance to Z0 at f_i
            target = 1.0 / (w_i * Z0)
            grid = C_grid
        best_idx, best_d = 1, None
        for idx in range(1, len(grid)):  # never omits -- always populates
            d = abs(grid[idx] - target)
            if best_d is None or d < best_d:
                best_d, best_idx = d, idx
        out.append(best_idx)
    print(" ".join(map(str, out)))


main()
