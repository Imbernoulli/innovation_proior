# TIER: greedy
# The "obvious first attempt": space the delay lines evenly by a small fixed step
# (2 samples apart) starting from Lmin -- never checks whether the resulting lengths
# are coprime -- and fit ONE global gain from the overall (endpoint) target decay
# level, applied uniformly to every line. This textbook recipe (uniform gain = single
# exponential decay, naive spacing) is exactly the two traps the family is built
# around: it cannot bend to a two-stage decay curve, and on tests where Lmin is even
# it produces an all-even delay set (every echo arrival time is a multiple of 2) that
# collides constantly and starves the echo-density curve.
import sys
GAIN_MIN, GAIN_MAX = 1e-3, 0.99


def main():
    tok = sys.stdin.read().split()
    p = 0
    N = int(tok[p]); p += 1
    T = int(tok[p]); p += 1
    Lmin = int(tok[p]); p += 1
    Lmax = int(tok[p]); p += 1
    K = int(tok[p]); p += 1
    ts = [int(tok[p + j]) for j in range(K)]; p += K
    target_db = [float(tok[p + j]) for j in range(K)]; p += K
    # target_density unused by the greedy recipe (it only reasons about decay level)
    p += K
    w_decay = float(tok[p]); p += 1
    w_density = float(tok[p]); p += 1

    step = 2
    L = [min(Lmin + step * i, Lmax) for i in range(N)]
    Lavg = sum(L) / N

    # Fit a single global gain to the LAST checkpoint's target dB level only (the
    # "quick and dirty" single-exponential fit an average coder reaches for first).
    slope = target_db[-1] / max(1, ts[-1])   # dB per sample
    g0 = 10.0 ** (slope * Lavg / 20.0)
    g0 = min(GAIN_MAX, max(GAIN_MIN, g0))
    g = [g0] * N

    out = []
    out.append(" ".join(str(x) for x in L))
    out.append(" ".join("%.6f" % x for x in g))
    sys.stdout.write("\n".join(out) + "\n")


main()
