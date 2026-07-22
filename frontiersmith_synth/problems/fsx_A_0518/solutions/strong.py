# TIER: strong
# The insight: a memoryless fit of the slow training branch leaves residuals that
# correlate with the drive DIRECTION, not its VALUE -- the tell of hidden state.
# So posit a Schmitt LATCH plus an actuation DELAY and recover them from the data:
#   * OFF/ON amplitudes from robust percentiles of the heater;
#   * the hysteresis edges by looking at WHERE the reconstructed latch flips:
#       heater turns ON while the drive is FALLING  -> lower edge L
#       heater turns OFF while the drive is RISING   -> upper edge H
#     (a static curve would flip at a single value; two direction-dependent
#      values ARE the band);
#   * the actuation delay k by the shift that best explains training (the slow
#     regime hides it faintly, so we search a small window).
# Emit a stateful DSL program: a latch on the drive + a delayed latch tap.
# It generalises to the fast held-out drive where the band + lag dominate.
import sys


def latch_roll(drive, L, H):
    s = 0
    out = []
    for d in drive:
        if d < L:
            s = 1
        elif d > H:
            s = 0
        out.append(s)
    return out


def mse(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) / len(a)


def main():
    data = sys.stdin.read().split()
    if not data:
        print("OUT 0.5"); return
    n = int(data[0])
    vals = data[2:]
    drive = []
    y = []
    for i in range(n):
        drive.append(float(vals[2 * i]))
        y.append(float(vals[2 * i + 1]))

    ys = sorted(y)
    off = ys[int(0.05 * len(ys))]
    hi = ys[int(0.95 * len(ys))]
    amp = max(1e-6, hi - off)
    mid = off + amp / 2.0

    best = None
    bestLHk = (0.40, 0.58, 2)
    for k in range(0, 7):
        Ls = []
        Hs = []
        for tau in range(1, n - k):
            s0 = 1 if y[tau - 1 + k] > mid else 0
            s1 = 1 if y[tau + k] > mid else 0
            if s0 == 0 and s1 == 1:      # turned ON while falling -> lower edge
                Ls.append(drive[tau])
            if s0 == 1 and s1 == 0:      # turned OFF while rising -> upper edge
                Hs.append(drive[tau])
        if not Ls or not Hs:
            continue
        L = sum(Ls) / len(Ls)
        H = sum(Hs) / len(Hs)
        if L >= H:
            continue
        S = latch_roll(drive, L, H)
        pred = [off + amp * (S[t - k] if t - k >= 0 else 0) for t in range(n)]
        e = mse(pred, y)
        if best is None or e < best:
            best = e
            bestLHk = (L, H, k)

    L, H, k = bestLHk
    print("LATCH %.6f - d | d - %.6f" % (L, H))
    if k <= 0:
        print("OUT %.6f + %.6f * S" % (off, amp))
    else:
        print("OUT %.6f + %.6f * Sk%d" % (off, amp, k))


if __name__ == "__main__":
    main()
