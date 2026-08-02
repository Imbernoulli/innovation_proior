# TIER: greedy
# The obvious recipe: scan the catalogue for what accelerating episodes USED TO
# LOOK LIKE right before they erupted. Take the low (5th) percentile of ACC and
# of SEIS seen across the ERUPTIVE rows only as an "alert threshold", and raise
# a high, fixed probability whenever a new episode clears BOTH thresholds. This
# never looks at the (far more numerous) failed-intrusion rows at all, so it
# never learns how often THEY also clear the same acceleration/seismicity
# levels -- ACC and SEIS respond to unrest intensity in general, not
# specifically to whether magma reaches the surface. The rule looks plausible
# on the training catalogue but has no way to express the real determinant of
# eruption (chamber inflation), and it over-forecasts hardest once episodes get
# louder across the board on the more intense held-out bout.
import sys


def percentile(xs, q):
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * q
    f = int(k); c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def pstdev(xs):
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / len(xs)
    return v ** 0.5


def main():
    data = sys.stdin.read().split()
    if not data:
        print("0.5"); return
    n = int(data[0])
    vals = data[2:]
    rows = []
    for i in range(n):
        ACC = float(vals[4 * i]); INFL = float(vals[4 * i + 1])
        SEIS = float(vals[4 * i + 2]); erupt = int(vals[4 * i + 3])
        rows.append((ACC, INFL, SEIS, erupt))

    pos = [r for r in rows if r[3] == 1]
    if len(pos) < 3:
        pos = rows

    accA = [r[0] for r in pos]
    seisS = [r[2] for r in pos]
    thrA = percentile(accA, 0.05)
    thrS = percentile(seisS, 0.05)
    spreadA = pstdev(accA) or 1.0
    spreadS = pstdev(seisS) or 1.0
    gA = 2.0 / spreadA
    gS = 2.0 / spreadS
    pmax = 0.6

    print("%.10g * sig ( %.10g * ( ACC - %.10g ) ) * sig ( %.10g * ( SEIS - %.10g ) )"
          % (pmax, gA, thrA, gS, thrS))


if __name__ == "__main__":
    main()
