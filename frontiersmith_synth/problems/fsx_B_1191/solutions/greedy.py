# TIER: greedy
"""The obvious first move: ordinary least squares of power on irradiance
alone, P = a*G + c, fit over the (never-clipped) winter training rows and
reported as an unbounded closed form. It fits the visible branch very well
(the branch really is close to linear there) but never applies the
nameplate N or any notion of a ceiling, so on the held-out summer rows --
where irradiance regularly exceeds the true clip -- its prediction keeps
climbing straight past the true (flat) output."""
import sys


def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    # data[1] = t, data[2] = N (unused by this tier -- that is the point)
    rows = data[3:]
    Sg = Sp = Sgg = Sgp = 0.0
    for i in range(n):
        G = float(rows[3 * i])
        # T = float(rows[3*i+1])  # ignored
        P = float(rows[3 * i + 2])
        Sg += G
        Sp += P
        Sgg += G * G
        Sgp += G * P
    denom = n * Sgg - Sg * Sg
    if abs(denom) < 1e-9:
        a, c = 0.0, Sp / n if n else 0.0
    else:
        a = (n * Sgp - Sg * Sp) / denom
        c = (Sp - a * Sg) / n
    print("%.8f * G + %.8f" % (a, c))


if __name__ == "__main__":
    main()
