# TIER: greedy
import sys

# The "obvious" move: pool ALL train rows and fit ONE effective term,
# N ~= c * (a*b) * lam, by ordinary least squares (no intercept). This
# nails the compact, near-square training pans (area term dominates there)
# but never looks at the perimeter at all -- so it silently assumes the
# boundary correction is always negligible. That assumption is exactly what
# breaks on very elongated "baking sheet" pans.


def read_rows():
    data = sys.stdin.read().split()
    it = iter(data)
    k = int(next(it))
    rows = []
    for _ in range(k):
        a = float(next(it)); b = float(next(it)); lam = float(next(it)); n = float(next(it))
        rows.append((a, b, lam, n))
    return rows


def main():
    rows = read_rows()
    sxy = 0.0
    sxx = 0.0
    for a, b, lam, n in rows:
        x = a * b * lam
        sxy += x * n
        sxx += x * x
    c = sxy / sxx if sxx > 1e-12 else 0.0
    print("%.8f * a * b * lam" % c)


if __name__ == "__main__":
    main()
