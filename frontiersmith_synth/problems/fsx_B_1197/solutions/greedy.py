# TIER: greedy
# The obvious recipe: "position error repeats roughly once a day" -> ASSUME the
# repeat period is EXACTLY one solar day (86400s, the only period explicitly
# named in the statement) and fit an intercept + 1st/2nd/3rd-harmonic Fourier
# series at that FIXED frequency by ordinary least squares on the training
# window.  Because the true orbital-repeat period is only a few hundred
# seconds off, this looks almost perfect on the few visible training days --
# but the fixed-24h phase has no mechanism to track the true near-day period,
# so it silently decoheres by the time the held-out horizon (weeks later)
# arrives.
import sys, math

SOLAR_PERIOD = 86400.0


def lstsq_solve(A, y):
    """Plain Gaussian-elimination normal-equations solve (no numpy dependency)."""
    m = len(A[0])
    ata = [[0.0] * m for _ in range(m)]
    aty = [0.0] * m
    for row, yv in zip(A, y):
        for i in range(m):
            aty[i] += row[i] * yv
            for j in range(m):
                ata[i][j] += row[i] * row[j]
    # Gaussian elimination with partial pivoting
    for col in range(m):
        piv = max(range(col, m), key=lambda r: abs(ata[r][col]))
        if abs(ata[piv][col]) < 1e-12:
            continue
        ata[col], ata[piv] = ata[piv], ata[col]
        aty[col], aty[piv] = aty[piv], aty[col]
        pivval = ata[col][col]
        for j in range(col, m):
            ata[col][j] /= pivval
        aty[col] /= pivval
        for r in range(m):
            if r != col:
                factor = ata[r][col]
                if factor != 0.0:
                    for j in range(col, m):
                        ata[r][j] -= factor * ata[col][j]
                    aty[r] -= factor * aty[col]
    return aty


def main():
    data = sys.stdin.read().split()
    if not data:
        print("0")
        return
    n = int(data[0])
    vals = data[2:]
    ts = [float(vals[2 * i]) for i in range(n)]
    ys = [float(vals[2 * i + 1]) for i in range(n)]

    w0 = 2 * math.pi / SOLAR_PERIOD
    # design: [1, cos(w0 t), sin(w0 t), cos(2 w0 t), sin(2 w0 t), cos(3 w0 t), sin(3 w0 t)]
    A = []
    for t in ts:
        row = [1.0]
        for i in (1, 2, 3):
            row.append(math.cos(i * w0 * t))
            row.append(math.sin(i * w0 * t))
        A.append(row)
    coef = lstsq_solve(A, ys)
    c0, a1, b1, a2, b2, a3, b3 = coef

    expr = (
        "%.8f + %.8f * cos ( %.10f * t ) + %.8f * sin ( %.10f * t ) "
        "+ %.8f * cos ( %.10f * t ) + %.8f * sin ( %.10f * t ) "
        "+ %.8f * cos ( %.10f * t ) + %.8f * sin ( %.10f * t )"
        % (c0,
           a1, w0, b1, w0,
           a2, 2 * w0, b2, 2 * w0,
           a3, 3 * w0, b3, 3 * w0)
    )
    print(expr)


if __name__ == "__main__":
    main()
