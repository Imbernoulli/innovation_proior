# TIER: greedy
# The obvious recipe: notice tension looks like it grows/shrinks roughly
# exponentially with depth, so bin the training rows by depth (3..6), take
# the mean observed tension in each bin (this washes out per-row noise), and
# fit a SINGLE aggregate exponential curve T(d) ~ c * r**d through those bin
# means by log-linear regression. This captures the right SHAPE (unlike a
# flat or linear guess) and is a clean, stable population-average fit on the
# training window. But it fits ONE global rate/amplitude across every
# design, so it throws away exactly what strong exploits: each design's OWN
# (p1,p2) fixes a distinct mixture of the recurrence's two eigen-modes (a
# per-row amplitude and a secondary correction), not a single population-
# wide exponent -- so this recipe is measurably biased on any individual
# held-out design, and the bias compounds badly once d reaches 10..14.
import sys, math, re


def spacify(expr):
    toks = re.findall(r"\*\*|\d+\.\d+(?:[eE][-+]?\d+)?|\d+(?:[eE][-+]?\d+)?"
                       r"|[A-Za-z_]\w*|[+\-*/(),]", expr)
    return " ".join(toks)


def solve2(X, y):
    # 2x2 normal equations, closed form
    m = len(X[0])
    XtX = [[0.0] * m for _ in range(m)]
    Xty = [0.0] * m
    for row, yv in zip(X, y):
        for i in range(m):
            Xty[i] += row[i] * yv
            for j in range(m):
                XtX[i][j] += row[i] * row[j]
    for i in range(m):
        XtX[i][i] += 1e-9
    a00, a01 = XtX[0]
    a10, a11 = XtX[1]
    b0, b1 = Xty
    det = a00 * a11 - a01 * a10
    if abs(det) < 1e-12:
        det = 1e-12
    w0 = (b0 * a11 - a01 * b1) / det
    w1 = (a00 * b1 - b0 * a10) / det
    return w0, w1


def main():
    data = sys.stdin.read().split()
    if len(data) < 2:
        print("0.0")
        return
    n = int(data[0])
    vals = data[2:]
    bins = {}
    idx = 0
    for _ in range(n):
        dep = int(vals[idx])
        row = vals[idx: idx + dep + 2]
        T = float(row[-1])
        bins.setdefault(dep, []).append(T)
        idx += dep + 2

    X, y = [], []
    for dep, Ts in sorted(bins.items()):
        mean_T = sum(Ts) / len(Ts)
        X.append([1.0, float(dep)])
        y.append(math.log(max(mean_T, 1e-3)))

    if len(X) < 2:
        print(spacify("%.10g" % (math.exp(y[0]) if y else 0.0)))
        return

    a, b = solve2(X, y)
    c = math.exp(a)
    r = math.exp(b)
    expr = "(%.10g)*(%.10g)**d" % (c, r)
    print(spacify(expr))


if __name__ == "__main__":
    main()
