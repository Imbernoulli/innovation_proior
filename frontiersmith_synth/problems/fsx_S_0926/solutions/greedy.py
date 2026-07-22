# TIER: greedy
# The obvious first idea: back out an estimated wear level from each row via
# the GIVEN observation formula (What_i = (T_i/BASE[mat_i] - 1) / ALPHA), then
# notice it trends upward over the training window and fit a plain ordinary-
# least-squares line against the job's POSITION in the sequence -- a classic
# time-indexed curve fit that completely ignores gap/load/material and the
# previous wear value (Wprev is never referenced). This nails the public
# training window (wear really does trend with position there) but the
# submitted "recursion" is not a recursion at all: on held-out sequences that
# are five times longer, position keeps climbing past anything seen in
# training and the linear trend runs away, while it also has no way to react
# to the much longer idle "maintenance" gaps or the heavier load mix that
# appear only in the extrapolation regime -- both of which the true tool
# actually reacts to (it recovers during the long gaps and saturates under
# heavy load).
import sys


def read_input():
    data = sys.stdin.read().split()
    pos = 0
    t = int(data[pos]); pos += 1
    n = int(data[pos]); pos += 1
    alpha = float(data[pos]); pos += 1
    base = [float(data[pos]), float(data[pos + 1]), float(data[pos + 2])]; pos += 3
    rows = []
    for i in range(n):
        gap = float(data[pos]); load = float(data[pos + 1])
        mat = int(data[pos + 2]); T = float(data[pos + 3])
        pos += 4
        rows.append((gap, load, mat, T))
    return t, n, alpha, base, rows


def main():
    t, n, alpha, base, rows = read_input()

    xs = []  # job index (1-based)
    ys = []  # backed-out wear estimate
    for i, (gap, load, mat, T) in enumerate(rows):
        what = (T / base[mat] - 1.0) / alpha
        xs.append(float(i + 1))
        ys.append(what)

    m = len(xs)
    sx = sum(xs); sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = m * sxx - sx * sx
    if abs(denom) < 1e-9:
        K, C = 0.0, sy / m
    else:
        K = (m * sxy - sx * sy) / denom
        C = (sy - K * sx) / m

    print("%.10f*idx + %.10f" % (K, C))


if __name__ == "__main__":
    main()
