# TIER: greedy
# The obvious recipe: notice the decay-rate column IS informative (it falls
# toward zero as R approaches onset), fit an honest ordinary-least-squares
# line decay_rate ~ alpha + beta*R, and extrapolate that same line past the
# zero-crossing to get a growth rate g(R) = -(alpha + beta*R) for R > Rc.
# Then report THAT growth rate directly as the predicted oscillation
# amplitude. This finds the onset location fine, but never asks what units
# a growth rate and an amplitude actually live in -- it ignores the given
# Landau constant L and the cubic saturation it encodes entirely. Amplitude
# grows like sqrt(g) near onset (steep) while g itself is still tiny there,
# so this systematically misjudges the shape of the post-onset branch.
import sys


def main():
    data = sys.stdin.read().split()
    if not data:
        print("0"); return
    M = int(data[0])
    # data[1] = t, data[2] = L (unused by this solution)
    vals = data[3:]
    Rs = []
    decays = []
    for i in range(M):
        Rs.append(float(vals[3 * i]))
        decays.append(float(vals[3 * i + 1]))

    n = len(Rs)
    mx = sum(Rs) / n
    my = sum(decays) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(Rs, decays))
    sxx = sum((x - mx) ** 2 for x in Rs)
    beta = sxy / sxx if sxx > 1e-12 else 0.0
    alpha = my - beta * mx

    # g(R) = -(alpha + beta*R); report it directly, clamped at 0, as amplitude
    print("relu ( ( -1.0 * %.10g ) + ( -1.0 * %.10g ) * R )" % (alpha, beta))


if __name__ == "__main__":
    main()
