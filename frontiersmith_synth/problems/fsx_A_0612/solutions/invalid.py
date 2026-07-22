# TIER: invalid
# The textbook inverse: naively invert F by running diffusion BACKWARD (u <- u - alpha*Laplacian).
# This is the ill-posed deconvolution the statement warns about: the tiny high-frequency factors
# get divided out and EXPLODE into wildly oscillating, NEGATIVE source values. Negative sources
# are infeasible, so the checker rejects this output (score 0) -- a faithful demonstration of the trap.
import sys

def main():
    toks = iter(sys.stdin.read().split())
    N = int(next(toks)); T = int(next(toks))
    alpha = float(next(toks)); cost = float(next(toks))
    y = [[float(next(toks)) for _ in range(N)] for _ in range(N)]

    s = [row[:] for row in y]
    for _ in range(T):                       # unstable reverse diffusion
        nxt = [[0.0] * N for _ in range(N)]
        for i in range(N):
            up = s[(i - 1) % N]; dn = s[(i + 1) % N]; me = s[i]
            for j in range(N):
                lap = up[j] + dn[j] + me[(j - 1) % N] + me[(j + 1) % N] - 4.0 * me[j]
                nxt[i][j] = me[j] - alpha * lap
        s = nxt

    out = []
    for i in range(N):
        out.append(" ".join("%.9g" % s[i][j] for j in range(N)))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
