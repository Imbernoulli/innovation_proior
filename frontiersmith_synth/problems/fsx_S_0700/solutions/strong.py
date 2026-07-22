# TIER: strong
"""Two-plane Gerchberg-Saxton: alternate projecting the CURRENT mask onto each
plane's amplitude-constraint set in turn (propagate to plane A, impose
sqrt(target_A) on the amplitude while keeping the phase, propagate back and
impose the FIXED uniform aperture amplitude by keeping only the phase; then do
the same for plane B using the updated mask). Because the mask must satisfy
BOTH planes' amplitude constraints with only phase degrees of freedom, no
single inverse transform can do it -- only the iterated projection converges
towards a shared compromise. A couple of restarts (deterministic, seeded from
the instance itself) and best-iterate tracking guard against GS stalls.
"""
import sys
import numpy as np


def _chirp(N, alpha, sign):
    c = (N - 1) / 2.0
    ii, jj = np.indices((N, N))
    return np.exp(sign * 1j * alpha * ((ii - c) ** 2 + (jj - c) ** 2))


def _forward(theta, alpha):
    N = theta.shape[0]
    field = np.exp(1j * theta) * _chirp(N, alpha, 1.0)
    return np.fft.fft2(field, norm="ortho")


def _backward(F, alpha):
    N = F.shape[0]
    field = np.fft.ifft2(F, norm="ortho")
    return field * _chirp(N, alpha, -1.0)


def _nmse(theta, alpha, T):
    N = theta.shape[0]
    F = _forward(theta, alpha)
    P = (np.abs(F) ** 2) / float(N * N)
    Q = T / T.sum()
    return float(np.sum((P - Q) ** 2)) / max(float(np.sum(Q * Q)), 1e-12)


def _tv(theta):
    d1 = theta[1:, :] - theta[:-1, :]
    d2 = theta[:, 1:] - theta[:, :-1]
    w1 = np.arctan2(np.sin(d1), np.cos(d1))
    w2 = np.arctan2(np.sin(d2), np.cos(d2))
    return float(np.sum(w1 ** 2) + np.sum(w2 ** 2)) / max(w1.size + w2.size, 1)


def _objective(theta, TA, TB, aA, aB, lam):
    return 0.5 * (_nmse(theta, aA, TA) + _nmse(theta, aB, TB)) + lam * _tv(theta)


def gs_two_plane(N, aA, aB, TA, TB, lam, iters=25, restarts=2, seed=555):
    QA = np.sqrt(TA / TA.sum() * (N * N))
    QB = np.sqrt(TB / TB.sum() * (N * N))
    rng = np.random.RandomState(seed)
    best_theta, best_obj = None, None
    for r in range(restarts):
        theta = np.zeros((N, N)) if r == 0 else rng.uniform(-np.pi, np.pi, size=(N, N))
        local_best, local_best_obj = None, None
        for _ in range(iters):
            FA = _forward(theta, aA)
            FA2 = QA * np.exp(1j * np.angle(FA))
            theta = np.angle(_backward(FA2, aA))
            FB = _forward(theta, aB)
            FB2 = QB * np.exp(1j * np.angle(FB))
            theta = np.angle(_backward(FB2, aB))
            obj = _objective(theta, TA, TB, aA, aB, lam)
            if local_best_obj is None or obj < local_best_obj:
                local_best_obj, local_best = obj, theta.copy()
        if best_obj is None or local_best_obj < best_obj:
            best_obj, best_theta = local_best_obj, local_best
    return best_theta


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    N = int(next(it))
    aA = float(next(it))
    aB = float(next(it))
    lam = float(next(it))
    TA = np.array([float(next(it)) for _ in range(N * N)]).reshape(N, N)
    TB = np.array([float(next(it)) for _ in range(N * N)]).reshape(N, N)

    theta = gs_two_plane(N, aA, aB, TA, TB, lam)

    out = [str(N)]
    for i in range(N):
        out.append(" ".join("%.10f" % v for v in theta[i]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
