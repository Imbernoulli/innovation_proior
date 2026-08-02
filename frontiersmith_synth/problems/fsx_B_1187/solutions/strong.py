# TIER: strong
"""The insight: use aspect-angle DIVERSITY across the dwell to unfold the
aliasing instead of trusting the raw spectral peak at any one look.

For every candidate class (known blade/scatterer count) and every possible
"anchor" angle, enumerate the small set of PRF-fold hypotheses that could
have produced the observed line there (true = k*PRF +/- observed, for a few
small integer k -- the aliasing order is bounded because rates are bounded
by the candidate's own stated range). Each hypothesis proposes a candidate
rotation rate; a WRONG hypothesis only explains the one anchor angle it was
built from and disagrees with the other K-1 aspect-angle observations
(different sin(theta) -> different fold behaviour), while the RIGHT
(class, rate) reproduces the forward model consistently at every angle in
the dwell. So: score every hypothesis against ALL angles and keep the best.
This is a reformulation (search over consistency across the diversity axis)
rather than "greedy plus more iterations"."""
import sys, math

FLOOR = 0.12
CAP = 6.0


def fold(f, prf):
    k = math.floor(f / prf + 0.5)
    return abs(f - k * prf)


def angle_score(resid):
    return max(FLOOR, min(1.0, 1.0 - resid / CAP))


def fit_quality(prf, angles, obs, blade, rate):
    F = 0.0
    for th, o in zip(angles, obs):
        s = math.sin(math.radians(th))
        f_pred = blade * rate * s
        f_pred_folded = fold(f_pred, prf)
        resid = abs(f_pred_folded - o)
        F += angle_score(resid)
    return F


def main():
    toks = sys.stdin.read().split()
    idx = 0
    prf = float(toks[idx]); idx += 1
    K = int(toks[idx]); idx += 1
    C = int(toks[idx]); idx += 1
    angles = [float(toks[idx + i]) for i in range(K)]; idx += K
    obs = [float(toks[idx + i]) for i in range(K)]; idx += K
    classes = []
    for _ in range(C):
        name = toks[idx]; idx += 1
        blade = int(toks[idx]); idx += 1
        rlo = float(toks[idx]); idx += 1
        rhi = float(toks[idx]); idx += 1
        classes.append((name, blade, rlo, rhi))

    best = None  # (F, class_id, rate)
    for cid, (name, blade, lo, hi) in enumerate(classes):
        for ai, th in enumerate(angles):
            s = math.sin(math.radians(th))
            if s < 1e-6:
                continue
            for k in range(0, 6):
                for sign in (1, -1):
                    cand_true = k * prf + sign * obs[ai]
                    if cand_true <= 0:
                        continue
                    rate = cand_true / (blade * s)
                    if not (lo <= rate <= hi):
                        continue
                    F = fit_quality(prf, angles, obs, blade, rate)
                    if best is None or F > best[0]:
                        best = (F, cid, rate)

    if best is None:
        # fallback (should not happen given the planted structure): checker's
        # own naive reference so we still output something feasible.
        name0, blade0, lo0, hi0 = classes[0]
        sys.stdout.write("0 %.6f\n" % ((lo0 + hi0) / 2.0))
        return

    _, cid, rate = best
    sys.stdout.write("%d %.6f\n" % (cid, rate))


if __name__ == "__main__":
    main()
