# TIER: greedy
"""The obvious ("textbook") approach: trust the loudest look.  Pick the aspect
angle with the largest sin(theta) (the most Doppler-sensitive look -- an
average coder's natural pick for "best SNR"), assume the observed peak there
is the TRUE frequency (no PRF aliasing), and for each candidate class solve
for the rotation rate that reproduces it.  Report the class whose implied
rate falls inside its own stated range (closest to the range's midpoint).

This is exactly the trap: whenever the true rotation rate's line exceeds
PRF/2 at the loudest angle, the "no aliasing" assumption is wrong and this
recipe locks onto whichever OTHER candidate class happens to explain the
folded value -- a confident, wrong, single-angle answer."""
import sys, math


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

    ref_i = max(range(K), key=lambda i: math.sin(math.radians(angles[i])))
    th = angles[ref_i]
    f = obs[ref_i]
    s = math.sin(math.radians(th))

    best = None  # (dist_to_mid, class_id, rate)
    for cid, (name, blade, lo, hi) in enumerate(classes):
        if s <= 1e-9:
            continue
        implied = f / (blade * s)
        if lo <= implied <= hi:
            mid = (lo + hi) / 2.0
            d = abs(implied - mid)
            if best is None or d < best[0]:
                best = (d, cid, implied)

    if best is None:
        # fallback: nothing matched at the reference angle -- clamp the
        # least-bad candidate into its own feasible range.
        best2 = None
        for cid, (name, blade, lo, hi) in enumerate(classes):
            implied = f / (blade * s) if s > 1e-9 else lo
            clamped = min(max(implied, lo), hi)
            d = abs(implied - clamped)
            if best2 is None or d < best2[0]:
                best2 = (d, cid, clamped)
        best = best2

    _, cid, rate = best
    sys.stdout.write("%d %.6f\n" % (cid, rate))


if __name__ == "__main__":
    main()
