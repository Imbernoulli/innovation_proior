#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy TRAIN trace to stdout.

The Alternate-Bearing Orchard.  A hidden second-order rational recurrence

    x(t+1) = (a + x(t)) / x(t-1)                (a > 0, hidden per test id)

drives an orchard's annual yield index.  This is the classical "Lyness"
recurrence: it algebraically preserves the quantity

    I(x, y) = (x+1)*(y+1)*(x+y+a) / (x*y)

step after step -- the estate's true "reserve capacity" -- which is what
keeps a century of boom/bust cycles bounded to one fixed orbit instead of
wandering off.  Real ledgers are never perfectly clean: each year the
realised yield is nudged off the exact recurrence by a small clipped
multiplicative shock (weather, pests, ...), so the invariant itself performs
a slow random walk rather than staying bit-exact -- there is a genuine
irreducible floor to how well ANY law can forecast the ledger.

Each test id fixes a DIFFERENT hidden (a, x0, x1) and a single realised
noisy trajectory of length T0 (printed, the "early records") + Textra
(held out, "late-stage ledger", never printed -- lives only in verify.py).

STDOUT prints ONLY: a header "<n_train> <test_id>" then n_train lines, each
one float (the observed yield that year).  The hidden a, x0, x1, seed, and
the reserve invariant are NEVER printed.
"""
import sys, random, math

A_LO, A_HI = 0.6, 2.6
KICK_LO, KICK_HI = 1.6, 2.4
NOISE_CLIP = 0.35


def hidden_params(t):
    """Hidden Lyness recurrence parameters for this test id (lives in gen AND verify).

    x0, x1 are deliberately displaced from the recurrence's fixed point
    x* = (1+sqrt(1+4a))/2 by a large multiplicative "kick" (>=1.6x or <=1/1.6x,
    on each tap independently) -- this guarantees a genuinely wide boom/bust
    orbit (not a near-fixed-point trickle) so the invariant-preserving law has
    real amplitude to protect, whatever 'a' turns out to be.
    """
    rng = random.Random(900001 + t * 7919)
    a = rng.uniform(A_LO, A_HI)
    xstar = (1.0 + math.sqrt(1.0 + 4.0 * a)) / 2.0

    def kick():
        m = rng.uniform(KICK_LO, KICK_HI)
        return m if rng.random() < 0.5 else 1.0 / m

    x0 = xstar * kick()
    x1 = xstar * kick()
    return a, x0, x1


def sigma_proc(t):
    return 0.010 + 0.0015 * (t - 1)


def train_len(t):
    return 55 + 2 * (t - 1)


def extra_len(t):
    return 90 + 8 * (t - 1)


def full_trajectory(t):
    """Regenerate the single realised (train+heldout) noisy trajectory for test id t."""
    a, x0, x1 = hidden_params(t)
    sp = sigma_proc(t)
    n_total = train_len(t) + extra_len(t)
    rng = random.Random(31337 + t * 104729)
    xs = [x0, x1]
    for i in range(1, n_total - 1):
        raw = (a + xs[i]) / xs[i - 1]
        eps = rng.gauss(0.0, sp)
        eps = max(-NOISE_CLIP, min(NOISE_CLIP, eps))
        xs.append(raw * (1.0 + eps))
    return xs


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    n = train_len(t)
    xs = full_trajectory(t)
    out = ["%d %d" % (n, t)]
    for i in range(n):
        out.append("%.6f" % xs[i])
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
