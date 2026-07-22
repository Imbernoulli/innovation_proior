# TIER: greedy
# The obvious "textbook system-ID" recipe: notice the residual of a pure-P
# fit still trends with the accumulated error, so recognise a PI structure
# and fit u = Kp*e + Ki*I by ordinary least squares (closed-form normal
# equations) on the calm log.  This captures the integral action and is an
# excellent fit to the TRAINING data (which never saturates), but it commits
# to an UNCLAMPED linear law -- no notion that the throttle has finite
# authority.  On the stormy held-out flight, large errors drive I (and hence
# u) far past what the real lazy throttle would ever produce, and the
# resulting runaway windup blows the rollout off the true trajectory.
import sys


def main():
    data = sys.stdin.read().split()
    if not data:
        print("OUT 0.0 * e")
        return
    n = int(data[0])
    vals = data[3:]
    es = []
    us = []
    Is = []
    I = 0.0
    for i in range(n):
        sp = float(vals[3 * i])
        y = float(vals[3 * i + 1])
        u = float(vals[3 * i + 2])
        e = sp - y
        I = I + e
        es.append(e)
        Is.append(I)
        us.append(u)

    See = sum(e * e for e in es)
    SeI = sum(e * Iv for e, Iv in zip(es, Is))
    SII = sum(Iv * Iv for Iv in Is)
    Seu = sum(e * u for e, u in zip(es, us))
    SIu = sum(Iv * u for Iv, u in zip(Is, us))
    det = See * SII - SeI * SeI
    if abs(det) < 1e-9:
        Kp, Ki = (Seu / See if See > 1e-9 else 0.0), 0.0
    else:
        Kp = (Seu * SII - SeI * SIu) / det
        Ki = (See * SIu - SeI * Seu) / det

    print("OUT %.6f * e + %.6f * I" % (Kp, Ki))


if __name__ == "__main__":
    main()
