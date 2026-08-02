# TIER: greedy
# The "obvious first attempt": chase precision-per-minute. Sort every alert by
# score/cost density, fill the time budget, done -- never checks the per-typology
# coverage floor at all, so it silently starves whichever typology the model scores
# low this round, even when the regulator requires covering it.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    test_id = int(next(it)); N = int(next(it)); K = int(next(it)); C = int(next(it))
    alerts = {}
    for _ in range(N):
        aid = int(next(it)); t = int(next(it)); cost = int(next(it)); score = int(next(it))
        alerts[aid] = (t, cost, score)
    for _ in range(K):
        next(it); next(it)  # coverage floors: read and ignore

    order = sorted(alerts, key=lambda a: (-(alerts[a][2] / alerts[a][1]), alerts[a][1], a))

    sel = []
    cost_used = 0
    for aid in order:
        c = alerts[aid][1]
        if cost_used + c <= C:
            sel.append(aid); cost_used += c

    out = [str(len(sel))]
    out.extend(str(a) for a in sel)
    print("\n".join(out))


if __name__ == "__main__":
    main()
