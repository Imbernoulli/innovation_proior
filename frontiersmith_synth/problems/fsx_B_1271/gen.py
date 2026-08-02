import sys, random

# Deterministic instance generator for credit-limit-assign (format C).
# `python3 gen.py <testId>` -> ONE instance on stdout. Seeded ONLY by testId.

TIERS = [0, 800, 1600, 3200, 6000, 11000, 18000]   # L[0..6]; L[0]=0 means "no card"
M = len(TIERS) - 1
BASE_TIER = 2                 # tier the checker's own trivial baseline uses

# difficulty ladder: (N, trap) -- trap=True forces the top-score band to skew
# heavily "responsive" (limit-sensitive default risk) so a pure score-ranking
# assignment burns the loss cap on borrowers whose marginal profit at high
# tiers is small or negative, while mid/low-score "stable" borrowers who are
# actually the most profitable at high tiers get starved.
LADDER = [
    (16, False), (20, False), (24, True), (28, False), (32, True),
    (36, False), (40, True), (44, False), (48, True), (60, True),
]


def profit_loss(m, d_bps, u_bps, apr_bps, sev_bps):
    if m == 0:
        return 0, 0
    L = TIERS[m]
    balance = L * u_bps // 10000
    revenue = balance * apr_bps // 10000 * (10000 - d_bps) // 10000
    loss = balance * d_bps // 10000 * sev_bps // 10000
    return revenue - loss, loss


def main():
    testId = int(sys.argv[1])
    idx = min(max(testId, 1), len(LADDER)) - 1
    N, trap = LADDER[idx]
    rng = random.Random(1000 * testId + 7)

    apr_bps = 1400 + 50 * (testId % 5)          # ~14%-16%
    sev_bps = 5000 + 40 * (testId % 6)          # ~50%-52% loss-given-default

    rows = []   # (score, [d_1..d_M], [u_1..u_M])
    for _ in range(N):
        score = rng.randint(300, 850)
        if trap:
            seg = 1 if (score >= 700 and rng.random() < 0.85) or \
                       (score < 700 and rng.random() < 0.15) else 0
        else:
            seg = 1 if rng.random() < 0.35 else 0

        if seg == 0:
            base_d = max(60, 900 - score)
        else:
            base_d = max(80, 950 - score)
        base_u = 5200 + rng.randint(-300, 300)
        comfort = 2 + (score % 3)   # tier index 2..4

        d_list = [0] * M
        u_list = [0] * M
        for m in range(1, M + 1):
            if seg == 0:                       # stable: gentle risk, saturating use
                d = base_d + 40 * m
                if m <= comfort:
                    u = base_u + 500 * m
                else:
                    u = base_u + 500 * comfort - 650 * (m - comfort)
            else:                               # responsive: convex risk, keeps spending
                d = base_d + 55 * m * m
                u = min(9700, base_u + 750 * m)
            d_list[m - 1] = max(10, min(9800, d))
            u_list[m - 1] = max(200, min(9800, u))
        rows.append((score, d_list, u_list))

    # loss cap: 0.3x the portfolio's total expected loss if EVERY applicant got
    # the top tier -- binding enough that the cap is a real constraint.
    total_top_loss = 0
    for score, d_list, u_list in rows:
        _, loss = profit_loss(M, d_list[M - 1], u_list[M - 1], apr_bps, sev_bps)
        total_top_loss += loss
    cap = max(50, int(total_top_loss * 0.30))

    out = []
    out.append("%d %d" % (N, M))
    out.append(" ".join(str(x) for x in TIERS[1:]))
    out.append("%d %d %d" % (apr_bps, sev_bps, cap))
    for score, d_list, u_list in rows:
        toks = [str(score)]
        for m in range(M):
            toks.append(str(d_list[m]))
            toks.append(str(u_list[m]))
        out.append(" ".join(toks))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
