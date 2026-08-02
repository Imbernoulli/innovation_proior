# TIER: trivial
# Reproduces the checker's own internal baseline: give every applicant the fixed
# tier BASE_TIER (2), processed in input order, skipping an applicant once doing
# so would break the loss cap. Ignores the response tables entirely.
import sys

BASE_TIER = 2


def profit_loss(L, m, d_bps, u_bps, apr_bps, sev_bps):
    if m == 0:
        return 0, 0
    balance = L[m] * u_bps // 10000
    revenue = balance * apr_bps // 10000 * (10000 - d_bps) // 10000
    loss = balance * d_bps // 10000 * sev_bps // 10000
    return revenue - loss, loss


def main():
    toks = sys.stdin.read().split()
    p = 0
    N = int(toks[p]); p += 1
    M = int(toks[p]); p += 1
    L = [0] * (M + 1)
    for m in range(1, M + 1):
        L[m] = int(toks[p]); p += 1
    apr_bps = int(toks[p]); p += 1
    sev_bps = int(toks[p]); p += 1
    cap = int(toks[p]); p += 1

    choices = []
    running_loss = 0
    for _ in range(N):
        score = int(toks[p]); p += 1
        d = [0] * (M + 1)
        u = [0] * (M + 1)
        for m in range(1, M + 1):
            d[m] = int(toks[p]); p += 1
            u[m] = int(toks[p]); p += 1
        _, loss = profit_loss(L, BASE_TIER, d[BASE_TIER], u[BASE_TIER], apr_bps, sev_bps)
        if running_loss + loss <= cap:
            running_loss += loss
            choices.append(BASE_TIER)
        else:
            choices.append(0)

    print(" ".join(str(x) for x in choices))


if __name__ == "__main__":
    main()
