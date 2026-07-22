# TIER: greedy
# Naive curve fit: average the day-over-day growth ratio on non-month-end
# days, round it to the nearest 1/1000 (a plausible decimal rate), then
# read off a fee/threshold from the month-end shortfalls implied by that
# rounded rate. Looks accurate on the training window but the rounded rate
# is essentially never the true rational r, so floor()'s rounding boundary
# drifts once rolled far past the training horizon.
import sys

L = 30


def main():
    data = sys.stdin.read().split()
    idx = 0
    test_id = int(data[idx]); idx += 1
    K = int(data[idx]); idx += 1
    T = int(data[idx]); idx += 1
    Lm = int(data[idx]); idx += 1
    accounts = []
    for _ in range(K):
        bal = [int(x) for x in data[idx:idx + T + 1]]
        idx += T + 1
        accounts.append(bal)

    # 1) estimate r via mean ratio on safe (non-month-end) days
    ratios = []
    for bal in accounts:
        for t in range(T):
            if t % Lm != Lm - 1:
                b, bn = bal[t], bal[t + 1]
                if b > 0:
                    ratios.append((bn - b) / b)
    r_hat = sum(ratios) / len(ratios) if ratios else 0.0
    qg = 1000
    pg = max(0, min(qg - 1, round(r_hat * qg)))

    # 2) estimate fee / theta from month-end shortfalls under the rounded rate
    fee_candidates = []
    fee_month_starts = []
    for bal in accounts:
        nmonths = T // Lm
        for m in range(nmonths):
            t_end = m * Lm + (Lm - 1)
            if t_end + 1 > T:
                continue
            b_last, b_next = bal[t_end], bal[t_end + 1]
            grown_est = (b_last * (qg + pg)) // qg
            delta = grown_est - b_next
            month_start = bal[m * Lm]
            if delta > 0:
                fee_candidates.append(delta)
                fee_month_starts.append(month_start)

    fee_g = round(sum(fee_candidates) / len(fee_candidates)) if fee_candidates else 0
    theta_g = round(sum(fee_month_starts) / len(fee_month_starts)) if fee_month_starts else 0

    print(f"{pg} {qg} {fee_g} {theta_g}")


if __name__ == "__main__":
    main()
