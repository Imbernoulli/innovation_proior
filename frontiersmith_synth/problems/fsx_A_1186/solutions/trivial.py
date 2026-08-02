# TIER: trivial
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    n = int(next(it)); m = int(next(it)); K = int(next(it)); L = int(next(it)); seed = int(next(it))
    p = int(next(it))
    row_sum = [0.0] * n
    row_cnt = [0] * n
    tot_sum, tot_cnt = 0.0, 0
    for _ in range(p):
        i = int(next(it)); j = int(next(it)); v = float(next(it))
        row_sum[i] += v
        row_cnt[i] += 1
        tot_sum += v
        tot_cnt += 1
    re_ = int(next(it))
    for _ in range(re_):
        next(it); next(it)
    ce_ = int(next(it))
    for _ in range(ce_):
        next(it); next(it)
    q = int(next(it))
    global_mean = tot_sum / tot_cnt if tot_cnt > 0 else 0.0
    row_mean = [(row_sum[i] / row_cnt[i]) if row_cnt[i] > 0 else global_mean for i in range(n)]

    out = []
    for _ in range(q):
        i = int(next(it)); j = int(next(it))
        out.append("%.4f" % row_mean[i])
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
