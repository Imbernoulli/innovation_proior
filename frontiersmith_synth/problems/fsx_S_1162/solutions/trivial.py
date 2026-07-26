# TIER: trivial
"""
Ignore temperature entirely: predict the mean of the training measurements
for every protocol, hot or cold, monotone or reheated.  This reproduces the
checker's own internal baseline construction almost exactly.
"""
import sys


def main():
    data = sys.stdin.read().split()
    N = int(data[2])
    idx = 4
    m_sum = 0.0
    for _ in range(N):
        K = int(data[idx])
        idx += 1 + K
        m_sum += float(data[idx])
        idx += 1
    mean_m = m_sum / N
    print("%.6f 0.0" % mean_m)
    print("%.6f 0.0" % mean_m)
    print("600.0 600.0")


if __name__ == "__main__":
    main()
