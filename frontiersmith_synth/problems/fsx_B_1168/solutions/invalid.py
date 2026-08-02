# TIER: invalid
# Declares more sensors than the budget allows -> must be rejected by feasibility (Ratio 0.0).
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    test_id = int(next(it))
    n = int(next(it)); T = int(next(it)); F_max = int(next(it))
    k = F_max + 1
    out = [str(k)]
    for i in range(k):
        out.append(f"{i} 0.0 0.0")
    print("\n".join(out))


if __name__ == "__main__":
    main()
