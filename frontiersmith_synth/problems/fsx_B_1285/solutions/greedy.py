# TIER: greedy
# The "obvious" recipe: compare the two NON-adaptive global strategies --
# settle right now at the opening offer, or commit to going all the way to
# trial on every branch -- and commit to whichever single strategy has the
# better expected value. This ignores that the decision could be revisited
# node-by-node as information arrives, so it cannot cut losses on branches
# that turn unfavorable, nor can it settle only after good news confirms a
# strong price.
import sys

def depth(idx):
    return (idx + 1).bit_length() - 1

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    T = int(next(it))
    cost = [int(next(it)) for _ in range(T)]
    M = 2 ** (T + 1) - 1
    S = [int(next(it)) for _ in range(M)]
    NLEAF = 2 ** T
    L = [int(next(it)) for _ in range(NLEAF)]

    settle_now = S[0]
    total_cost = sum(cost)
    trial_always = sum(L) / len(L) - total_cost

    if trial_always > settle_now:
        policy = ["C"] * M          # push every branch all the way to trial
    else:
        policy = ["S"] * M          # settle the whole case immediately

    print(M)
    print(" ".join(policy))

if __name__ == "__main__":
    main()
