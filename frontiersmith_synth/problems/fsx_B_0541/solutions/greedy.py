# TIER: greedy
# The obvious "average strong coder" approach:
#   * process jobs biggest-first (do large jobs while the wheel is fresh -- a
#     natural way to fight the (1+wear)^2 amplification), and
#   * treat wear as something you fix by RE-DRESSING: whenever wear climbs past a
#     threshold, pay T_r and reset.
# It never realises that the small soft jobs are free maintenance: size-priority
# buries them at the very end, so it pays T_r again and again.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    T_r = int(next(it))
    W_max = int(next(it))
    jobs = []
    for i in range(N):
        s = int(next(it))
        d = int(next(it))
        jobs.append((s, d, i + 1))

    order = sorted(jobs, key=lambda x: -x[0])  # biggest size first

    theta = 3  # re-dress once wear exceeds this
    out = []
    w = 0
    for (s, d, ix) in order:
        if w > theta:
            out.append(0)
            w = 0
        out.append(ix)
        w = w + d
        if w < 0:
            w = 0
    sys.stdout.write(" ".join(str(x) for x in out))


if __name__ == "__main__":
    main()
