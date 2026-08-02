# TIER: greedy
"""
The obvious "safe" recipe: gate as FINE-GRAINED as possible (one block per domain, or a plain
round-robin pack when the block count exceeds the domain limit) so each domain's idle windows
are as exclusively "its own" as they can be -- and gate EVERY idle run, no matter how short
(threshold 0), because "any time nobody in the domain is working, turn it off" sounds obviously
correct. It never looks at the idle-run-LENGTH distribution and never checks whether the
one-time wakeup energy is actually cheaper than just staying on through a short gap -- it pays
the fixed wakeup cost every single time a domain goes idle, however briefly.
"""
import sys


def main():
    data = sys.stdin.read().split()
    p = iter(data)

    def nx():
        return next(p)

    N = int(nx())
    D = int(nx())
    K = int(nx())
    T = int(nx())
    nx(); nx()  # L, W (unused -- that's the point: this recipe never looks at them)
    for _ in range(K * N):
        nx()  # consume trace rows, never inspected

    Du = min(N, D)
    dom = [(i % Du) + 1 for i in range(N)]
    theta = [0] * Du  # gate every idle run, however short

    out = [str(Du), " ".join(map(str, dom)), " ".join(map(str, theta))]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
