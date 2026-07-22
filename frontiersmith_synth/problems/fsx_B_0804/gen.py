#!/usr/bin/env python3
"""gen.py <testId> -- shared-humidity-coat-staggering instance generator.
Deterministic: all randomness seeded from testId only.
"""
import sys, random


def main():
    tid = int(sys.argv[1])
    rng = random.Random(20260 + 17 * tid)

    # difficulty / trap ladder. (P, chain-length range, congestion c=cn/cd).
    # cases 6-10 are engineered traps: P is large relative to the steady-state
    # optimal concurrency K* = 1/c implied by the quadratic congestion law, so
    # maximal fan-out (apply everything the instant it is eligible) is far from
    # the interior optimum.  cases 1-3 keep P close to K* so naive full
    # parallelism is already close to optimal (sanity / non-trap cases).
    # cases 1-3: c is tiny, so the steady-state optimum K*=1/c is already >= P --
    # maximal fan-out is close to optimal here (non-trap, sanity + a genuine
    # greedy-beats-trivial margin).
    # cases 4-10: c=0.3 fixed (K*~3.3), P escalates from 18 to 95 -- maximal
    # fan-out (greedy) drifts further and further from the fixed interior
    # optimum K* that `strong` locks onto, while `strong`'s achievable ratio
    # stays essentially flat (it always finds K* regardless of P).
    cfgs = {
        1:  dict(P=8,  chain=(1, 2), c=(1, 12)),
        2:  dict(P=12, chain=(1, 2), c=(1, 15)),
        3:  dict(P=15, chain=(1, 2), c=(1, 20)),
        4:  dict(P=18, chain=(3, 5), c=(3, 10)),   # trap 1
        5:  dict(P=26, chain=(3, 5), c=(3, 10)),   # trap 2
        6:  dict(P=36, chain=(3, 5), c=(3, 10)),   # trap 3
        7:  dict(P=48, chain=(3, 5), c=(3, 10)),   # trap 4
        8:  dict(P=62, chain=(3, 5), c=(3, 10)),   # trap 5
        9:  dict(P=78, chain=(3, 5), c=(3, 10)),   # trap 6
        10: dict(P=95, chain=(3, 5), c=(3, 10)),   # trap 7 (largest / adversarial)
    }
    cfg = cfgs[min(max(tid, 1), 10)]
    P = cfg["P"]
    lo, hi = cfg["chain"]
    cn, cd = cfg["c"]

    lines = [str(P), f"{cn} {cd}"]
    for _p in range(P):
        k = rng.randint(lo, hi)
        bases = [rng.randint(3, 18) for _ in range(k)]
        lines.append(str(k) + " " + " ".join(map(str, bases)))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
