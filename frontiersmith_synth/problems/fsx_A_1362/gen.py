#!/usr/bin/env python3
"""gen.py <testId> -- prints one fair-division instance to stdout.

Instance = n agents, m = n items, integer valuation matrix v[i][j] in [0,100].

Two planted regimes (alternate by testId), sizes n=3..7 as a difficulty ladder:
  ALIGNED  (odd testId 1,3,5,7,9): item k is the EXCLUSIVE favorite of agent k
           (high value only to its owner, low to everyone else) -- no two agents
           want the same item, so "give each item to its top bidder" happens to
           reconstruct the unique envy-free optimum.
  CONFLICT (even testId 2,4,6,8,10): item 0 is a shared "prize" that EVERY agent
           values highly (higher than any agent's own filler item), while items
           1..n-1 remain exclusive filler favorites of agents 1..n-1 (agent 0 has
           no exclusive item of its own -- it depends on getting a slice of the
           prize). Whichever agent an argmax-per-item rule hands the whole prize
           to is then envied by every other agent, since every other agent values
           the prize strictly more than their own entire bundle.

All randomness is seeded purely from testId -> fully deterministic & reproducible.
"""
import random
import sys


def sizes_and_regime(test_id: int):
    # n grows 3..7 across the ladder; regime alternates aligned/conflict.
    n = 3 + (test_id - 1) // 2
    n = min(n, 7)
    regime = "aligned" if test_id % 2 == 1 else "conflict"
    return n, regime


def build_matrix(rng: random.Random, n: int, regime: str):
    m = n
    v = [[0] * m for _ in range(n)]
    LOW_LO, LOW_HI = 1, 15
    if regime == "aligned":
        OWN_LO, OWN_HI = 70, 100
        for i in range(n):
            for j in range(m):
                if i == j:
                    v[i][j] = rng.randint(OWN_LO, OWN_HI)
                else:
                    v[i][j] = rng.randint(LOW_LO, LOW_HI)
    else:  # conflict
        PRIZE_LO, PRIZE_HI = 85, 100
        FILLER_LO, FILLER_HI = 55, 75
        # item 0 = shared prize, valued highly by every agent independently.
        for i in range(n):
            v[i][0] = rng.randint(PRIZE_LO, PRIZE_HI)
        # items 1..n-1 = exclusive filler favorites of agents 1..n-1.
        for i in range(n):
            for j in range(1, m):
                if i == j:
                    v[i][j] = rng.randint(FILLER_LO, FILLER_HI)
                else:
                    v[i][j] = rng.randint(LOW_LO, LOW_HI)
    return v


def main():
    test_id = int(sys.argv[1])
    n, regime = sizes_and_regime(test_id)
    # decorrelate the RNG stream per testId while staying fully reproducible.
    rng = random.Random(900000 + 97 * test_id + (1 if regime == "conflict" else 0))
    v = build_matrix(rng, n, regime)
    m = n
    out = [f"{n} {m}"]
    for i in range(n):
        out.append(" ".join(str(x) for x in v[i]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
