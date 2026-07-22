import math
import random
import sys

# Difficulty ladder: (K, Dcrit, P)
#   K     = number of arms sharing the breaker
#   Dcrit = distance of the ONE long "critical" chain (arm 0)
#   P     = shared instantaneous power cap (the breaker rating)
# All other K-1 arms get short chains (planted "slack" arms) drawn from a
# small range relative to Dcrit, so the instance always mixes one long chain
# with many short ones (the trap: naive per-arm max-speed serialization
# wastes the cap on the short arms instead of overlapping them with the
# critical arm's run). A is derived from P so the ladder stays in a
# calibrated makespan-vs-energy trade-off regime at every scale.
SIZES = [
    (8,  600,  150),
    (10, 900,  180),
    (14, 1500, 220),
    (16, 2000, 300),
    (20, 2600, 360),
    (22, 3200, 450),
    (26, 4000, 550),
    (28, 4800, 650),
    (32, 5500, 800),
    (36, 6000, 900),
]


def build(testId):
    K, Dcrit, P = SIZES[(testId - 1) % len(SIZES)]
    rnd = random.Random(1000 + 7 * testId)

    VMAX = math.isqrt(P)
    A = max(1, round(2.2 * VMAX))

    D = [Dcrit]
    hi = max(6, Dcrit // 25)
    for _ in range(K - 1):
        D.append(rnd.randint(5, hi))

    # Shuffle the non-critical arms' positions (but remember arm 0 is fixed
    # as the planted critical one) so a solver cannot assume sorted input.
    tail = D[1:]
    rnd.shuffle(tail)
    D = [D[0]] + tail

    out = []
    out.append(f"{K} {P} {A}")
    out.append(" ".join(str(x) for x in D))
    return "\n".join(out) + "\n"


def main():
    testId = int(sys.argv[1])
    sys.stdout.write(build(testId))


if __name__ == "__main__":
    main()
