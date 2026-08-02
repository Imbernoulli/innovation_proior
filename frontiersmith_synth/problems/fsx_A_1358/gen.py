import sys, random

# Difficulty ladder of 10 hand-tuned instances. Each entry:
#   (m, n, rule_type, weights_or_None, target, seed)
# rule_type: 0 = SCORE (positional scoring rule with vector `weights`)
#            1 = RUNOFF2 (Borda top-2, then pairwise-majority runoff)
# `seed` drives the same random.Random(seed)+shuffle profile construction used
# during authoring/tuning, so every case is a fully deterministic function of testId.
TESTS = {
    1:  (3,  8,  0, [1, 0, 0],          2, 100100002),
    2:  (4,  10, 0, [3, 2, 1, 0],       3, 200100000),
    3:  (4,  12, 0, [2, 1, 1, 0],       3, 300100000),
    4:  (5,  14, 0, [3, 2, 1, 0, 0],    3, 400100000),
    5:  (4,  16, 1, None,               2, 500100010),
    6:  (5,  20, 1, None,               2, 600100016),
    7:  (5,  22, 0, [4, 3, 2, 1, 0],    3, 700100000),
    8:  (5,  24, 1, None,               4, 800100005),
    9:  (6,  30, 1, None,               0, 900100003),
    10: (6,  34, 0, [5, 3, 2, 1, 1, 0], 1, 1000100003),
}


def main():
    tid = int(sys.argv[1])
    if tid not in TESTS:
        tid = ((tid - 1) % 10) + 1
    m, n, rule_type, weights, target, seed = TESTS[tid]

    rng = random.Random(seed)
    ballots = []
    for _ in range(n):
        perm = list(range(m))
        rng.shuffle(perm)
        ballots.append(perm)

    out = [f"{m} {n} {rule_type} {target}"]
    if rule_type == 0:
        out.append(" ".join(map(str, weights)))
    for b in ballots:
        out.append(" ".join(map(str, b)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
