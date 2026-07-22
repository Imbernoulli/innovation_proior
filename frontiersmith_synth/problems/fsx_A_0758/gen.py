import sys, random

# Roles played by the alphabet symbols. Exactly one "trigger" role (A) has an
# expand rule; exactly two ordered pairs have collapse rules: (C,D)->E (the
# canyon collapse, only reachable after expanding an A that is followed by a
# D) and (P,Q)->S (an immediately-available "bait" collapse, no expand needed).
ROLES = ["A", "B", "C", "D", "E", "P", "Q", "S", "F1", "F2"]
BASE_COST = {"A": 50, "B": 20, "C": 35, "D": 40, "E": 2,
             "P": 20, "Q": 20, "S": 1, "F1": 8, "F2": 6}

# difficulty ladder: (num_motifs, num_baits, num_fillers)
# t6, t7, t9 are motif-heavy "trap" cases (few baits relative to motifs) where
# a single-step-only greedy recipe captures almost none of the achievable gain.
LADDER = {
    1: (1, 2, 2),
    2: (2, 4, 3),
    3: (3, 6, 4),
    4: (6, 10, 8),
    5: (10, 15, 15),
    6: (20, 10, 30),
    7: (40, 8, 60),
    8: (80, 60, 120),
    9: (150, 25, 220),
    10: (250, 300, 400),
}


def main():
    tid = int(sys.argv[1])
    M, Pc, Ff = LADDER.get(tid, LADDER[10])
    rng = random.Random(19260817 + 97 * tid)

    roles = ROLES[:]
    rng.shuffle(roles)
    role_to_sym = {role: i + 1 for i, role in enumerate(roles)}
    K = len(ROLES)
    cost = [0] * (K + 1)
    for role in ROLES:
        cost[role_to_sym[role]] = BASE_COST[role]

    symA, symB, symC, symD, symE = (role_to_sym[r] for r in "ABCDE")
    symP, symQ, symS = (role_to_sym[r] for r in "PQS")
    symF1, symF2 = role_to_sym["F1"], role_to_sym["F2"]

    blocks = []
    for _ in range(M):
        blocks.append([symA, symD])
    for _ in range(Pc):
        blocks.append([symP, symQ])
    for _ in range(Ff):
        blocks.append([symF1 if rng.random() < 0.5 else symF2])
    rng.shuffle(blocks)

    s = [x for blk in blocks for x in blk]
    n = len(s)
    R = 2 * M + Pc + 2  # small slack over the exact budget the planted plan needs

    expand_rules = [(symA, symB, symC)]              # v -> x y
    collapse_rules = [(symC, symD, symE), (symP, symQ, symS)]  # x y -> z

    out = []
    out.append(f"{n} {K} {R}")
    out.append(" ".join(map(str, s)))
    out.append(" ".join(str(cost[v]) for v in range(1, K + 1)))
    out.append(str(len(expand_rules)))
    for v, x, y in expand_rules:
        out.append(f"{v} {x} {y}")
    out.append(str(len(collapse_rules)))
    for x, y, z in collapse_rules:
        out.append(f"{x} {y} {z}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
