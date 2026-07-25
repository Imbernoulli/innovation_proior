import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rnd = random.Random(seed)

    R = rnd.randint(1, 6)
    C = rnd.randint(1, 6)

    # weighted cell types: open '.', blocked '#', tower '*'
    types = ['.'] * 6 + ['#'] * 3 + ['*'] * 2

    rows = []
    has_tower = False
    cells = []
    for i in range(R):
        row = []
        for j in range(C):
            ch = rnd.choice(types)
            if ch == '*':
                has_tower = True
            row.append(ch)
        rows.append(row)
        cells.append(row)

    # Sometimes force NO tower at all (everything unreachable -> answer 0),
    # to exercise that corner; otherwise ensure at least one tower exists.
    force_no_tower = (rnd.random() < 0.15)
    if force_no_tower:
        for i in range(R):
            for j in range(C):
                if rows[i][j] == '*':
                    rows[i][j] = rnd.choice(['.', '#'])
    elif not has_tower:
        i = rnd.randrange(R); j = rnd.randrange(C)
        rows[i][j] = '*'

    out = [f"{R} {C}"]
    for i in range(R):
        out.append("".join(rows[i]))
    sys.stdout.write("\n".join(out) + "\n")

main()
