import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    random.seed(seed)

    q = random.randint(1, 14)
    # Small badge universe so inclusive/exclusive boundaries get exercised heavily,
    # including query bounds that fall between or outside actual badge values.
    MAXB = random.choice([3, 5, 8, 12])
    present = set()
    lines = [str(q)]
    for _ in range(q):
        # Decide op: bias toward queries, but keep insert/remove balanced with state.
        choices = ['?']
        # can insert if there's a free badge
        if len(present) < MAXB:
            choices.append('+')
        if present:
            choices.append('-')
        t = random.choice(choices)
        if t == '+':
            # pick a badge not present
            b = random.randint(1, MAXB)
            while b in present:
                b = random.randint(1, MAXB)
            present.add(b)
            lines.append(f"+ {b}")
        elif t == '-':
            b = random.choice(sorted(present))
            present.discard(b)
            lines.append(f"- {b}")
        else:
            # query bounds can be outside [1,MAXB] and can be lo>hi sometimes
            lo = random.randint(0, MAXB + 2)
            hi = random.randint(0, MAXB + 2)
            lines.append(f"? {lo} {hi}")
    sys.stdout.write('\n'.join(lines) + '\n')

main()
