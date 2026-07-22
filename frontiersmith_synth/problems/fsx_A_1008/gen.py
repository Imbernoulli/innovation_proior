import sys, random


def build_core_table(c, sigma, shift_idx, merge_idx):
    """Cerny-style core over states 0..c-1: `shift_idx` is a cyclic permutation,
    `merge_idx` merges state c-1 into 0 (else identity) -- guarantees the core is
    synchronizable. Every OTHER letter is the identity on this core (so a word
    built purely from another block's letters cannot touch this block at all)."""
    table = [list(range(c)) for _ in range(sigma)]
    table[shift_idx] = [(i + 1) % c for i in range(c)]
    table[merge_idx] = [0 if i == c - 1 else i for i in range(c)]
    return table


def build_automaton(n, c, sigma, core_table, shift_idx, merge_idx, rng):
    """states 0..c-1 = core (uses core_table verbatim), c..n-1 = peripheral.
    Under this block's own two active letters (shift/merge), every peripheral
    state jumps straight into the core (a random target). Under any other
    letter (another block's letters), every state -- core or peripheral -- is
    left exactly where it is (identity), so foreign letters are true no-ops."""
    table = [row[:] for row in core_table]  # currently length c per row
    for l in range(sigma):
        for p in range(c, n):
            if l == shift_idx or l == merge_idx:
                table[l].append(rng.randrange(c))
            else:
                table[l].append(p)
    return table  # table[l][s] for s in 0..n-1


def emit_automaton(table, n, sigma):
    lines = [str(n)]
    for l in range(sigma):
        lines.append(" ".join(str(x) for x in table[l]))
    return lines


def plan(t):
    """Return (Lmax, blocks) where blocks = list of (core_size, [peripheral_sizes...]).
    sigma is derived as 2*len(blocks) so every block gets its own disjoint
    (shift, merge) letter pair -- no two blocks ever share an active letter."""
    if t == 1:
        blocks = [(3, [1, 1]), (5, [1, 1])]
        Lmax = 26
    elif t == 2:
        blocks = [(3, [1, 1, 1]), (5, [1, 1])]
        Lmax = 30
    elif t == 3:
        blocks = [(3, [1, 1, 1, 1]), (6, [1, 1])]
        Lmax = 40
    elif t == 4:
        blocks = [(3, [1, 1, 1, 1, 1]), (6, [1, 1])]
        Lmax = 40
    elif t == 5:
        blocks = [(3, [1, 1, 1, 1, 1]), (6, [1, 1, 1])]
        Lmax = 40
    elif t == 6:
        blocks = [(3, [1, 1, 1, 1, 1, 1]), (6, [1, 1, 1])]
        Lmax = 40
    elif t == 7:
        blocks = [(3, [1, 1, 1, 1, 1, 1]), (7, [1, 1, 1])]
        Lmax = 48
    elif t == 8:
        blocks = [(3, [1, 1, 1, 1, 1, 1, 1]), (7, [1, 1, 1])]
        Lmax = 48
    elif t == 9:
        blocks = [(3, [1, 1, 1, 1, 1, 1, 1, 1]), (7, [1, 1, 1])]
        Lmax = 48
    else:  # t == 10, hardest trap: 9-automaton cheap block, 3-automaton pricier block
        blocks = [(3, [1, 1, 1, 1, 1, 1, 1, 1, 1]), (7, [1, 1, 1])]
        Lmax = 48
    return Lmax, blocks


def main():
    t = int(sys.argv[1])
    rng = random.Random(1000 + 7919 * t)
    Lmax, blocks = plan(t)
    sigma = 2 * len(blocks)

    automata = []  # list of (n, table)
    for bi, (core_size, peripherals) in enumerate(blocks):
        shift_idx, merge_idx = 2 * bi, 2 * bi + 1
        core_table = build_core_table(core_size, sigma, shift_idx, merge_idx)
        for p in peripherals:
            n = core_size + p
            table = build_automaton(n, core_size, sigma, core_table, shift_idx, merge_idx, rng)
            automata.append((n, table))

    k = len(automata)
    out = [f"{k} {sigma} {Lmax}"]
    for n, table in automata:
        out.extend(emit_automaton(table, n, sigma))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
