import sys, random, itertools

# gen.py <testId> -- prints ONE shared-XOR-circuit instance to stdout.
#
# Structure: partition C = m*s variables into m disjoint "atoms" of size s
# (a fixed but hidden group of s variables, labels scattered by a shuffle so
# atom membership is not index-contiguous). Each of the R targets is the XOR
# of k DISTINCT atoms chosen uniformly (atoms are variable-disjoint, so a
# target's variable set is exactly the union of its k atoms -- weight = k*s,
# no accidental GF(2) cancellation). Atom reuse factor u = R*k/m is kept
# >= ~4 so every atom is shared by several targets (the exploitable common
# subexpression), but no atom's full membership ever appears as any single
# target's variable set on its own (every target is a union of >=2 atoms) --
# so a per-target/per-row construction pass never exposes an atom as a
# reusable intermediate. s >= 3 additionally means a one-shot "most frequent
# raw-variable PAIR" pass can capture at most part of an atom (2 of its s
# variables), never the atom as a single reusable wire; only an iterative
# scheme that re-mines candidate wires (raw variables AND previously built
# wires) after each commitment can discover the full atom and reuse it.
#
# (atom_size s, atoms_per_target k, num_atoms m, num_targets R)
SPECS = {
    1:  (3, 2, 5, 10),
    2:  (3, 2, 6, 14),
    3:  (3, 3, 6, 16),
    4:  (3, 3, 7, 18),
    5:  (3, 3, 8, 24),
    6:  (4, 3, 8, 26),
    7:  (3, 4, 10, 30),
    8:  (4, 3, 10, 34),
    9:  (4, 4, 10, 38),
    10: (4, 4, 12, 44),
}


def main():
    tid = int(sys.argv[1])
    s, k, m, R = SPECS[tid]
    rng = random.Random(20000 + 37 * tid)

    C = m * s
    labels = list(range(1, C + 1))
    rng.shuffle(labels)
    atoms = [sorted(labels[i * s:(i + 1) * s]) for i in range(m)]

    combos = list(itertools.combinations(range(m), k))
    rng.shuffle(combos)
    chosen = combos[:R]

    rows = []
    for idxs in chosen:
        varset = set()
        for a_idx in idxs:
            varset.update(atoms[a_idx])
        rows.append(sorted(varset))

    out = ["%d %d" % (len(rows), C)]
    for row in rows:
        out.append("%d %s" % (len(row), " ".join(map(str, row))))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
