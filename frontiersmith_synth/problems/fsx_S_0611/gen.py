import sys

# Instance = design parameters for a coherent pixel typeface.
#   line1: N H W          (N glyphs on an H x W grid)
#   line2: inkLo inkHi    (per-glyph ink-count budget, inclusive)
#   line3: mh mw          (motif window height/width for the vocabulary count)
#   line4: gnum gden      (vocabulary penalty gamma = gnum/gden, a rational)
#
# The score maximizes  D^2 / (1 + gamma * V)  where
#   D = min pairwise Hamming distance between glyphs (distinctness), and
#   V = number of distinct mh x mw ink windows over the whole font (style cost).
# Larger gamma => coherence (small motif vocabulary) matters more than raw distance,
# so a distance-only construction is a trap: it buys distinctness with a huge vocabulary.

# Fixed grid / motif / budget across the ladder so the baseline stays feasible.
H, W = 7, 5
MH, MW = 2, 3
INK_LO, INK_HI = 12, 24
GDEN = 50

# (N, gnum) schedule: N grows (small -> full alphabet); gamma spans low->high.
# Higher-gamma cases are the traps where a distance-only greedy lands far from strong.
SCHED = [
    (10,  6),   # 1  gamma 0.12
    (14,  7),   # 2  0.14
    (18,  8),   # 3  0.16
    (22,  9),   # 4  0.18
    (26, 11),   # 5  0.22
    (26, 12),   # 6  0.24   trap
    (26, 14),   # 7  0.28   trap
    (26, 16),   # 8  0.32   trap
    (26, 10),   # 9  0.20
    (26, 15),   # 10 0.30   trap
]

def main():
    t = int(sys.argv[1])
    idx = (t - 1) % len(SCHED)
    N, gnum = SCHED[idx]
    out = []
    out.append("%d %d %d" % (N, H, W))
    out.append("%d %d" % (INK_LO, INK_HI))
    out.append("%d %d" % (MH, MW))
    out.append("%d %d" % (gnum, GDEN))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
