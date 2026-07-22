import sys

# Difficulty ladder: hand-specified (N, chain_k, free_k, cap) per testId.
# chain_k: strictly ascending divisor chain (each divides the next, last divides N) --
#   the mandated colotomic hierarchy (slowest .. fastest structural layer).
# free_k: independent (non-nested) instrument onset counts, each dividing N.
# cap: max instruments allowed to sound simultaneously on any single beat.
#
# cap == len(chain_k) is a TRAP configuration: the nested chain alone already reaches
# len(chain_k) simultaneous onsets at the coarsest beats (unavoidable, required by the
# hierarchy), so every free instrument must be phased to NEVER coincide with the chain's
# finest grid or with any other free instrument -- exactly the CRT-residue task. A solver
# that reuses the textbook "start every part on the downbeat" convention (phase 0 for all)
# collides immediately at beat 0 for literally every instrument.
TESTS = {
    1:  (72,   [3, 6],           [4, 4],                           3),
    2:  (90,   [3, 6],           [9, 9, 9, 9, 9],                  2),   # trap
    3:  (120,  [3, 6, 12],       [8, 8, 8],                        3),   # trap
    4:  (168,  [3, 6],           [7, 7, 7, 7],                     2),   # trap
    5:  (180,  [3, 6, 12],       [4, 4, 4, 5, 5],                  3),   # trap
    6:  (252,  [3, 6],           [7, 7, 7, 4, 4, 4],               2),   # trap
    7:  (240,  [3, 6],           [5, 5, 5, 5],                     2),   # trap
    8:  (280,  [5, 10, 20],      [4, 4, 4, 7, 7],                  4),
    9:  (360,  [3, 6, 12],       [8, 8, 8, 8, 5, 5],               3),   # trap
    10: (420,  [3, 6, 12],       [10, 10, 10, 10, 7, 7, 7],        3),   # trap, largest
}


def main():
    tid = int(sys.argv[1])
    tid = max(1, min(10, tid))
    N, chain, free, cap = TESTS[tid]
    D = len(chain)
    R = len(free)
    out = []
    out.append("%d %d %d %d" % (N, D, R, cap))
    out.append(" ".join(str(x) for x in chain))
    out.append(" ".join(str(x) for x in free))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
