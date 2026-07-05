import sys, random

# gen.py <testId> : print ONE instance of the reservoir-batch routing problem.
#
# A reservoir network is a rows x cols grid of reservoirs (nodes). Two reservoirs
# are directly connected by a canal iff they are orthogonally adjacent in the grid.
# Reservoir id = r*cols + c  (0-indexed, row-major).
#
# There are V = rows*cols labelled water BATCHES, batch i starts in reservoir i
# (identity placement). A list of K scheduled MIXING OPERATIONS is given; operation
# t requires batches a_t and b_t to sit in two directly-connected reservoirs at the
# moment it is performed. Operations may be performed in ANY order.
#
# Output schema:
#   rows cols
#   K
#   a_1 b_1
#   ...
#   a_K b_K

DIMS = {1:(3,3),2:(3,4),3:(4,4),4:(4,4),5:(4,5),
        6:(4,5),7:(5,5),8:(5,5),9:(5,6),10:(5,6)}
NOPS = {1:8,2:10,3:14,4:18,5:22,6:26,7:32,8:38,9:46,10:55}

def main():
    t = int(sys.argv[1])
    rows, cols = DIMS.get(t, (5,6))
    K = NOPS.get(t, 55)
    V = rows * cols
    rng = random.Random(1000 + t)
    ops = []
    for _ in range(K):
        a = rng.randrange(V)
        b = rng.randrange(V)
        while b == a:
            b = rng.randrange(V)
        ops.append((a, b))
    out = [f"{rows} {cols}", str(K)]
    for a, b in ops:
        out.append(f"{a} {b}")
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
