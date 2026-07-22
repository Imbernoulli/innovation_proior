import sys, random

# Industrial paint-line color-changeover sequencing with cumulative residue and full cleans.
# Instance ladder: testId 1..10, N = 8 + testId (9..18).
# Planted structure: asymmetric changeover matrix (dark->light expensive), per-job residue
# deposits, a residue cap R that partitions any run into epochs, and a full-clean cost.

LAM = 10          # darkness (dark->light) penalty weight  -> big asymmetric transitions
RMIN, RMAX = 6, 16
R_CAP = 60        # cumulative residue threshold
CCLEAN = 18       # cost of one full clean (epoch reset)
DMAX = 9

def main():
    testId = int(sys.argv[1])
    rng = random.Random(60600 + testId)
    N = 8 + testId

    d = [rng.randint(0, DMAX) for _ in range(N)]      # darkness of each color
    h = [rng.randint(0, 5) for _ in range(N)]         # hue family
    r = [rng.randint(RMIN, RMAX) for _ in range(N)]   # residue deposited by each job
    s = [rng.randint(2, 6) for _ in range(N)]         # startup cost from a clean machine
    for j in range(N):
        if r[j] > R_CAP:
            r[j] = R_CAP                              # every job fits alone in one epoch

    w = [[0]*N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            huecost = min(4, abs(h[i] - h[j]))
            darkpen = LAM * max(0, d[i] - d[j])       # painting light after dark is costly
            noise = rng.randint(0, 3)
            w[i][j] = 2 + huecost + darkpen + noise

    out = []
    out.append("%d %d %d" % (N, R_CAP, CCLEAN))
    out.append(" ".join(map(str, r)))
    out.append(" ".join(map(str, s)))
    for i in range(N):
        out.append(" ".join(map(str, w[i])))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
