# TIER: strong
import sys, math

L_CALIB = [10.8, 13.0, 14.6, 16.7, 18.9, 20.1]  # must match verify.py's L_CALIB
P = len(L_CALIB)

def nearest(L):
    best_p, best_d = 0, 1e18
    for p in range(P):
        d = abs(math.log(L_CALIB[p] / L))
        if d < best_d:
            best_d, best_p = d, p
    return best_p

def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    R = int(next(it)); seed = int(next(it))
    targets = [[int(next(it)) for _ in range(R)] for _ in range(R)]
    # the insight: global parameters only set an AVERAGE texture. Since each region has its
    # own requested wavelength, design the FIELD by solving each region's own nearest-match
    # against the published (feed,kill) calibration table independently.
    out = []
    for i in range(R):
        for j in range(R):
            out.append(str(nearest(targets[i][j])))
    sys.stdout.write(" ".join(out) + "\n")

if __name__ == "__main__":
    main()
