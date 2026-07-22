# TIER: greedy
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
    targets = [int(next(it)) for _ in range(R * R)]
    # the obvious first approach: fit ONE global (feed,kill) formulation to the average
    # requested texture, and paint it everywhere -- global parameters, average texture.
    mean_L = sum(targets) / len(targets)
    p = nearest(mean_L)
    out = [str(p)] * (R * R)
    sys.stdout.write(" ".join(out) + "\n")

if __name__ == "__main__":
    main()
