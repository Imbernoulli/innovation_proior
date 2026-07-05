# TIER: trivial
# Single ring of N equal zones on the mid-annulus circle -- reproduces the checker baseline B.
import sys, math

def main():
    toks = sys.stdin.read().split()
    N = int(toks[0]); R = float(toks[1]); r_in = float(toks[2])
    rmid = (r_in + R) / 2.0
    w = R - r_in
    r_ang = rmid * math.sin(math.pi / N) if N >= 2 else w / 2.0
    r = min(w / 2.0, r_ang)
    # shrink a hair to stay strictly inside tolerances
    r *= (1.0 - 1e-9)
    out = [str(N)]
    for i in range(N):
        a = 2.0 * math.pi * i / N
        out.append("%.10f %.10f %.10f" % (rmid * math.cos(a), rmid * math.sin(a), r))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
