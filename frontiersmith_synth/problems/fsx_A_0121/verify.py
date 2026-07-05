#!/usr/bin/env python3
"""
Deterministic checker for the "reservoir dam network" resonance-free scheduling
problem (greedy-priority-construction / cap-set family, format C).

CLI: python3 verify.py <in> <out> <ans>     (ans is an empty placeholder, ignored)

Instance (<in>):  a single integer n = number of rivers.
Artifact (<out>): first integer M, then M rows of n integers, each in {0,1,2}
                  (a discharge profile over the n rivers, level in {low=0,mid=1,high=2}).

Feasibility (STRICT):
  * every profile is a length-n vector over {0,1,2};
  * all M profiles are distinct;
  * NO three distinct profiles a,b,c "resonate": a[i]+b[i]+c[i] == 0 (mod 3) for
    every river i.  (Equivalently: the profile set is a cap set in F_3^n.)
  ANY violation  ->  Ratio: 0.0

Objective (maximize): F = M = number of scheduled profiles.
Internal baseline B: the "conservative" construction {0,1}^{n-1} x {0}  (fix the
  last river to low, use only low/mid on the rest) -- always a valid resonance-free
  family of size 2^(n-1).
Score:  sc = min(1000, 100 * F / max(1e-9, B)) ; Ratio = sc/1000
        (trivial ~ 0.1 ; a 10x-better schedule caps at 1.0)
"""
import sys


def read_ints(path):
    with open(path) as f:
        toks = f.read().split()
    out = []
    for t in toks:
        try:
            out.append(int(t))
        except ValueError:
            return None
    return out


def fail(reason):
    print("Resonance/feasibility violation: %s  Ratio: 0.0" % reason)
    sys.exit(0)


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]

    inp = read_ints(in_path)
    if not inp:
        fail("bad instance")
    n = inp[0]

    toks = read_ints(out_path)
    if toks is None or len(toks) == 0:
        fail("output not a list of integers")

    M = toks[0]
    if M < 0:
        fail("negative count")
    # hard cap: a distinct subset of F_3^n cannot exceed 3^n profiles
    if M > 3 ** n:
        fail("count exceeds 3^n")
    body = toks[1:]
    if len(body) != M * n:
        fail("expected %d profile entries, got %d" % (M * n, len(body)))

    profiles = []
    for r in range(M):
        row = tuple(body[r * n:(r + 1) * n])
        for v in row:
            if v < 0 or v > 2:
                fail("discharge level out of {0,1,2}")
        profiles.append(row)

    S = set(profiles)
    if len(S) != M:
        fail("duplicate profiles")

    # cap-set check: for each distinct pair (a,b) the unique completing profile c
    # with a+b+c == 0 (mod 3) must NOT be present.  For a != b, c is automatically
    # distinct from both a and b.
    plist = profiles
    for i in range(M):
        a = plist[i]
        for j in range(i + 1, M):
            b = plist[j]
            c = tuple((-(a[k] + b[k])) % 3 for k in range(n))
            if c in S:
                fail("resonant triple detected")

    F = float(M)
    B = float(2 ** (n - 1))
    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("rivers=%d scheduled=%d baseline=%d  Ratio: %.6f" % (n, M, int(B), sc / 1000.0))


if __name__ == "__main__":
    main()
