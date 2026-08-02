# TIER: strong
"""The insight: deliberately form the WEAK, reversible (target) bonds first
so that mistakes can still anneal out, and hold the strong (decoy) bonds
back as long as possible so they never get a chance to freeze onto a
monomer the target structure actually needs. Target bonds are enabled at
step 1 -- they keep re-forming and breaking harmlessly while hot, then lock
into the CORRECT configuration once the temperature finally drops below
their own strength. Decoys are enabled only at the very last step, by which
point every target monomer they might have grabbed is already frozen in
place, so they simply have nowhere left to bind."""
import sys


def main():
    data = sys.stdin.read().split()
    idx = 0
    N = int(data[idx]); idx += 1
    M = int(data[idx]); idx += 1
    Tmax = int(data[idx]); idx += 1
    theta0 = int(data[idx]); idx += 1
    types = []
    for _ in range(M):
        idx += 3  # u, v, s
        typ = data[idx]; idx += 1
        types.append(typ)

    times = [1 if typ == 'T' else Tmax for typ in types]
    print(" ".join(map(str, times)))


if __name__ == "__main__":
    main()
