# TIER: invalid
"""Deliberately infeasible: every tug just stays parked at its start position, but we
still claim every ship is served (with start tick 0). Since a parked tug is essentially
never exactly at the required rendezvous point, the checker's position check must reject
this -> Ratio: 0.0."""
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    T = int(next(it)); N = int(next(it)); L = int(next(it))
    pos = [int(next(it)) for _ in range(T)]
    coeff = int(next(it)); pen = int(next(it))
    jobs = []
    for _ in range(N):
        a = int(next(it)); b = int(next(it)); k = int(next(it))
        rel = int(next(it)); w = int(next(it)); nw = int(next(it))
        windows = [(int(next(it)), int(next(it))) for _ in range(nw)]
        jobs.append({"a": a, "b": b, "k": k, "release": rel, "weight": w})

    out = []
    for t in range(T):
        out.append(f"1 0 {pos[t]}")
    out.append(str(N))
    for j in range(1, N + 1):
        k = jobs[j - 1]["k"]
        team = [(t) % T for t in range(k)]
        out.append(f"{j} 0 " + " ".join(map(str, team)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
