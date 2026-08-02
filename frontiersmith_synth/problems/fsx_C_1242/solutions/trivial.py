# TIER: trivial
"""Reproduces the checker's own baseline construction: pin K and N stationary,
stream M, on EVERY layer regardless of shape (classic textbook
weight-stationary default). No adaptation to the instance at all."""
import sys


def read_instance():
    toks = sys.stdin.read().split()
    i = 0

    def nxt():
        nonlocal i
        v = toks[i]
        i += 1
        return v

    P = int(nxt()); Q = int(nxt()); L = int(nxt())
    RELOAD = int(nxt()); SWITCH = int(nxt())
    layers = []
    for _ in range(L):
        m = int(nxt()); k = int(nxt()); n = int(nxt())
        layers.append((m, k, n))
    return L, layers


def main():
    L, layers = read_instance()
    out = [str(L)]
    out += ["KNM"] * L
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
