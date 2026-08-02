# TIER: invalid
"""Emits a structurally broken artifact: a header that lies about the number
of layers, plus code lines that are not valid permutations of M,K,N. Must
score 0 under the checker's strict schema validation."""
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
    return L


def main():
    L = read_instance()
    out = [str(L + 1)]        # header lies about the count
    out += ["MMM"] * L        # not a permutation of M,K,N (repeated letter)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
