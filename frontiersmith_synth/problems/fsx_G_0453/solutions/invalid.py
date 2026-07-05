# TIER: invalid
# Emits a structurally invalid placement (all logical qubits claimed to sit on
# physical qubit 0) -> the checker's placement validity gate rejects it -> 0.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    P = int(next(it)); E = int(next(it)); M = int(next(it)); L = int(next(it))
    # placement line: all zeros (logical 0 placed P times) -> invalid
    out = [" ".join("0" for _ in range(P))]
    for _ in range(M):
        out.append("0")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
