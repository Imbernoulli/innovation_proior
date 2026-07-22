# TIER: trivial
"""
Trivial baseline: install exactly one amplifier, at the transmitter (node 0),
and launch at the SMALLEST allowed power. No spacing decision, no power
tuning -- the simplest possible feasible plan.
"""
import sys


def main():
    data = sys.stdin.read().split()
    ptr = 0
    N = int(data[ptr]); ptr += 1
    ptr += N  # xs (unused)
    ptr += 4  # alpha c0 c_ase c_kerr (unused)
    ptr += 1  # thresh (unused)
    K = int(data[ptr]); ptr += 1
    allowed = [int(v) for v in data[ptr:ptr + K]]; ptr += K

    p_min = allowed[0]
    out = []
    out.append("1")
    out.append("0")
    out.append(str(p_min))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
