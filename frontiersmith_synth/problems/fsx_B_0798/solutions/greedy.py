# TIER: greedy
"""
Greedy "textbook recipe": amplify EVERY candidate node (maximum density --
more repeaters can only help, right?) and launch every span at the LARGEST
allowed power (maximum signal -- more power can only help, right?). This
ignores that the Kerr nonlinear penalty grows with the CUBE of launch power
times span length, and that every extra amplifier adds its own fixed noise
floor -- so "more amplifiers, more power, everywhere" is not actually the
recipe that maximizes reach.
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

    p_max = allowed[-1]
    out = []
    out.append(str(N))
    out.append(" ".join(str(i) for i in range(N)))
    out.append(" ".join(str(p_max) for _ in range(N)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
