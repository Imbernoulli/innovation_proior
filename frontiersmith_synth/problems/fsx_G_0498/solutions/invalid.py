# TIER: invalid
# Emits garbage: m rows of the non-binary character '2'. Every "decoding" fails the
# checker's strict length-n {0,1} parse, so the whole artifact is rejected -> Ratio 0.0.
import sys


def main():
    data = sys.stdin.read().split("\n")
    idx = 0
    while data[idx].strip() == "":
        idx += 1
    n, r, m, T = map(int, data[idx].split())
    out = ["2" * n for _ in range(m)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
