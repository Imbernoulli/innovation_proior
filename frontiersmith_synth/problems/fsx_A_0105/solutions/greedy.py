# TIER: greedy
# Uniform full-support profile: ping equally in every segment.
# Its autoconvolution is a triangle peaking at the center -> C1 = 2.0 exactly, beating the
# naive half-fill baseline (B = 4.0) for a Ratio ~ 0.2.
import sys

def main():
    n = int(sys.stdin.read().split()[0])
    f = [1.0] * n
    sys.stdout.write(" ".join("%.6f" % x for x in f) + "\n")

if __name__ == "__main__":
    main()
