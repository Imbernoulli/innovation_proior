# TIER: trivial
# Constant strength, "always release": r_t = 1 for every tick. Every bud
# passes its founding check and the plant grows fully, uniformly bushy in
# creation order -- exactly the grader's own internal baseline (-> ~0.1).
import sys


def main():
    toks = sys.stdin.read().split()
    H = int(toks[0])
    sys.stdout.write(" ".join(["1"] * H) + "\n")


if __name__ == "__main__":
    main()
