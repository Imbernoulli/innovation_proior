# TIER: trivial
# The identity permutation S(x)=x. It is affine, so its nonlinearity is 0 and
# its linearity equals 2^n -- exactly the checker's baseline B (=> Ratio 0.1).
import sys

def main():
    n = int(sys.stdin.read().split()[0])
    N = 1 << n
    sys.stdout.write("\n".join(str(x) for x in range(N)) + "\n")

if __name__ == "__main__":
    main()
