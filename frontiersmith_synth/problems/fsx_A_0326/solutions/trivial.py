# TIER: trivial
# Reproduces the checker's internal reference design M0 (diag +1, off -1).
import sys

def main():
    N = int(sys.stdin.read().split()[0])
    out = []
    for i in range(N):
        out.append(" ".join("1" if i == j else "-1" for j in range(N)))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
