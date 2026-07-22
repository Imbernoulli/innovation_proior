# TIER: trivial
# Bolt nothing -- reproduces the checker's bolt-nothing baseline (-> ~0.1).
import sys


def main():
    toks = sys.stdin.read().split()
    E = int(toks[2])                 # line2: N E L p q Bmax  -> E is token index 2
    out = "\n".join("0" for _ in range(E))
    sys.stdout.write(out + "\n")


if __name__ == "__main__":
    main()
