# TIER: invalid
# Emits words containing a symbol outside the alphabet -> the checker's feasibility
# gate must reject the whole submission (Ratio 0.0).
import sys

def main():
    toks = sys.stdin.read().split()
    k = int(toks[0])
    bad = str(k)          # digit k is NOT in the alphabet {0..k-1}
    sys.stdout.write((bad + "\n") * 5)

if __name__ == "__main__":
    main()
