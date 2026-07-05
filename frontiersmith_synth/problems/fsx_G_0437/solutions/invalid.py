# TIER: invalid
# Emits a single rank-1 term of all-ones -> does not reconstruct T -> score 0.
import sys

def main():
    tok = sys.stdin.read().split()
    a = int(tok[0]); b = int(tok[1]); c = int(tok[2])
    out = ["1"]
    out.append(" ".join(["1"] * a))
    out.append(" ".join(["1"] * b))
    out.append(" ".join(["1"] * c))
    sys.stdout.write("\n".join(out) + "\n")

main()
