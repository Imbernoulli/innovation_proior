# TIER: invalid
# Emits a well-formed-looking circuit whose outputs DO NOT realise the target rows
# (wrong wiring) -> the checker's exact-equivalence gate must score this 0.
import sys

def main():
    tok = sys.stdin.read().split()
    p = 0
    def nx():
        nonlocal p
        v = int(tok[p]); p += 1; return v
    m = nx(); n = nx()
    for _ in range(m * n):
        nx()
    # one gate (0 ^ 1); route every output to input node 0 (mask = x_0), which does not
    # equal the dense target rows.
    gates = [(0, 1)]
    outputs = [0] * m
    out = [str(len(gates))]
    for a, b in gates:
        out.append("%d %d" % (a, b))
    out.append(" ".join(map(str, outputs)))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
