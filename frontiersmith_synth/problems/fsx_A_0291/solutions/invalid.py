# TIER: invalid
# Emit the entire space F_3^n, which is packed with conflict triples -> scores 0.
import sys

def main():
    tok = sys.stdin.read().split()
    n = int(tok[0])
    space = 3 ** n

    def dec(e):
        d = []
        for _ in range(n):
            d.append(e % 3); e //= 3
        return tuple(d)

    out = [str(space)]
    for e in range(space):
        out.append(' '.join(map(str, dec(e))))
    sys.stdout.write('\n'.join(out) + '\n')

main()
