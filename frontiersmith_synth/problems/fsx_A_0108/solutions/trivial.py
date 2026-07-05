# TIER: trivial
# One separable operating mode per nonzero (injection i, production j) fiber:
#   term = e_i (x) e_j (x) H[i][j][:].   R = number of nonzero mode-3 fibers = baseline B.
import sys


def read_tensor():
    d = sys.stdin.read().split()
    a = int(d[0]); b = int(d[1]); c = int(d[2])
    it = iter(d[3:])
    H = [[[int(next(it)) for _ in range(c)] for _ in range(b)] for _ in range(a)]
    return a, b, c, H


def main():
    a, b, c, H = read_tensor()
    terms = []
    for i in range(a):
        for j in range(b):
            fib = H[i][j]
            if any(x != 0 for x in fib):
                u = [0] * a; u[i] = 1
                v = [0] * b; v[j] = 1
                w = list(fib)
                terms.append((u, v, w))
    out = [str(len(terms))]
    for (u, v, w) in terms:
        out.append(" ".join(str(x) for x in u + v + w))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
