# TIER: strong
# THE INSIGHT: ignore the program text entirely.  Treat the ritual as a black box,
# probe it, and rediscover WHAT it computes.
#   1) Evaluate the given program at all 2^8 boolean input points.
#   2) Mobius (inclusion-exclusion) transform -> the multilinear polynomial's
#      monomial coefficients a_S.  (The hidden function is multilinear.)
#   3) Recognise the closed-form identity: it is a tridiagonal continuant with
#      diagonal x_0..x_{n-1}.  Read off each off-diagonal constant directly:
#         c_k = -a_{full \ {k-1,k}}      (a single adjacent domino deleted).
#   4) Emit the compact 3-term recurrence  D_k = x_k*D_{k-1} - c_k*D_{k-2}.
# This collapses the ritual to O(n) ops -- unreachable by any syntactic optimiser.
import sys

NIN = 8
P = 2147483647

def read_prog():
    it = iter(sys.stdin.read().split())
    p = int(next(it)); L = int(next(it))
    prog = []
    for _ in range(L):
        op = next(it)
        if op == "const":
            prog.append(("const", int(next(it)) % P, None))
        else:
            prog.append((op, int(next(it)), int(next(it))))
    return prog

def evaluate(prog, xs):
    vals = list(xs)
    for ins in prog:
        op = ins[0]
        if op == "const":
            vals.append(ins[1])
        elif op == "add":
            vals.append((vals[ins[1]] + vals[ins[2]]) % P)
        elif op == "sub":
            vals.append((vals[ins[1]] - vals[ins[2]]) % P)
        else:
            vals.append((vals[ins[1]] * vals[ins[2]]) % P)
    return vals[-1]

def main():
    prog = read_prog()

    # 1) values on the boolean cube
    f = [0] * 256
    for mask in range(256):
        xs = [1 if (mask >> b) & 1 else 0 for b in range(8)]
        f[mask] = evaluate(prog, xs)

    # 2) Mobius transform: a_S = sum_{T subset S} (-1)^{|S\T|} f(T)
    a = [0] * 256
    for S in range(256):
        s = 0
        T = S
        while True:
            popc = bin(S ^ T).count("1")
            s += f[T] if popc % 2 == 0 else -f[T]
            if T == 0:
                break
            T = (T - 1) & S
        a[S] = s % P

    # 3) active variables + recover continuant constants
    active = set()
    for S in range(256):
        if a[S] % P != 0:
            for b in range(8):
                if (S >> b) & 1:
                    active.add(b)
    n = (max(active) + 1) if active else 1
    full = (1 << n) - 1
    c = {}
    for k in range(1, n):
        Sk = full & ~((1 << (k - 1)) | (1 << k))
        c[k] = (-a[Sk]) % P

    # 4) emit the recurrence program
    out = []
    def emit(line):
        out.append(line)
        return NIN + len(out) - 1

    if n == 1:
        z = emit("const 0")
        emit("add 0 %d" % z)               # result = x0
        sys.stdout.write("\n".join([str(len(out))] + out) + "\n")
        return

    # D_1 = x1*x0 - c1
    t = emit("mul 1 0")
    c1 = emit("const %d" % (c[1] % P))
    Dprev1 = emit("sub %d %d" % (t, c1))
    Dprev2 = 0                              # D_0 = x0 (index 0)
    for k in range(2, n):
        m1 = emit("mul %d %d" % (k, Dprev1))
        ck = emit("const %d" % (c[k] % P))
        m2 = emit("mul %d %d" % (ck, Dprev2))
        Dk = emit("sub %d %d" % (m1, m2))
        Dprev2, Dprev1 = Dprev1, Dk

    sys.stdout.write("\n".join([str(len(out))] + out) + "\n")

if __name__ == "__main__":
    main()
