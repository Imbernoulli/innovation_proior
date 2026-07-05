# TIER: invalid
# Emits a well-formed but WRONG circuit: it runs basic MMD synthesis and then drops
# the final gate, so the circuit no longer computes the target permutation. The
# checker's exact-equivalence gate must reject it -> score 0.
import sys


def synth_basic(perm, n):
    N = 1 << n
    t = [0] * N
    for x in range(N):
        t[perm[x]] = x
    gates = []
    for i in range(N):
        ns = i & ~t[i]; b = 0
        while ns:
            if ns & 1:
                cmask = t[i]; mask = 1 << b
                for x in range(N):
                    if (t[x] & cmask) == cmask:
                        t[x] ^= mask
                gates.append((b, cmask))
            ns >>= 1; b += 1
        nc = t[i] & ~i; b = 0
        while nc:
            if nc & 1:
                cmask = i; mask = 1 << b
                for x in range(N):
                    if (t[x] & cmask) == cmask:
                        t[x] ^= mask
                gates.append((b, cmask))
            nc >>= 1; b += 1
    return gates


def emit(gates, n):
    out = [str(len(gates))]
    for (tgt, cmask) in gates:
        controls = [c for c in range(n) if (cmask >> c) & 1]
        out.append(" ".join([str(len(controls)), str(tgt)] + [str(c) for c in controls]))
    sys.stdout.write("\n".join(out) + "\n")


def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    N = 1 << n
    perm = [int(v) for v in data[1:1 + N]]
    gates = synth_basic(perm, n)
    if gates:
        gates = gates[:-1]        # drop last gate -> wrong permutation
    emit(gates, n)


if __name__ == "__main__":
    main()
