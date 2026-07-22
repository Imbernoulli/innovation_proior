# TIER: strong
"""
The insight: the public state numbering is an ARBITRARY label. Re-encoding the
states is free (it costs no gates -- it is just which b-bit code we call each
state), but it changes the Algebraic Normal Form of the transition function
completely. Some relabelings make the dynamics (near-)affine over GF(2), in
which case the ANF collapses to a handful of XORs. So: search the space of
state encodings (bijections state -> b-bit code) for one that minimizes total
ANF nonlinearity (a stochastic local search, restarts + hill-climbing, seeded
purely from the instance so it stays deterministic), THEN compile the winning
encoding's ANF to a circuit -- same compiler greedy.py uses, but now fed a
representation where the circuit is (almost) trivial.
"""
import sys, random, hashlib


def anf_transform(truth, n):
    a = truth[:]
    size = 1 << n
    for i in range(n):
        bit = 1 << i
        for x in range(size):
            if x & bit:
                a[x] ^= a[x ^ bit]
    return a


def emit_bit_from_anf(anf, n, emit):
    monomials = [mask for mask in range(1, 1 << n) if anf[mask]]
    term_wires = []
    for mask in monomials:
        bits = [i for i in range(n) if (mask >> i) & 1]
        if len(bits) == 1:
            term_wires.append(bits[0])
        else:
            cur = bits[0]
            for t in bits[1:]:
                cur = emit("AND %d %d" % (cur, t))
            term_wires.append(cur)
    if not term_wires:
        w0 = 0
        nw = emit("NOT %d" % w0)
        if anf[0] == 0:
            return emit("AND %d %d" % (w0, nw))
        else:
            return emit("OR %d %d" % (w0, nw))
    cur = term_wires[0]
    for t in term_wires[1:]:
        cur = emit("XOR %d %d" % (cur, t))
    if anf[0] == 1:
        cur = emit("NOT %d" % cur)
    return cur


def bit_cost(anf, n):
    monomials = [mask for mask in range(1, 1 << n) if anf[mask]]
    cost = 0
    for mask in monomials:
        pc = bin(mask).count("1")
        cost += max(0, pc - 1)
    if not monomials:
        cost += 2
    else:
        cost += max(0, len(monomials) - 1)
        if anf[0] == 1:
            cost += 1
    return cost


def total_cost(perm, S, K, b, m, delta, n):
    Y = [0] * (1 << n)
    for s in range(S):
        cs = perm[s]
        for k in range(K):
            Y[cs | (k << b)] = perm[delta[s][k]]
    tot = 0
    for j in range(b):
        truth = [(Y[x] >> j) & 1 for x in range(1 << n)]
        anf = anf_transform(truth, n)
        tot += bit_cost(anf, n)
    return tot


def search_encoding(S, K, b, m, delta, n, rng):
    identity = list(range(S))
    best_perm = identity[:]
    best_cost = total_cost(identity, S, K, b, m, delta, n)

    restarts = 6
    iters_per_restart = max(200, 1400 // max(1, S // 4))
    for r in range(restarts):
        if r == 0:
            cur = identity[:]
        else:
            cur = list(range(S))
            rng.shuffle(cur)
        cur_cost = total_cost(cur, S, K, b, m, delta, n)
        if cur_cost < best_cost:
            best_cost, best_perm = cur_cost, cur[:]
        for _ in range(iters_per_restart):
            i, j = rng.sample(range(S), 2)
            cur[i], cur[j] = cur[j], cur[i]
            c = total_cost(cur, S, K, b, m, delta, n)
            if c <= cur_cost or rng.random() < 0.05:
                cur_cost = c
                if c < best_cost:
                    best_cost, best_perm = c, cur[:]
            else:
                cur[i], cur[j] = cur[j], cur[i]
    return best_perm


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    S = int(next(it)); K = int(next(it))
    delta = [[0] * K for _ in range(S)]
    for s in range(S):
        for k in range(K):
            delta[s][k] = int(next(it))

    b = (S - 1).bit_length()
    m = (K - 1).bit_length()
    n = b + m

    h = hashlib.sha256(repr((S, K, delta)).encode()).digest()
    seed = int.from_bytes(h[:8], "big")
    rng = random.Random(seed)

    code_of = search_encoding(S, K, b, m, delta, n, rng)

    Y = [0] * (1 << n)
    for s in range(S):
        cs = code_of[s]
        for k in range(K):
            Y[cs | (k << b)] = code_of[delta[s][k]]

    gates = []
    wire_count = [n]

    def emit(line):
        gates.append(line)
        w = wire_count[0]
        wire_count[0] += 1
        return w

    outwires = []
    for j in range(b):
        truth = [(Y[x] >> j) & 1 for x in range(1 << n)]
        anf = anf_transform(truth, n)
        outwires.append(emit_bit_from_anf(anf, n, emit))

    out = []
    out.append("%d %d" % (b, m))
    out.append(str(S))
    for s in range(S):
        out.append("%d %d" % (s, code_of[s]))
    out.append(str(len(gates)))
    out.extend(gates)
    out.append(" ".join(str(w) for w in outwires))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
