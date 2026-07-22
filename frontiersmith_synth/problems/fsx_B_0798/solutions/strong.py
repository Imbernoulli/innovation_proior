# TIER: strong
"""
Strong: reformulate the joint (amplifier placement x per-span power) choice
as a DAG. A node in the DAG is "the last active amplifier is at index i,
having already banked base_noise (the value of the accumulated-noise
expression evaluated right at i, distance 0 into the next span)". An edge
i -> i' with chosen power P covers every downstream node in (i, i'] using
that span's own local SNR curve -- which is UNIMODAL in P (linear signal
gain vs cubic Kerr penalty), so trying every allowed power per edge lets
each span sit near its own interior optimum instead of reusing one global
power for the whole line.

Because base_noise only ever increases along any path, and only the PAIR
(reached-so-far, base_noise) matters for everything downstream, we keep a
Pareto frontier per DAG node (maximize reached, minimize base_noise) and
prune dominated states. This is exact for the given action space and runs
in O(N^2 * K) time -- far cheaper than enumerating placements.
"""
import sys


def main():
    data = sys.stdin.read().split()
    ptr = 0
    N = int(data[ptr]); ptr += 1
    xs = [int(v) for v in data[ptr:ptr + N]]; ptr += N
    alpha = float(data[ptr]); ptr += 1
    c0 = float(data[ptr]); ptr += 1
    c_ase = float(data[ptr]); ptr += 1
    c_kerr = float(data[ptr]); ptr += 1
    thresh = float(data[ptr]); ptr += 1
    K = int(data[ptr]); ptr += 1
    allowed = [int(v) for v in data[ptr:ptr + K]]; ptr += K

    NEG_NOISE = c0 * 1
    states = [[] for _ in range(N)]
    parent = [dict() for _ in range(N)]
    states[0] = [(0, NEG_NOISE)]

    def insert(lst, key_i, reached, noise, from_key, prev_i, P):
        for (r2, n2) in lst:
            if r2 >= reached and n2 <= noise and (r2, n2) != (reached, noise):
                return
        lst[:] = [(r2, n2) for (r2, n2) in lst if not (reached >= r2 and noise <= n2)]
        lst.append((reached, noise))
        parent[key_i][(reached, noise)] = (from_key[0], from_key[1], prev_i, P)

    best_overall = 0
    best_key = (0, NEG_NOISE)
    best_i = 0
    for i in range(N):
        if not states[i]:
            continue
        for (reached_i, noise_i) in states[i]:
            if reached_i > best_overall:
                best_overall, best_key, best_i = reached_i, (reached_i, noise_i), i
            if i == N - 1:
                continue
            for ip in range(i + 1, N):
                ell = xs[ip] - xs[i]
                for P in allowed:
                    gain = 0
                    for j in range(i + 1, ip + 1):
                        d = xs[j] - xs[i]
                        noise = noise_i + c_ase * d + c_kerr * d * (P ** 3)
                        atten = 10.0 ** (-alpha * d / 10.0)
                        sig = P * atten
                        snr = sig / max(noise, 1e-9)
                        if snr >= thresh:
                            gain += 1
                    new_reached = reached_i + gain
                    new_noise = noise_i + c0 + c_ase * ell + c_kerr * ell * (P ** 3)
                    insert(states[ip], ip, new_reached, new_noise, (reached_i, noise_i), i, P)

    # reconstruct the best path
    path_idx = []
    path_pow = []
    cur_i, cur_key = best_i, best_key
    while True:
        entry = parent[cur_i].get(cur_key)
        if entry is None:
            break
        pr, pn, prev_i, P = entry
        path_idx.append(prev_i)
        path_pow.append(P)
        cur_i, cur_key = prev_i, (pr, pn)
    path_idx.reverse()
    path_pow.reverse()
    if not path_idx:
        # degenerate: only node 0 ever active
        path_idx = [0]
        path_pow = [allowed[0]]

    out = []
    out.append(str(len(path_idx)))
    out.append(" ".join(str(v) for v in path_idx))
    out.append(" ".join(str(v) for v in path_pow))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
