# TIER: strong
"""The insight: partition by ARITHMETIC INTENSITY (flops per byte), not by raw
byte volume. A group is worth keeping on Host only if its intensity clears a
threshold derived from the instance's own device rates and fetch cost:

    tau = L_fetch * N_rate * H_rate / (H_rate - N_rate)      [H_rate > N_rate]

(tau is the intensity at which the extra compute time paid by running on the
slower near-memory device exactly equals the host's home-memory fetch cost;
above tau, Host wins.) This makes the same "big bytes" kernel route to Host
when near-memory is weak (trap cases: tau is small, so only truly low-
intensity kernels are worth sending away) and to NM when near-memory is
competitive. A short bounded local search then repairs the residual
interaction effects (cross-device edge costs, device-queue serialization,
and the coarse group granularity) that the closed-form per-group rule
ignores in isolation."""
import sys


def ceil_div(a, b):
    return -(-a // b)


def main():
    data = sys.stdin.buffer.read().split()
    it = iter(data)

    def nxt():
        return int(next(it))

    n = nxt(); m = nxt(); g = nxt()
    H_rate = nxt(); N_rate = nxt(); L_fetch = nxt(); L_edge = nxt()
    group_of = [nxt() for _ in range(n)]
    flops = [0] * n
    mem_bytes = [0] * n
    for i in range(n):
        flops[i] = nxt()
        mem_bytes[i] = nxt()
    preds = [[] for _ in range(n)]
    for _ in range(m):
        u = nxt(); v = nxt(); b = nxt()
        preds[v].append((u, b))

    grp_flops = [0] * g
    grp_bytes = [0] * g
    for i in range(n):
        grp_flops[group_of[i]] += flops[i]
        grp_bytes[group_of[i]] += mem_bytes[i]

    if H_rate > N_rate:
        tau = L_fetch * N_rate * H_rate / float(H_rate - N_rate)
    else:
        tau = 0.0  # near-memory at least as fast: never worth the fetch cost

    assign = []
    for k in range(g):
        b = grp_bytes[k]
        f = grp_flops[k]
        intensity = (f / b) if b > 0 else float('inf')
        assign.append(0 if intensity > tau else 1)

    def simulate(device):
        finish = [0] * n
        dev_free = [0, 0]
        makespan = 0
        for v in range(n):
            ready = 0
            for (u, bb) in preds[v]:
                t = finish[u]
                if device[u] != device[v]:
                    t += bb * L_edge
                if t > ready:
                    ready = t
            dv = device[v]
            rate = H_rate if dv == 0 else N_rate
            proc = ceil_div(flops[v], rate)
            if dv == 0:
                proc += mem_bytes[v] * L_fetch
            start = ready if ready > dev_free[dv] else dev_free[dv]
            fin = start + proc
            finish[v] = fin
            dev_free[dv] = fin
            if fin > makespan:
                makespan = fin
        return makespan

    def expand(a):
        return [a[group_of[i]] for i in range(n)]

    best = simulate(expand(assign))
    passes = 0
    improved = True
    while improved and passes < 4:
        improved = False
        passes += 1
        for k in range(g):
            assign[k] ^= 1
            cur = simulate(expand(assign))
            if cur < best:
                best = cur
                improved = True
            else:
                assign[k] ^= 1

    print(" ".join(map(str, assign)))


if __name__ == "__main__":
    main()
