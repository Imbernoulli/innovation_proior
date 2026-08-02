# TIER: greedy
"""The obvious first idea: minimize data movement by offloading anything
"data-heavy" to near-memory, using a fixed absolute byte cutoff. It never
looks at flops or at the device rates given in the instance -- so it cannot
tell a genuinely data-heavy kernel (low flops, should go to NM) apart from a
kernel that merely has a lot of bytes but is also extremely compute-heavy
(should stay on Host once near-memory is slow). This is the "offload
everything data-heavy" trap called out in the brief."""
import sys


def main():
    data = sys.stdin.buffer.read().split()
    it = iter(data)

    def nxt():
        return int(next(it))

    n = nxt(); m = nxt(); g = nxt()
    _H_rate = nxt(); _N_rate = nxt(); _L_fetch = nxt(); _L_edge = nxt()
    group_of = [nxt() for _ in range(n)]
    mem_bytes = [0] * n
    for i in range(n):
        _flops = nxt()
        mem_bytes[i] = nxt()
    # edges / rates are read but intentionally unused by this heuristic

    grp_bytes = [0] * g
    grp_count = [0] * g
    for i in range(n):
        grp_bytes[group_of[i]] += mem_bytes[i]
        grp_count[group_of[i]] += 1

    CUTOFF = 150  # fixed, instance-agnostic byte threshold
    assign = []
    for k in range(g):
        avg_bytes = grp_bytes[k] / max(1, grp_count[k])
        assign.append(1 if avg_bytes >= CUTOFF else 0)

    print(" ".join(map(str, assign)))


if __name__ == "__main__":
    main()
