# TIER: strong
# Insight: total execution time = (I + stalls + branch flushes) * cycle_time is a
# JOINT function of the stage partition, not just of stage COUNT -- "peak clock
# frequency" (S=N) also maximizes every hazard/branch stage-gap. Because N,K are
# small (N<=13, K<=6), decompose exactly: enumerate every contiguous partition
# (which cut points are used), and for each partition solve the forwarding
# choice as an exact 0/1 knapsack (K<=6 -> 2^K subsets) instead of a greedy
# value/cost purchase. Keep the global minimum over the whole joint space.
import sys


def main():
    data = sys.stdin.read().split()
    idx = [0]

    def nxt():
        v = data[idx[0]]
        idx[0] += 1
        return v

    N = int(nxt()); K = int(nxt()); L = int(nxt())
    Br = int(nxt()); Mb = int(nxt()); resolve_block = int(nxt())
    Budget = int(nxt()); I = int(nxt())
    c = [int(nxt()) for _ in range(N)]
    haz = []
    for _ in range(K):
        need_b = int(nxt()); res_b = int(nxt()); dist = int(nxt()); freq = int(nxt())
        haz.append((need_b, res_b, dist, freq))

    positions = list(range(1, N))  # candidate cut points
    best_time = None
    best_S, best_cuts, best_chosen = 1, [], []

    for mask in range(1 << (N - 1)):
        cuts = [positions[i] for i in range(N - 1) if (mask >> i) & 1]
        S = len(cuts) + 1
        boundaries = [0] + cuts + [N]
        stage_of = [0] * (N + 1)
        for s_idx in range(1, S + 1):
            lo, hi = boundaries[s_idx - 1] + 1, boundaries[s_idx]
            for b in range(lo, hi + 1):
                stage_of[b] = s_idx
        stage_delay = [0] * (S + 1)
        for b in range(1, N + 1):
            stage_delay[stage_of[b]] += c[b - 1]
        T = max(stage_delay[1:S + 1]) + L

        gaps = []
        for k in range(K):
            need_b, res_b, dist, freq = haz[k]
            gap = stage_of[res_b] - stage_of[need_b]
            stall_wo = max(0, gap - dist)
            gaps.append((gap, stall_wo, freq))

        branch_penalty = Mb * (stage_of[resolve_block] - 1)

        best_local_stall = None
        best_local_sub = 0
        for sub in range(1 << K):
            cost = 0
            stall = 0
            ok = True
            for k in range(K):
                gap, stall_wo, freq = gaps[k]
                if (sub >> k) & 1:
                    cost += gap
                    if cost > Budget:
                        ok = False
                        break
                else:
                    stall += freq * stall_wo
            if not ok:
                continue
            if best_local_stall is None or stall < best_local_stall:
                best_local_stall = stall
                best_local_sub = sub

        total_cycles = I + best_local_stall + branch_penalty
        total_time = total_cycles * T
        if best_time is None or total_time < best_time:
            best_time = total_time
            best_S, best_cuts = S, cuts
            best_chosen = [k + 1 for k in range(K) if (best_local_sub >> k) & 1]

    print(best_S)
    print(" ".join(map(str, best_cuts)))
    print(len(best_chosen))
    print(" ".join(map(str, best_chosen)))


if __name__ == "__main__":
    main()
