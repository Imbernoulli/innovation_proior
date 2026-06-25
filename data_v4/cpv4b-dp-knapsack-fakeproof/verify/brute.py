#!/usr/bin/env python3
# Independent brute force for "Resonant Knapsack".
#
# Problem (restated):
#   We have n crystal shards. Shard i has integer mass w_i (>=1) and an
#   integer "phase" p_i in {0,...,M-1}.  We pick a (possibly empty) subset S.
#   The subset is RESONANT at target mass W and target phase q iff
#       sum_{i in S} w_i == W   AND   ( sum_{i in S} p_i ) mod M == q.
#   Count the number of resonant subsets, modulo 1_000_000_007.
#
# Brute force: enumerate all 2^n subsets, accumulate.
import sys

MOD = 1_000_000_007

def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    W = int(data[idx]); idx += 1
    M = int(data[idx]); idx += 1
    q = int(data[idx]); idx += 1
    w = []
    p = []
    for _ in range(n):
        w.append(int(data[idx])); idx += 1
        p.append(int(data[idx])); idx += 1

    count = 0
    for mask in range(1 << n):
        sw = 0
        sp = 0
        for i in range(n):
            if mask & (1 << i):
                sw += w[i]
                sp += p[i]
        if sw == W and (sp % M) == q:
            count += 1
    print(count % MOD)

if __name__ == "__main__":
    main()
