# TIER: strong
import sys
from collections import defaultdict, deque


def settle_group(group):
    """group: list of (party_id, value) with sum(value) == 0. Returns len(group)-1
    transfers that zero the whole group out via a local prefix-carry chain."""
    transfers = []
    carry = 0
    for t in range(len(group) - 1):
        carry += group[t][1]
        if carry != 0:
            a = group[t][0]; b = group[t + 1][0]
            # transfer (u, v, amt): u pays v amt -> contributes -amt to u, +amt to v.
            if carry > 0:
                transfers.append((b, a, carry))
            else:
                transfers.append((a, b, -carry))
    return transfers


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); m = int(next(it))
    balance = [0] * (n + 1)
    for _ in range(m):
        d = int(next(it)); c = int(next(it)); a = int(next(it))
        balance[d] -= a
        balance[c] += a

    # ---- the insight: minimum transfers == n - (max # of disjoint zero-sum
    # subsets). Every already-zero party is a free size-1 group. Beyond that,
    # hunt for zero-sum SUBSETS of the balance values directly (not amount
    # magnitude, not party adjacency): first exact pairs (hashmap complement
    # lookup), then exact triples (pair-sum hashmap), then settle whatever is
    # left with a single fallback chain. Each recovered k-subset saves k-1
    # transfers versus treating its members individually.

    nonzero = [i for i in range(1, n + 1) if balance[i] != 0]

    # --- pass 1: exact-pair zero-sum subsets via value hashmap ---
    value_to_ids = defaultdict(deque)
    for pid in nonzero:
        value_to_ids[balance[pid]].append(pid)

    transfers = []
    matched = set()
    for val in list(value_to_ids.keys()):
        if val <= 0:
            continue
        dq_pos = value_to_ids[val]
        dq_neg = value_to_ids.get(-val, deque())
        while dq_pos and dq_neg:
            p_pos = dq_pos.popleft()
            p_neg = dq_neg.popleft()
            transfers.append((p_neg, p_pos, val))
            matched.add(p_pos)
            matched.add(p_neg)

    remaining = [pid for pid in nonzero if pid not in matched]

    # --- pass 2: exact-triple zero-sum subsets via pair-sum hashmap ---
    L = len(remaining)
    pairsum = defaultdict(deque)
    for a_i in range(L):
        pa, va = remaining[a_i], balance[remaining[a_i]]
        for b_i in range(a_i + 1, L):
            pb, vb = remaining[b_i], balance[remaining[b_i]]
            pairsum[va + vb].append((pa, pb))

    used = set()
    for pid in remaining:
        if pid in used:
            continue
        target = -balance[pid]
        dq = pairsum.get(target)
        if not dq:
            continue
        while dq:
            p1, p2 = dq.popleft()
            if p1 in used or p2 in used or p1 == pid or p2 == pid:
                continue
            used.add(pid); used.add(p1); used.add(p2)
            group = [(pid, balance[pid]), (p1, balance[p1]), (p2, balance[p2])]
            transfers.extend(settle_group(group))
            break

    # --- fallback: whatever is left, settle with a single chain ---
    leftover = [(pid, balance[pid]) for pid in remaining if pid not in used]
    transfers.extend(settle_group(leftover))

    out = [str(len(transfers))]
    for u, v, a in transfers:
        out.append(f"{u} {v} {a}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
