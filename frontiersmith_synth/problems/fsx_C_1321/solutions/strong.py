# TIER: strong
"""Insight: what matters is not nominal site count but spacing sites near
the diffusion length lam = sqrt(D/v_max), trading off crowding penalty
against how much of the strip each site can draw reactant from without
starving its neighbors. Rather than commit to a single formula, this
searches a handful of evenly-spaced lattice placements at strides/offsets
scaled around the estimated diffusion length (plus a plain uniform B-way
partition of the strip as a fallback), scoring EACH candidate with an
internal copy of the exact reaction-diffusion-poisoning protocol from
statement.md, and keeps whichever real placement scores highest."""
import sys
import math

T = 12
R = 25
DT = 0.1


def simulate(L, active, D, v_max, gamma, r_screen, poison_rate, C0):
    poison = [0.0] * L
    c = [C0] * L
    total = 0.0
    for _t in range(T):
        mult = [1.0] * L
        for i in range(L):
            if not active[i]:
                continue
            cnt = 0
            for dd in range(1, r_screen + 1):
                if i - dd >= 0 and active[i - dd]:
                    cnt += 1
                if i + dd < L and active[i + dd]:
                    cnt += 1
            mult[i] = 1.0 / (1.0 + gamma * cnt)
        k = [0.0] * L
        for i in range(L):
            if active[i]:
                k[i] = v_max * mult[i] * (1.0 - poison[i])
        turnover = [0.0] * L
        for _m in range(R):
            new_c = [0.0] * L
            for i in range(L):
                left = c[i - 1] if i - 1 >= 0 else C0
                right = c[i + 1] if i + 1 < L else C0
                produced = k[i] * c[i] * DT
                turnover[i] += produced
                val = c[i] + DT * D * (left + right - 2.0 * c[i]) - produced
                if val < 0.0:
                    val = 0.0
                elif val > C0:
                    val = C0
                new_c[i] = val
            c = new_c
        for i in range(L):
            if active[i]:
                total += turnover[i]
                poison[i] = min(1.0, poison[i] + poison_rate * turnover[i])
    return total


def spaced_placement(L, B, stride, offset):
    if stride <= 0:
        stride = 1
    active = [0] * L
    seen = set()
    cur = offset
    guard = 0
    while len(seen) < B and guard < 6 * L:
        p = cur % L
        if p not in seen:
            seen.add(p)
        cur += stride
        guard += 1
    for p in seen:
        active[p] = 1
    # top up if the lattice couldn't produce B distinct cells (small L / large stride)
    if sum(active) < B:
        for j in range(L):
            if sum(active) >= B:
                break
            if active[j] == 0:
                active[j] = 1
    return active


def uniform_partition(L, B):
    active = [0] * L
    if B <= 1:
        active[L // 2] = 1
        return active
    for i in range(B):
        p = round(i * (L - 1) / (B - 1))
        active[p] = 1
    if sum(active) < B:
        for j in range(L):
            if sum(active) >= B:
                break
            if active[j] == 0:
                active[j] = 1
    return active


def main():
    data = sys.stdin.read().split()
    pos = 0

    def nxt():
        nonlocal pos
        v = data[pos]
        pos += 1
        return v

    L = int(nxt())
    B = int(nxt())
    D = float(nxt())
    v_max = float(nxt())
    gamma = float(nxt())
    r_screen = int(nxt())
    poison_rate = float(nxt())
    C0 = float(nxt())

    lam = math.sqrt(max(D, 1e-9) / max(v_max, 1e-9))

    best_active = None
    best_score = -1.0

    candidates = []
    for stride_mul in (0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0):
        stride = max(1, int(round(lam * stride_mul)))
        candidates.append((stride, 0))
        candidates.append((stride, stride // 2))
    candidates.append(("uniform", 0))

    for stride, offset in candidates:
        if stride == "uniform":
            active = uniform_partition(L, B)
        else:
            active = spaced_placement(L, B, stride, offset)
        if sum(active) != B:
            continue
        score = simulate(L, active, D, v_max, gamma, r_screen, poison_rate, C0)
        if score > best_score:
            best_score = score
            best_active = active

    if best_active is None:
        best_active = uniform_partition(L, B)

    sys.stdout.write(" ".join(str(x) for x in best_active) + "\n")


if __name__ == "__main__":
    main()
