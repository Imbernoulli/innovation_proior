import sys, random

# Generator for "quota-chain-schoolmatch": apprentice-to-workshop seat quotas.
# testId 1..10 = difficulty ladder (N, M grow). Deterministic in testId only.
#
# PLANTED STRUCTURE (why demand-proportional quotas trap the solver):
#  * 2 "magnet" workshops (ids 0,1) sit on EVERY apprentice's list as their #1
#    choice. Each magnet's priority is dominated by a FIXED, SMALL, hidden elite
#    (a global subset unrelated to district) -- so first-choice demand for a
#    magnet vastly overstates how much quota it can usefully absorb: capacity
#    beyond the elite's size is handed out by raw aptitude, i.e. it upgrades
#    apprentices who already had a solid fallback, while the SAME seats spent on
#    a district workshop can rescue apprentices with NO fallback at all.
#  * R regional (district) workshops (ids 2..R+1) have POWER-LAW skewed
#    populations (one "mega" district, several small ones) and are chained in a
#    RING: a rejected resident re-proposes to the next district in the ring.
#    Within a district, "ambitious" apprentices only fall back to it after both
#    magnets reject them; "loyal" apprentices list it directly with ring
#    fallback; "desperate" apprentices list ONLY their home district -- no
#    fallback at all -- and sit at the BOTTOM of that district's own priority
#    queue (below every loyal/ambitious resident), so they are rescued only once
#    a district's quota clears its OWN loyal population -- a real per-district
#    threshold that a flat, count-proportional split routinely misses.
#  * D further "catalogue" workshops (ids R+2..M-1) attract NO applicants this
#    cycle at all -- real seats that exist on paper but that raw demand counts
#    cannot tell apart from a genuinely under-served district.
# A capacity cut at one district does not just cost that district's own
# marginal resident: it changes ring slack two hops away and changes which
# districts ever clear their desperate-rescue threshold -- effects invisible to
# any count of first choices.

LADDER = [
    (40, 6), (60, 8), (90, 8), (130, 10), (180, 10),
    (240, 12), (300, 14), (380, 16), (480, 18), (600, 20),
]

W = [100, 45, 8, 3, 1]  # rank-1..rank-5 guild-point weights (steep drop-off)


def main():
    tid = int(sys.argv[1])
    idx = min(max(tid, 1), len(LADDER)) - 1
    N, M_real = LADDER[idx]
    rng = random.Random(31337 + 101 * tid)

    R = M_real - 2                      # regional (district) workshops
    D = int(round(3.4 * M_real))        # uninteresting catalogue workshops
    M = M_real + D
    T = max(M_real, int(round(0.60 * N)))

    # ---- power-law skewed district populations (one mega + several minor) ----
    skew_power = 1.6
    weights = [1.0 / ((r + 1) ** skew_power) for r in range(R)]
    tot_w = sum(weights)
    raw_sizes = [N * wv / tot_w for wv in weights]
    group_sizes = [max(1, int(round(x))) for x in raw_sizes]
    diff = N - sum(group_sizes)
    i = 0
    while diff != 0:
        j = i % R
        if diff > 0:
            group_sizes[j] += 1; diff -= 1
        elif group_sizes[j] > 1:
            group_sizes[j] -= 1; diff += 1
        i += 1

    home = [0] * N
    p = 0
    for r in range(R):
        for _ in range(group_sizes[r]):
            home[p] = r
            p += 1

    # ---- capacity upper bounds (physical room limits) ----
    fair = -(-T // M)  # ceil(T/M)
    elite_frac = 0.08
    elite_size = max(2, int(round(elite_frac * N)))
    cap_max = [0] * M
    cap_max[0] = elite_size
    cap_max[1] = elite_size
    for r in range(R):
        cap_max[2 + r] = max(fair + 1, group_sizes[r])
    for d in range(D):
        cap_max[2 + R + d] = fair + 1

    # ---- FIXED global elite pools (independent of district) drive magnet priority ----
    ids = list(range(N))
    rng.shuffle(ids)
    eliteA = set(ids[:elite_size])
    eliteB = set(ids[elite_size:2 * elite_size])

    # ---- apprentice classes: ambitious / normal-loyal / desperate ----
    pA, pL = 0.55, 0.15
    apt = [0] * N; cre = [0] * N; sen = [0] * N; cls = [0] * N
    for i in range(N):
        apt[i] = rng.randint(0, 299) + (1000 if i in eliteA else 0)
        cre[i] = rng.randint(0, 299) + (1000 if i in eliteB else 0)
    for i in range(N):
        u = rng.random()
        if u < pA:
            cls[i] = 0        # ambitious
        elif u < pA + pL:
            cls[i] = 1        # normal-loyal
        else:
            cls[i] = 2        # desperate
    for i in range(N):
        sen[i] = rng.randint(0, 199) if cls[i] == 2 else rng.randint(300, 999)

    # ---- preference lists (variable length) ----
    prefs = [None] * N
    for i in range(N):
        r = home[i]
        homew = 2 + r
        nextw = 2 + ((r + 1) % R)
        if cls[i] == 0:
            prefs[i] = [0, 1, homew, nextw]
        elif cls[i] == 1:
            prefs[i] = [homew, nextw, 0, 1]
        else:
            prefs[i] = [homew]  # desperate: home or nothing

    # ---- priority orders (full permutations, highest priority first) ----
    LOCAL_BONUS = 10 ** 7
    order = [None] * M
    order[0] = sorted(range(N), key=lambda i: (-apt[i], i))
    order[1] = sorted(range(N), key=lambda i: (-cre[i], i))
    for r in range(R):
        w = 2 + r
        order[w] = sorted(
            range(N),
            key=lambda i: (-(sen[i] + (LOCAL_BONUS if home[i] == r else 0)), i),
        )
    for d in range(D):
        w = 2 + R + d
        order[w] = sorted(range(N), key=lambda i: (rng.random(), i))

    out = ["%d %d %d" % (N, M, T), " ".join(map(str, cap_max))]
    for i in range(N):
        L = len(prefs[i])
        out.append(str(L) + " " + " ".join(map(str, prefs[i])))
    for w in range(M):
        out.append(" ".join(map(str, order[w])))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
