# TIER: strong
import sys
from fractions import Fraction as Fr


def simulate(order, songs, K, alpha, decay, gamma):
    E = Fr(0)
    M = [Fr(0)] * K
    peak = Fr(0)
    for idx in order:
        e_milli, d, style = songs[idx]
        e = Fr(e_milli, 1000)
        sim = sum(Fr(style[j]) * M[j] for j in range(K)) / K
        fatigue = gamma * sim * E
        E = E + e * (1 - E) - fatigue
        if E < 0:
            E = Fr(0)
        if E > 1:
            E = Fr(1)
        if E > peak:
            peak = E
        for j in range(K):
            M[j] = decay * M[j] + (1 - decay) * Fr(style[j])
    final = E
    return alpha * peak + (1 - alpha) * final


def main():
    data = sys.stdin.read().split()
    p = 0
    def nxt():
        nonlocal p
        v = data[p]; p += 1
        return v
    N = int(nxt()); T = int(nxt()); K = int(nxt())
    alpha = Fr(int(nxt()), 1000)
    decay = Fr(int(nxt()), 1000)
    gamma = Fr(int(nxt()), 1000)
    songs = []
    for _ in range(N):
        e = int(nxt()); d = int(nxt())
        style = [int(nxt()) for _ in range(K)]
        songs.append((e, d, style))

    # INSIGHT: E += e*(1-E) is a saturating update -- without fatigue, final E only
    # depends on the *set* played (1 - prod(1-e_i)), never the order, and E is monotone
    # so peak == final. Order (and selection) only matter because similarity-fatigue can
    # make the state DIP. A dip right before a closer is not a cost, it's a purchase: it
    # restores (1-E) headroom so the closer's marginal gain e*(1-E) is large again. So:
    #   1. find the "closer group" -- the style class holding the single highest-energy
    #      song -- and reserve its best few songs to play dead LAST (peak and final both
    #      land on the loudest, freshest note).
    #   2. everything else is an "opening" candidate. Accept an opening song only if,
    #      simulated against the reserved closers, it does not *hurt* the exact final
    #      score (an exchange/marginal-value test) -- this is what naturally excludes
    #      long same-style "bait" runs that only build fatigue for no payoff, and instead
    #      favors cheap, diverse, low-energy filler that manufactures a low state at
    #      negligible fatigue cost.
    #   3. a bounded, deterministic relocation pass polishes the exact order.
    if N == 0:
        print(0); print("")
        return

    best_e = max(s[0] for s in songs)
    closer_style = None
    for i, s in enumerate(songs):
        if s[0] == best_e:
            closer_style = tuple(s[2])
            break
    closer_pool = [i for i in range(N) if tuple(songs[i][2]) == closer_style]

    chosen_closers = []
    tot_closer = 0
    for i in sorted(closer_pool, key=lambda i: -songs[i][0]):
        d = songs[i][1]
        if tot_closer + d <= T:
            chosen_closers.append(i)
            tot_closer += d
    chosen_closers.sort(key=lambda i: songs[i][0])  # ascending -> loudest plays absolute last

    remaining = T - tot_closer
    others = [i for i in range(N) if i not in chosen_closers]
    others.sort(key=lambda i: songs[i][0])  # cheap/diverse candidates first

    opening = []
    tot_open = 0
    for i in others:
        d = songs[i][1]
        if tot_open + d > remaining:
            continue
        base_score = simulate(opening + chosen_closers, songs, K, alpha, decay, gamma)
        trial_score = simulate(opening + [i] + chosen_closers, songs, K, alpha, decay, gamma)
        if trial_score >= base_score:
            opening.append(i)
            tot_open += d

    order = opening + chosen_closers

    # bounded deterministic local-search polish: relocate one song at a time
    passes = 0
    improved = True
    while improved and passes < 3 and len(order) <= 40:
        improved = False
        passes += 1
        cur = simulate(order, songs, K, alpha, decay, gamma)
        for pos in range(len(order)):
            for newpos in range(len(order)):
                if pos == newpos:
                    continue
                trial = order[:]
                x = trial.pop(pos)
                trial.insert(newpos, x)
                s = simulate(trial, songs, K, alpha, decay, gamma)
                if s > cur:
                    order = trial
                    cur = s
                    improved = True

    print(len(order))
    print(" ".join(map(str, order)))


if __name__ == "__main__":
    main()
