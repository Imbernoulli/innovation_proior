import sys, random

# fsx_C_0650 -- crowd-energy-hysteresis-setlist
# Instance layout:
#   line 1: N T K
#   line 2: alpha_milli decay_milli gamma_milli   (all /1000)
#   next N lines: e_milli d s_1 .. s_K            (song energy(/1000), duration, style bits)
#
# Three planted song classes (order in the file is SHUFFLED, class identity is not labeled):
#   filler  -- cheap, low energy, style bits drawn independently at random (diverse)
#   bait    -- moderate/high energy, ALL share the SAME style vector style_A -> stacking
#              them lets similarity-fatigue-memory crash the crowd state if mishandled
#   finale  -- the highest-energy songs, style vector style_B has ZERO bit-overlap with
#              style_A -> playing a finale song right after a bait run gets the full
#              (1-E) rebound with no fatigue penalty (the "buy back slope" trick)


def main():
    tid = int(sys.argv[1])
    rng = random.Random(20260 + 7 * tid)

    K = 5
    style_A = [1, 1, 1, 0, 0]
    style_B = [0, 0, 0, 1, 1]

    # difficulty ladder: counts grow with tid (scale: large by tid=10)
    filler_n = 3 + tid // 2          # 3 .. 8
    bait_n = 3 + tid                 # 4 .. 13
    finale_n = 1 + (tid % 3 == 0)    # mostly 1, occasionally 2

    songs = []  # (e_milli, d, style)

    for _ in range(filler_n):
        e = rng.randint(30, 120)
        d = rng.randint(1, 2)
        style = [rng.randint(0, 1) for _ in range(K)]
        songs.append((e, d, style))

    for _ in range(bait_n):
        e = rng.randint(280, 600)
        d = rng.randint(2, 5)
        songs.append((e, d, list(style_A)))

    for _ in range(finale_n):
        e = rng.randint(650, 830)
        d = rng.randint(2, 4)
        songs.append((e, d, list(style_B)))

    rng.shuffle(songs)
    N = len(songs)

    total_dur = sum(d for _, d, _ in songs)
    # budget: tight enough that not everything fits, generous enough finale + a real
    # bait/filler mix can be assembled. Ratio drifts with tid to vary the trade-off.
    frac = 0.50 + 0.02 * (tid % 5)   # 0.50 .. 0.58
    T = max(6, int(total_dur * frac))

    alpha_milli = 350 + 30 * (tid % 7)     # 350..530 (/1000)
    decay_milli = 500 + 20 * (tid % 6)     # 500..600 (fast-forgetting memory -> recent run dominates)
    gamma_milli = 850 + 20 * (tid % 8)     # 850..990 (strong fatigue bite)

    out = [f"{N} {T} {K}", f"{alpha_milli} {decay_milli} {gamma_milli}"]
    for e, d, style in songs:
        out.append(f"{e} {d} " + " ".join(map(str, style)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
