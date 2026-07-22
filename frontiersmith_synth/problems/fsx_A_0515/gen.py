import sys, random

# ---------------------------------------------------------------------------
# Boutique clearance sale over T=30 days.
#   Seller sets an integer price p_t for each day.
#   Each buyer i has: value v, arrival a, deadline h, discount permille D (d=D/1000).
#   Forward-looking best response: among days a<=t<=h with per-day capacity left and
#   global stock left, buy 1 unit on the day maximising d^t*(v-p_t) if that is positive.
#   Buyers processed in the given (shuffled) order; total stock K, per-day cap s.
#
# The 10 cases plant a 3-cluster "screening" structure so a NON-MONOTONE hump
# (early low -> mid spike -> late low) strictly dominates every monotone markdown.
# ---------------------------------------------------------------------------

T = 30
PMAX = 120

def main():
    i = int(sys.argv[1])
    rng = random.Random(90515 + 1000 * i)

    # difficulty ladder: population size + cluster balance grow with i
    scale = 1 + (i - 1) // 2          # 1..5
    n_early = 4 + 2 * scale + (i % 2)     # impatient, low value, must buy early
    n_mid   = 4 + 2 * scale               # mid-window, HIGH value  (spike target)
    n_late  = 4 + 3 * scale               # very patient, low value (floor crowd)
    n_noise = 2 + (i % 3)                 # a few random buyers (open-endedness)

    buyers = []  # (v,a,h,D)

    # ---- early-impatient cluster: value LOW, window early, heavy discount ----
    for _ in range(n_early):
        v = rng.randint(48, 58)
        a = 1
        h = rng.randint(2, 4)
        D = rng.randint(560, 660)
        buyers.append((v, a, h, D))

    # ---- mid-window HIGH-value cluster: only in market mid-run (a spike target) ----
    mid_a_lo, mid_a_hi = 7, 9
    mid_h_lo, mid_h_hi = 14, 17
    for _ in range(n_mid):
        v = rng.randint(95, min(PMAX, 112))
        a = rng.randint(mid_a_lo, mid_a_hi)
        h = rng.randint(mid_h_lo, mid_h_hi)
        D = rng.randint(860, 930)
        buyers.append((v, a, h, D))

    # ---- late-patient bargain crowd: value LOW, long horizon, near-patient ----
    for _ in range(n_late):
        v = rng.randint(30, 40)
        a = rng.randint(1, 4)
        h = 30
        D = rng.randint(975, 998)
        buyers.append((v, a, h, D))

    # ---- noise buyers: arbitrary params (defeat pure pattern-match) ----
    for _ in range(n_noise):
        v = rng.randint(25, PMAX)
        a = rng.randint(1, 20)
        h = rng.randint(a, 30)
        D = rng.randint(600, 999)
        buyers.append((v, a, h, D))

    rng.shuffle(buyers)   # processing order (matters under scarcity)
    N = len(buyers)

    # inventory scarcity: total stock K < N in most cases (seller must choose whom
    # to serve), per-day capacity s binds the cheap "sale" days -> overflow.
    if i <= 2:
        K = N                       # slack cases (ladder sanity)
        s = N
    else:
        K = int(round(N * (0.72 - 0.02 * (i - 3))))   # 0.72 -> 0.58
        K = max(n_mid + 3, min(N, K))
        s = max(3, (n_early + 1) // 1 - 1)             # cheap day can't absorb all
        s = min(s, N)

    p0 = 60   # baseline single reference price (checker's constant-price baseline)

    lines = ["%d %d %d %d %d %d" % (T, N, K, s, PMAX, p0)]
    for (v, a, h, D) in buyers:
        lines.append("%d %d %d %d" % (v, a, h, D))
    sys.stdout.write("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
