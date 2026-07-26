import sys, math

# Deterministic difficulty/regime ladder, seeded ONLY by testId.
#   T      = number of dispatch timesteps (a "day" of the plant's operation)
#   mode   = demand-mix regime that controls whether cascade (turbine->absorption)
#            windows are present:
#     'dedicated'  : chill demand stays near zero -> LP steam has nowhere useful to
#                    go -> cascading is a bad trade (traps a solver that always cascades)
#     'cascade'    : power AND chill demand both track each other and grow with T
#                    -> cascading is a strong trade (traps a solver that never cascades)
#     'alt'        : alternates per-timestep between a dedicated-like and a
#                    cascade-like sub-pattern (regime literally changes within one day)
#     'adversarial': cycles tiny / spike / starved sub-patterns for stress coverage
LADDER = [
    (6,   'mixed'),
    (8,   'dedicated'),
    (10,  'cascade'),
    (16,  'cascade'),
    (20,  'cascade'),
    (24,  'dedicated'),
    (40,  'cascade'),
    (60,  'alt'),
    (90,  'adversarial'),
    (150, 'cascade'),
]

# --- fixed hospital-plant hardware (same equipment every day; only the demand
#     profile and derived capacity/pricing curvature differ per instance) ---
A_B = 1.10       # boiler: base fuel per HP-steam unit
EPS_P = 0.08     # turbine: electricity extracted per unit HP steam routed through it
EPS_S = 0.86     # turbine: LP (exhaust) steam per unit HP steam routed through it
COP_ABS = 0.85   # absorption chiller: chill units per LP-steam unit
COP_ELEC = 3.40  # electric chiller: chill units per electricity unit
A_G = 4.60       # grid import: base fuel-equivalent per electricity unit


def lcg_stream(seed):
    s = seed & 0x7fffffff
    while True:
        s = (1103515245 * s + 12345) & 0x7fffffff
        yield s / 0x7fffffff


def gen_demands(T, mode, rnd):
    S, Pw, Ch = [], [], []
    for t in range(T):
        ang = 2.0 * math.pi * t / max(T, 1)
        jS = int(8 * next(rnd))
        base_S = 35 + int(round(15 * math.sin(ang))) + jS
        S_t = max(6, base_S)

        sub = mode
        if mode == 'mixed':
            # 2:1 cascade-weighted alternation -- the regime literally changes
            # mid-day, but cascade windows still dominate the day's total.
            sub = 'dedicated' if (t % 3 == 0) else 'cascade'
        elif mode == 'alt':
            sub = 'cascade' if (t % 3 != 2) else 'dedicated'
        elif mode == 'adversarial':
            r = t % 3
            sub = 'dedicated' if r == 0 else ('cascade' if r == 1 else 'starved')

        if sub == 'dedicated':
            jP = int(10 * next(rnd))
            Pw_t = 25 + jP
            jC = int(3 * next(rnd))
            Ch_t = jC  # near-zero chill: LP steam would be wasted if cascaded
        elif sub == 'cascade':
            jP = int(10 * next(rnd))
            Pw_t = 30 + int(round(20 * math.sin(ang + 1.0))) + jP
            Pw_t = max(8, Pw_t)
            jC = int(8 * next(rnd))
            Ch_t = max(0, int(round(0.85 * Pw_t)) + jC)
        else:  # starved: tiny everything, degenerate edge case
            Pw_t = int(3 * next(rnd))
            Ch_t = int(3 * next(rnd))

        S.append(S_t)
        Pw.append(Pw_t)
        Ch.append(Ch_t)
    return S, Pw, Ch


def main():
    i = int(sys.argv[1])
    idx = min(max(i, 1), len(LADDER)) - 1
    T, mode = LADDER[idx]
    rnd = lcg_stream(90000 + 131 * i)

    S, Pw, Ch = gen_demands(T, mode, rnd)

    max_S = max(S)
    max_Pw = max(Pw)
    max_Ch = max(Ch)
    Cap_b = float(3.0 * max_S + 10)
    c_b = A_B / (12.0 * Cap_b)
    peakE = max_Pw + max_Ch / COP_ELEC
    c_g = A_G / (1.0 * max(peakE, 1.0))

    out = []
    out.append("%d %.6f %.8f %.6f %.6f %.6f %.6f %.6f %.6f %.8f" %
                (T, A_B, c_b, Cap_b, EPS_P, EPS_S, COP_ABS, COP_ELEC, A_G, c_g))
    for t in range(T):
        out.append("%d %d %d" % (S[t], Pw[t], Ch[t]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
