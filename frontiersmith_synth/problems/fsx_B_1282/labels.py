"""
Shared, deterministic instance generator for fsx_B_1282 (market-abuse-surveil).

Both gen.py (prints the PUBLIC event stream to stdout) and verify.py (needs the
HIDDEN ground-truth manipulator set to score) call generate_instance(testId) and
get bit-for-bit identical results, because everything is seeded from testId alone.
gen.py never prints the role/label information -- only the raw event stream plus
N, W, K. verify.py re-derives the hidden manipulator windows by replaying the same
seeded generation, exactly like format-E checkers regenerate a held-out split.
"""
import random

# (N, W, frac_mm, frac_manip, hard) per testId 1..10 -- ladder small -> large,
# with >=3 explicit MM-flood "trap" cases (frac_mm large relative to frac_manip).
_TABLE = {
    1:  (10, 3, 0.30, 0.25, False),
    2:  (12, 3, 0.45, 0.20, False),   # trap: MM flood
    3:  (12, 4, 0.28, 0.28, False),
    4:  (14, 4, 0.50, 0.20, False),   # trap: MM flood
    5:  (14, 4, 0.32, 0.26, False),
    6:  (16, 4, 0.45, 0.20, False),   # trap: MM flood
    7:  (16, 5, 0.28, 0.28, False),
    8:  (18, 5, 0.42, 0.20, True),    # trap: MM flood + harder
    9:  (18, 5, 0.30, 0.26, True),    # hardest: weak signal + MM lookalikes
    10: (20, 5, 0.35, 0.24, True),    # largest: weak signal + MM lookalikes
}

BUY, SELL = "B", "S"


def generate_instance(test_id: int):
    N, W, frac_mm, frac_manip, hard = _TABLE[int(test_id)]
    rng = random.Random(20260000 + int(test_id) * 7919)

    n_mm = max(1, round(N * frac_mm))
    n_manip = max(1, round(N * frac_manip))
    if n_mm + n_manip >= N:
        n_mm = max(1, N - n_manip - 1) if N - n_manip - 1 >= 1 else n_mm
    n_noise = max(0, N - n_mm - n_manip)
    roles = ["mm"] * n_mm + ["manip"] * n_manip + ["noise"] * n_noise
    while len(roles) < N:
        roles.append("noise")
    roles = roles[:N]
    rng.shuffle(roles)

    manip_windows = {}   # pid -> set(window) where this pid actually manipulates
    M = set()
    for pid, role in enumerate(roles):
        if role == "manip":
            k = rng.choice([1, 2, 2, 3])
            k = min(k, W)
            wins = set(rng.sample(range(W), k))
            manip_windows[pid] = wins
            for w in wins:
                M.add((w, pid))

    events = []  # (w, pid, t, side, action, size)

    def emit(w, pid, t, side, action, size):
        events.append((w, pid, t, side, action, size))

    for pid, role in enumerate(roles):
        for w in range(W):
            if role == "mm":
                # Two-sided quote maintenance, done as several EPISODES: each
                # episode is a tight, rapid-fire run of cancels concentrated on
                # ONE side (re-skewing a quote in response to a tick) -- locally
                # it looks exactly like a layering burst (same short-window
                # rate). But episodes alternate sides across the window, so
                # tallied over the WHOLE cell the cancel flow is close to
                # side-BALANCED. A detector that only looks at local burst rate
                # cannot tell this apart from a manipulator; a detector that
                # tallies the side split over the full window can.
                n_ep = rng.randint(4, 6)
                t = 0
                cb_tot = cs_tot = 0
                for e_i in range(n_ep):
                    # alternate sides with mild noise so it isn't perfectly 50/50
                    if e_i % 2 == 0:
                        side = BUY if rng.random() < 0.85 else SELL
                    else:
                        side = SELL if rng.random() < 0.85 else BUY
                    burst_len = rng.randint(3, 6)
                    emit(w, pid, t, side, "P", rng.randint(1, 40)); t += 1
                    for _ in range(burst_len):
                        emit(w, pid, t, side, "C", rng.randint(1, 40)); t += 1
                        if side == BUY:
                            cb_tot += 1
                        else:
                            cs_tot += 1
                    t += rng.randint(1, 2)  # gap before next episode
                # rare, timing-uncorrelated inventory trade (not right after a
                # same-side cancel cluster) -- an occasional false-lookalike
                if rng.random() < (0.22 if hard else 0.08):
                    side = rng.choice([BUY, SELL])
                    emit(w, pid, t, side, "T", rng.randint(1, 40)); t += 1
            elif role == "manip" and w in manip_windows.get(pid, ()):
                # Layering burst: rapid CANCELS concentrated on one side (a tight,
                # near-unbroken run, deliberately sized to overlap a market
                # maker's total volume so raw cancel COUNT alone cannot separate
                # them), a little off-side noise, then a short gap and a
                # same-side AGGRESSIVE trade.
                side = rng.choice([BUY, SELL])
                off_side = SELL if side == BUY else BUY
                n_burst = rng.randint(8, 18) if hard else rng.randint(11, 27)
                n_off = rng.randint(0, 3)
                t = 0
                # a couple of placements to seed the burst, then near-unbroken cancels
                emit(w, pid, t, side, "P", rng.randint(1, 40)); t += 1
                for _ in range(n_burst):
                    emit(w, pid, t, side, "C", rng.randint(1, 40)); t += 1
                    if rng.random() < 0.15:
                        emit(w, pid, t, side, "P", rng.randint(1, 40)); t += 1
                for _ in range(n_off):
                    emit(w, pid, t, off_side, "C", rng.randint(1, 40)); t += 1
                gap = rng.randint(1, 3)
                t += gap
                n_trade = rng.randint(1, 3)
                for _ in range(n_trade):
                    emit(w, pid, t, side, "T", rng.randint(1, 40)); t += 1
            else:
                # light noise activity: a manip pid outside its manipulation
                # window(s), or a genuine low-activity noise participant.
                n_ev = rng.randint(1, 5)
                t = 0
                for _ in range(n_ev):
                    side = rng.choice([BUY, SELL])
                    action = rng.choice(["P", "C", "P", "C", "T"])
                    emit(w, pid, t, side, action, rng.randint(1, 40)); t += 1

    events.sort(key=lambda e: (e[0], e[1], e[2]))

    K = max(4, round(1.6 * len(M)))
    K = min(K, N * W)  # never exceed the universe of participant-windows

    return {
        "N": N, "W": W, "K": K,
        "events": events,
        "M": M,
        "roles": roles,
    }
