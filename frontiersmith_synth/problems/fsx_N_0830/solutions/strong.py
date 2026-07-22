# TIER: strong
# The insight: don't trust the raw interview score's sign. Infer, from the
# EARLY portion of the stream's own score distribution, which of the two
# disclosed value regimes governs this instance (Rising instances cluster
# scores high; Fading instances cluster them low), convert every score into
# an ESTIMATED true value under that inferred regime (folding in the
# disclosed drift term, so later arrivals are correctly recognized as
# richer), then maintain a threshold that always targets exactly the K
# richest ESTIMATED candidates still ahead -- reserving slots for the
# high-drift tail instead of spending them on the low-drift head. Early
# passes made while still calibrating are not wasted: they stay in the
# recall pool and get grabbed back the moment they clear the (now
# regime-corrected) bar, as long as the window hasn't expired, and any
# still-open slots near the end of the horizon sweep the best remaining
# held candidates before their windows lapse.
import sys, json

inst = json.load(sys.stdin)
N, K, W, decay, drift = inst["N"], inst["K"], inst["W"], inst["decay"], inst["drift"]
scores = inst["score"]


def drift_factor(i):
    return 1.0 + drift * (i / max(1, N - 1))


# --- calibration prefix: infer the active regime from its mean score ---
m = max(15, round(0.15 * N))
warm = scores[:m]
mean_o = sum(warm) / len(warm) if warm else 0.5
if mean_o >= 0.5:
    A, B = 1.2, 0.15    # inferred regime: Rising
else:
    A, B = -1.2, 1.35   # inferred regime: Fading

est = [max(0.0, (A * scores[i] + B) * drift_factor(i)) for i in range(N)]

actions = [0] * N
recalls = []
held = []   # list of (j, est_j) still within their recall window
hired = 0

for i in range(N):
    held = [(j, ev) for (j, ev) in held if i - j <= W]
    slots_left = K - hired
    if slots_left <= 0:
        break

    future_sorted = sorted(est[i:], reverse=True)
    take = min(len(future_sorted), slots_left)
    thresh = future_sorted[take - 1] if take > 0 else float("inf")

    if est[i] >= thresh:
        actions[i] = 1
        hired += 1
    else:
        held.append((i, est[i]))
        if held and hired < K:
            held.sort(key=lambda x: -x[1])
            bj, bv = held[0]
            if bv >= thresh and (i - bj) <= W:
                recalls.append([i, bj])
                held.pop(0)
                hired += 1

if hired < K:
    held = [(j, ev) for (j, ev) in held if (N - 1) - j <= W]
    held.sort(key=lambda x: -x[1])
    for (j, ev) in held:
        if hired >= K:
            break
        recalls.append([N - 1, j])
        hired += 1

print(json.dumps({"actions": actions, "recalls": recalls}))
