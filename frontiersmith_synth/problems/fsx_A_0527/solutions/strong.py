# TIER: strong
# Insight: gate admission on the SHADOW PRICE of a mooring credit, not on the fee
# you happen to see.  Three moves, composed:
#   (1) opportunity-cost reservation -- read the FULL class schedule (public) and
#       reserve enough credits for the scarce deep-sea (class 2) demand; while the
#       berth still holds more than that reserve line, admit decent traffic, but
#       once remaining drops to the reserve line, slam the bar shut so the last
#       credits are held for whales.  (bar keyed on rem_bucket = shadow price.)
#   (2) regime-drift probe -- the running-fee signal (sig_bucket) tells you when
#       the tariff has drifted UP; on a detected drift, DROP the bar to grab the
#       now-lucrative vessels instead of hoarding.
#   (3) deep-sea is never turned away -- its own bar stays low across all state.
# This reserves capacity BEFORE the scarce high-fee class arrives, which a flat
# early-fitted bar structurally cannot do.
import sys, json

inst = json.load(sys.stdin)
K = inst["K"]; R = inst["R_buckets"]; S = inst["S_buckets"]
B = inst["B"]; g = float(inst["prior_g"])
classes = inst["classes"]; sizes = inst["sizes"]

# --- reserve line: credits to hold for the deep-sea class we can see coming ---
deep_credits = sum(w for c, w in zip(classes, sizes) if c == 2)
reserve = deep_credits
if reserve > B - 1:
    reserve = B - 1
reserve_frac = reserve / B if B else 0.0
reserve_bucket = int(R * reserve_frac)      # rem_bucket at/below which we clamp

bars = [[[0.0 for _ in range(S)] for _ in range(R)] for _ in range(K)]
for c in range(K):
    for rb in range(R):
        for sb in range(S):
            if c == 2:
                bar = 0.4 * g                    # always welcome the whales
            elif sb >= 2:
                bar = 0.6 * g                    # drift detected -> grab high fees
            else:
                if rb > reserve_bucket:
                    bar = 0.95 * g               # above reserve line: admit decent
                else:
                    bar = 3.2 * g                # at/below reserve line: hold credits
            bars[c][rb][sb] = bar

print(json.dumps({"bars": bars}))
