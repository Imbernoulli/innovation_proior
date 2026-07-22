# TIER: strong
# INSIGHT (the genuine leverage, not "post a tighter spread"):
#  (1) The mid path m and the flow S_t,B_t are PUBLIC, so the coming move is
#      ANTICIPATED.  Reformulate the problem as INVENTORY CONTROL: pick a target
#      inventory q*_t that leans the way the upcoming move pays -- LONG ahead of an
#      up-move, SHORT ahead of a down-move -- rather than a spread to post.
#  (2) The fill model is INVERTIBLE: to trade a target quantity on one side you set the
#      half-spread to hit exactly that fill fraction (h = W*(1 - qty/flow)) and cap the
#      size, and you WITHDRAW the other side (size 0) so the adverse burst cannot push
#      you the wrong way.  This is the "skew": refuse the short the buy-burst wants to
#      hand you, and build the long the move will reward -- starting EARLY, while the
#      base sell-interest is still there, because the pre-move burst chokes exactly the
#      interest you need.
#  (3) After the move passes, the target relaxes to 0, so you UNWIND into the post-move
#      flow and bank the directional gain, paying the convex holding charge for as few
#      steps as the position must be carried.
#  When no move is anticipated (calm), q*_t ~ 0 and you fall back to symmetric spread
#  capture (h=W/2) -- there the myopic recipe is already right, so you tie it.
import sys, json

inst = json.load(sys.stdin)
T = inst["T"]
m = inst["m"]
S = inst["S"]
B = inst["B"]
W = float(inst["W"])
Qmax = float(inst["Qmax"])

GAMMA = 0.75      # lookahead discount on future mid increments
KPOS = 40.0       # inventory desired per unit of discounted future move
THRESH = 6.0      # below this desired change, just capture spread symmetrically


def future_signal(t):
    s = 0.0
    w = 1.0
    for k in range(t + 1, T + 1):
        s += w * (m[k] - m[k - 1])
        w *= GAMMA
    return s


hb = [W] * T
ha = [W] * T
zb = [0.0] * T
za = [0.0] * T

q = 0.0
for t in range(T):
    fut = future_signal(t)
    qstar = KPOS * fut
    if qstar > Qmax:
        qstar = Qmax
    elif qstar < -Qmax:
        qstar = -Qmax
    dq = qstar - q

    if dq > THRESH:
        # want to BUY dq: hit the bid, withdraw the ask
        want = dq
        avail = S[t]
        if avail <= 1e-9:
            buy = 0.0
        else:
            buy = want if want < avail else avail
        if buy > 0.0:
            frac = buy / avail            # in (0,1]
            h = W * (1.0 - frac)
            if h < 0.0:
                h = 0.0
            if h > 0.99 * W:
                h = 0.99 * W
            hb[t] = h
            zb[t] = buy
            q += buy                       # (position cap is never breached here)
        za[t] = 0.0
    elif dq < -THRESH:
        # want to SELL -dq: lift the ask, withdraw the bid
        want = -dq
        avail = B[t]
        if avail <= 1e-9:
            sell = 0.0
        else:
            sell = want if want < avail else avail
        if sell > 0.0:
            frac = sell / avail
            h = W * (1.0 - frac)
            if h < 0.0:
                h = 0.0
            if h > 0.99 * W:
                h = 0.99 * W
            ha[t] = h
            za[t] = sell
            q -= sell
        zb[t] = 0.0
    else:
        # neutral: symmetric spread capture (matches the myopic recipe here)
        h = W / 2.0
        hb[t] = h
        ha[t] = h
        # symmetric full-size fills keep inventory ~flat while banking spread
        big = Qmax * 4.0
        zb[t] = big
        za[t] = big
        # mirror the resulting inventory drift for the next step's target math
        fb = 1.0 - hb[t] / W
        fa = 1.0 - ha[t] / W
        buy = S[t] * fb
        sell = B[t] * fa
        nq = q + buy
        if nq > Qmax:
            buy = Qmax - q
        q += buy
        if q - sell < -Qmax:
            sell = q + Qmax
        q -= sell

print(json.dumps({"hb": hb, "ha": ha, "zb": zb, "za": za}))
