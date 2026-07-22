# TIER: greedy
# The obvious recipe: at EVERY step post the spread that maximizes myopic per-step
# capture, symmetric, at full size.  For the fill model f(h)=1-h/W the expected spread
# capture per unit of flow is h*(1-h/W), maximized at h = W/2 -- so quote hb=ha=W/2 on
# both sides with size large enough to take all the flow.  This ignores the (public)
# price path entirely: on an adverse instance the pre-move buy burst lifts the ask and
# loads a big SHORT right before the price rises (or a big LONG right before it drops),
# and the maker bleeds the move while pocketing only a sliver of spread.
import sys, json

inst = json.load(sys.stdin)
T = inst["T"]
W = inst["W"]
Qmax = inst["Qmax"]

h = W / 2.0
big = Qmax * 4.0  # size never binds; the flow and the fill fraction cap the fills
print(json.dumps({
    "hb": [h] * T, "ha": [h] * T,
    "zb": [big] * T, "za": [big] * T,
}))
