# TIER: greedy
# The obvious "maximize round-trip efficiency" recipe: cycle ONLY the small,
# efficient battery (90% round trip) and never route a single watt through the
# huge, lossy fuel store. From a myopic efficiency standpoint a converter that
# spends 80-90% of its life idle and loses 40% of whatever passes through it looks
# like pure waste, so a reasonable-looking first attempt just never bothers with it:
# surplus beyond the battery's small cap is curtailed, and any deficit beyond the
# battery's small reserve is simply unserved. This tracks day-to-day noise fine but
# is defenseless against a multi-tick drought.
import sys, json

inst = json.load(sys.stdin)
T = inst["T"]
load = inst["load"]
gen = inst["gen"]
bat = inst["battery"]
bcap = bat["cap"]
brate = bat["rate"]
bin_ = bat["eta_in"]

bc = [0.0] * T
bd = [0.0] * T
fc = [0.0] * T
fd = [0.0] * T
blev = 0.0
for t in range(T):
    net = gen[t] - load[t]
    if net >= 0.0:
        want = net
        if want > brate: want = brate
        room = (bcap - blev) / bin_ if bin_ > 0.0 else 0.0
        if want > room: want = room
        if want < 0.0: want = 0.0
        bc[t] = want
        blev += bin_ * want
    else:
        need = -net
        take = need
        if take > brate: take = brate
        if take > blev: take = blev
        bd[t] = take
        blev -= take

print(json.dumps({"bc": bc, "bd": bd, "fc": fc, "fd": fd}))
