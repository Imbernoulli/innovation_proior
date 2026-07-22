# TIER: strong
# INSIGHT (not "use both stores more"): the two stores serve different probability
# STRATA. The battery is the fast, cheap, everyday buffer -- for ordinary noise it
# is already the right (and sufficient) answer, so this policy ties the "obvious"
# battery-only recipe whenever a window's total shortfall is within what the
# battery could plausibly cover. Only when a drought's total energy shortfall
# CLEARLY exceeds the battery's entire capacity does this policy treat the fuel
# store as what it actually is: insurance. It scans the WHOLE public trace for such
# genuine drought windows, and for each one, reallocates ordinary surplus AWAY from
# the (more efficient) battery and INTO the (lossier, but much bigger) fuel store
# during a long lead-in before the drought -- because the lossy store needs far more
# raw surplus, over far more ticks, to bank the same delivered energy. The battery
# is topped back off only in the last few ticks before the drought hits (it recovers
# fast, so it does not need the long lead). During the drought itself, both stores
# discharge together: fuel carries the steady bulk of the shortfall, battery smooths
# whatever spikes remain. Away from any drought this collapses to exactly the
# battery-first recipe -- the insight only pays when there is a real tail to hedge.
import sys, json

inst = json.load(sys.stdin)
T = inst["T"]
load = inst["load"]
gen = inst["gen"]
bat = inst["battery"]
fuel = inst["fuel"]
bcap = bat["cap"]; brate = bat["rate"]; bin_ = bat["eta_in"]; bout = bat["eta_out"]
fcap = fuel["cap"]; frate = fuel["rate"]; fin_ = fuel["eta_in"]; fout = fuel["eta_out"]

net = [gen[t] - load[t] for t in range(T)]
is_drought = [net[t] < -3.0 for t in range(T)]

windows = []
t = 0
while t < T:
    if is_drought[t]:
        s = t
        while t < T and is_drought[t]:
            t += 1
        windows.append((s, t))
    else:
        t += 1

LEAD = 45      # ticks of lead-in eligible for fuel-priority provisioning
TOPOFF = 3     # last few ticks before the drought: revert to topping off the battery
pre_drought = [False] * T
for (s, e) in windows:
    total_deficit = 0.0
    for k in range(s, e):
        d = -net[k]
        if d > 0.0:
            total_deficit += d
    # only worth diverting surplus to the lossy store if the small battery
    # clearly could not cover this window on its own
    if total_deficit > 1.3 * bcap * bout:
        for tt in range(max(0, s - LEAD), max(0, s - TOPOFF)):
            pre_drought[tt] = True

bc = [0.0] * T
bd = [0.0] * T
fc = [0.0] * T
fd = [0.0] * T
blev = 0.0
flev = 0.0
for t in range(T):
    n = net[t]
    if n >= 0.0:
        if pre_drought[t]:
            want_f = n
            if want_f > frate: want_f = frate
            room_f = (fcap - flev) / fin_ if fin_ > 0.0 else 0.0
            if want_f > room_f: want_f = room_f
            if want_f < 0.0: want_f = 0.0
            fc[t] = want_f
            flev += fin_ * want_f
            leftover = n - want_f
            want_b = leftover
            if want_b > brate: want_b = brate
            room_b = (bcap - blev) / bin_ if bin_ > 0.0 else 0.0
            if want_b > room_b: want_b = room_b
            if want_b < 0.0: want_b = 0.0
            bc[t] = want_b
            blev += bin_ * want_b
        else:
            want_b = n
            if want_b > brate: want_b = brate
            room_b = (bcap - blev) / bin_ if bin_ > 0.0 else 0.0
            if want_b > room_b: want_b = room_b
            if want_b < 0.0: want_b = 0.0
            bc[t] = want_b
            blev += bin_ * want_b
            leftover = n - want_b
            want_f = leftover
            if want_f > frate: want_f = frate
            room_f = (fcap - flev) / fin_ if fin_ > 0.0 else 0.0
            if want_f > room_f: want_f = room_f
            if want_f < 0.0: want_f = 0.0
            fc[t] = want_f
            flev += fin_ * want_f
    else:
        need = -n
        want_f = need
        if want_f > frate: want_f = frate
        if want_f > flev: want_f = flev
        fd[t] = want_f
        flev -= want_f
        need -= fout * want_f
        if need > 0.0:
            take_b = need
            if take_b > brate: take_b = brate
            if take_b > blev: take_b = blev
            bd[t] = take_b
            blev -= take_b

print(json.dumps({"bc": bc, "bd": bd, "fc": fc, "fd": fd}))
