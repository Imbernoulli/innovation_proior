# TIER: strong
# Insight: since the grade is the MINIMUM hit rate over five traces, spending the pin
# budget on whatever is globally loudest (the greedy recipe) is the wrong objective --
# a maximin resource-allocation problem wants the budget spent where it moves the
# CURRENTLY worst trace. Two-stage water-filling:
#   1) Cheap insurance first: pin whichever keys show up (however lightly) across the
#      MOST distinct traces -- the true cross-trace "intersection working set" -- since
#      protecting those helps every personality simultaneously for one unit of budget.
#   2) Spend any remaining budget greedily: repeatedly re-simulate all five traces
#      under the pin set chosen so far, find the currently-worst trace, and add
#      whichever still-unpinned candidate key raises THAT trace's hit rate the most
#      (a local re-simulation, not a static count) -- directly targeting the binding
#      constraint instead of the loudest signal.
# Weights are a single fixed hedge (moderate LRU + moderate MRU to blunt the loop
# trace's cyclic worst case, a touch of LFU, and a scan-admission gate keyed off the
# windowed "fraction of first-time keys" signal) -- one static formula whose *inputs*
# (recency/frequency/scan-signal) shift with whichever trace is currently replaying,
# so it reacts differently to a steady zipf trace than to a scan sweep or a loop
# without ever branching on which trace it is.
import sys, json
from collections import Counter

inst = json.load(sys.stdin)
C = inst["capacity"]
P = inst["pin_budget"]
traces = inst["traces"]
names = list(traces.keys())

W_LRU, W_MRU, W_LFU, W_SCAN = 1.0, 1.5, 0.5, 3.0
SCAN_WINDOW = 24


def simulate(trace, pin, C):
    pinset = set(pin)
    managed_cap = max(0, C - len(pinset))
    resident = set()
    last_access = {}
    freq = {}
    ever_seen = set()
    window = []
    hits = 0
    n = len(trace)
    for t, k in enumerate(trace):
        first_time = k not in ever_seen
        ever_seen.add(k)
        scan_signal = (sum(window) / len(window)) if window else 0.0
        if k in pinset or k in resident:
            hits += 1
            freq[k] = freq.get(k, 0) + 1
            last_access[k] = t
        else:
            admit = True
            if W_SCAN > 0 and first_time and (scan_signal * W_SCAN >= 1.0):
                admit = False
            if admit:
                if len(resident) < managed_cap:
                    resident.add(k)
                else:
                    maxrec = 0
                    maxfreq = 0
                    recs = {}
                    freqs = {}
                    for j in resident:
                        recs[j] = t - last_access.get(j, t - 1)
                        freqs[j] = freq.get(j, 0)
                        if recs[j] > maxrec:
                            maxrec = recs[j]
                        if freqs[j] > maxfreq:
                            maxfreq = freqs[j]
                    maxrec = maxrec or 1
                    maxfreq = maxfreq or 1
                    best_key = None
                    best_score = None
                    for j in sorted(resident):
                        recnorm = recs[j] / maxrec
                        freqnorm = freqs[j] / maxfreq
                        score = W_LRU * recnorm + W_MRU * (1 - recnorm) - W_LFU * freqnorm
                        if best_score is None or score > best_score:
                            best_score = score
                            best_key = j
                    resident.discard(best_key)
                    resident.add(k)
            freq[k] = freq.get(k, 0) + 1
            last_access[k] = t
        window.append(1 if first_time else 0)
        if len(window) > SCAN_WINDOW:
            window.pop(0)
    return hits / n if n else 0.0


# Stage 1: rank candidate keys by cross-trace coverage (how many of the five traces
# they appear in at all), tie-broken by total frequency -- the cheap-insurance core.
coverage = Counter()
total_freq = Counter()
for nm in names:
    seen_in_trace = set(traces[nm])
    for k in seen_in_trace:
        coverage[k] += 1
    total_freq.update(traces[nm])

candidates = sorted(total_freq.keys(), key=lambda k: (-coverage[k], -total_freq[k]))
pool = candidates[:60]  # cap search width for speed

pin = []
for k in pool:
    if coverage[k] == len(names) and len(pin) < P:
        pin.append(k)

# Stage 2: maximin water-filling for any remaining budget.
while len(pin) < P:
    rates = {nm: simulate(traces[nm], pin, C) for nm in names}
    worst_name = min(rates, key=lambda n: rates[n])
    best_key = None
    best_rate = rates[worst_name]
    for k in pool:
        if k in pin:
            continue
        trial = pin + [k]
        r = simulate(traces[worst_name], trial, C)
        if r > best_rate:
            best_rate = r
            best_key = k
    if best_key is None:
        for k in pool:
            if k not in pin:
                best_key = k
                break
    if best_key is None:
        break
    pin.append(best_key)

print(json.dumps({"pin": pin, "w_lru": W_LRU, "w_mru": W_MRU, "w_lfu": W_LFU, "w_scan": W_SCAN}))
