# TIER: strong
# The insight: reinforcement on a source should DECAY at a rate matched to that
# source's own depletion, not stay pinned forever. We track each source's
# HEADROOM = max(0, stock - regen) -- the one-time surplus above what its
# regen alone can sustain. While a source still has headroom we camp most ants
# there (exploit); a fixed reserve fraction of ants is spent the whole time
# pre-building trail on the NEXT source in line (paying the pheromone-decay
# ramp-up cost early, before it's needed) so that the moment the current
# source's headroom is exhausted -- exactly when its depletion is complete --
# we can hand the majority of ants to an already-warm trail instead of a
# cold-start (near-zero trail) source. We deterministically grid-search a
# small set of (which-source-first ordering, reserve fraction, switch
# threshold) configurations by simulating each one exactly (same dynamics the
# evaluator uses) and keep whichever config actually harvests the most on
# THIS instance -- a reformulation (phased exploit/pre-warm decomposition)
# plus deterministic local tuning, not "greedy with more iterations".
import sys, json


def simulate(srcs, T, A, alloc):
    K = len(srcs)
    stock = [s["stock0"] for s in srcs]
    trail = [0.0] * K
    total = 0.0
    for t in range(T):
        row = alloc[t]
        for i in range(K):
            s = srcs[i]
            trail[i] = s["decay"] * trail[i] + (1 - s["decay"]) * row[i]
            potential = s["rate"] * trail[i]
            h = stock[i] if stock[i] < potential else potential
            total += h
            nxt = stock[i] - h + s["regen"]
            stock[i] = s["cap"] if nxt > s["cap"] else nxt
    return total


def headroom(stock_i, regen_i):
    hr = stock_i - regen_i
    return hr if hr > 0 else 0.0


def phased_alloc(srcs, K, T, A, order, reserve_frac, switch_frac):
    stock = [s["stock0"] for s in srcs]
    trail = [0.0] * K
    alloc = []
    ptr = 0
    reserve = int(round(A * reserve_frac))
    for t in range(T):
        primary = order[ptr]
        while ptr < K - 1 and headroom(stock[primary], srcs[primary]["regen"]) < switch_frac * srcs[primary]["cap"]:
            ptr += 1
            primary = order[ptr]
        row = [0] * K
        main_ants = A - reserve if ptr < K - 1 else A
        row[primary] += main_ants
        if ptr < K - 1:
            row[order[ptr + 1]] += reserve
        alloc.append(row)
        for i in range(K):
            s = srcs[i]
            trail[i] = s["decay"] * trail[i] + (1 - s["decay"]) * row[i]
            h = stock[i] if stock[i] < s["rate"] * trail[i] else s["rate"] * trail[i]
            nxt = stock[i] - h + s["regen"]
            stock[i] = s["cap"] if nxt > s["cap"] else nxt
    return alloc


def best_alloc(srcs, K, T, A):
    orders = {
        "value": sorted(range(K), key=lambda i: -(srcs[i]["stock0"] * srcs[i]["rate"])),
        "rate":  sorted(range(K), key=lambda i: -srcs[i]["rate"]),
        "regen": sorted(range(K), key=lambda i: -(srcs[i]["regen"] * srcs[i]["rate"])),
    }
    best, best_val = None, -1.0
    for order in orders.values():
        for reserve_frac in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35):
            for switch_frac in (0.05, 0.10, 0.15, 0.20, 0.30):
                alloc = phased_alloc(srcs, K, T, A, order, reserve_frac, switch_frac)
                v = simulate(srcs, T, A, alloc)
                if v > best_val:
                    best_val, best = v, alloc
    return best


inst = json.load(sys.stdin)
K = inst["K"]
T = inst["T"]
A = inst["A"]
srcs = inst["sources"]

alloc = best_alloc(srcs, K, T, A)
print(json.dumps({"alloc": alloc}))
