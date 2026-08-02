# TIER: greedy
# The obvious recipe: read each source's per-ant yield "rate" from the public
# instance, weight foraging effort proportional to rate, and commit to that
# split for the WHOLE horizon -- i.e. reinforce the best-looking trail(s) once
# and never revisit the decision. This ignores stock/regen/decay entirely, so
# it never notices when a heavily-favored source's one-time stock has run dry
# and settled into its slow regen-bound trickle: it keeps camping ants there
# instead of shifting them to a source with untapped stock. Beats the trivial
# equal split (it does use the one visible quality signal), but on a "patchy"
# layout with one deceptively high-rate, low-stock/low-regen source, this
# static weighting keeps feeding the drained source forever and starves.
import sys, json

inst = json.load(sys.stdin)
K = inst["K"]
T = inst["T"]
A = inst["A"]
srcs = inst["sources"]

w = [s["rate"] for s in srcs]
wsum = sum(w)
row_f = [A * wi / wsum for wi in w]
row = [int(x) for x in row_f]                 # floor
rem = A - sum(row)
order = sorted(range(K), key=lambda i: -(row_f[i] - row[i]))
for i in range(rem):
    row[order[i % K]] += 1

print(json.dumps({"alloc": [row[:] for _ in range(T)]}))
