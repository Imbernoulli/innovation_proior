# TIER: strong
# THE INSIGHT: proportioning under unknown rescaling is impossible from absolute
# concentration, but it is trivial from RELATIVE POSITION.  A cell's RANK fraction
# -- how many cells carry less morphogen than it, divided by the tissue length --
# is invariant to ANY monotone rescaling of the gradient: amplitude, offset, AND
# slope/shape exponent all cancel, because they are all monotone maps.  So we read
# rank, not concentration, and cut at the pattern's own proportions 0.25 / 0.75.
# A one-cell neighbour average (the GRN reading its neighbours) denoises local rank
# swaps.  This survives every rescaling; only the low-frequency developmental bump
# and readout noise keep it below a perfect score -- leaving headroom for a policy
# that tunes the smoothing window / cut points against the residual bias.
import sys, json

json.load(sys.stdin)  # the network is scale-free; it needs nothing from the field
print(json.dumps({"feature": "rank", "smooth": 1, "cuts": [0.25, 0.75]}))
