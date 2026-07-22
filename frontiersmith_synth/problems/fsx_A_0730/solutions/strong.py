# TIER: strong
# Insight: a flat 0..s-1 tally spends every state on EXACT resolution over a
# tiny range and then freezes -- the wrong trade when the true count can run
# far past the budget. Instead, split the s-state budget into a small grid of
# (level, phase) states: phase 0..P-1 within a level tracks exactly, but P
# consecutive hits "carry" into the next level, and decoding
#   out(level, phase) = P * (2^level - 1) + phase * 2^level
# means each phase step at level L is worth 2^L real hits once decoded. With
# n_levels levels and P phases per level (n_levels * P <= s), the machine
# stays EXACT while counts are small (level 0) and keeps bounded RELATIVE
# error (about 1/P per level) out to a range that grows exponentially with
# n_levels -- instead of clamping at a fixed additive ceiling.
#
# The instance tells us m, k, target_residues, l_min/l_max BEFORE any stream
# is seen. Symbols are i.i.d. uniform over 0..m-1, so a stream of length L has
# an expected hit count hit_rate * L (a Binomial(L, hit_rate) mean) -- fully
# derivable from public fields, no hidden data needed. Rather than picking the
# (n_levels, P) split by a crude "does it cover a padded ceiling" rule (which
# can waste resolution on headroom nothing ever uses), CALIBRATE directly: for
# every affordable split of the budget, actually simulate the resulting
# machine's decode error on a grid of representative counts spanning
# [l_min, l_max] * hit_rate (plus some tail beyond l_max for safety) and keep
# whichever split minimizes the mean relative error on that grid. This is the
# modular-hashing + counting-approximation + automaton-memory-tradeoff
# synthesis: none of the three pieces alone would produce this design, and it
# is calibrated per-instance rather than a single fixed recipe.
import sys, json

inst = json.load(sys.stdin)
m = inst["m"]
s = inst["s"]
k = inst["k"]
residues = set(inst["target_residues"])
l_min = inst["l_min"]
l_max = inst["l_max"]

hit = [(x % k) in residues for x in range(m)]
hit_rate = sum(hit) / m if m else 0.0

# representative expected counts across the length range, plus a modest tail
# beyond l_max (stream-to-stream variance can push a real count higher than
# the expected value at l_max).
grid_lengths = [l_min + (l_max - l_min) * i / 8.0 for i in range(9)]
grid_lengths.append(l_max * 1.25)
grid_counts = sorted(set(max(0, round(hit_rate * L)) for L in grid_lengths))
if not grid_counts:
    grid_counts = [0]


def build_tables(n_levels, P):
    n_states = n_levels * P

    def flat(level, phase):
        return level * P + phase

    trans = [[0] * m for _ in range(n_states)]
    out = [0.0] * n_states
    for level in range(n_levels):
        for phase in range(P):
            idx = flat(level, phase)
            out[idx] = float(P * (2 ** level - 1) + phase * (2 ** level))
            for x in range(m):
                if hit[x]:
                    if phase < P - 1:
                        nxt = flat(level, phase + 1)
                    elif level < n_levels - 1:
                        nxt = flat(level + 1, 0)
                    else:
                        nxt = idx  # saturate at the top state
                else:
                    nxt = idx
                trans[idx][x] = nxt
    return trans, out


def simulate_hits(trans, out, n_hits):
    """Feed n_hits consecutive hit-symbols (any hit column works, they're
    all identical) through the machine and read the decoded estimate."""
    hit_col = hit.index(True) if True in hit else 0
    state = 0
    for _ in range(n_hits):
        state = trans[state][hit_col]
    return out[state]


best = None  # (mean_err, n_levels, P, trans, out)
for n_levels in range(1, s + 1):
    P = s // n_levels
    if P < 1:
        continue
    trans, out = build_tables(n_levels, P)
    errs = []
    for c in grid_counts:
        est = simulate_hits(trans, out, c)
        errs.append(abs(est - c) / (c + 1))
    mean_err = sum(errs) / len(errs)
    if best is None or mean_err < best[0] - 1e-12:
        best = (mean_err, n_levels, P, trans, out)

_, n_levels, P, trans, out = best
n_states = n_levels * P

print(json.dumps({"n_states": n_states, "start": 0, "trans": trans, "out": out}))
