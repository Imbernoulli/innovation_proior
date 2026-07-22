# TIER: strong
# The insight: trade code minimality for a spaced, self-correcting genome.
#   1. DEVELOPMENTAL COMPRESSION -- run-length-encode the target: grow each
#      maximal same-type run with one CKPT(start)+SET(type) "seed" and then
#      DIV the daughter cell forward across the rest of the run (1 byte per
#      extra cell instead of 2), which frees up length budget.
#   2. ERROR-CORRECTING REDUNDANCY -- every cell is addressed ABSOLUTELY via
#      CKPT (never accumulated MOVEs), so one corrupted instruction can only
#      ever mis-vote the ONE cell it targets -- no cascade. With the freed
#      budget, spend extra CKPT(pos)+SET(type) "votes" round-robin across
#      cells (up to 3 independent votes/cell) so the vote-tally rule majority-
#      corrects a single-bit typo in any one repeat.
#   3. Redundancy is spent round-robin (not front-loaded) so a tight length
#      budget still raises the WORST cell's vote count instead of maxing a
#      few cells while leaving others at a single, fragile vote.
import sys, json

inst = json.load(sys.stdin)
A = inst["A"]
target = inst["target"]
L_max = inst["L_max"]
P = len(target)


def encode(op, arg):
    return ((op & 7) << 5) | (arg & 31)


# --- 1. run-length decomposition ---
runs = []
i = 0
while i < P:
    j = i
    while j + 1 < P and target[j + 1] == target[i]:
        j += 1
    runs.append((i, j, target[i]))
    i = j + 1

tape = []
for (a, b, t) in runs:
    tape.append(encode(4, a))          # CKPT(a)
    tape.append(encode(2, t % A))      # SET(t)          -- vote #1 for cell a
    for pos in range(a + 1, b + 1):
        tape.append(encode(3, 0))      # DIV(+1)         -- vote #1 for cell pos

# --- 2/3. round-robin redundancy with leftover budget ---
reps = [1] * P
order = list(range(P))
progressed = True
while len(tape) + 2 <= L_max and progressed:
    progressed = False
    for p in order:
        if reps[p] >= 3:
            continue
        if len(tape) + 2 > L_max:
            break
        tape.append(encode(4, p))       # CKPT(p)
        tape.append(encode(2, target[p] % A))  # SET(target[p]) -- extra vote
        reps[p] += 1
        progressed = True

tape = tape[:L_max]
if not tape:
    tape = [encode(0, 0)]
print(json.dumps({"tape": tape}))
