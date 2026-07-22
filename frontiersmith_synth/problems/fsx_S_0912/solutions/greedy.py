# TIER: greedy
# The obvious, minimal-length genome: CKPT(0) to zero the write head, then walk
# left to right with (SET(target[i]), MOVE(+1)) pairs. This exactly reconstructs
# the target when unmutated and is the shortest natural encoding -- but every
# MOVE is load-bearing: a single corrupted MOVE permanently shifts the head, so
# everything written after that point lands on the wrong cell. No redundancy,
# no re-anchoring -> a textbook "just draw it" genome that the noisy tape
# punishes hard.
import sys, json

inst = json.load(sys.stdin)
A = inst["A"]
target = inst["target"]
L_max = inst["L_max"]


def encode(op, arg):
    return ((op & 7) << 5) | (arg & 31)


tape = [encode(4, 0)]                 # CKPT(0)
n = len(target)
for i, t in enumerate(target):
    tape.append(encode(2, t % A))     # SET(t)
    if i != n - 1:
        tape.append(encode(1, 5))     # MOVE(+1)  ((5 % 9) - 4 == 1)

tape = tape[:L_max]
print(json.dumps({"tape": tape}))
