import sys, random

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
random.seed(seed)

# Small cases so the O(n!) brute force is feasible: n up to 7.
n = random.randint(0, 7)

# Mix value regimes to exercise the trap (ties, equal t, equal w, ratio collisions).
mode = random.randint(0, 3)
if mode == 0:
    T_HI, W_HI = 6, 6          # tiny values -> many ties / equal ratios
elif mode == 1:
    T_HI, W_HI = 20, 3         # heavy weights, varied times
elif mode == 2:
    T_HI, W_HI = 3, 20         # varied weights, similar times
else:
    T_HI, W_HI = 12, 12

lines = [str(n)]
for _ in range(n):
    t = random.randint(1, T_HI)
    w = random.randint(1, W_HI)
    lines.append(f"{t} {w}")

sys.stdout.write("\n".join(lines) + "\n")
