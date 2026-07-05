import sys

# Mountain-rescue relay placement: emit an instance = (m relays, d=3 dimensions).
# The rescue region is the unit cube [0,1]^3 (normalized latitude, longitude, altitude).
# Difficulty ladder is deterministic in testId ONLY; dimension fixed at 3.
# Medium scale: relay count grows from small to moderate.
LADDER = [10, 12, 14, 16, 18, 20, 22, 24, 27, 30]

i = int(sys.argv[1])
idx = min(max(i, 1), len(LADDER)) - 1
m = LADDER[idx]
d = 3
print("%d %d" % (m, d))
