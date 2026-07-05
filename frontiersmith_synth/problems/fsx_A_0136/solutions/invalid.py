# TIER: invalid
# Emits an out-of-alphabet entry (value 2) -> feasibility fails -> scores 0.
import sys

N = int(sys.stdin.read().split()[0])
rows = []
for i in range(N):
    row = [1] * N
    rows.append(row)
rows[1][1] = 2  # not in {-1,+1}
out = "\n".join(" ".join(str(v) for v in row) for row in rows)
sys.stdout.write(out + "\n")
