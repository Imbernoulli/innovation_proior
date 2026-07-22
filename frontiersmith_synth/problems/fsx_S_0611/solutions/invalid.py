# TIER: invalid
# Emits an all-blank font: every glyph has ink 0, violating both the ink budget
# and 4-connectivity. The checker must reject it with Ratio 0.
import sys

def main():
    d = sys.stdin.read().split()
    N, H, W = int(d[0]), int(d[1]), int(d[2])
    out = ["0"] * (N * H * W)
    sys.stdout.write(" ".join(out) + "\n")

main()
