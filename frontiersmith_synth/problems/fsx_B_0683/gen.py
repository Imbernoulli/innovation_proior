#!/usr/bin/env python3
# gen.py <testId>  -- prints ONE instance of "Nested Dyadic Quantization Under a Split Budget".
#
# Deterministic in testId only (no external randomness). Builds C = T + M channels, each an
# empty-at-start dyadic partition of [0,1) with max depth D:
#   - T "gateway" channels: four weighted point-clusters positioned so the *root* split (which
#     must be committed before anything else in that channel is reachable) yields a tiny SSE
#     gain, but the *second-level* splits it unlocks (splitting each depth-1 child) each yield a
#     MUCH larger gain. A myopic best-first grower never sees enough value in the root move to
#     take it before cheaper-looking channels absorb the whole budget.
#   - M "single-shot" channels: two well-separated clusters resolved completely by ONE split,
#     with per-split value between the gateway root gain and the gateway's true per-split value.
# Total split budget S is shared and kept scarce (S < M) so a naive best-first grower spends
# every unit on single-shot channels and never affords a single gateway root split, while the
# globally-optimal allocation sacrifices a few single-shot channels to fully unlock several
# gateways (whose 3-split average payoff beats the single-shot rate).
import sys


def emit(testId: int) -> None:
    t = testId
    D = 3                                   # max dyadic depth (root=0 .. leaves at depth<=3)

    T = 2 + t // 2                          # number of gateway (trap) channels
    leftover = 4 + t                        # extra single-shot splits the optimal plan affords
    S = 3 * T + leftover                    # shared split budget
    M = S + 5 + t                           # number of single-shot channels (M > S: always plenty)

    # Gateway channel template (scaled mildly with t to vary magnitudes across cases).
    a = 4000 + 150 * t                      # weight of the near-boundary sub-cluster
    b = 90 + 4 * t                          # weight of the far sub-cluster
    # positions: left pair straddles 0.5 from below, right pair straddles 0.5 from above,
    # each pair itself straddles the 0.25/0.75 quarter boundary so a SECOND split resolves it.
    posA, posB = 0.499, 0.001                # inside [0,0.5): A near center, B far (near 0)
    posC, posE = 0.501, 0.999                # inside [0.5,1): C near center, E far (near 1)

    # Single-shot channel template: two clusters straddling 0.5, resolved by ONE split.
    p = 27 + t                               # weight of each single-shot cluster
    posL, posR = 0.2, 0.8

    lines = [f"{T + M} {S} {D}"]

    for i in range(T):
        pts = [(posA, a), (posB, b), (posC, a), (posE, b)]
        lines.append(str(len(pts)))
        for v, w in pts:
            lines.append(f"{v:.6f} {w}")

    for j in range(M):
        pts = [(posL, p), (posR, p)]
        lines.append(str(len(pts)))
        for v, w in pts:
            lines.append(f"{v:.6f} {w}")

    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    tid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    tid = max(1, min(10, tid))
    emit(tid)
