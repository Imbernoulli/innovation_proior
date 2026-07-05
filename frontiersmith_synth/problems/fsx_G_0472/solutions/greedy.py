# TIER: greedy
# Conservative plain GDA: shrink the step size a lot (theta=0, alpha=0) so the
# rotation-driven divergence is tamed by damping alone.  A small constant step
# converges slowly but reliably on these strongly-convex-strongly-concave
# saddles, beating the default baseline -- but it never uses optimism or
# momentum, so it stays far from the achievable gap.
import sys, json

json.load(sys.stdin)
print(json.dumps({"eta_x": 0.008, "eta_y": 0.008, "theta": 0.0, "alpha": 0.0}))
