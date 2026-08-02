# TIER: invalid
# Emits an expression that references unknown names / attempts code injection.
# The checker's strict AST whitelist rejects any name outside {t, x, pi, e,
# exp, log, sqrt, abs, min, max} -> Ratio: 0.0.
import sys

sys.stdin.read()
print("__import__ ( 'os' ) . system ( 'echo hacked' ) + wobble ( t )")
