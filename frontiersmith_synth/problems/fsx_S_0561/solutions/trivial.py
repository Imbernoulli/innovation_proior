# TIER: trivial
# Resubmit the target circuit verbatim: it trivially computes the right function,
# but with the full obfuscated gate count -> reproduces the checker baseline (~0.1).
import sys
sys.stdout.write(sys.stdin.read())
