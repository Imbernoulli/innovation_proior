# TIER: invalid
import sys
toks=sys.stdin.read().split()
n=int(toks[0])
# emit n obviously out-of-range unit counts (also wrong total) -> checker rejects
sys.stdout.write(" ".join(["1000000000"]*n)+"\n")
