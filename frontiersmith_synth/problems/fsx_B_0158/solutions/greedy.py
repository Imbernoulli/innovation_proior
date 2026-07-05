# TIER: greedy
# Marginal-frequency ordering: put the clause that appears in the MOST directives
# first, then the next-most-frequent, and so on (ties broken by higher compute
# weight, then lower index). Hoisting the widely-shared clauses to the front makes
# many directives share long leading prefixes, so this is far better than the
# catalog order -- but it is blind to co-occurrence structure (two clauses that are
# each frequent yet rarely appear together still get placed adjacently), so it
# leaves prefix-sharing on the table.
import sys, json

inst = json.load(sys.stdin)
C = inst["n_clauses"]
weights = inst["weights"]
directives = inst["directives"]

freq = [0] * C
for d in directives:
    for c in d:
        freq[c] += 1

order = sorted(range(C), key=lambda c: (-freq[c], -weights[c], c))
print(json.dumps({"order": order}))
