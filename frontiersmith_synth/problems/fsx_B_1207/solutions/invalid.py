# TIER: invalid
# Emits an expression referencing an unknown name -> the checker's strict
# AST whitelist rejects it and prints Ratio: 0.0.
import sys

sys.stdin.read()
print("t + resistance_gene_pool")
