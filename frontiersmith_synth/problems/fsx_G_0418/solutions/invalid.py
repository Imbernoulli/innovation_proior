# TIER: invalid
# Emits an expression referencing variables that do not exist in the problem
# (only T and C are allowed) -> the checker rejects it -> score 0.
print("pressure ** 2.0 + flowrate * 3.0")
