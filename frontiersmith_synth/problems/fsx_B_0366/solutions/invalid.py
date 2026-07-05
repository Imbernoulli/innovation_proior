# TIER: invalid
# Emits an expression referencing variables that do not exist in the problem
# (only k and V are allowed) -> the checker rejects it -> score 0.
print("crew ** 1.2 * volume ** 0.6 + overhead")
