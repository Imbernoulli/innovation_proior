# TIER: invalid
# Emits an expression referencing variables that do not exist in the problem
# (only n and c are allowed) -> the checker rejects it -> score 0.
print("berth ** 2.0 + throughput * 3.0")
