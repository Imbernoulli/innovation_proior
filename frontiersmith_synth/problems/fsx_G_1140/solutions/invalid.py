# TIER: invalid
"""
Infeasible on purpose: an automaton with zero accepting states can never
finish a trip, so no cost is ever defined. Must score 0.
"""
print("1 0")
print("0")
print("0")
