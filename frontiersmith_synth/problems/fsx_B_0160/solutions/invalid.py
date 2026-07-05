# TIER: invalid
# Quote an absurdly long service time at every node (S_i = 500). The net replenishment
# time tau_i = SI_i + T_i - S_i goes negative (and the station cap s_max is blown), so
# every node is infeasible under the guaranteed-service structure -> score 0 on all
# instances.
import sys, json

inst = json.load(sys.stdin)
print(json.dumps({"S": [500] * inst["n"]}))
