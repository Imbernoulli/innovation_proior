# TIER: invalid
# Emit a graph that contains a directed cycle (0->1->2->0). A propagation graph
# must be acyclic, so the grader's DAG check rejects it -> score 0.
import sys, json
inst = json.load(sys.stdin)
n = inst["n"]
edges = [[0, 1], [1, 2], [2, 0]]
# pad with more edges; the cycle alone already forces rejection
for i in range(3, n):
    edges.append([0, i])
print(json.dumps({"edges": edges}))
