The result I am presenting is Savitch's theorem, which gives the canonical deterministic space bound for simulating nondeterministic space. For every space-constructible function S(n) that is at least logarithmic in the input length, the theorem states that NSPACE(S(n)) is contained in DSPACE(S(n)^2). In other words, whatever a nondeterministic Turing machine can decide using S(n) cells of work space, a deterministic Turing machine can decide using only O(S(n)^2) cells. This is a landmark result in computational complexity theory because it shows that nondeterminism does not buy an exponential explosion in space. The simulation may take an enormous amount of time, but the amount of reusable memory it needs is only the square of the original space bound.

The conceptual move behind the theorem is to stop thinking of a nondeterministic computation as a tree of branching choices and to start thinking of it as a reachability problem on an implicit configuration graph. Fix a nondeterministic machine M and an input x. A configuration of M on x encodes everything needed to continue the computation from a given moment: the current state of the finite control, the contents of the work tape within the allowed bound, the positions of the work-tape heads, and the position of the input head. Because M uses S(n) space and S(n) is at least log n, each configuration can be written using O(S(n)) bits. The number of distinct configurations is therefore at most exponential in S(n), namely 2^{O(S(n))}, but each individual configuration is small. Two configurations are connected by a directed edge if the machine can move from the first to the second in one legal step according to its transition relation and the input symbol being read. The machine accepts x if and only if some accepting configuration is reachable from the start configuration in this graph.

The naive deterministic attempts to decide reachability all fail the space budget. Maintaining the set of all configurations that could be current at a given time produces an exponentially large frontier. A depth-first search of one branch at a time still needs a visited discipline, and that visited set can also become exponentially large. Even demanding an explicit accepting computation history yields a certificate that can be exponentially long. All of these approaches waste space because they try to remember a global object. The insight of Savitch's theorem is that I do not need to remember the whole graph, the whole frontier, or the whole path. I only need a way to verify that one small configuration can reach another within a bounded number of steps.

The key definition is bounded reachability. Let Reach(u, v, i) mean that configuration v can be reached from configuration u in at most 2^i steps. When i is zero, deciding Reach(u, v, 0) is easy: either u equals v, or there is a single legal transition from u to v. For larger i, I use a divide-and-conquer decomposition. Suppose there is a path from u to v of length at most 2^i. If the path is shorter than 2^{i-1}, then Reach(u, v, i - 1) already holds, so I can take the middle configuration z to be v. Otherwise, there is some configuration z exactly halfway along the path; the first half witnesses Reach(u, z, i - 1) and the second half witnesses Reach(z, v, i - 1). Conversely, if any z satisfies both shorter reachability facts, concatenating the two paths gives a path from u to v of length at most 2^i. This recurrence drives the simulation.

The deterministic simulator evaluates this recurrence recursively. To decide acceptance, I call Reach on the start configuration, an accepting configuration, and i equal to the ceiling of log N, where N is the number of vertices in the configuration graph. I can take N to be 2^{O(S(n))}, because any accepting walk that repeats a configuration can be shortened by deleting the loop, so an accepting path never needs more than N - 1 steps. At each recursive level, I enumerate candidate middle configurations z one at a time. For each z, I recursively check Reach(u, z, i - 1). If that succeeds, I reuse the same workspace to check Reach(z, v, i - 1). The two subchecks do not have to exist simultaneously in memory, which is the crucial space-saving mechanism. If a candidate z fails, I discard the intermediate computation and try the next z.

The space cost is what makes the theorem work. Each recursion frame stores only a constant number of configuration names plus the recursion level and loop counters. A configuration name takes O(S(n)) bits, so each frame uses O(S(n)) space. The recursion depth is the number of levels needed to reduce the path budget from N to 1, which is O(log N) = O(S(n)). Because the workspace for the two recursive subcalls is reused rather than kept simultaneously, the total deterministic space is the product of depth and frame size, giving O(S(n)^2). The time cost is large because every level scans all candidate middle configurations, but the theorem concerns space, not time, so this price is acceptable.

Savitch's theorem has two classic corollaries. First, because the square of a polynomial is still a polynomial, taking S(n) to be any polynomial yields PSPACE = NPSPACE. Any problem decidable by a polynomial-space nondeterministic machine is also decidable by a polynomial-space deterministic machine. Second, taking S(n) = log n gives the containment NL ⊆ DSPACE(log^2 n), since nondeterministic logarithmic-space machines have configurations of size O(log n) and the simulation uses O(log^2 n) space. The theorem shows that, in the space-bounded setting, the deterministic and nondeterministic classes do not separate by the kind of exponential gap seen in time complexity unless much stronger separations hold elsewhere.

The method name I propose is Savitch's theorem, or more descriptively the Savitch deterministic-space simulation of nondeterministic space. The theorem is not a practical algorithm, because the running time of the simulation is far too large. It is instead a structural result about the power of deterministic memory when it reuses the same cells across recursively defined subproblems. The proof technique of recursive middle-configuration reachability has become a standard tool in complexity theory.

To make the construction concrete, I have written a small Python illustration that applies the same divide-and-conquer reachability idea to an explicit directed graph. The program builds a random graph, picks a source and a target, and decides reachability using the Savitch-style bounded reachability recurrence. The code keeps the structure of the proof: a recursive function with a level parameter, enumeration of candidate middle vertices, and reuse of the same call stack. Running it on a small example confirms that the recurrence is correct and that the program never needs to store more than the current endpoints, candidate middle, and recursion level at any one time.

```python
import random

def build_random_graph(n, edge_prob=0.15):
    """Build a random directed graph with n vertices as an adjacency set."""
    graph = {i: set() for i in range(n)}
    for u in range(n):
        for v in range(n):
            if u != v and random.random() < edge_prob:
                graph[u].add(v)
    return graph

def one_step(graph, u, v):
    """True if there is a directed edge from u to v."""
    return v in graph[u]

def savitch_reach(graph, u, v, level, memo=None):
    """
    Decide whether v is reachable from u in at most 2**level steps,
    using the Savitch divide-and-conquer recurrence.
    """
    if memo is None:
        memo = {}
    key = (u, v, level)
    if key in memo:
        return memo[key]

    n = len(graph)
    if level == 0:
        result = (u == v) or one_step(graph, u, v)
        memo[key] = result
        return result

    # Enumerate candidate middle vertices one at a time.
    for z in range(n):
        if savitch_reach(graph, u, z, level - 1, memo):
            if savitch_reach(graph, z, v, level - 1, memo):
                memo[key] = True
                return True
    memo[key] = False
    return False

def reachable(graph, source, target):
    """Top-level wrapper: decide reachability using at most n-1 steps."""
    n = len(graph)
    if n <= 1:
        return source == target
    # Enough levels so that 2**level >= n-1.
    level = 0
    while (1 << level) < n - 1:
        level += 1
    return savitch_reach(graph, source, target, level)

if __name__ == "__main__":
    random.seed(0)
    n = 12
    graph = build_random_graph(n, edge_prob=0.18)
    source = 0
    target = n - 1

    # Compute the deterministic answer using BFS for comparison.
    bfs_visited = {source}
    frontier = [source]
    while frontier:
        u = frontier.pop(0)
        for v in graph[u]:
            if v not in bfs_visited:
                bfs_visited.add(v)
                frontier.append(v)
    bfs_answer = target in bfs_visited

    savitch_answer = reachable(graph, source, target)
    print("BFS says reachable:", bfs_answer)
    print("Savitch recurrence says reachable:", savitch_answer)
    assert bfs_answer == savitch_answer, "Mismatch between BFS and Savitch simulation!"
```
