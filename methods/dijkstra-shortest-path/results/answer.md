# Dijkstra's Shortest Path Algorithm

## Problem

Given a graph whose edges (roads) have **nonnegative** lengths — possibly directed — find the
minimum total length of a path from a source node `P` to a target `Q`, and recover the path
itself. The search state should avoid a sorted or repeatedly scanned full edge list.

## Key idea

Grow the set `A` of nodes whose shortest distance from `P` is known, one node at a time, in
order of increasing distance. For a frontier node `v`, its tentative distance `t[v]` is the
best known route that reaches some settled node and then crosses one edge into `v`.

If `u` has the smallest tentative distance, then `u` is final. Any path from `P` to `u` first
leaves `A` at some frontier node `y` via an edge `(x, y)` with `x` in `A`. The prefix to `x`
costs at least the known shortest distance `d[x]`, so the route's prefix through this
particular crossing costs at least `d[x] + length(x, y)`. The tentative distance `t[y]` is
the best crossing into `y` from the settled set, so that prefix costs at least `t[y]`. Since
`t[y] >= t[u]`, and the remaining suffix from `y` to `u` has nonnegative length, every path
to `u` costs at least `t[u]`. Since `t[u]` is itself a real path, it equals the true shortest
distance. This is exactly where nonnegative lengths are required.

## Algorithm

Maintain settled nodes, best tentative distances, predecessors, and a min-heap keyed by
tentative distance.

1. Put `P` in the frontier with distance 0.
2. Extract the smallest heap entry. If it is stale or already settled, skip it; otherwise
   settle that node.
3. If the node is `Q`, trace predecessors backward to recover the route.
4. For each outgoing edge `(u, v)` with `v` unsettled, compute `d(u) + length(u, v)`.
   If that improves `v`'s best known distance, record the new distance and predecessor and
   push the new heap entry.
5. Repeat until `Q` is settled or the frontier empties.

Works for directed edges. Correctness requires nonnegative lengths.

With a linear scan to choose the minimum frontier node, the time is `O(V^2 + E)` and the live
branch records stay `O(V)` besides the external road source. The heap implementation below
inspects each outgoing edge of a settled node once; each successful improvement pushes one
heap entry, and lazy deletion leaves stale entries in the heap. Its time is
`O((E + V) log(E + 1))` and its extra heap space can be `O(E)`. For a simple graph,
`log(E + 1) = O(log V)`, so this is commonly written `O((E + V) log V)` and often
`O(E log V)` when the edge term dominates.

## Code

```python
import heapq

# outgoing_roads(node) -> iterable of (neighbour, length).
# Lengths are nonnegative; roads may be directed.

def shortest_path(outgoing_roads, start, end):
    """Return the minimum total length and route from start to end.

    Precondition: every returned edge length is nonnegative.
    Return None when end is unreachable.
    """
    heap = [(0, start)]
    best = {start: 0}
    predecessor = {start: None}
    settled = set()

    while heap:
        dist, u = heapq.heappop(heap)
        if dist > best.get(u, float("inf")):
            # stale key left over from an older, worse route
            continue
        if u in settled:
            continue
        settled.add(u)

        if u == end:
            path = []
            node = end
            while node is not None:
                path.append(node)
                node = predecessor[node]
            path.reverse()
            return dist, path

        for v, length in outgoing_roads(u):
            if v in settled:
                continue
            candidate = dist + length
            if candidate < best.get(v, float("inf")):
                best[v] = candidate
                predecessor[v] = u
                heapq.heappush(heap, (candidate, v))

    return None


if __name__ == "__main__":
    road_map = {
        "Rotterdam": [("Utrecht", 57), ("Amsterdam", 78)],
        "Amsterdam": [("Utrecht", 40), ("Zwolle", 112)],
        "Utrecht":   [("Zwolle", 90), ("Amsterdam", 40)],
        "Zwolle":    [("Groningen", 100)],
        "Groningen": [],
    }

    def outgoing_roads(city):
        return road_map.get(city, ())

    print(shortest_path(outgoing_roads, "Rotterdam", "Groningen"))
```

The heap may contain stale entries after a better tentative distance is found; comparing the
popped distance with `best[u]` and skipping settled nodes gives lazy deletion without an
explicit decrease-key.
