The demonstration calls for a shortest road route between two Dutch cities on a map small enough for a 1956 audience to check, yet the ARMAC has almost no memory. A road map is just points connected by segments with nonnegative lengths, possibly one-way, and the goal is the minimum-length route from a source to a target. The brute-force idea of enumerating every route is out of reach, and the known alternatives are too heavy for this machine. Ford's label-correcting method keeps a tentative distance for every node and repeatedly scans the entire edge set to relax violations, which means storing and re-scanning all branches many times before anything stabilizes. Kruskal's and Loberman-Weinberger's spanning-tree ideas sort all edges up front, so they too require the whole branch list to be resident at once. What is needed is a way to commit to answers one city at a time while touching only a tiny active slice of the map.

The method is Dijkstra's shortest-path algorithm. Instead of aiming straight at the destination, it grows a settled set of cities whose true shortest distance from the source is already known, starting with the source at distance zero. For every city not yet settled but reachable by a single road from the settled set, it keeps a tentative distance equal to the best known route that reaches some settled city and then crosses one road into that frontier city. The city on the frontier with the smallest tentative distance is moved into the settled set, and its distance becomes final. The reason this is safe is that all road lengths are nonnegative: any route to that city must leave the settled set somewhere, and the first crossing already costs at least as much as the smallest frontier tentative distance, while the remaining suffix cannot make the total any smaller. Once a city is settled it is never revisited, so the algorithm only ever needs one candidate branch per frontier city and one predecessor branch per settled city, rather than the full edge list.

When a city is newly settled, the only fresh information comes from its outgoing roads. For each neighbor not yet settled, the algorithm computes the candidate distance through the newly settled city; if this improves the neighbor's best known distance, the neighbor's tentative distance and predecessor are updated. Choosing the next city to settle is naturally implemented by a min-priority queue keyed by tentative distance. Stale queue entries left behind by later improvements are harmless: when one surfaces, it is discarded if the city is already settled or if its key is worse than the current best distance. This keeps the bookkeeping frugal and the code simple. A linear scan over the frontier would also work and uses only O(V) live records, while the heap version runs in O((E + V) log V) time for sparse graphs.

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
