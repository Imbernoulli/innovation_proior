## Research question

A new computer, the ARMAC, is about to be inaugurated at the Mathematical Centre in
Amsterdam, and it needs a public demonstration. The predecessor machine was so unreliable
that the only demo anyone dared show was random-number generation; the ARMAC is reliable
enough to attempt something with an answer a layperson can check. The constraint is sharp:
the problem must be statable to non-computing people and its answer must be understandable
by them. The natural choice is the shortest road route between two given cities of the
Netherlands — *what is the shortest way to travel from Rotterdam to Groningen?* The goal is
an algorithm that, given a road map with distances, produces the minimum-total-length route
between two named places, and that is small and frugal enough to actually run on the ARMAC:
the map is reduced to 64 cities so that a city fits in 6 bits, and the machine has very
little memory.

## Background

A road map is a set of points (cities) joined by segments (roads), each segment carrying a
length. At least one route exists between any two cities. Road lengths are nonnegative — a
road never has negative length — and a road may be one-way, so the length from X to Y need
not equal the length from Y to X. The quantity wanted is the minimum total length over all
routes from a source P to a target Q.

A structural fact about this problem is **optimal substructure**: if R lies on a
minimum-length route from P to Q, then the portion from P to R is itself a minimum-length
route from P to R. Equivalently, a shortest path is built out of shortest paths to its
intermediate points.

At the time, this kind of problem is barely regarded as mathematics. The prevailing attitude
is that there is a finite number of ways of going from A to B and obviously one of them is
shortest; discrete, combinatorial algorithms have not yet acquired mathematical
respectability and there are no journals that obviously want them.

A second, sibling problem from the same setting: on the back panel of a machine, many points
must be tied to the same voltage with copper wire; minimizing the total wire is the
minimum-total-length **tree** spanning a set of points.

## Baselines

**Ford, "Network flow theory," Rand P-923 (1956), as described by Berge (1958, pp. 68–69).**
The label-correcting approach. Keep a tentative distance `d[v]` for *every* node (∞ at
start, 0 at the source). Repeatedly look for any edge `(u, v)` that violates
`d[v] ≤ d[u] + length(u, v)` and "relax" it by setting `d[v] := d[u] + length(u, v)`; stop
when no edge violates. Under the road-distance assumptions it is correct. It keeps a label
for the whole node set, and to find violations it scans the edge data; a node's label can be
corrected several times before it stabilizes.

**Kruskal (1956); Loberman and Weinberger (1957)** — for the spanning-tree sibling. Both
first sort all of the up-to ½n(n−1) edges by length and then add edges cheapest-first
(skipping any that would form a cycle), storing the edges together for the sort. An edge's
length may itself be a computable function of the endpoints' coordinates.

## Evaluation settings

The natural yardstick is the demonstration itself: a reduced road map of the Netherlands with
64 cities (6-bit city codes) and the inter-city road lengths, run on the ARMAC, asked for the
shortest route between two named cities such as Rotterdam and Groningen. A correct method
returns the minimum total length and the route achieving it. The figures of merit that matter
on this machine are how many branch records must be held at any moment and how much
arithmetic and bookkeeping is done. The map lengths are nonnegative and roads may be
directed.

## Code framework

The caller supplies a way to enumerate the outgoing roads of one city and calls one route
routine.

```python
# outgoing_roads(city) -> iterable of (neighbour, length).
# Lengths are nonnegative; roads may be directed.
# Example reduced road map (cities as strings; on the machine, 6-bit codes).
road_map = {
    "Rotterdam": [("Utrecht", 57), ("Amsterdam", 78)],
    "Amsterdam": [("Utrecht", 40), ("Zwolle", 112)],
    "Utrecht":   [("Zwolle", 90), ("Amsterdam", 40)],
    "Zwolle":    [("Groningen", 100)],
    "Groningen": [],
}


def outgoing_roads(city):
    return road_map.get(city, ())


def shortest_path(outgoing_roads, start, end):
    """Return the minimum total length and route from start to end.

    Return None when end is unreachable.
    """
    # TODO: choose and maintain the search state, update it from outgoing
    # roads, and reconstruct the route when `end` is reached.
    pass


if __name__ == "__main__":
    print(shortest_path(outgoing_roads, "Rotterdam", "Groningen"))
```
