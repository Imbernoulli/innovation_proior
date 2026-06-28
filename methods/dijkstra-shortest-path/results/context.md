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
little memory. The deliverable is a single self-contained C++17 program that reads the map
from stdin and writes the route answer to stdout.

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

## Input-output contract

The program reads `n m s t`, then `m` directed roads `u v w`, from stdin. Node codes are
0-based integers, and each road length `w` is nonnegative. It prints the minimum total
length from `s` to `t` followed by one shortest route as space-separated node codes, or
prints `UNREACHABLE` when `t` cannot be reached.

```python
#include <bits/stdc++.h>
using namespace std;

int main(){
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m, s, t;
    if (!(cin >> n >> m >> s >> t)) return 0;

    vector<vector<pair<int, long long>>> adj(n);
    for (int i = 0; i < m; ++i) {
        int u, v;
        long long w;
        cin >> u >> v >> w;
        adj[u].push_back({v, w});
    }

    bool reachable = false;
    long long total_length = 0;
    vector<int> path;

    // TODO: compute whether t is reachable from s, the minimum total length,
    // and one corresponding route as node codes in order.

    if (!reachable) {
        cout << "UNREACHABLE\n";
        return 0;
    }

    cout << total_length << "\n";
    for (size_t i = 0; i < path.size(); ++i) {
        cout << path[i] << " \n"[i + 1 == path.size()];
    }
    return 0;
}
```
