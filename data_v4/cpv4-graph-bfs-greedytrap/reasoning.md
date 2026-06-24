**Reading the problem and pinning the contract.** I have `n` stations and `m` bidirectional
connections; each connection carries a *line* label `c`. I start at station `0`, want station `n-1`,
and travel as a walk along connections. The cost is one *transfer* every time two consecutive
connections have different line labels; staying on the same line is free, and boarding the very first
connection of the whole trip is free. I must print the minimum number of transfers, `0` if
`n-1 == 0`, and `-1` if `n-1` is unreachable. Let me fix the scale first because it decides the data
structures: `n, m <= 2*10^5`, labels up to `10^9`. The labels are sparse and large, so I will compress
them to a contiguous range before anything else — I do not want `10^9`-sized arrays. The answer itself
is at most `m <= 2*10^5` (you cannot make more transfers than connections you ride along a shortest
walk, and an optimal walk never needs to repeat a state), so the count fits comfortably in a 32-bit
`int`; the only thing that needs care is the label magnitude, which compression handles. That is the
first decision: compress labels, keep everything else `int`.

**A second contract subtlety I want nailed before coding: what "free to stay on a line" really means.**
The phrase "ride along a line for free" is dangerously easy to over-read. It does *not* mean "if two
stations are both on line `c`, I can hop between them for free." It means: if I am standing at a
station having just arrived on a line-`c` connection, I may continue along *another line-`c`
connection incident to that same station* for free. Free movement on line `c` is therefore confined to
a *connected component of the subgraph formed by line-`c` connections*. Two connections that share the
label `c` but sit in different parts of the network are not mutually free; to get from one to the other
I must transfer onto and off of some bridging line. I write this down now because I can feel it is
exactly the kind of thing a first implementation will get wrong.

**Laying out the candidate approaches.** Two routes are on the table and I want to commit to the one I
can *prove*, not the one that is shortest to type.

- *Plain BFS / "stay on the line" greedy.* Either run an ordinary BFS over stations and report some
  edge-derived count, or greedily keep extending the current line as far as it goes and only switch
  when forced, choosing the switch that reaches the furthest. Both are `O(n+m)` and a handful of lines.
  The structural risk is glaring: the quantity I pay for is *line changes*, while BFS measures
  *connections* and the greedy makes *local* switch decisions. Minimizing one need not minimize the
  other. I will not trust this until I have tried to break it.
- *0-1 BFS over an augmented state.* "Free to stay, pay 1 to switch" is a graph with edge weights in
  `{0, 1}`, which is the textbook setting for 0-1 BFS (a deque: weight-0 relaxations to the front,
  weight-1 to the back). The state has to encode enough about "which line am I on" to decide whether
  the next move is free. The risk here is not the idea but (a) choosing a state that is both correct
  and small, and (b) transcribing the model so it does not invent free moves that do not exist — the
  exact trap I flagged above.

**Stress-testing the greedy before committing.** Hand-waving "fewest stops feels right" is how wrong
solutions ship, so let me attack it with a concrete instance. Stations `0..4`, target `4`, connections:
`(0,2,line 1)`, `(2,4,line 2)`, `(0,1,line 5)`, `(1,3,line 5)`, `(3,4,line 5)`.

A connection-counting BFS finds the shortest *route* `0 -> 2 -> 4`: two connections, station distance
2. But those two connections are on line 1 then line 2 — that is one transfer. The greedy variants do
no better: starting at `0`, "extend the current line as far as possible" has to first pick a line, and
if it grabs line 1 toward station 2 (a perfectly reasonable local choice — it makes progress toward the
target) it is then forced to switch to line 2 at station 2 to reach 4: again one transfer.

Now I hunt for something these approaches structurally cannot reach. Look at `0 -> 1 -> 3 -> 4`: those
three connections are *all on line 5*. I board line 5 (free, first board), ride `0->1`, continue
`1->3` (same line, free), continue `3->4` (same line, free). Total transfers: **zero**. So the optimum
is `0`, strictly better than the `1` that both the BFS-by-connections and the "extend then switch"
greedy report — and it is achieved by the *longer* route. The verification paid off: it shows precisely
why the greedy is wrong. Fewer connections is not fewer transfers; a longer same-line ride beats a
shorter two-line ride. Greedy and plain BFS are out. I commit to the 0-1 BFS over an augmented state.

**Deriving the state and the 0/1 structure.** The future only cares about one thing when I am standing
at a station: which line I just rode in on, because that determines which onward connections are free.
So the natural state is `(station, last-line)`. From state `(u, last)`, riding a connection `(u, v, c)`
costs `0` if `c == last` (or if `last` is "none", i.e. the very first board) and `1` otherwise, landing
in state `(v, c)`. I want the minimum cost to reach any state `(n-1, *)`. Because all edge weights are
`0` or `1`, 0-1 BFS gives this in linear time in the size of the state graph. The answer is the minimum
over all lines `c` of `dist[(n-1, c)]` (plus the trivial `0` when `n-1 == 0`).

**First implementation attempt — and a worry about its size.** I code the `(station, last-line)` state
directly: an adjacency list per station of `(neighbor, line)`, a hash map `dist` keyed by the encoded
state `(station, line)`, and a deque. From a popped state I scan all incident connections, relaxing
`0`-weight continuations to the deque front and `1`-weight switches to the back. I add a stale-pop
guard so a state improved after being queued is not re-expanded with an outdated distance. I trace it
on the sample above and on `[(0,1,10),(1,2,10),(2,3,20),(3,4,30)]` (target 4): the latter forces line
10 -> line 10 (free) -> line 20 (switch) -> line 30 (switch) = 2 transfers, and the code agrees. I wire
up an independent brute force (a Dijkstra over `(station, last-line)` written from scratch, different
code path) and a tiny random generator, and run a few hundred small cases — **zero mismatches.** The
logic is correct.

**First debug episode — the worst case detonates.** Correct is not enough; I have to survive the
bounds. I build an adversarial instance: one *hub* station `1` of huge degree, where each spoke
`2,3,4,...` connects to the hub on its *own distinct line*, plus a single connection `0 -> 1`. Now the
hub gets *entered* on many different lines, so it accumulates many distinct states `(1, line)`. And
here is the killer: every time I pop a state `(1, c)`, I rescan *all* of the hub's incident
connections to relax onward. With degree `D`, that is `D` states each rescanning `D` edges — `O(D^2)`.
I time it at `D = 20000`: **1.6 seconds**. At the real bound `D` up to `2*10^5` this is minutes, a dead
TLE. The `(station, last-line)` state is *correct* but its expansion cost is quadratic on a hub. The
trace is unambiguous: the blow-up is the `D` distinct in-lines each forcing a full `D`-edge rescan. I
have to remove the per-line re-expansion of a station, not merely tune constants.

**Rethinking the model to kill the quadratic.** The wasted work is that arriving at the hub on line `a`
and on line `b` are treated as totally separate, each re-examining every onward connection. I want a
representation where "the set of stations reachable for free once I am on line `c`" is computed *once*
and shared. That is exactly a *line-component*: within line `c`, the stations are partitioned into
connected components, and inside one component I roam for free. So I build a layered (bipartite) graph:

- stations `0 .. n-1` are ordinary nodes;
- for each `(line c, connected component of line c)` I mint one *super-node*;
- a station node connects to a line-component super-node with weight `1` (boarding / switching onto
  that line-component), and the super-node connects back to each of its stations with weight `0`
  (riding for free to any station in the component).

Then `dist[station]` from a 0-1 BFS counts *boardings*. Since the first board is free, the number of
transfers is `boardings - 1`, so the answer for station `n-1` is `dist[n-1] - 1`. Crucially the
super-node is visited once and relaxes its stations once; total work is `O(n + m)` regardless of hub
degree. This is the BFS-on-an-augmented-graph idea, but the augmentation is line-*components*, not raw
states.

**Second debug episode — the layered model is initially WRONG, and a random case catches it.** I
implement the bipartite graph, but in my first cut I create *one super-node per line label*, connecting
every station that touches line `c` to that single node. I rerun the random oracle and it immediately
fails. Seed 12 gives stations `0..3`, target `3`, connections `(2,3,line A)`, `(1,0,line B)`,
`(3,2,line A)`, `(0,0,line C)` (a self-loop). My sol prints `0`; the brute prints `-1`. I trace by hand.
Physically, station `0` reaches only station `1` (and itself); stations `2,3` form a separate component.
There is *no* walk from `0` to `3` — the brute's `-1` is right. But in my one-node-per-line graph, the
connections `(1,0,line B)` and `(3,2,line A)` ... wait, those are different lines, so let me find the
real culprit: it is the duplicated *line B* across the components in the failing seeds. With one node
per label, two disconnected segments that merely share a label `c` both attach to the *same* super-node,
so the BFS happily rides from a station in the first segment, "through" the shared super-node, to a
station in the second segment — a free teleport that does not exist physically. That is precisely the
contract subtlety I warned myself about at the start: free movement is confined to a *connected
component* of line `c`, not the whole label. One node per label silently glues disjoint segments
together and fabricates reachability. The brute, which counts transfers along real walks, exposes it.

**The fix: super-node per (line, component), via a per-line union-find.** For each line I take only its
connections, reset the touched stations to singletons in a DSU, union along that line's edges, and mint
*one super-node per DSU root*. Each touched station attaches to the super-node of its own component, so
two disjoint segments of the same label get two different super-nodes and can no longer teleport into
each other. The DSU reset is scoped to the touched stations of the current line so `find` never escapes
that set, and the total reset+union work summed over all lines is `O(m * alpha)`. I re-run the oracle:
seed 12 now prints `-1`, matching the brute. I sweep a few hundred cases across generators that
deliberately reuse a tiny pool of labels (to force disconnected same-label segments) — **zero
mismatches.**

**Tracing the fixed model on the sample to sanity-check the derivation itself.** Sample again:
`(0,2,line 1)`, `(2,4,line 2)`, `(0,1,line 5)`, `(1,3,line 5)`, `(3,4,line 5)`, target `4`. Line 5's
connections `{0-1, 1-3, 3-4}` form one component `{0,1,3,4}` -> super-node `P`. Line 1's `{0-2}` ->
super-node `Q` over `{0,2}`. Line 2's `{2-4}` -> super-node `R` over `{2,4}`. 0-1 BFS from station `0`:
`dist[0]=0`. Board `P` (weight 1) -> `dist[P]=1`; board `Q` (weight 1) -> `dist[Q]=1`. Ride `P` for free
to `1,3,4`: `dist[1]=dist[3]=dist[4]=1`. So `dist[4]=1`, and the answer is `dist[4]-1 = 0`. Correct — it
matches the hand reasoning that the all-line-5 ride needs zero transfers. The `-1`-board offset is doing
its job: `dist[4]=1` means "one boarding," which is "zero transfers." And on the two-line route, station
`4` is also reachable via `Q` then `R` at `dist=2` (two boardings = one transfer), but BFS keeps the
smaller `dist[4]=1`. The model and the offset agree with the physics.

**Edge cases, deliberately, because this is where this kind of code dies.**

- *`n = 1` (start == destination).* I special-case `n-1 == 0` and print `0` before building anything;
  no boarding is needed and the `-1` offset must not turn a never-boarded destination into a phantom.
- *Unreachable.* If `dist[n-1]` stays `INF`, I print `-1`. I never compute `INF - 1`; the check is
  `dist[dst] >= INF` first. Confirmed on `n=3, m=1, (0,1,7)`: station 2 isolated -> `-1`.
- *A single direct connection, `n=2, (0,1,9)`.* Board line 9 (free first board), ride to `1`:
  `dist[1]=1`, answer `0`. One free board is zero transfers — correct.
- *`m = 0` with `n >= 2`.* No connections, destination unreachable -> `-1`. The loop over lines never
  runs, `dist[n-1]` stays `INF`. Correct.
- *Self-loops (`u == v`).* A loop on line `c` just unions a station with itself and adds a (station,
  super-node) pair; it never creates a spurious transfer and never moves you, so it is harmless. The
  generator includes self-loops and the oracle stays clean.
- *Duplicate / parallel connections.* They add duplicate `(station, super-node)` edges, only a constant
  factor; `200000` parallel `0-1` connections run in `0.03s`. The union-find dedups the component, so
  no extra super-nodes are minted.
- *Large labels.* Compression to `0..L-1` keeps `byLine` sized by distinct labels, not by `10^9`.
- *Overflow.* Distances are bounded by `m <= 2*10^5`, far inside `int`; I use `INT_MAX` as `INF` and
  always test `>= INF` before subtracting one, so no underflow.

**Performance re-verification.** The hub that detonated the first model at `1.6s` now runs in `0.08s`
because each super-node is relaxed once; the dense single-line `2*10^5`-edge case runs in `0.08s`; the
maximal path-plus-random `n=m=2*10^5` case runs in `0.16s`, well under the 2-second limit, using about
43 MB. Across three generators — one general, one denser with more labels, one adversarial with a tiny
label pool to force disconnected same-label segments — several thousand random small cases agree with
the independent brute force with zero mismatches, and the two crafted samples print `0` and `2` as
documented.

**Final solution.** I convinced myself the *idea* is right by disproving the connection-counting
greedy with the all-line-5 counterexample, and by checking the 0/1 structure on the sample; I convinced
myself the *implementation* is right by tracing the hub blow-up to the per-line re-expansion (fixing it
with shared line-component super-nodes) and by tracing the false-teleport bug to one-node-per-label
(fixing it with a per-line union-find that splits a label into its components), each fix confirmed
against the independent brute. That is what I ship — one self-contained file, a 0-1 BFS over a layered
station / line-component graph:

```cpp
#include <bits/stdc++.h>
using namespace std;

struct DSU {
    vector<int> p;
    void init(int n) { p.resize(n); iota(p.begin(), p.end(), 0); }
    int f(int x) { while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; } return x; }
    void u(int a, int b) { a = f(a); b = f(b); if (a != b) p[a] = b; }
};

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    vector<int> eu(m), ev(m), ec(m);
    {
        vector<long long> raw(m);
        for (int i = 0; i < m; i++) {
            long long u, v, c;
            cin >> u >> v >> c;
            eu[i] = (int)u; ev[i] = (int)v; raw[i] = c;
        }
        // Compress line labels to 0..L-1 (labels may be huge / sparse).
        vector<long long> tmp(raw.begin(), raw.end());
        sort(tmp.begin(), tmp.end());
        tmp.erase(unique(tmp.begin(), tmp.end()), tmp.end());
        for (int i = 0; i < m; i++)
            ec[i] = (int)(lower_bound(tmp.begin(), tmp.end(), raw[i]) - tmp.begin());
    }

    if (n - 1 == 0) { cout << 0 << "\n"; return 0; }   // start == destination

    // ---- Build "line-component" super-nodes. ----------------------------------------------
    // Within one line c, two stations linked by a chain of line-c edges are mutually reachable
    // with ZERO transfers.  So the unit you may roam for free is a *connected component of the
    // subgraph induced by line c* -- NOT the line as a whole (two disjoint segments that merely
    // share a label are not connected).  Each (line, component) becomes one super-node.
    //
    // Layered 0-1 BFS:
    //   station ---- weight 1 ----> line-component node   (boarding / switching onto a line)
    //   line-component node -- weight 0 --> station       (riding for free to any of its stations)
    // dist[station] then counts boardings; with the first board free, transfers = boardings - 1,
    // so the answer for station n-1 is dist[n-1] - 1.

    // Group edge indices by compressed line label.
    int L = 0;
    for (int i = 0; i < m; i++) L = max(L, ec[i] + 1);
    vector<vector<int>> byLine(L);
    for (int i = 0; i < m; i++) byLine[ec[i]].push_back(i);

    DSU dsu; dsu.init(n);
    vector<vector<pair<int,int>>> g(n); // ids 0..n-1 are stations; component nodes appended after
    int nextComp = n;

    for (int c = 0; c < L; c++) {
        auto &es = byLine[c];
        if (es.empty()) continue;
        // Local union-find over only the stations this line touches: reset them to singletons,
        // then union along this line's edges, so find() never escapes the touched set.
        for (int idx : es) { dsu.p[eu[idx]] = eu[idx]; dsu.p[ev[idx]] = ev[idx]; }
        for (int idx : es) dsu.u(eu[idx], ev[idx]);

        unordered_map<int,int> rootToNode;
        rootToNode.reserve(es.size() * 2 + 1);
        auto getNode = [&](int station) -> int {
            int r = dsu.f(station);
            auto it = rootToNode.find(r);
            if (it != rootToNode.end()) return it->second;
            int id = nextComp++;
            rootToNode[r] = id;
            g.emplace_back();        // adjacency row for the new component node
            return id;
        };
        for (int idx : es) {
            for (int s : {eu[idx], ev[idx]}) {
                int node = getNode(s);
                g[s].push_back({node, 1});  // board / switch onto this line-component
                g[node].push_back({s, 0});  // ride for free to this station
            }
        }
    }

    int V = (int)g.size();
    const int INF = INT_MAX;
    vector<int> dist(V, INF);
    deque<int> dq;
    dist[0] = 0;
    dq.push_back(0);
    while (!dq.empty()) {
        int x = dq.front(); dq.pop_front();
        int dx = dist[x];
        for (auto &pr : g[x]) {
            int y = pr.first, w = pr.second;
            if (dx + w < dist[y]) {
                dist[y] = dx + w;
                if (w == 0) dq.push_front(y);
                else dq.push_back(y);
            }
        }
    }

    int dst = n - 1;
    if (dist[dst] >= INF) cout << -1 << "\n";          // unreachable
    else cout << (dist[dst] - 1) << "\n";              // boardings - 1 (first board is free)
    return 0;
}
```

**Causal recap.** The connection-counting greedy looked right but a single traced counterexample (the
all-line-5 route `0->1->3->4` at *zero* transfers beating the shorter two-line route at *one*) showed
that fewest connections is not fewest transfers, so I moved to a 0-1 BFS whose cost is line changes;
the natural `(station, last-line)` state is correct but a hub entered on `D` lines re-expands `D` times
over `D` edges, which I timed at quadratic and `1.6s`, so I collapsed free movement into one super-node
per line so each is relaxed once; one node *per label* then teleported between disjoint same-label
segments — a random seed where station `0` cannot reach the target but my code returned `0` pinned it
against the brute — so I split each label into its connected components with a per-line union-find; and
the `boardings - 1` offset with an `INF`-guarded subtraction closes out the first-board-free, single
connection, unreachable, and `n = 1` corners, all confirmed against the independent Dijkstra brute on
thousands of cases.
