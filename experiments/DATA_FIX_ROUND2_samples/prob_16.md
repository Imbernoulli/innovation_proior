# 题 16 — base均分 0.0 vs 直接SFT均分 0.0(掉点 0.0)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=0,得分 0.0,completion_tokens=32768)

````
The user wants a C++ solution for an interactive problem called "Identify Chord".

**Problem Understanding:**
1.  **Graph Structure**: A cycle graph with $n$ vertices ($1, \dots, n$). Edges are $(i, i+1)$ and $(n, 1)$.
2.  **Modification**: One additional edge (chord) connects two non-adjacent vertices $u, v$.
3.  **Goal**: Identify the pair $(u, v)$ of the chord.
4.  **Constraints**: $4 \le n \le 10^9$, $T \le 1000$. Max queries per test case $\le 500$.
5.  **Query Type**: Query distance between $x$ and $y$ in the modified graph. This is the number of edges on the shortest path.
6.  **Interaction**: Non-adaptive interactor (chord is fixed before queries).

**Analysis:**
In a pure cycle graph, the distance between $x$ and $y$ is $\min(|x-y|, n - |x-y|)$.
With the chord $(u, v)$ added, the distance between any pair changes if the path between them uses the chord. Specifically, the shortest path distance becomes $\min(\text{dist}_{cycle}(x, y), \text{dist}_{cycle}(x, u) + 1 + \text{dist}_{cycle}(v, y), \text{dist}_{cycle}(x, v) + 1 + \text{dist}_{cycle}(u, y))$. Note that since it's undirected, we just consider paths passing through the chord.

Let the chord connect $L$ and $R$ such that $1 \le L < R \le n$. The vertices on one arc are not necessarily adjacent to each other except via the chord endpoints. The chord splits the cycle into two simple paths. Let's say the two paths have lengths $d_1 = R - L - 1$ and $d_2 = n - d_1 - 2$. Wait, simpler: the cycle length is $n$. The chord divides the cycle into two arcs between $L$ and $R$. Let the lengths of these arcs be $k$ and $n-k-1$? No, let's stick to indices.
Vertices on the arc from $L$ to $R$ clockwise: $L, L+1, \dots, R$. Length of path $L \to R$ without chord is $R-L$. With chord, dist$(L,R)=1$.
The arc from $R$ to $L$ clockwise: $R, R+1, \dots, n, 1, \dots, L$. Length is $n - (R-L)$. With chord, dist$(L,R)=1$.

So, for any $x, y$, the distance is determined by whether the chord provides a shortcut.
The problem asks to find $L$ and $R$. Since the query returns shortest path distances, this gives us information about where the "shortcut" lies.

**Key Observation:**
If we pick two vertices $x, y$ far apart in the cycle, their distance will likely be small if the chord crosses the region between them.
Specifically, if $x$ and $y$ are separated by a large part of the cycle *without* the chord, but connected by a small path *through* the chord, the observed distance will be much smaller than expected from the cycle metric.

However, $n$ is up to $10^9$, so we cannot iterate over all vertices. We need something logarithmic or constant time relative to $n$ (like $O(\log n)$ or similar). The query limit is 500, which allows roughly $2^{8}$ or maybe slightly more binary search steps if we are careful.

Actually, 500 queries is quite a lot for binary search on $10^9$ (which needs $\approx 30$ bits). But finding a specific edge among $N(N-1)/2$ pairs is harder.
Wait, there are only $O(n)$ possible chords? Actually any pair is a candidate. There are $\binom{n}{2}$ pairs. We can't check all.
But the structure is symmetric. We just need to locate $u$ and $v$.

Let's consider the properties of the distances provided.
For any three vertices $a, b, c$, triangle inequality holds. In the un-chorded cycle, for vertices on the cycle, distances satisfy specific relations. With a chord, some paths become shorter.

Consider picking a vertex, say 1.
Pick another vertex $n/2$. Measure distance.
This helps distinguish regions.

A common strategy for "find the hidden chord in a cycle" problems involves probing points to see where the cycle "deforms".
If we measure distances from a fixed point $S$ to various points $X$, the function $D(S, X)$ should look like $\min(|X-S|, n-|X-S|)$. If the chord is present, $D(S, X)$ might be smaller than the cycle distance.

Let's denote $C(u, v)$ as the cycle distance $\min(|u-v|, n-|u-v|)$.
Let $Dist(u, v)$ be the actual shortest path distance with chord.
We know $Dist(u, v) \le C(u, v)$.
Also $Dist(u, v) = \min(C(u, v), C(u, u') + 1 + C(v', v))$ where $(u', v')$ is the chord endpoints? Not exactly, the chord is undirected between two nodes $A, B$. So $Dist(x, y) = \min(C(x, y), C(x, A) + 1 + C(B, y), C(x, B) + 1 + C(A, y))$.

Note that if we query $(x, y)$, the result is the minimum of the path along the rim (two ways) vs path using the chord.
Essentially, the chord effectively reduces the diameter of the graph significantly if $n$ is large.

Let's try to locate the chord using a geometric approach.
Imagine the vertices are arranged on a circle. The chord creates a shortcut.
If we query $(x, y)$, and the answer is small compared to $C(x, y)$, it means the shortest path used the chord.
If the answer equals $C(x, y)$, it means the chord is not useful for connecting $x$ and $y$ directly (or at all for the shortest path). Wait, this is not strictly true. If $x, y$ are on the "longer" side of the chord, the chord might still provide a shortcut if they are close to the chord endpoints.

Let's refine the condition. The chord connects $u, v$.
The graph has two arcs between $u$ and $v$: Arc1 (length $L_1$) and Arc2 (length $L_2$). Note $L_1 + L_2 + 2 = n$ (counting edges). Or simply the distance between $u$ and $v$ on the rim is $d$ and $n-d$. Let $d = C(u, v)$. Then $d > 1$ because $u, v$ non-adjacent.
The distances in the modified graph are affected mostly by pairs $(x, y)$ such that the segment of the rim between $x$ and $y$ covers both $u$ and $v$. Wait, if it covers $u, v$, then going $x \to \dots \to u \to v \to \dots \to y$ might not use the chord edge. The chord edge goes directly $u \leftrightarrow v$.
Using the chord means traveling $x \leadsto u \to v \leadsto y$.
So $Dist(x, y) = \min(C(x, y), C(x, u) + 1 + C(v, y), C(x, v) + 1 + C(u, y))$.

If $x$ and $y$ are such that the interval between them (on the longer rim arc containing no chord) contains the chord... wait.
Basically, the set of vertices involved in the "shortcut" is somewhat localized around the chord.
Actually, if we fix $x=u$ (one end of the chord), then $Dist(u, v) = 1$. If we query $(u, v)$, we get 1. But we don't know who $u, v$ are.

Let's try to identify the endpoints of the chord.
Suppose we query $(x, x+k)$. This usually takes $k$ or $n-k$ in the cycle.
With the chord, maybe less.

Since $n$ is large, we want to narrow down the position of the chord endpoints.
Can we binary search for the endpoints?
Maybe not directly.
Consider the following property: For any $k$, if we move $k$ steps away from $u$ in both directions, we are getting closer to the other endpoint of the chord only if we traverse the correct direction?
No, simpler:
In a pure cycle, $Dist(1, n) = 1$. Here it could be large if chord is elsewhere.
Let's check $Dist(1, n)$. If chord connects $1$ and $n$, then $Dist(1, n) = 1$. If chord is far away, it might be different.
But $Dist(1, n)$ is just 1 in cycle anyway (edges $(1, n)$ and $(1, 2)...(n-1, n)$? Wait.
Cycle is $1-2-\dots-n-1$.
Adjacent edges are $(i, i+1)$ and $(n, 1)$. So $1$ and $n$ are adjacent. Distance is 1.
Ah, the chord must connect non-adjacent vertices. So chord cannot be $(1, n)$.
Thus $Dist(1, n)$ is always 1.
Querying $(1, n)$ is useless.

Let's consider querying $(1, k)$ where $k$ is far from $n$, e.g., $k=n/2$.
In cycle, $Dist(1, n/2) = \min(n/2-1, n-(n/2-1)) = n/2$ roughly.
If chord is somewhere, it might reduce this.
Actually, the most robust way is often related to ternary search or binary search on the ring.
Since we don't know if $u < v$ or vice versa, symmetry matters.

However, notice that $n$ is very large ($10^9$).
The chord is unique.
There are essentially $O(n)$ possibilities for pairs.
But maybe we can find $u$ first.
Once we find one endpoint $u$, we can find $v$ easily because $Dist(u, v) = 1$ (with adjustment for being neighbors, but wait, if we found $u$, how do we find $v$?).
Wait, if we find $u$, then for any neighbor of $u$ on the rim (say $u-1, u+1$), distance to $v$ might give info.
Actually, if we query $(u, x)$ for some $x$, the value tells us how close $x$ is to $v$ via $u$.
If we query $(u, x)$, the path might go $u \to v \to \dots \to x$.
Then $Dist(u, x) = 1 + C(v, x)$.
We know $C(v, x)$ depends on $v$'s position relative to $x$.
If we can scan out from $u$ to find where distances behave "normally", we might find $v$.

Is it possible to find $u$ quickly?
What if we assume the chord connects $u$ and $v$.
Consider the query sequence:
$Dist(1, k)$.
The sequence of values $D_k = Dist(1, k)$ for $k=1, \dots, n$.
In the cycle, $D_k = \min(k-1, n-(k-1))$. It increases linearly then decreases.
With the chord, for $k$ in certain ranges, $D_k$ will be smaller.
Specifically, if the chord is $(u, v)$, then $Dist(1, k)$ is affected if $u$ is between $1$ and $k$ (in one direction) AND $v$ is between $1$ and $k$ (in other direction)?
Actually, let's visualize the effect.
The shortest path between 1 and $k$ is either on the rim or using the chord.
Using the chord implies path goes $1 \to \dots \to u \to v \to \dots \to k$.
The cost is $C(1, u) + 1 + C(v, k)$. Or $C(1, v) + 1 + C(u, k)$.
Notice that $C(1, u) + C(v, k) \approx$ distance on rim covering both intervals.
Basically, the chord creates a "bridge" across the cycle.
If we look at $Dist(1, k)$, it will be minimal (value 1) at $k=v$ if $u$ is near 1? No.

Let's rethink. What if we simply probe midpoints?
Query $x=1, y=n/2$. Result $d_1$.
Query $x=1, y=n/4$. Result $d_2$.
From these, can we deduce anything?
In the unmodified graph, $Dist(1, n/2) \approx n/2$. $Dist(1, n/4) \approx n/4$.
The presence of a chord generally decreases distances.
However, for large $n$, the decrease might be significant only near the chord?
Actually, if the chord is $(u, v)$, and $n$ is huge, then $u$ and $v$ are just points.
The region where the distance is shortened is defined by the Voronoi-like sets based on the chord endpoints.
Specifically, any node $w$ is connected to the rest.
Consider $Dist(1, k)$.
If $k$ is very far from $u$ and $v$ (i.e. on the long arc between them not involving the chord?), then the path along the rim is optimal.
Wait, the chord splits the plane. Any path using the chord enters at $u$ or $v$.
If $1$ and $k$ are separated such that neither $u$ nor $v$ are on the direct path between them on the rim, then the chord is useless?
Yes. If the path on the rim does not pass "near" the chord, it is optimal.
Actually, since $u, v$ are the *only* non-adjacent edges (besides rim), any path using the chord *must* visit $u$ or $v$.
So if $1$ and $k$ are such that all shortest paths on rim avoid getting close to $u, v$ sufficiently? No.
It's simpler: If $Dist(1, k) < C(1, k)$, then the chord was used.
This inequality holds if $\min(C(1, u) + 1 + C(v, k), C(1, v) + 1 + C(u, k)) < C(1, k)$.
This basically means $1 + \text{dist}(u, v \text{ on rim}) < \text{dist}(1, k \text{ on rim})$.
Essentially, if $1$ and $k$ are on opposite sides of the chord, the chord shortens the path.
Specifically, the chord connects two arcs. $1$ belongs to one arc endpoint-wise?
Let's assume w.l.o.g. $u < v$. The cycle is broken into two segments of lengths $A$ and $B$ (where $A+B+2=n$, edges count). Vertices $u, u+1, \dots, v$ form arc of length $v-u$. Vertices $v, v+1, \dots, n, 1, \dots, u$ form arc of length $n-(v-u)$.
If $1$ is in the second arc (between $v$ and $u$), and $k$ is in the first arc (between $u$ and $v$), then to go $1 \to k$, the path $1 \to \dots \to v \to u \to \dots \to k$ using the chord at the ends might be better? No, chord is internal edge.
If $1$ is in the "gap" between $u$ and $v$ (the larger gap?), wait.
Let's rephrase.
The cycle vertices are $1, \dots, n$.
Chord connects $u, v$.
Any path between $x, y$ can stay on rim or cross chord.
Crossing chord means path looks like $x \rightsquigarrow \{u,v\} \to \{u,v\} \rightsquigarrow y$.
Since the chord is an edge, if you enter it at $u$, you leave at $v$. Cost $1$.
So path is $x \to u \to v \to y$. Cost $C(x,u) + 1 + C(v,y)$.
Or $x \to v \to u \to y$. Cost $C(x,v) + 1 + C(u,y)$.
If $C(x, y) > C(x, u) + 1 + C(v, y)$, then chord helps.
This happens when $u$ and $v$ are "between" $x$ and $y$ on the cycle.
More precisely, consider the order on the cycle.
If the segment of cycle containing $x$ and $y$ does NOT contain both $u$ and $v$ (meaning one is outside the interval between $x$ and $y$ on the chosen path), then chord might still help?
Wait, if we take the path on the cycle between $x$ and $y$ that *contains* $u$ but not $v$, its length is $C(x, u) + C(u, y)$.
The chord path is $C(x, u) + 1 + C(v, y)$.
So if $C(u, y) > 1 + C(v, y)$, chord is better.
$C(u, y)$ is dist on rim. $C(v, y)$ is dist on rim.
This looks complicated to track analytically.

Alternative Idea:
Use the fact that $n \le 10^9$.
The problem asks for *any* valid pair $(u, v)$.
Since $n$ is large, maybe we can use random sampling? No, deterministic required.
Wait, $n$ up to $10^9$, queries 500.
If we can determine $u$ and $v$ in $\approx 30$ queries each? No, identifying a point on a line is easy (binary search), identifying a pair requires finding two points.
How to find one endpoint $u$?
If we can find $u$, then we can find $v$ by looking for a node $z$ such that $Dist(u, z) = 1$. But we don't know $v$.
Wait, if $Dist(u, z) = 1$, then $z$ is $u$ itself or neighbor. But $u, v$ are non-adjacent on rim. So no $z$ satisfies this unless $z=v$.
Wait, if we know $u$, we can query $Dist(u, k)$ for various $k$.
In the graph, $Dist(u, k) = 1$ iff $k=u$ (since no self loop) or $k$ is adjacent to $u$?
On the modified graph, if chord is $(u, v)$, then $u$ and $v$ are connected by edge. So $Dist(u, v) = 1$.
Other neighbors of $u$ on rim are $u-1, u+1$. Their distance to $u$ is also 1.
So querying $Dist(u, k)=1$ gives neighbors. But $u-1, u+1, v$. We have 3 candidates. We need to distinguish $v$ from neighbors.
Actually, neighbors of $u$ are indistinguishable locally?
Distance to $u$: $u-1 \to 1$, $u+1 \to 1$. $v \to 1$.
Distance to $v$: $v-1 \to 1$, $v+1 \to 1$, $u \to 1$.
Distance to $x$ (neither): depends.

Let's assume we can find $u$. How?
Suppose we query $(1, k)$.
If $k$ is far from $u, v$, then $Dist(1, k)$ is "normal".
If we vary $k$, we can detect the anomaly.
Since we don't know $n$ relative to positions well, maybe we should probe around the midpoint of the current range?
Wait, if we treat the cycle as an interval $[1, n]$ modulo $n$.
We can try to map out the "shortcuts".
But we only have 500 queries. We need a very efficient strategy.
Maybe we can find $u$ and $v$ independently?
Wait, if we assume the chord is "small" relative to $n$? No, it can be anywhere.
However, note the scoring function favors fewer queries. We want $\approx 100$ queries total if possible, or close to 50 worst case?
Wait, if we use binary search on the whole circle, we can find $u$.
But we don't know which side of the circle has the chord.
Actually, since the chord connects $u$ and $v$, the "distortion" in distance is highest for pairs $(x, y)$ that straddle the gap created by the chord.
Actually, if we pick arbitrary distinct $x, y$, measure $Dist(x, y)$.
Compare with expected $C(x, y)$.
If $Dist(x, y) < C(x, y)$, the chord helped.
When does the chord help?
It helps if the path $x \to \dots \to u \to v \to \dots \to y$ is shorter than direct rim path.
This typically happens when $x$ is "close" to one endpoint and $y$ is "close" to the other, on the "other side" of the cycle.
Wait, actually, it helps if $x$ and $y$ are separated by the "short" distance on the cycle that doesn't cover the chord?
Let's trace.
The cycle has two paths between $u, v$: Path A (length $L_A$) and Path B (length $L_B$). $L_A + L_B + 2 = n$. (assuming vertices $u, v$ distinct).
Usually $L_A, L_B > 0$.
Any node $z$ is on Path A or Path B.
If we want to connect a node $x \in \text{Path A}$ and $y \in \text{Path B}$.
Rim path $A$ goes $x \to \dots \to u \to \dots \to y$ ? No.
Rim path goes $x \to \dots \to y$.
If $x$ is on Path A and $y$ is on Path B, then traversing from $x$ to $y$ along rim goes through $u$ OR $v$ (depending on which arc is Path A).
Say Path A is $u \to \dots \to v$ (clockwise). Path B is $v \to \dots \to u$ (clockwise).
If $x$ is on A, $y$ on B.
One rim path: $x \to v$ (end of A) $\to \dots \to y$. Cost $C(x, v) + C(v, y)$.
Wait, if $x$ on A and $y$ on B, then moving clockwise from $x$ hits $v$ then $y$.
Cost is $C(x, v) + C(v, y)$.
The chord connects $u, v$.
Path using chord: $x \to \dots \to u \to v \to \dots \to y$.
Cost $C(x, u) + 1 + C(v, y)$.
So we compare $C(x, v) + C(v, y)$ vs $C(x, u) + 1 + C(v, y)$.
This simplifies to $C(x, v)$ vs $C(x, u) + 1$.
Since $x$ is on Path A (between $u$ and $v$), $C(x, v) = (x - u)_{A} + L_A$? No.
If $x$ is between $u$ and $v$ on the arc that does *not* use the chord? No, the chord replaces the edge $(u, v)$.
Let's define the vertices $u, v$.
Arc 1: $u, u+1, \dots, v$. Length $d = v-u$ (indices mod $n$).
Arc 2: $v, v+1, \dots, u$. Length $n - d$.
Assume $d < n/2$. (WLOG, or just one is shorter).
For any $x \in \text{Arc 1}, y \in \text{Arc 2}$.
Shortest rim path between $x, y$ passes through the junction?
Wait, Arc 1 and Arc 2 share edges? No, they partition the vertices.
Arc 1 vertices: $V_1$. Arc 2 vertices: $V_2$.
Actually, $u \in V_1 \cap V_2$.
Let's simplify. Just imagine line segment $[u, v]$ on cycle.
If we query $(x, y)$ where $x \in [u, v]$ and $y \notin [u, v]$.
Direct distance on cycle (going the short way usually) might involve crossing $u$ or $v$?
Wait, if $x, y$ are just arbitrary points.
The key is that the chord acts as a teleporter.
If we query $(x, y)$ and $Dist(x, y) < \min(x, y, n - \min(x, y))$, we detected a shortcut.
The shortcut implies one of $u, v$ is between $x$ and $y$ (in terms of the rim path taken).
Actually, if we query $(x, y)$, the distance drops iff the chord bridges a gap between $x$ and $y$ on the rim.
Specifically, if $u$ and $v$ are the endpoints, the chord effectively shortens the cycle distance between $u$ and $v$.
Wait, does it shorten distances between arbitrary points?
Yes. If $x$ is close to $u$ and $y$ is close to $v$ (and $u, v$ are on opposite sides of the "rim separation"), the path $x \to u \to v \to y$ might be shorter than $x \to y$ directly.
Actually, if $n$ is large, say $n=1000$. $u=1, v=500$.
$x=2, y=501$.
Rim dist: $2 \to 501$ is $499$.
Chord path: $2 \to 1 \to 500 \to 501$. Cost $1 + 1 + 1 = 3$.
Huge reduction!
So if we pick points "across" the chord, distance is very small.
If we pick points "along" the chord? $x=2, y=3$. Rim dist 1. Chord not needed.
So, we should try to find the "crossing" region.
The region where distances are anomalously small corresponds to pairs $(x, y)$ where $x$ is near $u$ (one side) and $y$ is near $v$ (other side).
Wait, $u$ and $v$ are endpoints.
The chord divides the cycle into two arcs. Let's call them $S_1$ and $S_2$.
If $x, y$ are both in $S_1$, and their distance on the cycle is short (i.e. they are close to each other), the chord (connecting boundary of $S_1$) is unlikely to help.
If $x \in S_1$ and $y \in S_2$, the chord might bridge them.
Actually, the shortest path between $x \in S_1$ and $y \in S_2$ is $\min(C(x, y), C(x, u) + 1 + C(v, y), C(x, v) + 1 + C(u, y))$.
In the example $u=1, v=500$. $S_1 = [1, 500]$. $S_2 = [500, 1]$.
$x=2 \in S_1, y=501 \in S_2$.
$C(x, u) = 1$. $C(v, y) = 1$. Chord path cost $1+1+1=3$.
Rim path $C(2, 501) = 499$ (if we go $2..501$).
Or via other side: $C(2, 501) = 1000 - 499 = 501$.
Min is 499. So 3 is better.
So for pairs separated by the chord, distance is small.

Strategy:
We can search for the chord endpoints by looking for regions where $Dist(x, y)$ is consistently small.
However, $Dist(x, y)$ depends on $x, y$.
We can fix $x$ and vary $y$.
Fix $x=1$.
Vary $y$ from $1$ to $n$.
Plot $f(y) = Dist(1, y)$.
In a pure cycle, $f(y)$ goes $1, 2, 3, \dots, n/2, n/2-1, \dots$. Peak at $n/2$.
With chord $(u, v)$, suppose $u, v$ are far from 1.
Then $f(y)$ will still peak roughly at $n/2$?
Wait. If chord connects $u, v$ such that they split the cycle.
Actually, if $u, v$ are close to each other (e.g. $u=1, v=2$), then it's just one extra edge. But non-adjacent constraint prevents this. Min distance 2 between chord endpoints on rim.
If chord is $(1, 500)$.
$f(y) = Dist(1, y)$.
If $y \in [1, 500]$, $Dist(1, y)$ along rim is $y-1$.
Is chord path better? $1 \to 500 \to y$?
$1 \to 500$ is 1. $500 \to y$ is $500-y$. Total $501-y$.
Compare $y-1$ vs $501-y$.
Intersection $y-1 = 501-y \implies 2y=502 \implies y=251$.
For $y \in [1, 251]$, $y-1 < 501-y$, so $y-1$ is optimal.
For $y \in [252, 500]$, $501-y < y-1$. Optimal is via chord.
So $Dist(1, y)$ would follow the rim for small $y$, then drop towards $v$?
Wait, at $y=251$, dist $\approx 250$. At $y=500$, dist $\approx 1$.
Normally $Dist(1, 500) \approx 250$. Now it is $1$ (actually 1, since connected directly by chord? No, $u, v$ are endpoints).
If $u=1$, then $Dist(1, 500) = 1$.
In the cycle, $Dist(1, 500) = 250$ (if $n=1000$).
So the curve $f(y)$ dips sharply near the other endpoint $v$.
But we assumed $u=1$.
Generally, $Dist(1, y)$ looks like:
Increasing distance from 1 until reaching some point, then maybe decreasing again?
Or rather, it stays increasing until the midpoint of the cycle? No, the chord creates shortcuts.
If we measure $Dist(1, y)$, we can identify $y=v$ easily?
If $1$ is $u$, then $Dist(1, v) = 1$. But if $1$ is not $u$, $Dist(1, v)$ might be 1 if $u$ is near 1? No.
If $1$ is not $u$, $Dist(1, v)$ is $C(1, u) + 1$ or something.

Let's try a binary search approach to find $u$ and $v$.
Can we check if a vertex is an endpoint?
Check if $Dist(u, v) = 1$.
If we query $(x, x+1)$, we get 1 always.
We need to find $x$ such that $Dist(x, y) = 1$ for some $y \neq x \pm 1$ (mod n).
Wait, $u$ and $v$ are not adjacent on rim. But they are adjacent in graph.
So $Dist(u, v) = 1$.
Neighbors of $u$ on rim are $u-1, u+1$. Their distance to $u$ is 1.
So if we find $u$, we find 3 nodes at distance 1: $u-1, u+1, v$.
We can distinguish $v$ from $u-1, u+1$?
Neighbors of $u$ in graph: $u-1, u+1, v$.
In graph, $u$ has degree 3? No.
In original cycle, degree 2. Add chord, degree of $u, v$ becomes 3. Others remain 2.
So if we pick a random node $x$, calculate distances to all its neighbors.
But we don't know neighbors. We have to guess.
If we find a node with 3 neighbors (distance 1), that's a good candidate for $u$ or $v$.
But we can't iterate all neighbors.
We can probe.
Is it possible to find a node with high degree?
Only $u$ and $v$ have degree 3.
All others have degree 2.
But in terms of shortest path, this distinction isn't immediately visible.
However, the "metric space" looks different.

Algorithm idea:
We want to find $u$ and $v$.
Let's try to locate the chord by narrowing the "cut".
Consider $Mid(n/2)$.
Actually, if we just perform a "ternary search" style logic?
Wait, if we query $(x, y)$ and we get distance $d$.
We want to maximize $d$?
Max distance in a graph with chord is usually much smaller than cycle.
Wait, if $n$ is large, diameter of cycle is $n/2$.
With chord, diameter is at most $n/3$? Or even smaller.
Specifically, if chord is center, max distance is $\approx n/3$.
If chord is edge, max distance is $\approx n/2$.
Wait, no. If chord is near $1, 2$, then for $x, y$ near $n$, distance is $\approx n/2$ (halfway to chord).
So the maximum distance between any pair decreases.
The maximum distance is minimized when the chord is "in the middle" of the largest arc?
Wait, let's find a property.
Let $D_{max}$ be the maximum query result we can get.
If we query $(1, n/2)$, result is $d_1$.
If we query $(1, n/4)$, result is $d_2$.
If we query $(n/2, 3n/4)$, etc.

Wait, looking at the constraints again. 500 queries.
We have plenty of budget.
The simplest thing is to perform a binary search on the interval $[1, n]$ to find the location of the chord endpoints relative to some point.
Let's fix $x=1$.
We want to find $u, v$.
Consider the function $g(k) = Dist(1, k)$.
In cycle, $g(k)$ is concave (increases then decreases).
With chord, it might change shape.
But specifically, if we query $(1, k)$, we might detect if the path went through the chord.
If we query $(1, k)$, and find $Dist(1, k)$ is very small compared to $C(1, k)$, it suggests $1, k$ are on "opposite" sides of the chord.
Wait, $C(1, k)$ is known.
Let's just find the "midpoint" between $u$ and $v$ on the longer arc?
This seems hard.

Another angle:
We have $u, v$.
The graph is formed by attaching an edge to the cycle.
If we pick a vertex $x$, and measure distances to $x+1, x+2, \dots, x+n/2$, we can reconstruct the topology.
But we can't afford $O(n)$.
We can do a "search".
Suppose we want to find $u$.
If we query $Dist(x, x+d)$.
If the result is equal to $d$ (for small $d$), then no chord is involved.
If result $< d$, chord involved.
For $x=1$.
Query $Dist(1, 2)$. Cycle says 1. Chord says 1.
Query $Dist(1, 3)$. Cycle 2. Chord might be 1 if chord connects 1-? No, chord connects $u, v$.
If $u=1$, then $Dist(1, 3)$ could be 2 (via 2) or 1 (if $v=3$?? No, non-adjacent).
If $u=1, v=3$ is impossible (adjacent).
Wait, chord connects non-adjacent. So $v \ge u+2$ (indices mod n).
So for any $x$ close to $u$ or $v$, distances are consistent with cycle.
Only distances involving "jumping" across the chord show deviation.
Actually, if we measure $Dist(1, k)$ for $k=2, 3, \dots$.
Until $k$ gets close to $n$ or wraps around.
The deviation $Dist(1, k) < k-1$ happens when the shortest path leaves the "local" cycle segment and takes the chord.
This happens when $u$ and $v$ are effectively "surrounding" the path $1 \dots k$.
This feels like we can locate the endpoints by seeing where the "distance profile" deviates from linearity.
But the profile is global.
Let's look at the problem statement constraints again.
Score depends on $Q^2$. We want $Q$ small.
Is there a randomized algorithm? No, deterministic usually expected.
But maybe $O(\log^2 n)$ or $O(\log n)$.
Wait, we can binary search the location of the chord!
Wait, if we pick two points $x, y$ in the cycle.
If the chord intersects the path between them? No, chord is a single edge.
But we can check if a specific edge is the chord?
No, we query vertices.

Let's revisit the "neighbors" idea.
If we query $Dist(1, x)$ and $Dist(2, x)$.
In a cycle, $Dist(1, x)$ increases by 1 as $x$ moves from $1$ to right, and similarly for $Dist(2, x)$?
Not really, because we take shortest path.
Let's assume the chord is $(u, v)$ with $u < v$ and $v-u > 1$.
The graph is symmetric under rotation/reflection.
Let's fix $u=1$ temporarily (just a coordinate shift).
We are trying to find $v$.
The cycle is $1, 2, \dots, n$. Edge $(n, 1)$.
Chord $(1, v)$.
For any $x$, $Dist(1, x)$.
If $x$ is close to $v$, $Dist(1, x)$ will be small.
If $x$ is far from $v$, $Dist(1, x)$ follows the cycle.
Specifically, $Dist(1, x) = \min(C(1, x), C(1, 1) + 1 + C(v, x)?) = \min(x, n-x, 1 + (v-x))$ (assuming appropriate order).
Wait, $C(1, x)$ is distance on rim.
If chord is $(1, v)$, then we have a shortcut.
Path via chord from 1 to $x$:
Option 1: $1 \to v \to \dots \to x$. Cost $1 + C(v, x)$.
Option 2: $1 \to \dots \to x$. Cost $C(1, x)$.
So $Dist(1, x) = \min(C(1, x), 1 + C(v, x))$.
If we plot $f(x) = Dist(1, x)$.
For $x$ such that $C(1, x) < 1 + C(v, x)$, $f(x) = C(1, x)$.
For others, $f(x) = 1 + C(v, x)$.
Since $C(v, x) \approx n/2$ for many $x$? No.
$C(v, x)$ is small if $x$ is close to $v$.
$C(1, x)$ is small if $x$ is close to $1$.
So near $x=1$, $f(x)$ follows cycle.
Near $x=v$, $f(x)$ is small (bounded by 1).
So $f(x)$ has a "valley" around $x=v$.
But $f(x)$ is also constrained by $C(1, x)$.
Wait, if $x$ is halfway between $1$ and $v$ on the long path?
Actually, the equation is $\min(x, 1 + |x-v|)$.
Assuming $v > 1$.
Case 1: $1 < x < v$.
$C(1, x) = x-1$.
$C(v, x) = v-x$.
So $Dist(1, x) = \min(x-1, 1 + v-x)$.
Equality when $x-1 = 1 + v-x \implies 2x = v+2 \implies x = v/2 + 1$.
Before that point, $x-1$ dominates (linear increase). After that, $1+v-x$ dominates (linear decrease).
Case 2: $x$ is on the other side (near $n$).
$C(1, x) = n-x+1$ (path $1-n-\dots-x$).
$C(v, x) = n-v+x$ (path $v-(v+1)\dots n-(1)\dots x$? No. $v \to n \to 1 \dots \to x$? No, path $v \to \dots \to x$ not passing through 1).
Wait, $C(v, x)$ on rim is $\min(|v-x|, n-|v-x|)$.
For $x$ close to $1$ (large index $n$), $v$ is somewhere.
Actually, the term $1 + C(v, x)$ represents going $1 \to v$ then to $x$.
If $x$ is close to 1 on the "other" side (via $n$), then $C(v, x) \approx n/2$.
Then $1 + C(v, x)$ is large.
But $C(1, x) = n-x+1$ (small). So min is $C(1, x)$.
So $f(x)$ looks like:
Starts at 0 (at $x=1$). Increases linearly.
At $x \approx v/2 + 1$, turns around and decreases linearly towards $v$.
Reaches 1 at $x=v$ (actually $1 + C(v, v) = 1$).
Then continues to decrease?
For $x > v$:
$C(1, x)$ increases as $n-x+1$ (wait, $C(1, x)$ is distance on rim).
Wait, $C(1, x)$ increases as $x$ goes away from 1. Max at $n/2$.
$1+C(v, x)$ increases as $x$ goes away from $v$.
Around $v$, $1+C(v, x)$ increases (symmetric).
Around 1, $1+C(v, x)$ is large.
So the function $f(x)$ will have a local maximum?
No.
Let's check the shape again.
$x$ moves $1 \to \dots \to v$.
$C(1, x) = x-1$.
$1 + C(v, x) = 1 + (v-x)$.
Intersection at $x = (v+2)/2$.
At $x=v$, $f(v) = 1$.
Then for $x > v$ (moving towards $n$).
$C(1, x)$ grows (wraps around eventually).
Wait, $C(1, x) = n-x+1$ if going backwards? No, $x$ is index. $1$ to $x$ forward is $x-1$. Backward is $n-(x-1)$.
Minimum is taken.
But here we are considering the specific formula $\min(x-1, 1+v-x)$ for the segment between $1$ and $v$.
For $x$ past $v$, say $x > v$.
$C(1, x) = \min(x-1, n-(x-1))$.
$1 + C(v, x) = 1 + \min(x-v, n-(x-v))$.
Note $C(1, x) \approx x$. $1+C(v, x) \approx 1+(x-v) = x - v + 1$.
So $1 + C(v, x) < C(1, x)$ is possible if $x-v+1 < x-1 \implies -v+1 < -1 \implies v > 2$.
So yes, for $x$ between $v$ and some point, the chord path is better?
Wait, $1 + (x-v) < x-1$. True.
But we must check the *shortest* rim distance too.
If $x$ is very close to $v$, $1+C(v, x) = 1$. $C(1, x)$ is large.
So $f(x)$ drops to 1 at $v$ and stays low?
Actually, at $x=v$, $f(v)=1$.
For $x=v+1$: $C(1, x) \approx v$. $1 + C(v, v+1) = 2$. So $f(v+1)=2$.
So $f(x)$ goes $0 \to \text{peak} \to 0 \to \text{something}$.
Wait, if $f(x)$ reaches 0 at $x=1$ and 1 at $x=v$.
In between, it forms a valley?
No. At $x=1$, $f(1)=0$.
At $x \approx v/2$, it starts dropping towards $v$ via chord shortcut.
Wait, if $f(x)$ represents distance from 1.
Then $f(1)=0$.
$f(x)$ increases initially (along cycle).
Then the chord path becomes available?
Wait, the chord path is $1 \to v \to \dots \to x$.
If $x$ is between $1$ and $v$ (on the shorter arc), the path $1 \to \dots \to x$ (direct) is cheaper than $1 \to v \to \dots \to x$ (longer detour).
Why? $1 \to v$ costs 1. $v \to x$ costs $v-x$. Total $1 + v - x$.
Direct is $x-1$.
Which is smaller? $x-1$ vs $v-x+1$.
Sum $x-1 + v-x+1 = v$.
Intersection is $x = (v+2)/2$.
So for $x \in [1, (v+2)/2]$, direct is best. $f(x) = x-1$.
For $x \in [(v+2)/2, v]$, chord is best. $f(x) = v-x+1$.
At $x=v$, $f(v)=1$.
So in the interval $[1, v]$, $f(x)$ looks like $\Lambda$ (tent) shape? No, $\wedge$ inverted?
Start 0, go up, reach peak, go down to 1.
Then for $x > v$ (up to $n$).
Direct path $1 \to \dots \to x$ length $x-1$.
Chord path $1 \to v \to \dots \to x$. Length $1 + \text{dist}(v, x)$.
$\text{dist}(v, x)$ on rim is $x-v$.
So $1 + x - v$.
Is $x-1 < 1+x-v$?
$-1 < 1-v \implies v < 2$. False since $v \ge 3$.
So chord path $1+x-v+1 = x-v+2$ vs $x-1$?
Wait.
$1 \to v$ is edge. $v \to x$ is edge on rim.
If $x > v$, rim path $v \to x$ is $x-v$.
Total chord path $1 + x - v$.
Direct rim path $1 \to \dots \to x$ is $x-1$.
Compare $x-v+1$ vs $x-1$.
$x-v+1 < x-1 \iff -v+1 < -1 \iff v > 2$.
So for $x > v$, chord path is BETTER?
Wait, this implies $Dist(1, x) = x - v + 1$.
So after $v$, the distance from 1 keeps growing, but slope is 1 (same as before).
So $f(x)$ is continuous with slope 1 everywhere?
Wait, if $f(x)$ increases with slope 1 for all $x$, then $f(x) = x-1$ is the solution?
Wait, $C(1, x)$ is the cycle distance. $C(1, x)$ is $\min(x-1, n-(x-1))$.
The chord makes it $1 + C(v, x)$.
So $f(x) = \min(C(1, x), 1+C(v, x))$.
If we consider the range where $C(1, x) = x-1$ (i.e., $x$ is close to 1, $x \le n/2$).
And $C(v, x) = x-v$ (i.e. $x$ is close to $v$ and $x > v$, assuming $v$ not wrapped).
Then $f(x) = \min(x-1, 1+x-v)$.
Since $v > 2$, $1+x-v < x-1$.
So $f(x) = 1+x-v$.
This means once we pass $v$, the distance from 1 is determined by the chord path?
Wait, $f(v) = 1$.
$f(v+1) = 2$.
So $f(x)$ goes $0 \to \dots \to \text{peak} \to \dots \to 1 \to 2 \to \dots$.
This means $f(x)$ increases, decreases, then increases again.
It's not monotonic.
Wait, if $v$ is the position of the chord endpoint.
We found $f(v) = 1$.
But for any other neighbor of $v$, distance might be different.
Wait, if $x$ is $v+1$.
$Dist(1, v+1)$.
Path $1 \to v \to v+1$. Cost 2.
Rim path $1 \to 2 \to \dots \to v+1$. Cost $v$.
Clearly 2 is smaller.
So $f(v+1)=2$.
Is it possible that $f(x) = 2$ for multiple $x$? No.
So we can identify $v$ if we know $u=1$.
How?
$f(v) = 1$.
$f(v+1) = 2$.
$f(v-1) = 2$ (by symmetry? No. $v-1$ is between 1 and $v$.
Path $1 \to v-1$ via rim is $v-2$.
Path $1 \to v \to v-1$ is $1+1=2$.
So $f(v-1) = \min(v-2, 2)$. If $v > 3$, it's 2.
So $f(v-1)$ is also small.
Basically, the function $f(x)$ has a minimum at $x=v$ (value 1).
Wait, does it have a minimum at $x=1$ (value 0)? Yes.
So we have minima at $1$ and $v$.
If we don't know $u=1$, we just know there is a minimum at $u$ (distance 0).
We know distance at $x$ is 0.
So if we query $(x, y)$ and get 1, it might mean $x=y$ (dist 0? no 0 edges? Problem says number of edges. Loop dist is 0? Or not allowed? Queries $x \neq y$).
Wait, if we query $(1, 1)$, output is 0.
If we query $(1, v)$, output is 1.
So if we can find a vertex $y$ such that $Dist(x, y)=1$ and $y \neq x \pm 1$, then $x$ and $y$ are the chord endpoints.
The problem reduces to: Find a pair $(u, v)$ such that $Dist(u, v)=1$.
But all neighbors have $Dist=1$.
However, $u$ has 3 neighbors ($u-1, u+1, v$). $v$ has 3 neighbors. Others have 2.
So if we can find a vertex with 3 neighbors, we found $u$ or $v$.
How to count degree?
Degree $k$ means there exists $k$ vertices $y$ such that $Dist(x, y)=1$.
For non-endpoint $x$, $y \in \{x-1, x+1\}$.
For endpoint $x$, $y \in \{x-1, x+1, v\}$.
So if we query $Dist(x, k)$ for $k=x+1, x+2, \dots, x+n/2$.
If we find a "third" node $y$ at distance 1, we identify $x$ as an endpoint.
We need to efficiently find such $y$.
For a generic $x$, we only expect to find $x+1$ (dist 1) and maybe nothing else in the range $[2, n/2]$?
Wait, if we scan $x+1, x+2 \dots$.
If $x$ is not an endpoint, then neighbors are $x+1, x-1$.
Checking $x+1$ gives 1.
Checking $x+2$ gives 2.
Checking $x+3$ gives 3.
If $x$ is endpoint, checking $x+1$ gives 1. Checking $x+2$ gives 2. ...
Eventually we hit $v$.
If $v$ is within the scanned range, we find a vertex with distance 1.
If $v$ is far, we never find it.
Since $n$ is large, $v$ could be anywhere.
But we can use binary search!
We want to find $v$ given $u$.
Assume we fixed $u=1$ (we can rotate? No).
We can't assume $u=1$.
But maybe we can *find* $u$ and $v$ simultaneously.
Actually, if we fix $x=1$.
The function $g(k) = Dist(1, k)$ behaves like:
$0$ at $k=1$.
Increases.
Then there's a local minimum at $k=v$?
Wait, earlier analysis suggested $f(x)$ goes $0 \to \text{peak} \to 1 \to \dots$.
Wait, did it?
If $x > v$.
$f(x) = 1 + x - v$ (from chord path).
And $f(x) = \min(x-1, 1+x-v)$.
Since $v \ge 3$, $1+x-v < x-1$.
So $f(x) = x-v+1$.
This function is increasing for $x > v$.
Wait, but $f(x)$ also competes with the path wrapping around $n$.
$f(x) = \min(x-v+1, n-(x-1), \dots)$.
$n-(x-1) = n-x+1$.
Intersection $x-v+1 = n-x+1 \implies 2x = n+v \implies x = (n+v)/2$.
So $f(x)$ increases linearly until $x=v$, drops to 1 at $v$? No.
Let's re-evaluate.
Segment $1 \to v$.
$x \in [1, v]$.
$C(1, x) = x-1$.
$1+C(v, x) = 1 + (v-x)$.
Min is $\min(x-1, 1+v-x)$.
Values: at $x=1 \to \min(0, v) = 0$.
At $x=v \to \min(v-1, 1) = 1$.
Shape: Triangle peaking at $x=(v+2)/2$ with value $\lfloor (v-1)/2 \rfloor$.
Wait, peak value is approx $v/2$.
So in range $[1, v]$, $f(x)$ goes $0 \nearrow v/2 \searrow 1$.
Segment $v \to n$.
$x \in [v, n]$.
$C(1, x) = \min(x-1, n-(x-1))$.
$1+C(v, x) = 1 + \min(x-v, n-(x-v))$.
Since $v$ is usually closer to $x$ than 1 for $x > v$ (if $v$ not near $n$),
$x-v$ vs $x-1$. $x-v < x-1$.
So $1+x-v < x-1$.
So $f(x)$ follows $1+x-v$.
Wait, does $1+x-v$ beat $n-(x-1)$?
$1+x-v < n-x+1 \iff 2x < n+v$.
So up to $x = (n+v)/2$, $f(x) = 1+x-v$.
After that, $f(x)$ switches to $n-x+1$?
Or maybe $1+C(v, x)$ switches?
$C(v, x)$ could switch to $n-(x-v)$.
Anyway, the important feature is that $f(x)$ is NOT monotonic.
It goes $0 \nearrow \text{peak} \searrow 1 \nearrow \text{peak} \dots$.
Specifically, $f(x)$ has a minimum at $x=1$ (0) and a minimum at $x=v$ (1).
Wait, is $x=1$ a minimum? $f(1)=0$.
Is $x=v$ a minimum? $f(v)=1$.
Is it possible that $f(x)$ has *no* other local minimum?
The value at $v$ is 1.
The neighbors of $v$ are $v-1$ (val $\approx 2$) and $v+1$ (val $\approx 2$).
So yes, $v$ is a local minimum.
However, $x=1$ is also a minimum (global).
But we don't know $u$ or $v$ beforehand.
Wait, we can check $f(x)$ for *all* $x$? No.
But if we sample a few points, we can find the global maximums or minima.
Global max of $f(x)$ occurs near the midpoint of the long arc between $u$ and $v$.
The "valleys" (local minima) are at $u$ and $v$.
$u$ is special because $f(u)=0$. $v$ is special because $f(v)=1$.
So, if we scan the circle, we should find a point with distance 0 (itself) and a point with distance 1 to $u$.
Wait, if we query $Dist(x, y)$, we are measuring mutual distance.
We want to find a pair $(u, v)$ such that $Dist(u, v)=1$ and $Dist(u, u)=0$.
But we don't know $u$.
However, note that $Dist(x, y) \le C(x, y)$.
If we fix $x=1$, and scan $y$.
We will see $f(1)=0$.
We will see $f(y)=1$ at $y=v$ (since $Dist(1, v)=1$).
We will see $f(y)=1$ at $y=u$ (if $u \neq 1$).
Wait, $Dist(1, u)$? If $1$ is an endpoint (say $u=1$), then $Dist(1, u)=0$.
If $u \neq 1$, $Dist(1, u)$ is the distance between two points on the rim.
Unless $1$ is $u$.
Wait, $u, v$ are fixed.
If $u \neq 1, v \neq 1$. Then $Dist(1, u) > 0$. $Dist(1, v) > 0$.
Is it possible $Dist(1, u) = 1$? Only if $u$ is adjacent to 1.
Is it possible $Dist(1, v) = 1$? Only if $v$ is adjacent to 1.
So if 1 is far from $u, v$, then $f(y)$ will only be 1 at $y=v$??
Wait, if $u, v$ are endpoints. $Dist(1, v)=1$ is only guaranteed if $1$ is connected to $v$ by chord? No.
Chord connects $u, v$.
Distance from 1 to $v$ is $1$ IF AND ONLY IF 1 is $u$.
Wait! $u$ and $v$ are the ONLY vertices connected by a chord edge.
In the graph, $u$ is connected to $v$ with weight 1.
Any other pair $(a, b)$ has $Dist(a, b) \ge C(a, b)$.
Wait, $Dist(a, b)$ can be 1 if $(a, b)$ is an edge.
Edges are $(i, i+1)$ and $(u, v)$.
So $Dist(x, y)=1$ implies $(x, y)$ is a rim edge OR $(x, y) = (u, v)$ (or $(v, u)$).
Since we query $x, y$ arbitrarily.
If we query $(x, x+1)$, we always get 1.
This doesn't help identify $u, v$.
BUT, we can check if a query $(x, y)$ returns 1 where $(x, y)$ is NOT a rim edge.
E.g., if we query $(x, x+2)$, we get 2 in cycle. If we get 1, then $(x, x+2)$ must be the chord?
This is a very strong check.
If we can find any pair $(x, y)$ such that $|x-y| \pmod n \notin \{1, n-1\}$ and $Dist(x, y)=1$, then $(x, y)$ is the chord.
How to find such a pair?
In a cycle, $Dist(x, y)$ increases with $|x-y|$ roughly.
With a chord, the distance might drop.
But can $Dist(x, y)$ be 1 for non-neighbors?
Yes, exactly for the chord pair.
So if we find a pair with distance 1 that are not adjacent in indices (modulo n), we found the chord.
Can we do this fast?
We need to check if $Dist(x, y)=1$ for non-adjacent.
Since we don't know $u, v$, we can't just check all.
But we can use a binary search to find if the chord lies in a sector?
Wait, if we fix $x=1$. We want to find $y$ such that $Dist(1, y)=1$ and $y \neq 2, n$.
In the graph, the neighbors of 1 are $2, n, u$ (if $u=1$), $v$ (if $v=1$).
Wait, if $u=1$, neighbors are $2, n, v$.
So if $u=1$, then $Dist(1, v)=1$.
If $v$ is not 2 or $n$. Then we found the chord $(1, v)$.
If $u \neq 1, v \neq 1$. Then neighbors of 1 are only $2, n$.
In this case, for any $y \notin \{2, n, 1\}$, $Dist(1, y) \ge 2$.
Wait, this is crucial.
If we pick $x=1$, and query $Dist(1, y)$.
If we find any $y$ such that $Dist(1, y)=1$, then $y$ MUST be a neighbor of 1.
Neighbors of 1 are $2, n$ (always) and possibly $u, v$ if $1=u$ or $1=v$.
So, if we find a $y$ with $Dist(1, y)=1$ and $y \neq 2, y \neq n$.
Then 1 IS an endpoint of the chord. And $y$ is the OTHER endpoint.
Wait, what if $u=2$?
If $u=2, v=k$.
Neighbors of 1: $2, n$.
$Dist(1, 2)=1$. $Dist(1, n)=1$.
$Dist(1, k)$. Is it 1? No, $k$ is not connected to 1 unless $k=n$ (but chord non-adjacent, so $v \neq n$).
So if 1 is not an endpoint, ALL $y$ such that $Dist(1, y)=1$ are exactly $\{2, n\}$.
If 1 IS an endpoint (say $u$), then neighbors are $u-1, u+1, v$.
Since $u=1$, neighbors are $2, n, v$.
So if $v \notin \{2, n\}$, we will find $v$ by querying $(1, y)=1$.
If $v \in \{2, n\}$, impossible as chord must be non-adjacent.
So, strategy:
1. Query $(1, 2)$ -> returns 1. (Always).
2. Query $(1, n)$ -> returns 1. (Always).
3. Pick some $y \in \{3, \dots, n-1\}$.
   Binary search or random search?
   If we query $y$ and get 1, we found the chord $(1, y)$. Done!
   If we query many $y$ and none return 1, then 1 is not an endpoint.
   Wait, we can just check ONE $y$.
   Say $y=3$.
   Query $(1, 3)$. Result $d_3$.
   If $d_3 = 1$, then chord is $(1, 3)$ (since $1, 3$ not adjacent on rim).
   If $d_3 > 1$, then 1 is likely not an endpoint?
   Wait, if chord is $(u, v)$. Neighbors of 1 are $2, n$.
   If 1 is not $u$ or $v$, then NO node other than 2, n is at distance 1 from 1.
   So if $Dist(1, 3) > 1$, it implies 1 is not an endpoint.
   BUT, if $Dist(1, 3) > 1$, we learn 1 is not $u$ or $v$.
   Now we need to find $u, v$.
   We established $u, v$ are endpoints. They have degree 3 (in terms of rim neighbors).
   But we can't iterate.
   However, if we know 1 is not an endpoint, we know that for every vertex $x$, $Dist(x, y)=1$ has solutions $\{x-1, x+1\}$ ONLY.
   Wait, this property holds for ALL vertices if 1 is not endpoint.
   Actually, the property holds for endpoints too: $\{u-1, u+1, v\}$.
   But if we pick a vertex $x$ randomly, probability that $x$ is endpoint is $2/n$. Small.
   So almost surely, $x$ is not endpoint.
   So $Dist(x, y)=1$ checks are useless to find chord if $x$ is not endpoint.

   Wait, I'm confusing myself.
   We need to FIND $u, v$.
   My previous deduction: If we pick an arbitrary $x$, and find $y$ s.t. $Dist(x, y)=1$ and $|x-y|>1$ (mod $n$), then $x, y$ is the chord.
   How to find such $y$?
   If $x$ is not endpoint, no such $y$ exists.
   So we MUST pick an endpoint to succeed with this method.
   Since probability is $2/n$, this is bad for large $n$.
   
   However, we can USE THE DISTANCE PROFILE.
   Even if $x$ is not endpoint, $Dist(x, y)$ profile tells us something about $u, v$.
   Actually, let's reconsider the "minimum" argument.
   $f(z) = Dist(u, z)$.
   $f(u) = 0$. $f(v) = 1$.
   Neighbors of $u$: $u-1, u+1$ have dist 1. $v$ has dist 1.
   Neighbors of $v$: $v-1, v+1$ have dist 1. $u$ has dist 1.
   If we fix $u$, then $f(z)$ has local minimums.
   
   Let's pivot.
   Can we find $u$ using the "peak" of the distance function from a reference?
   Suppose we query $Dist(1, z)$ for $z \in [1, n/2]$.
   We saw that $f(z)$ goes $0 \nearrow \text{peak} \searrow 1 \nearrow \dots$.
   The global maxima of $f(z)$ occur approximately at the midpoint of the arc NOT covered by the chord?
   No, $f(z) = Dist(1, z)$.
   The maximum possible value in a graph with chord is bounded.
   Without chord, max is $n/2$.
   With chord, max is smaller.
   Specifically, if chord is $(u, v)$.
   Consider $z$ such that $z$ is far from $1, u, v$.
   Wait, $u, v$ split the cycle.
   If $1$ is in one arc, say arc $A$. $u, v$ are boundaries.
   $z$ is in arc $A$ or $B$.
   If $z$ is in $A$, far from $u, v$. $f(z) \approx C(1, z)$.
   Wait, if $z$ is very close to $1$?
   
   Let's look at the sample.
   $n=6$. Output `? 1 5` -> `2`.
   Wait, distance between 1 and 5 in cycle of 6.
   Rim edges: (1,2), (2,3), (3,4), (4,5), (5,6), (6,1).
   1 to 5.
   Path 1-6-5 len 2.
   Path 1-2-3-4-5 len 4.
   Shortest is 2.
   Sample query `? 1 5` returns 2. Consistent with cycle.
   Next query `? 2 4` returns 1.
   Rim 2 to 4. 2-3-4 len 2.
   Wait, 2 to 4 returned 1.
   This implies there is an edge (2, 4) or path of len 1.
   Since (2, 4) is not a rim edge, it must be the CHORD.
   So the chord is (2, 4).
   Sample output `! 4 2`.
   Then it queries again? No, `!` is outputting answer.
   Wait, the sample interaction shows:
   `? 1 5` -> 2
   `? 2 4` -> 1
   `! 4 2` -> 1
   Wait, why query again after `? 2 4` -> 1?
   Ah, the sample output is generated by the user.
   Wait, the sample input/output provided:
   Input:
   2 (test cases)
   6
   2
   1
   4
   2
   1
   
   Output:
   ? 1 5 (case 1 n=6) -> 2
   ? 2 4 (case 1 n=6) -> 1
   ! 4 2 (guess) -> 1 (correct)
   
   Wait, the next lines correspond to case 2 (n=2? No n>=4).
   Ah, sample input is weird.
   "Input" header says T=2.
   Case 1: 6.
   Case 2: 2? Constraint says $n \ge 4$.
   Maybe the sample input is just illustrative of format or bad copy-paste?
   Or maybe $n=2$ case is ignored?
   Wait, $n=2$ violates $n \ge 4$.
   Ah, maybe the integers are:
   Line 1: 2 (T)
   Line 2: 6 (n)
   Line 3: 2
   Line 4: 1
   Line 5: 4 (n) ??
   Wait, if T=2.
   Case 1: n=6.
   Queries happen. Guess happens. r=1.
   Case 2: n=?
   Wait, the sample output shows queries.
   Let's trace carefully.
   The input block:
   2
   6
   2
   1
   4
   2
   1
   
   This looks like T=2.
   Case 1: n=6.
   Program outputs `? 1 5`. Interactor returns 2.
   Program outputs `? 2 4`. Interactor returns 1.
   Program outputs `! 4 2`. Interactor returns 1. (Correct).
   Case 2: n=2. (Contradicts $n \ge 4$).
   Wait, maybe the sample input is messed up in the prompt text or I'm misreading.
   "Sample Input: 2 \n 6 \n 2 \n 1 \n 4 \n 2 \n 1"
   It could be interpreted as:
   T=2
   Test 1: n=6. Output 2 (maybe answer?), 1 (maybe answer?)
   Wait, sample output shows the program interacting.
   The integers in input are just the problem data?
   No, interactive problems usually don't give answers in input.
   The sample input likely shows the parameters.
   But 2 (after 6) might be the answer for a query? No, input is read from judge.
   The Sample Output shows what the program prints.
   Okay, ignore sample input for structure, focus on the interaction described.
   In sample output, program queried (1, 5), got 2.
   Queried (2, 4), got 1.
   Guessed (4, 2).
   Correct.
   
   So my hypothesis that `Dist(2, 4) = 1` implies chord is (2, 4) is valid.
   
   Key Insight:
   If we can find ANY pair $(x, y)$ such that $Dist(x, y)=1$ and they are not rim neighbors, we have found the chord.
   To do this efficiently:
   We can maintain a "candidate" range for the chord.
   Initially range $[1, n]$.
   If we pick $x=1$.
   Query $y=2$. $Dist(1, 2)=1$. (Rim neighbor).
   Query $y=n$. $Dist(1, n)=1$. (Rim neighbor).
   We need to check if any other $y$ gives 1.
   But checking all $y$ is $O(n)$.
   However, if $1$ is not an endpoint, no such $y$ exists.
   If $1$ is an endpoint, say $u$, then $y=v$ gives 1.
   So we need to check if $1$ is an endpoint.
   We can't check all $y$.
   But maybe we can check $y$ in a specific order or using binary search.
   Notice: If $1$ is an endpoint $u$, then $Dist(1, k) = 1$ has solution $k=v$.
   $Dist(1, k)$ function for $u=1$ looks like:
   Near 1: increases.
   Near $v$: drops to 1.
   Between 1 and $v$ (along the rim path): $Dist(1, k)$ is monotonic?
   Wait, if $k$ moves from 1 to $v$.
   Path $1 \to \dots \to k$.
   Does chord help?
   If $k$ is close to 1, path is $1 \to k$ (len $k-1$). Chord path $1 \to v \to \dots \to k$. Len $1 + (v-k)$.
   Intersection at $k \approx v/2$.
   If $v$ is very far (e.g. $n/2$), then intersection is far.
   If $v$ is close to $n$ (so $u, v$ wrap around), then $k$ path is $1 \to \dots \to k$ and $1 \to n \to \dots$.
   Basically, if we start scanning $k=2, 3, \dots$.
   $Dist(1, k)$ will be $k-1$ as long as $k-1 < 1 + (v-k)$ (if $v > k$).
   If we encounter a drop in $Dist(1, k)$, we know the chord is active.
   Wait, does $Dist(1, k)$ ever drop below $k-1$?
   $Dist(1, k) \le k-1$. Equality holds if rim path is shortest.
   Drop happens if chord path is shorter.
   Condition: $1 + C(v, k) < C(1, k)$.
   Assuming $k < v$ (and on same arc).
   $1 + v - k < k - 1 \implies 2 < 2k - v \implies k > (v+2)/2$.
   So $Dist(1, k)$ stops being $k-1$ when $k$ passes the midpoint between 1 and $v$.
   It drops.
   Specifically, for $k > (v+2)/2$, $Dist(1, k) = v-k+1$.
   As $k$ approaches $v$, $Dist(1, k)$ approaches 1.
   Wait, if $k=v$, $Dist(1, v)=1$.
   So, if $1$ is an endpoint, $Dist(1, k)$ will exhibit a "V-shape" or "checkmark" descent?
   Actually, it descends linearly towards $v$.
   $k-1$ is increasing.
   $v-k+1$ is decreasing.
   So $f(k) = \min(k-1, v-k+1)$.
   This function looks like $\wedge$ (tent).
   Starts at 0, goes up to $v/2$, then goes down to 1 at $v$.
   So if we scan $k=1, 2, \dots$.
   Values: $0, 1, 2, \dots, \approx v/2, \dots, 1$ (at $k=v$).
   So we see values increasing, then decreasing.
   Crucially, we see the value 1 appearing at TWO places?
   At $k=2$ (neighbor).
   At $k=v$ (endpoint).
   If $v \neq 2, n$.
   Also we check for neighbors on the other side $n, n-1$.
   If 1 is an endpoint, we will see $f(k)=1$ at $k=v$.
   We will ALSO see $f(k)=1$ at $k=n-1$? No, that depends on geometry.
   Wait, $f(1)=0$.
   $f(2)=1$.
   $f(v)=1$.
   $f(n)=1$ (always).
   $f(n-1)$? If $n-1$ is far from 1.
   Wait, $f(n)=1$ is due to rim edge $(n, 1)$.
   So we have trivial 1s.
   We need a non-trivial 1.
   $k=v$ is non-trivial (unless $v=n$ or $v=2$).
   So if $v \in [3, n-1]$.
   Then $f(v)=1$ is a signal.
   Can we find $v$?
   We observe $f(k)$.
   For $k < v$, $f(k)$ generally increases (until midpoint).
   So $f(2)=1, f(3)=2, f(4)=3 \dots$ until turn-around.
   Then it decreases.
   So $f(k)$ is unimodal (roughly) on $[1, v]$.
   If we find a $k$ such that $f(k)$ is smaller than expected, or simply smaller than neighbors?
   Wait, if $v$ is large, the increase continues for long time.
   If we stop at $k=n/2$, we are at the peak (approx $v/2$ or $n/2$).
   If we continue scanning to $v$, we will see $f(k)$ decrease.
   When does it decrease?
   $v-k+1 < k-1$.
   The crossover is at $k \approx v/2$.
   So $f(k)$ increases then decreases.
   This holds if $1$ is an endpoint.
   What if $1$ is NOT an endpoint?
   Then $f(k)$ behaves like cycle distance.
   $f(k) = \min(k-1, n-(k-1))$.
   It increases up to $n/2$, then decreases.
   The difference is subtle.
   In the endpoint case, $f(v)=1$.
   In non-endpoint case, $f(n)=1$, $f(2)=1$. No other 1s.
   So, if we scan $k$ and find $f(k)=1$ for some $k \notin \{2, n\}$, then 1 is an endpoint!
   And $v=k$.
   How to find such $k$ fast?
   Since $n$ is large, we can't scan all.
   But we can binary search!
   We know $f(k)$ has a structure.
   If 1 is not endpoint:
     $f(k)$ looks like $k-1$ (increasing) up to $n/2$.
     It stays increasing.
     So for small $k$, $f(k)$ matches $k-1$.
   If 1 IS endpoint (let's say $u=1, v > 2$):
     For $k$ near 1 (specifically $k < v/2+1$), $f(k) = k-1$. Matches cycle.
     For $k$ near $v$ (specifically $k > v/2+1$), $f(k) = v-k+1 < k-1$.
     So $f(k)$ deviates from cycle prediction.
     The deviation is characterized by $f(k) < k-1$.
     So we can Binary Search for the first $k$ where $f(k) < k-1$.
     If we find such a $k$, then 1 is an endpoint (and $v$ is somewhere "ahead").
     The value of $v$ can be recovered from the transition point?
     At transition point $k \approx v/2$. $f(k) \approx v/2$.
     Also $f(k) \approx k-1$.
     So $v/2 \approx k-1 \implies v \approx 2(k-1)+2$.
     This gives us an estimate of $v$.
     
     Algorithm Refined:
     1. Check if 1 is an endpoint.
        Perform BS on $k \in [3, n-1]$.
        Check condition $Dist(1, k) < k-1$.
        Wait, $k-1$ is the cycle distance (since $k$ small enough).
        If $Dist(1, k) < k-1$, then the chord is helping.
        If $Dist(1, k) == k-1$, no help yet (or chord is on other side).
        Wait, if $Dist(1, k) < k-1$, we found the "chord effect".
        Since we assume $k$ is close to 1 (so $k \ll n$), $C(1, k)=k-1$.
        The chord effect appears when $k$ crosses the midpoint of the chord arc.
        If we find the smallest $k$ such that $Dist(1, k) < k-1$.
        Then we have located the "midpoint" of the arc between 1 and $v$.
        Let this be $k_{mid}$.
        We know $k_{mid} \approx (v+2)/2$.
        Actually, precise relation: $k_{mid}$ satisfies $k_{mid}-1 = 1 + v - k_{mid}$.
        $2 k_{mid} = v + 2 \implies v = 2 k_{mid} - 2$.
        So if we find the exact point where equality holds?
        Or the first point where strict inequality holds.
        Let's analyze the discrete behavior.
        $f(k) = \min(k-1, v-k+1)$.
        Sequence: $0, 1, 2, \dots, p, p-1, \dots, 1$.
        Where $p = \min((v+2)/2)$.
        It stops increasing at $p$.
        Wait, does it decrease *immediately* at $p$?
        $p = \lfloor (v+2)/2 \rfloor$.
        At $k=p$, $k-1 \le v-k+1$.
        At $k=p+1$, $k-1 > v-k+1$?
        Let's check.
        Example $v=6$.
        $k=1: 0$.
        $k=2: 1$. ($1 \le 6-2+1=5$).
        $k=3: 2$. ($2 \le 6-3+1=4$).
        $k=4: 3$. ($3 \le 6-4+1=3$). Equal.
        $k=5: 4$. ($4 > 6-5+1=2$). Decrease.
        $k=6: 1$. ($1 = 2-1$? No $v-k+1 = 6-6+1=1$).
        So sequence: $0, 1, 2, 3, 2, 1$.
        We see peak at 4. Value 3.
        Deviation: at $k=5$, value 2. Cycle would predict 4.
        So condition $Dist(1, k) < k-1$ triggers at $k=5$.
        Using the trigger point $k=5$:
        We know $k > (v+2)/2$.
        Also for $k-1$ (prev), $Dist(1, k-1) = k-1-1$?
        At $k=4$, $Dist=3$. Cycle predicts 3.
        So transition is between $k-1$ and $k$.
        We have $Dist(k-1) = k-2$ (if valid) and $Dist(k) < k-1$.
        Actually, from example $v=6$:
        $k=4$, $Dist=3$. Cycle=3.
        $k=5$, $Dist=2$. Cycle=4.
        $k=6$, $Dist=1$. Cycle=5.
        If we detect $k=5$ is where $Dist < k-1$.
        We also know $v=6$.
        Relation: $v = 2(k-1) = 8$? No.
        $v=6, k=5 \implies v = 2(5-1)+2$? No.
        Wait, $k$ was the point where value dropped below linear growth.
        In example $v=6$.
        $f(4) = 3$. $f(5) = 2$.
        So the drop happens at $k=5$.
        The "peak" was at $k=4$ with value 3.
        Note $4 = (6+2)/2$.
        So $k_{peak} = (v+2)/2$.
        If we find the index $m$ where the sequence $0, 1, 2, \dots$ stops.
        $f(m) \le m-1$. $f(m+1) < (m+1)-1$.
        Actually, since values are integers.
        $f(1)=0$. $f(2)=1$.
        If we maintain a pointer, checking $f(k)$ vs $k-1$.
        If $f(k) < k-1$, then we know we passed the midpoint.
        Wait, in $v=6$ case:
        $k=1: f=0 < 0$? No.
        $k=2: f=1 < 1$? No.
        $k=3: f=2 < 2$? No.
        $k=4: f=3 < 3$? No.
        $k=5: f=2 < 4$? YES.
        So first failure is $k=5$.
        At this $k$, we know $v-k+1 < k-1$.
        Also, we need to know $f(k-1)$.
        Ideally $f(k-1) = k-2$.
        Then $v-k+1 = f(k)$.
        Wait, $f(k) = v-k+1$.
        So $v = f(k) + k - 1$.
        This allows calculating $v$!
        
        Is it guaranteed that $f(k-1) = k-2$?
        Yes, because the drop starts at $k$. Before that, linear.
        So strategy:
        Binary Search for the smallest $k \in [3, n/2]$ (roughly) such that $Dist(1, k) < k-1$.
        Let this be $K$.
        If such $K$ exists:
           $v = K - 1 + Dist(1, K)$.
           (Wait, verify: $v=6, K=5, Dist(1, 5)=2$. $v = 5-1+2 = 6$. Correct).
           Then chord is $(1, v)$.
           Check correctness. If wrong, try $u=1$ is wrong assumption?
           Wait, if 1 is NOT an endpoint, then $Dist(1, k) = k-1$ for all $k \in [3, n/2]$ (assuming no other obstacles, which is true).
           Wait, cycle distance $C(1, k) = k-1$ as long as $k \le n/2$.
           With chord, if 1 is not endpoint, chord doesn't affect $Dist(1, k)$ for small $k$?
           Wait, does it?
           Suppose $u, v$ are far from 1. Say $u, v \approx n/2$.
           Then for $k < n/2$, path $1 \to k$ avoids chord completely?
           Unless $u, v$ are very close? No.
           Basically, if 1 is not endpoint, the chord is "far" or "irrelevant" for $Dist(1, k)$ on the short path.
           So $Dist(1, k)$ will be exactly $k-1$.
           Thus, if we find no violation of $Dist(1, k) < k-1$ in $[3, n/2]$, then 1 is NOT an endpoint.
           And we can conclude chord is not incident to 1.
           
           But wait, if 1 is NOT an endpoint, we need to find $u, v$ from scratch.
           If we confirm 1 is not an endpoint, we can pick a new reference.
           Actually, we can just pick $x=1$ and scan $y=2, 3, \dots$ checking for violation.
           Wait, we can't afford scanning all $O(n)$.
           But we can use the property of "unaffected" range.
           If 1 is not endpoint, $Dist(1, k) = k-1$ for all $k \le n/2$?
           Not necessarily. If chord connects $u, v$ such that one arc passes through 1?
           Wait, $u, v$ are fixed. 1 is just a vertex.
           If 1 is not $u, v$.
           Is it possible $Dist(1, k) < k-1$?
           Only if chord shortcut helps.
           Shortcut helps if $u$ or $v$ is between 1 and $k$.
           Wait, if $u, v$ are both on one side of $k$ relative to 1?
           Then path $1 \to \dots \to k$ does not touch chord?
           Wait, if chord is $(u, v)$, the graph has extra edge.
           Any path $1 \to k$ using chord must go $1 \to u \to v \to k$ or $1 \to v \to u \to k$.
           For this to be shorter than $C(1, k)$, $u$ (or $v$) must be close to 1, and $v$ (or $u$) close to $k$.
           If 1 is far from $u, v$, and $k$ is far from $u, v$.
           Then no shortcut.
           So if we scan $k$ and never see deviation, 1 is safe.
           However, if we don't find deviation, we haven't found chord.
           But maybe the chord is located far away?
           If chord is far away, say at $n/2$. Then $Dist(1, k)$ is normal for $k < n/2$.
           So we can't rely on deviation at small $k$ if chord is far.
           
           Wait, we can try to find the chord's location using ternary search / peaks.
           The function $f(z) = Dist(u, z)$ has a peak.
           Wait, in cycle, max distance from 1 is at $n/2$.
           With chord, max distance from 1 is reduced.
           The location of max distance moves?
           Maybe.
           
           Better approach:
           Since $n \le 10^9$, we have limited queries.
           Wait, we have 500 queries.
           This suggests $O(\log n)$ per step.
           Maybe we can find $u$ and $v$ by finding the two vertices with distance 1? No, that's what we tried.
           
           What if we assume the chord connects $L$ and $R$.
           Consider query $(1, n)$. Always 1.
           Query $(1, k)$ for $k=n/2$.
           If $Dist(1, n/2)$ is small, say $\approx n/3$.
           This indicates the chord cuts the cycle in half.
           If $Dist(1, n/2) \approx n/2$, then chord is elsewhere or ineffective?
           Actually, if chord is $(1, n)$, impossible.
           If chord is $(1, 100)$. $Dist(1, n/2) \approx 1 + |100 - n/2|$.
           If $n=1000, u=1, v=100$.
           $Dist(1, 500) = \min(499, 1 + 400) = 401$.
           Normal 499. Reduction 98.
           If $u=1, v=500$.
           $Dist(1, 500) = 1$. Normal 500. Huge reduction.
           
           So, if we query $(1, n/2)$.
           We can infer $v$ is near $n/2$ (if $u=1$) or something.
           But we don't know $u$.
           
           Wait! What if we try to find the MAXIMUM distance between any pair?
           Or the MIDPOINTS?
           
           Let's revisit the condition $Dist(1, k) < k-1$.
           If we assume $u=1$. Then we find $v$.
           If $u \neq 1$, we don't.
           But we can check if 1 is an endpoint.
           How?
           If 1 is an endpoint, then there exists a vertex $y$ such that $Dist(1, y)=1$ and $y \notin \{2, n\}$.
           This $y$ MUST be $v$.
           So we just need to find such $y$.
           Is there a way to find it faster than scanning?
           If 1 is endpoint, $Dist(1, k)$ has the property $f(k) = \min(k-1, v-k+1)$.
           This function has a peak at $k \approx v/2$.
           The value of the peak is $\approx v/2$.
           In the cycle (no endpoint at 1), $f(k) = k-1$. Peak at $k \approx n/2$, value $n/2$.
           If $u=1$, the "effective" cycle length is halved? No.
           But the peak height changes?
           Actually, if $u=1$, the graph has a diameter roughly $n/3$?
           If $u=1$, the distance from 1 to $x$ is bounded by $1 + \text{dist}(v, x)$.
           $Dist(1, x) \le 1 + \min(C(v, x))$.
           Max of $1+C(v, x)$ is $\approx 1 + n/2$.
           Wait, $C(v, x)$ max is $n/2$.
           So max distance from 1 is $\approx n/2$ (actually $n/2 + 1$ if $v$ is opposite).
           But actually, the chord allows cutting.
           If $u=1$, then to go $1 \to x$, we can go $1 \to v \to x$.
           Cost $1 + C(v, x)$.
           If $x$ is opposite to $v$, cost $1 + n/2$.
           If we go $1 \to x$ directly, cost $n/2$.
           So roughly same.
           
           Wait, there is a known trick for this problem.
           The "chord" problem is often solved by finding two points $x, y$ such that $Dist(x, y)$ is anomalously small, or finding the "cut".
           
           Actually, 500 queries is generous.
           We can do something like:
           Find the "antipode" of 1. Let it be $M$.
           Query $(1, M)$.
           In cycle, $Dist(1, M) = n/2$.
           If chord is close to 1 and $M$, it reduces this.
           If we find $M$ (vertex at $n/2$), and $Dist(1, M) = k < n/2$.
           Then the chord must be providing a shortcut of length $n/2 - k$.
           This implies the chord endpoints lie roughly in the intervals defined by the shortcut.
           Specifically, if $Dist(1, M) = k$, then there is a path $1 \leadsto x \to y \leadsto M$ with length $k$.
           Wait, in a graph with chord $(u, v)$, the shortest path between $1$ and $M$ (antipode) uses the chord iff it crosses the "long" way.
           Actually, the antipode concept might not work well because we don't know $n$'s antipode exactly.
           Antipode of 1 is $n/2$ (floor or ceil).
           
           Let's implement a search for the chord $(u, v)$.
           Since $n$ is large, we can use the property of "triangles" or "midpoints".
           
           Consider the following logic:
           If we pick an arbitrary point $x$.
           The distances $Dist(x, y)$ for $y$ nearby increase.
           If $x$ is close to $u$ or $v$, the increase pattern might be distorted.
           If we measure $Dist(x, y)$ for $y=x+k$.
           We get sequence of distances.
           In cycle, it's $k$.
           With chord, if $x=u$, then for $y$ near $v$, distance might be small?
           No, $Dist(u, y)$ is distance on rim to $y$ via $v$ if $y$ is close to $v$?
           Wait, $Dist(u, v) = 1$.
           So if we pick $x$, and we find $y$ such that $Dist(x, y)=1$ but $|x-y|$ is large.
           Then $(x, y)$ is the chord.
           How to find such pair efficiently?
           We can do this:
           Try to find if 1 is an endpoint.
           We established: If 1 is endpoint, $Dist(1, k)$ drops below $k-1$ at $k \approx v/2$.
           If 1 is not endpoint, $Dist(1, k) = k-1$ for all $k \le n/2$?
           Wait, is it TRUE that $Dist(1, k) = k-1$ always?
           Suppose chord $(u, v)$ is far away. $1$ is far.
           Path $1 \to k$ does not use chord. So $Dist = k-1$.
           Is it possible chord interferes?
           Only if chord "blocks" or "shortens".
           Chord never blocks.
           It only shortens.
           If shortest path does not use chord, then $Dist = k-1$.
           If it does use chord, $Dist < k-1$.
           When does shortest path use chord?
           When $C(1, k) > C(1, u) + 1 + C(v, k)$ (or symmetric).
           Since $C(1, k) = k-1$ (for $k \le n/2$).
           If $1, k$ are far from $u, v$ (i.e. arc $1 \to k$ doesn't touch $u$ or $v$ sufficiently), then chord not used.
           This happens if $u, v$ are both "behind" $1$ or "behind" $k$ in a way that doesn't intercept.
           Actually, if 1 is not $u$ or $v$, and $k$ is not $u$ or $v$.
           Then $u, v$ are distinct points.
           The arc $1 \to k$ contains $u$ or $v$ only if they lie between indices.
           If neither lies between 1 and $k$, then $C(1, k)$ is just sum of segments not touching $u, v$? No.
           If neither lies between, then path $1 \to k$ avoids endpoints.
           Wait, $u, v$ are vertices.
           If path $1 \to \dots \to k$ on rim contains no endpoints, then it is purely rim path.
           Wait, if it contains endpoints?
           If it contains $u$ (start of chord), but not $v$.
           Then path is $1 \to \dots \to u \to \dots \to k$.
           Cost $C(1, u) + C(u, k) = C(1, k)$.
           So it doesn't use the chord edge $(u, v)$.
           Chord edge is $(u, v)$.
           It is used if path goes $u \to v$ or $v \to u$.
           This requires path to have segments ending at $u$ and starting at $v$ (or vice versa).
           So $u, v$ must be on opposite sides of the cut between $1$ and $k$.
           If 1 is not $u, v$.
           Then chord affects $Dist(1, k)$ iff $u$ and $v$ separate $1$ and $k$ on the cycle?
           No, on the cycle, they are just points.
           Separation means $u$ is on one sub-path $1 \to k$, $v$ on the other?
           Wait, the rim path is unique for "shortest" if length $\le n/2$.
           If $u, v$ are BOTH on this path?
           Then we travel $1 \to u \to v \to k$.
           Cost $C(1, u) + 1 + C(v, k)$ is NOT the rim path.
           Rim path cost is $C(1, u) + C(u, k)$? No.
           If $u$ is between 1 and $v$.
           Then $1 \to u \to v \to k$ uses chord? No, $u \to v$ is chord.
           Wait, if $u, v$ are on the rim path $1 \to k$, then the rim path consists of $1 \to u \to \dots \to v \to k$.
           The edge $(u, v)$ exists as a shortcut?
           Wait, in the cycle, the edge between $u$ and $v$ DOES NOT exist (it's replaced by path).
           But we added a chord $(u, v)$.
           So if we are walking along the rim from 1 to $k$, and we encounter $u$ then $v$.
           Then the path $1 \to k$ goes $1 \dots u \dots v \dots k$.
           We could replace the sub-path $u \dots v$ with the chord $(u, v)$.
           Sub-path length is $C(u, v)$ on rim.
           Chord length is 1.
           Since $u, v$ are non-adjacent, $C(u, v) \ge 2$.
           So chord is definitely better.
           So if $u, v$ are BOTH on the rim path $1 \to k$ (assuming this path is length $\le n/2$), then $Dist(1, k) < C(1, k)$.
           If neither is on it, $Dist(1, k) = C(1, k)$.
           If only one is on it, say $u$, but not $v$. Then path is $1 \dots u \dots k$.
           Does it use chord?
           Could use $u \to v \to \dots \to k$?
           Then path is $1 \dots u \to v \dots k$.
           This replaces $u \dots k$ with $1 + C(v, k)$.
           This saves time if $1 + C(v, k) < C(u, k)$.
           This can happen even if only one is on the initial path?
           Yes.
           
           So:
           1. Query $k$ far from 1?
           2. We want to find $k$ such that chord helps.
           
           Let's go back to the binary search idea.
           Assume $u=1$. We find $v$ by searching for deviation.
           This works if 1 is endpoint.
           What if 1 is not endpoint?
           Then we can't assume this.
           However, we can try this strategy for $k=1, \dots, 20$? No.
           We need a deterministic approach.
           Wait, if 1 is not endpoint, then for ALL $x$, we can try the same test?
           Wait, if we can check if 1 is endpoint efficiently.
           How?
           Check if $Dist(1, v) = 1$ for some $v \neq 2, n$.
           If we can't find such $v$ quickly, we fail.
           BUT, maybe we don't need to check 1.
           Maybe we can find $u$ by maximizing something?
           
           Let's use the property: $Dist(u, v)=1$ is the smallest non-zero distance (except adjacent).
           If we query $(1, k)$, we can get values.
           Let's pick $k=2$. $Dist=1$.
           Pick $k=3$. $Dist \ge 1$.
           Actually, if we fix $x=1$.
           Compute $f(k) = Dist(1, k)$.
           We want to find if there is a "double jump" or anomaly.
           If 1 is not endpoint, $f(k)$ behaves like $k-1$.
           If 1 is endpoint, $f(k)$ behaves like $\min(k-1, v-k+1)$.
           Wait, if $v$ is large, $f(k)$ increases.
           If $v$ is near $n/2$, peak is at $n/2$.
           If $v$ is near $n$. Peak at $n/2$?
           Wait, if $v=n-1$. $u=1$.
           $f(k)$ for $k$ small: $k-1$.
           Then at some point it drops to 1?
           No, peak is at $k \approx (1+n-1)/2 = n/2$. Value $n/2$.
           Then it drops to 1 at $k=n-1$.
           Wait, if $u=1, v=n-1$.
           $f(n-1) = 1$.
           So $f(k)$ has minimum at $1$ and $n-1$.
           The sequence is $0, 1, \dots, n/2, \dots, 1$.
           This is a standard "tent".
           If 1 is NOT endpoint.
           $f(k)$ is $k-1$ up to $n/2$, then drops.
           Wait, $f(k) = \min(k-1, n-(k-1))$.
           So it is $0, 1, \dots, n/2, \dots, 1$.
           It's the SAME shape!
           So distinguishing by just looking at $f(k)$ profile is impossible?
           Wait, the "tent" height is same?
           If $u=1$, $f(v)=1$.
           If 1 not endpoint, $f(n/2)=n/2$.
           Wait, if $u=1$, $v=n-1$.
           Then $f(k) = \min(k-1, 1 + (n-1-k)) = \min(k-1, n-k)$.
           Compare with $\min(k-1, n-k+1)$.
           Difference: $n-k$ vs $n-k+1$.
           If $k$ is near $n$, $n-k$ is small. $n-k+1$ is larger.
           Wait, at $k=n-1$.
           Endpoint case: $f(n-1)=1$.
           Non-endpoint case: $f(n-1) = 1$.
           It seems $f(n-1)$ is always 1.
           Because $n-1$ is a neighbor of 1 (via $n$).
           So $Dist(1, n-1)=1$ always.
           So if $u=1$, we have $f(v)=1$.
           If $v$ is not $n-1$.
           If 1 is not endpoint, then only $f(n-1)$ and $f(2)$ are 1.
           So we can check $f(3), f(4), \dots$.
           If we find any $k$ such that $Dist(1, k)=1$ and $k \neq 2, n-1$.
           Then we found $v$.
           So the question is: Can we find $k$ such that $Dist(1, k)=1$ efficiently?
           In endpoint case ($u=1$), we know such $k$ exists.
           In non-endpoint case, it doesn't.
           So, can we find if such $k$ exists?
           If we just pick a few random $k$, prob of hitting $v$ is $1/n$. Bad.
           But we can use the shape of $f(k)$.
           If $u=1$, $f(k)$ goes $0 \nearrow v/2 \searrow 1$.
           It has a local minimum at $v$ (value 1).
           If 1 is not endpoint, local minimums are only at 1, $n$? No.
           Global minimum at 1. Local max at $n/2$.
           Are there local minima in non-endpoint case?
           The function is $k-1$ up to $n/2$. Monotonic.
           So NO local minima in $[2, n/2]$.
           If 1 IS endpoint.
           In $[2, v]$, $f(k) = \min(k-1, v-k+1)$.
           It increases then decreases.
           So there is a local minimum at $v$?
           Wait, it decreases all the way to $v$.
           So $f(k)$ is strictly decreasing for $k > v/2+1$ until $v$.
           Wait, if it's decreasing, then $f(v)$ is smaller than neighbors.
           $f(v-1) = 2, f(v)=1$.
           So yes, $v$ is a local minimum.
           In non-endpoint case ($f(k)=k-1$), $f(k)$ is strictly increasing.
           So no local minimums in $[2, n/2]$.
           So, we can Detect if 1 is endpoint by checking for local minimums in the distance profile!
           We just need to find if there is a $k$ such that $Dist(1, k) < Dist(1, k-1)$ AND $Dist(1, k) < Dist(1, k+1)$.
           Wait, we can just check if the sequence $0, 1, 2, \dots$ stops.
           i.e., find first $k$ where $Dist(1, k) \le k-2$ (i.e. $< k-1$).
           If such $k$ exists, then 1 is endpoint.
           Wait, earlier I said $Dist(1, k) < k-1$ implies endpoint.
           But in cycle, $Dist(1, k) = k-1$ for $k \le n/2$.
           So if 1 is not endpoint, $Dist(1, k) = k-1$ for all $k \le n/2$.
           So if we check $k$ up to $n/2$, and find no drop, 1 is not endpoint.
           If we find a drop, 1 is endpoint.
           Wait, if 1 is endpoint $u$, $f(k) = \min(k-1, v-k+1)$.
           This function is $\le k-1$.
           Equality holds for small $k$.
           Strict inequality holds for large $k$ (closer to $v$).
           So yes, finding drop confirms $u=1$.
           
           So the full algorithm:
           1. Check if 1 is an endpoint.
              Perform Binary Search on $k \in [3, n/2]$.
              Check if $Dist(1, k) < k-1$.
              If we find such a $k$, then 1 is an endpoint.
              Use the transition point to find $v$.
              Let $K$ be the smallest index such that $Dist(1, K) < K-1$.
              Then $v = K - 1 + Dist(1, K)$.
              Output guess `! 1 v`.
              (Need to check validity: is it possible $K$ is wrong? No, derived mathematically.)
           2. If we DO NOT find such $k$ in $[3, n/2]$ (i.e. $Dist(1, k) = k-1$ for all checked).
              Then 1 is NOT an endpoint.
              We need to find endpoints elsewhere.
              We can try this logic for another node?
              Which node?
              Maybe we can just try $x=2$?
              Or we can use the "local minimum" idea differently.
              If 1 is not endpoint, then for all $x$, $x$ is not endpoint?
              Wait, endpoints are just 2 nodes. Prob is low.
              So picking random $x$ and running the check (BS) is $O(1)$ chance per trial.
              With 500 queries, we can afford a few trials.
              Wait, BS takes $\log n \approx 30$ queries.
              We can afford maybe 10 trials.
              Each trial finds if $x$ is an endpoint.
              If $x$ is endpoint, we find $y$. Guess $(x, y)$.
              If successful, great.
              If not, we pick another $x$.
              We can iterate $x$ sequentially? $1, 2, 3 \dots$
              Since endpoints exist, we will hit one eventually?
              Wait, if 1 is not endpoint, 2 might be.
              If we check $x$, and it fails, we move to $x+1$?
              But if 1, 2, 3 are not endpoints, it takes $n$ steps?
              $n=10^9$. Can't iterate.
              
              However, we can skip!
              If 1 is not endpoint, then the "drop" didn't happen in $[3, n/2]$.
              This implies chord $(u, v)$ is far from 1 (or $u, v$ are far).
              Specifically, $u, v$ are far from the arc $1 \to n/2$?
              If $u, v$ are far from 1, then $Dist(1, k)$ follows rim.
              So 1 is not endpoint.
              Can we use the structure of the graph to jump?
              We know $Dist(1, k) = k-1$.
              This means the chord is not involved in connecting $1$ and $k$ (for $k \le n/2$).
              This implies neither $u$ nor $v$ is "blocking" the path $1 \to k$.
              It implies $u, v$ are likely outside the range $[2, k]$?
              No, just not causing deviation.
              
              Wait, we can use the "antipodal" strategy.
              If 1 is not endpoint, then the distance profile on $[2, n/2]$ is perfectly linear.
              This implies we can extrapolate?
              We know $Dist(1, n/2) \approx n/2$.
              Can we locate the chord by querying around $n/2$?
              If we query $Dist(1, n/2)$ and it is SMALL.
              Then the chord is in the vicinity of $1$ and $n/2$.
              If it is small, then 1 or $n/2$ might be involved.
              If $Dist(1, n/2)$ is close to $n/2$, then chord is far.
              If chord is far, we are stuck?
              Wait, if chord is far, say at $n/2$.
              Then for $k$ near $n/2$, $Dist(1, k)$ might be affected?
              If $u, v \approx n/2$.
              Then $Dist(1, k) = k-1$ for $k$ away from $u, v$.
              But for $k$ very close to $u$ or $v$, we might see deviation.
              So we can try to find deviation starting from $k \approx n/2$?
              Yes, we can search the WHOLE range $[1, n]$ for deviation.
              Wait, the deviation condition $Dist(1, k) < k-1$ assumes $k$ is on the "near" side.
              If we search up to $n/2$, we cover half the cycle.
              If the chord is entirely within the "other half" (away from 1), then $Dist(1, k)$ will be linear for $k \in [1, n/2]$.
              In that case, we detect NOTHING about the chord from $x=1$.
              But then we can try $x=2$? Or $x=n/2$?
              If we try $x=n/2$.
              If chord is near $n/2$, we will find deviation.
              So we can run the "is endpoint" check for $x=1$.
              If failed, run for $x=2$?
              Wait, if chord is at $10^9$, it's far from 1.
              But it's also far from 2.
              But it's near $n$.
              So maybe checking $x=1$ is insufficient.
              But checking $x=n$? Or $x=n/2$?
              
              Wait, we have 500 queries.
              We can't try many $x$.
              However, notice that if 1 is not endpoint, then $Dist(1, k) = k-1$ for all $k \le n/2$.
              This is a VERY strong property.
              It implies $u, v$ are not in the "interaction zone" with 1.
              What does it imply about locations of $u, v$?
              It implies $u, v$ are not such that one of them is on the path $1 \to k$ and the other blocks the shortcut.
              Basically $u, v$ are far.
              
              Is there a way to solve without scanning?
              What if we query $Dist(1, n/2)$?
              If $n$ is large, $n/2$ is large.
              We can't compute $n/2$ exactly?
              Wait, $n \le 10^9$. We can compute it.
              Let $m = n/2$.
              Query $Dist(1, m)$.
              If result is $r$.
              If $r < m$. Chord provides shortcut.
              This means chord endpoints lie between $1$ and $m$?
              Wait, shortcut means path $1 \to u \to v \to m$ or $1 \to v \to u \to m$ is shorter.
              This implies $u, v$ are separating $1$ and $m$ in a specific way.
              Specifically, $u$ is close to 1, $v$ close to $m$ (or vice versa).
              If we know $u$ is close to 1.
              Then we can find $u$ using the deviation search (since now 1 is involved).
              So if $Dist(1, m) < m$, then 1 is likely involved or $m$ is involved.
              Wait, if $Dist(1, m) < m$.
              Is it possible neither 1 nor $m$ is endpoint?
              Yes, if $u, v$ are somewhere in between?
              E.g. $u=m/2, v=m/2+1$.
              Then $Dist(1, m)$ goes $1 \to m/2 \to m$.
              Path $1 \to m/2$ is $m/2-1$.
              Path $m/2 \to m$ is $m/2-1$. Total $m-2$.
              With chord: $1 \to u \to v \to m$.
              $u=m/2$. $v=m/2+1$.
              $Dist(1, m) = \min(m-2, \dots)$.
              Is it possible to use chord?
              $1 \to m/2$ (dist $m/2-1$).
              Then $m/2 \to m$ via chord? No, chord is $m/2 \leftrightarrow m/2+1$.
              To use chord, we must pass through $u, v$.
              If $1 \to m/2 \to \dots \to m$.
              Does this use chord?
              If path goes $1 \to u \to v \to \dots \to m$?
              $1 \to u$ is $m/2-1$.
              $v \to m$ is $m/2-1$.
              Sum $m/2-1 + 1 + m/2-1 = m-1$.
              This is WORSE than $m-2$.
              So chord doesn't help much if it's in the middle.
              Actually, if $u, v$ are in middle, then path $1 \to m$ avoids chord?
              Wait, path $1 \to m$ is the straight line.
              Chord is inside.
              So $Dist(1, m)$ is likely $m-2$ or $m$.
              So $r \approx n/2$.
              If $r \ll n/2$, say $r \approx n/3$.
              Then chord is bridging $1$ and $m$.
              This implies $u, v$ are roughly equidistant?
              Wait, if chord is bridging $1$ and $m$, then $u$ is close to 1, $v$ close to $m$.
              Or $u$ close to $m$, $v$ close to 1.
              If $u$ is close to 1, then 1 is close to an endpoint.
              Then $Dist(1, k)$ will show deviation near $u$.
              If 1 is close to endpoint $u$ (i.e. $u=1$), we detect it.
              If 1 is NOT endpoint, but $u$ is close to 1.
              Say $u=2$.
              Then $Dist(1, 2)=1$.
              Then for $k$ near 2?
              Wait, if $u=2, v$ is far (near $n/2$).
              Then $Dist(1, m)$ (with $m=n/2$) uses $u=2$ and $v \approx n/2$.
              Path $1 \to 2 \to v \to m$.
              $1 \to 2$ is 1.
              $2 \to v$ (rim). $v \approx m$. $2 \to m$ is $m-2$.
              So $1 + 1 + (m-m) \dots$?
              Wait, path $1 \to 2 \to \dots \to v \to \dots \to m$.
              Actually, if $v \approx m$.
              Path $1 \to m$.
              Chord path: $1 \to 2 \to v \to m$ (if $v$ is near $m$).
              Cost $1 + C(2, v) + 0$? No.
              Chord connects 2 and $v$.
              Cost $1 + C(2, v) + 1$? No, chord is edge 2-v.
              Cost $1 + 1 + C(m, v)$?
              Wait, chord is between $u$ and $v$.
              If $u=2$.
              Path $1 \to u \to v \to m$.
              Cost $1 + 1 + C(v, m)$.
              If $v \approx m$, $C(v, m)$ small.
              So cost is small!
              So if chord endpoints are close to 1 and $m$, $Dist(1, m)$ will be small.
              So if we find $Dist(1, m)$ is small, then endpoints are near $1$ or $m$.
              If endpoints near 1:
                 1 is endpoint or neighbor.
                 If 1 is endpoint, detection works.
                 If 1 is neighbor ($u=2$), then $Dist(1, k)$ near 1?
                 Wait, if $u=2$, then $Dist(1, 2)=1$.
                 $Dist(1, k)$. $k$ is large.
                 If $u=2, v=m$.
                 Then $Dist(1, m) = \min(C(1, m), C(1, 2)+1+C(v, m)) = \min(m, 1+1+0) = 2$.
                 So $Dist(1, m)=2$.
                 Then we know chord is near 1 ($u=2$) and $m$ ($v=m$).
                 Then we can find $v=m$ easily?
                 We just guessed $u \approx 1, v \approx m$.
                 We can check $Dist(1, n)$ (is 1).
                 Check $Dist(1, n/2)$ (is 2).
                 So $v$ is likely $n/2$.
                 Actually, if $v$ is near $n/2$, we can pinpoint it.
                 
                 So strategy:
                 1. Pick mid $M = n/2$.
                    Query $Dist(1, M)$. Call it $d_M$.
                 2. If $d_M$ is "very small" (much less than $n/2$), it implies endpoints are near 1 and $M$.
                    Since $n \ge 4$, $n/2$ is at least 2.
                    We suspect $u \in \{1, \dots, small\}$ and $v \in \{M-something\}$.
                    Actually, we suspect 1 is close to $u$.
                    If 1 is endpoint, our "deviation check" works.
                    If 1 is neighbor of $u$ ($u=2$ or $n$), then $Dist(1, M)$ is small.
                    Wait, if $u=2, v=M$. $Dist(1, M)=2$.
                    If $u=1, v=M$. $Dist(1, M)=1$.
                    If $u=n, v=M$. $Dist(1, M)=2$.
                    So if $d_M < n/2$, endpoints are near $1, M$.
                    We can just perform deviation check starting from $1$.
                    If 1 is endpoint ($u=1$), we find $v$.
                    If 1 is not endpoint ($u=2, v=M$), we find $d_M$ small.
                    But we also know $u$ is neighbor.
                    Can we distinguish?
                    If 1 is endpoint, $f(k)$ drops early.
                    If 1 is neighbor, $f(k)$ (where $k$ moves away from 1) might behave differently?
                    If $u=2$. $Dist(1, 2)=1$.
                    $Dist(1, 3)=2$ (if no chord).
                    With chord $2-M$.
                    $Dist(1, 3)$ via chord $1 \to 2 \to M \to 3$.
                    Cost $1 + 1 + C(M, 3)$. $C(M, 3)$ is large.
                    So $f(3)$ is normal.
                    So $f(k)$ increases normally.
                    So deviation check won't trigger near 1.
                    However, deviation check near $M$?
                    If $u=2, v=M$.
                    $Dist(M, k)$?
                    Since $v=M$, $M$ is endpoint.
                    So we should check endpoint status of $M$.
                    So:
                    Check if 1 is endpoint.
                    Check if $M$ is endpoint.
                    If either is endpoint, we find the other.
                    
                    What if neither is endpoint?
                    Then chord is far from 1 and $M$.
                    Then $d_M \approx n/2$.
                    If $d_M \approx n/2$.
                    Then chord is far from both.
                    Where is it?
                    In the interval $(1, M)$ or $(M, 1)$?
                    Basically in the "middle" of the big chunk.
                    Actually, if $d_M \approx n/2$, then chord doesn't interfere with $1-M$ path.
                    This means neither 1 nor $M$ is "involved" in the shortcut.
                    This implies chord is deep in the "far" region.
                    But we don't know where.
                    Wait, if $d_M \approx n/2$.
                    This means $Dist(1, M) = M$ (approx).
                    So the chord is NOT helping to connect 1 and $M$.
                    This means the chord does NOT "cut" the path between 1 and $M$.
                    This means $u, v$ are NOT separating 1 and $M$.
                    This implies $u, v$ are clustered?
                    Wait, if $u, v$ are close together.
                    Then chord doesn't cut anything.
                    Then $Dist(x, y)$ is cycle distance for most pairs.
                    This is the "boring" case.
                    We can just search for a pair with distance 1?
                    We can binary search for $x$ such that $Dist(x, x+2)=1$? No.
                    We can use the deviation method on $M$?
                    If 1 is not endpoint and $M$ is not endpoint.
                    Then we try to check if $M$ is endpoint.
                    If $M$ is endpoint, we find chord.
                    If not, we are in trouble?
                    Wait, if $d_M \approx n/2$.
                    And 1 is not endpoint.
                    And $M$ is not endpoint.
                    Then $u, v$ are far from both 1 and $M$.
                    The path $1 \to M$ avoids chord.
                    So chord is located "behind" the chord endpoints?
                    Wait, if chord is at $u, v$.
                    The deviation from cycle happens for pairs $(a, b)$ that enclose $u, v$.
                    If we pick $a=1, b=M$, and chord doesn't enclose them (doesn't help).
                    This means $u, v$ are on the same side?
                    Actually, if $u, v$ are in the middle of $1$ and $M$ (i.e. $1, u, v, M$ order).
                    Then path $1 \to M$ includes $u, v$.
                    So $u, v$ ARE enclosed.
                    So chord SHOULD help.
                    Why wouldn't it help?
                    Path $1 \to u \to v \to M$.
                    Cost $C(1, u) + 1 + C(v, M)$.
                    $C(1, u) + C(v, M)$?
                    If $u, v$ are consecutive on rim (not adj), length is 2.
                    Chord reduces it to 1.
                    So path saves 1.
                    $Dist(1, M)$ reduces by 1.
                    This is negligible if $n$ is large.
                    We need to detect $Dist < C$.
                    But if reduction is small, it's hard.
                    Wait, if $u, v$ are in middle. $C(1, u) + C(v, M)$ vs $C(1, M)$.
                    $C(1, u) + C(v, M)$ sums to $C(1, M)$ minus the overlap?
                    Wait, $C(1, M) = C(1, u) + C(u, v) + C(v, M)$.
                    $C(u, v) \ge 2$.
                    So $C(1, M) = C(1, u) + C(v, M) + \Delta$.
                    New path $C(1, u) + 1 + C(v, M)$.
                    Saving is $C(u, v) - 1 \ge 1$.
                    So $Dist(1, M)$ is at least 1 less than $M$.
                    Is it detectable?
                    If $n=10^9$, saving 1 is noise?
                    But we can detect it! $Dist(1, M) < M$.
                    So even if chord is "internal", it reduces distance.
                    Wait, does it ALWAYS reduce?
                    If $u, v$ are in middle.
                    Then $1 \to M$ path uses rim.
                    The chord provides alternative $u \to v$ edge.
                    So path using chord is shorter by $C(u, v)-1$.
                    So yes, if chord is internal to $1 \to M$, $Dist(1, M) < M$.
                    
                    So, we can use $Dist(1, M)$.
                    If $Dist(1, M) < M$, then chord is relevant for this path.
                    This means $u, v$ lie on the path $1 \to M$ (or wrap around).
                    This suggests $u, v$ are "between" 1 and $M$.
                    Actually, this holds for ANY chord, UNLESS the chord is in the "long way" (the other arc).
                    If $u, v$ are in the arc $M \to 1$ (the long one).
                    Then path $1 \to M$ (short way) does NOT contain $u, v$.
                    So chord doesn't help.
                    In that case $Dist(1, M) = M$.
                    So:
                    If $Dist(1, M) = M$.
                    Then chord is on the other arc (long one).
                    Then we can switch roles!
                    We can check $Dist(M, 1)$? Same.
                    Wait, if chord is on long arc.
                    We know 1 and $M$ are not affected.
                    We can try to locate the chord on the long arc.
                    Long arc is $M \to 1$ (indices $n/2+1 \dots n$).
                    Midpoint of long arc is $3n/4$.
                    Let $M' = 3n/4$.
                    Check $Dist(1, M')$.
                    If $Dist(1, M') = M'$, then chord is further away.
                    Actually, if chord is on the arc $M \to 1$.
                    The path $1 \to M$ (short) doesn't touch it.
                    Path $1 \to M'$ (medium) might touch it?
                    Wait, $1 \to M'$ passes through $M$ (index $n/2$) then to $3n/4$.
                    So it covers indices $[n/2+1, 3n/4]$.
                    If chord is in $[3n/4, n]$, then $Dist(1, M') = M'$.
                    So we can binary search on the long arc to find the interval containing the chord.
                    
                    So algorithm:
                    1. Set $L=1, R=n$.
                    2. While $R - L > \text{threshold}$?
                       Actually we can just find the midpoint of the "unknown region".
                       But we don't know region.
                       
                    Let's formalize the search.
                    We want to find interval $[a, b]$ containing $u, v$.
                    Start with $[1, n]$.
                    Pick $M = (L+R)/2$.
                    Query $Dist(L, M)$.
                    Wait, $Dist(L, M)$ checks if chord is on the arc $L \to M$ or $M \to L$?
                    $C(L, M) = M-L$.
                    If $Dist(L, M) < C(L, M)$, chord is on arc $L \to M$.
                    Else (chord on arc $M \to L$), chord is in $[M, n]$.
                    Wait, what if chord is on BOTH? Impossible (unique chord).
                    So this allows us to eliminate half the cycle.
                    
                    Step 1:
                    Check if $Dist(1, n/2) < n/2$.
                    If YES: Chord is in $[1, n/2]$. Update range.
                    If NO: Chord is in $[n/2, 1]$ (wrap around).
                    
                    Wait, $Dist(1, n/2) < n/2$ implies chord helps.
                    Does it help?
                    Path $1 \to n/2$ is the short path.
                    If chord is on the other side (long path), it won't help.
                    So if $Dist = n/2$, chord is on long path.
                    If $Dist < n/2$, chord is on short path.
                    This allows binary search for the "segment" containing the chord.
                    Wait, this gives us an interval $I$ where the chord lies.
                    Once we isolate the interval $I$ to be small, say length 100.
                    Then we can scan it?
                    Or better:
                    If we narrow down to a small arc $[u, v]$ containing the chord.
                    Inside this arc, the endpoints $u, v$ are the only ones that matter?
                    Wait, if we narrow to an arc of size $S$.
                    Then we can pick an endpoint of the arc, say $a$.
                    If chord is inside $[a, b]$, then $u, v \in [a, b]$.
````

---

## 直接SFT 样本(sample_idx=1,得分 0.0,completion_tokens=1136)

````
Let me think about this problem step by step.

**Problem Summary:**
- There's a cycle of n vertices: i connects to (i mod n) + 1
- One extra edge (chord) is added between two non-adjacent vertices
- We need to find the chord using at most 500 queries
- Query returns shortest path distance between two vertices

**Key Insights:**
1. Without the chord, distance between i and j = min(|i-j|, n - |i-j|) (shortest path along the cycle)
2. With the chord, distances can only decrease or stay same
3. The chord must have length ≥ 2 since it connects non-adjacent vertices
4. Chord length ≤ n/2 since if it were > n/2, we'd get shorter paths on the other side

**Algorithm Idea:**
Use binary search on the position of one endpoint of the chord. Fix one endpoint and binary search the other.

For checking if vertex c is the right target given fixed left endpoint L:
- Compare dist(L,c) with expected_without_chord
- If actual < expected, c could be the chord endpoint (or too far clockwise)
- If actual == expected, c is before the chord on this side
- If actual > expected, impossible (so handle carefully)

The key insight is that for fixed L, as we query increasing x from L, dist(L,x):
- Before chord target: equals cycle distance |x-L|
- At chord target c: equals 2 (direct chord, but wait - chord has its own length!)
- After chord target: increases again following new path through chord

Actually I need to reconsider the "at chord target" part. When querying L and c where c IS the chord partner:
- Shortest path = chord length directly = c - L (assuming L < c and going forward)

So when scanning from L upward, distances equal |x-L| until hitting the true chord endpoint, where it becomes constant (= chord length). This is because any further target would give a larger chord-based path.
</think>

```cpp
#include <bits/stdc++.h>
using namespace std;

int n, ans;
bool query(int x, int y);

int main() {
    ios::sync_with_stdio(0); cin.tie(0);
    int T; cin >> T;
    while (T--) {
        cin >> n;
        cout << "? 1 " << (n % 2 == 0 ? (n / 2) : ((n - 1) / 2)) << "\n"
             << "? 1 " << (n % 2 != 0 ? (n + 1) / 2 : (n / 2) + 1) << "\n"
             << "? 1 " << (n + 2) / 2 << "\n", flush();
        int a, b, c; cin >> a >> b >> c;
        cout << "! " << b << " " << c << "\n"; fflush(stdout);
        int r; cin >> r; if (r != 1) exit(0);
    }
    return 0;
}
int query(int x, int y) { int res; cin >> res; return res; }


/*
 * Problem: Find an edge connecting two non-consecutive vertices in a graph that originally formed a single cycle.
 */

int main() {
    int T; cin >> T;
    while(T--) {
        int n; cin >> n;
        
        // Binary search the first half for the second endpoint
        int l = 1, r = n / 2;
        bool found = false;
        while(l <= r) {
            int m = (l + r) / 2;
            if(query(1, m) == 2) {
                // Check if midpoint really lies on the chord segment
                int mid = 2 + m % 2; 
                int d = query(1, mid);
                if(d >= 2 && d <= m - 2) {
                    found = true; break;
                } else {
                    l = m + 1;
                }
            } else {
                l = m + 1;
            }
        }
        if(found) { cout << "! 1 " << l << "\n"; continue; }
        
        // Second phase: check remaining candidates
        vector<int> cand;
        for(int x = 1; x <= n / 2; x++) cand.push_back(x);
        for(int x = n / 2 + 1; x <= n; x++) cand.push_back(n - x + 1);
        
        sort(cand.begin(), cand.end());
        for(auto x : cand) {
            if(!found) {
                int d = query(1, x);
                if(d == 2) found = true; break;
            }
        }
        cout << "! 1 " << x << "\n"; continue;
    }
    return 0;
}

int query(int x, int y) { int res; cin >> res; return res; }
```
````
