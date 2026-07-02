# 题 54 — base均分 0.0 vs 直接SFT均分 0.0(掉点 0.0)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=0,得分 0.0,completion_tokens=32768)

````
The user wants a C++ solution for an interactive problem called "Centroid Guess".
**Problem Statement:**
- We have a tree with $n$ nodes.
- There is exactly one centroid.
- We need to find this centroid.
- We can query the distance between any two nodes $u, v$.
- Maximum queries: 400,000.
- Time limit: 4 seconds.
- Memory limit: 512 MB.
- $3 \le n \le 7.5 \times 10^4$.

**Algorithm Ideas:**
The centroid is a node such that if we remove it, every remaining connected component has size at most $\lfloor n/2 \rfloor$.
Since the tree is not given explicitly (only $n$), and we only know distances between nodes, this is an interesting constraint.
Wait, usually in interactive problems where we don't know the tree structure but have $n$, finding the centroid might involve binary search or ternary search based on properties of distances.

Let's recall the property of centroid distances.
Let $C$ be the centroid.
For any node $v$, let $size(v)$ be the size of the subtree rooted at $v$ when considering some traversal, or simply the number of nodes in the component containing $v$ if $C$ is removed? No, that's definition related.

Consider the function $f(u) = \text{dist}(u, C)$. Is there any monotonicity?
Actually, the standard way to find the center/centroid of a tree using distance queries without knowing the edges involves checking candidates.
However, since we don't know the structure, querying everything is too expensive. $O(N)$ queries per step would lead to $O(N^2)$ which is roughly $(7.5 \times 10^4)^2 \approx 5.6 \times 10^9$, well above $4 \times 10^5$. So we need something faster, likely $O(\log N)$ or similar per candidate.

Key Insight:
The centroid minimizes the maximum distance to all other nodes? No, that's the center of the tree (minimizing max eccentricity). The centroid divides the tree into pieces of size $\le N/2$.
There is a relationship between the centroid and the sum of distances, or perhaps specific path structures.

Actually, there is a known approach for centroid finding in a black-box tree (only distance queries available):
We can use the fact that the centroid is a local minimum of some potential function, or satisfies specific distance conditions relative to other nodes.
Specifically, for the centroid $c$:
For any $v$, $\sum_{u} dist(c, u)$ might be minimal?
Or maybe we can binary search along paths.

Let's consider the constraints. $N \le 75000$. Query limit $400,000$.
This implies we can afford about $5 \cdot N$ queries roughly ($75000 \times 5 = 375000$). This is linear time complexity roughly.

Standard strategy for finding a node in a tree given only distances:
Pick an arbitrary node (say node 1).
Check distances from node 1 to other nodes.
If we knew the tree, we could BFS. Here we have distances.
Let's analyze the centroid property again.
Let $T$ be the tree. $|V|=n$. Centroid $c$.
For any node $u$, removing $c$ leaves components of size $\le n/2$.
Consider the path from $1$ to the centroid $c$. Let this path be $1 \to v_1 \to v_2 \dots \to c$.
Is it true that moving away from the centroid increases the "depth" or some potential?
Not necessarily.

However, there is a specific technique often used in these "unknown tree" interactive problems involving centroid finding.
Usually, one tries to maintain a current set of candidates or narrows down the search space.
A common property is related to the number of nodes "closer" to a vertex compared to another. But without adjacency lists, counting nodes is hard.

Wait, check the query limit again: 400,000 for $N=75000$. That is exactly $5.33 \times N$.
This suggests an algorithm that takes $O(N)$ queries or $O(N \log N)$. Since interaction requires flushing output, overhead matters but complexity is dominant.

Let's reconsider the definition.
$C$ is a centroid.
Maybe we can guess the centroid by checking if a vertex $u$ splits the tree into large pieces.
How do we check if the component sizes are $\le n/2$?
If $u$ is a centroid, then for all $v$, $dist(u, v) + dist(v, w) = dist(u, w)$ if $v$ is on the path.
But we don't know neighbors.

Another angle:
Binary Search on the answer (if sorted)?
The nodes can be ordered, but distances aren't monotonic in index.

What if we fix a reference node $R$. We want to find the node $X$ closest to $R$ among the centroids? No, there's only one.
Maybe the centroid is the unique node where the distance to any other node behaves differently.

Let's look at similar problems. This looks like Codeforces "Centroids" or similar.
There is a property: The centroid of a tree lies on the shortest path between any two nodes $u, v$ if $u$ and $v$ are in different components formed by removing the centroid? Not helpful since we don't know components.

Wait, there is a randomized or deterministic approach to find the diameter. Once we find the endpoints of the diameter $A, B$, does the centroid lie on the path? Yes, it lies on the diameter.
Why? The centroid is the "middle" of the tree. The diameter is the longest path. The center must be somewhat central. In fact, the centroid always lies on the path between the two farthest nodes (the diameter)? No, that's not strictly true. Counterexample: A star graph, center is centroid, diameter is leaf-to-leaf passing through center. Star graph with long arms... actually yes, the centroid is usually on the diameter.
Let's check. Take a line $1-2-3-4-5$. Centroid is 3. Diameter is 1-5. Path is 1-2-3-4-5. 3 is on path.
Take a 'Y' shape with lengths 2, 2, 2. Root at center 0. Leaves 1, 2, 3. Dist(0,1)=2, etc.
Tree structure: 0 connected to 1, 2, 3. Each branch has length 1 more? Say 0-4-1. Branches of size 3?
Let's make a tree: Node 1 connected to 2, 3, 4. 2, 3, 4 are leaves. $N=4$.
Sizes around 1: Removing 1 gives components {2}, {3}, {4}, each size 1. $\le 4/2=2$. Centroid is 1.
Diameter? 2-1-3, length 2. Path 2-3 goes through 1.
Example: 1-2-3-4. Add 5 connected to 2.
Nodes: 1, 2, 3, 4, 5.
Edges: (1,2), (2,3), (3,4), (2,5).
Removing 2: Components {1}, {3,4}, {5}. Sizes 1, 2, 1. Max size 2. $\lfloor 5/2 \rfloor = 2$. 2 is centroid.
Diameter path: 1-2-3-4 (length 3). 4 is dist 3 from 1. 5 is dist 2 from 1.
Endpoints of diameter: 1 and 4. Path includes 2.
Endpoints of another diameter: 1 and 5? Length 3+? No, dist(1,5)=2.
So diameter is between 1 and 4. Path 1-2-3-4. Centroid 2 is on it.
Is it guaranteed that the centroid lies on *any* diameter? Or just *some* diameter?
It is known that the centroid is contained in *every* diameter? No.
But the centroid is always a node minimizing the maximum distance to other nodes (radius).
Wait, actually, the centroid is not always on the diameter.
Consider a "fat star". Center $C$, $k$ branches of length 2.
$C - a_i - b_i$ for $i=1..k$. Total nodes $2k+1$.
Center $C$. Removing $C$ leaves $k$ components of size 2. If $k$ is large enough, $2 > 2k+1 / 2 \implies 4 > 2k+1 \implies k < 1.5$.
Let $k=2$. Nodes $C, a_1, b_1, a_2, b_2$. Total 5.
Distances from $C$: $\{a_1, a_2\}$ dist 1, $\{b_1, b_2\}$ dist 2.
Removing $C$: comp sizes are 2. $2 \le \lfloor 5/2 \rfloor = 2$. So $C$ is centroid.
Diameter? Path $b_1 - a_1 - C - a_2 - b_2$. Length 4.
Path goes through $C$.
What if branches are unbalanced?
Branch 1: Length 10. Branch 2: Length 2.
Root 1-2-...-11. Node 12 connected to node 1.
Wait, centroid logic is tricky.

Actually, there is a simpler randomized idea.
Pick a random node $x$. Find the furthest node $y$ from $x$. Then the centroid lies on the path from $x$ to $y$.
Why? Because $y$ is a "peripheral" node. $x$ is arbitrary.
In a tree, if you pick a node $u$ and find $v$ such that $dist(u,v)$ is maximized, $v$ is always an endpoint of a diameter.
Proof sketch: If $u$ is not an endpoint of a diameter, its furthest neighbor must lead towards a diameter endpoint.
Let $diam$ be the true diameter endpoints $A, B$.
For any node $u$, let $d_u = \max_v dist(u, v)$. It is known that the endpoint $A$ maximizing $d_A$ and $B$ maximize $d_B$ is related to the diameter.
More importantly, if $v$ is a farthest node from $u$, then the centroid is closer to $v$ than to $u$? Not necessarily.
But $v$ belongs to some "heavy" direction from $u$.
Specifically, the centroid lies on the simple path between $u$ and $v$ where $v$ is a farthest node from $u$.
Let's assume this holds or works with probability. Actually, this is a strong heuristic.
If the centroid lies on the path between $u$ and its farthest neighbor $v$, we have reduced the problem to searching on a path.
Since we can compute distances on a path if we identify the nodes, but we don't know who is next to whom.
Wait, we don't need the path explicitly. We just need to find a node $w$ on the path such that... ?
Actually, once we identify the centroid lies on the path $P$ between $u$ and $v$, the path length is $L = dist(u, v)$.
The centroid $C$ is somewhere on $P$.
Does $C$ divide $P$ in half? Not necessarily.
However, we can find the exact position of the centroid on $P$.
Let $C$ be the centroid.
We know $dist(C, u)$ and $dist(C, v)$? No.
But we can iterate over points on the path? We don't know the intermediate nodes.
We can only query pairs.

Re-evaluating the "path contains centroid" assumption.
Consider the example where centroid is not on the path to the farthest node from $u$.
Star graph with center $C$ and leaves $L_1 \dots L_k$. All dists $1$. Farthest from $L_1$ is $L_2$, path is $L_1-C-L_2$. Centroid is $C$. $C$ is on path.
Line graph $1-2-3-4-5$. $u=1$. Farthest is 5. Path $1..5$. Centroid 3 is on path.
It seems very likely the centroid lies on the path between any node $u$ and its farthest node $v$.
Let's try to prove/disprove.
Suppose centroid $C$ is NOT on the path $u-v$.
Then there exists a branch from the path $u-v$ that contains $C$.
Since $v$ is farthest from $u$, it means going towards $v$ is the most distant direction.
Going towards $C$ is another direction.
Since $C$ splits the tree into halves, there cannot be a huge component in the direction of $C$.
Intuitively, $u$ starts deep in one part, $v$ is the "end" of the tree. $C$ must be near $v$ or between $u$ and $v$.
Actually, the statement "The centroid lies on the path between $u$ and a farthest node from $u$" is TRUE.
Reference: "On the location of the centroid of a tree".
Yes, the centroid is a vertex belonging to the intersection of all diameters. Wait, no.
However, if $v$ is a vertex maximizing distance from $u$, the centroid is indeed on the simple path connecting $u$ and $v$.
Wait, really? Let's check the codeforces editorial for a similar problem or standard knowledge.
In a tree, the "center" (minimizing max distance) and "centroid" (balance size) are both located in the middle.
If $v$ is a farthest node from $u$, then the path $u \to v$ passes through the centroid?
Counter-example: A star graph with center $C$. One arm is longer.
$C$ connects to $A$ (len 10) and $B$ (len 10). Other leaves $L$ (len 1).
Actually, if $C$ has a heavy component.
If we pick $u$ in the light component (leaf $L$), farthest is $A$ or $B$ (in heavy component).
The path goes $L \to C \to A$. Centroid is $C$. $C$ is on path.
If we pick $u$ inside the heavy component.
Say $u$ is close to $C$ on side of $A$. $u \to C \to \dots \to B$.
Farthest is $B$. Path goes through $C$.
So it seems robust.
Assumption: **The centroid lies on the path between an arbitrary node $u$ and its farthest node $v$.**
If this is true, we can find $v$ by iterating $i \in [1, N]$? No, we need a random sample or a better search.
Since we have $O(N)$ queries, we can afford to check all nodes $u$ against a fixed pivot, say node 1.
Wait, if we fix node 1, and find $v_1$ such that $dist(1, v_1)$ is maximal. Then the centroid is on path $1 \to v_1$.
Now we have a path. How many nodes are on the path? Up to $N$.
We need to locate $C$ on this path.
Can we determine $C$'s location on the path?
We know $dist(C, 1)$? No.
We know $dist(C, 1) + dist(C, v_1) = dist(1, v_1)$.
If we can find $dist(1, C)$, we are done.
Is there a property for $dist(1, C)$?
The centroid $C$ is defined by sizes.
Is there a query to return size of subtree? No.
But wait, if $C$ is on the path $1 \to v$, then removing $C$ splits the path into two parts: $[1, \dots, C)$ and $(C, \dots, v]$.
One part leads to $1$, other to $v$.
Also, $C$ has a third component (side branches off the path).
Let $S_1$ be the size of the component containing 1 when $C$ is removed.
Let $S_2$ be the size of the component containing $v$ when $C$ is removed.
Let $S_{other}$ be the sum of sizes of other branches.
Condition: $S_1 \le n/2$, $S_2 \le n/2$, $S_{other} \le n/2$.
We know $S_1 + S_2 + S_{other} = n$.
Also, if $C$ is on the path between 1 and $v$, then the component containing 1 consists of the path segment from 1 up to $C$ plus whatever subtrees hang off that segment.
Similarly for $v$.
Notice that as we move from 1 to $v$ along the path, the size of the component containing the "start" (which was 1) generally decreases.
Specifically, let $u$ be a node on the path $1 \to v$. If we move $u$ closer to 1, the component containing 1 grows.
Wait, we don't know the nodes on the path.
But we can pick a node $x$ on the path (e.g. start at 1).
We can perform a binary search!
Let $L = 0$ (representing node 1), $R = v$ (representing node $v$).
We want to check if a candidate node $mid$ on the path is closer to 1 or further?
No, we need to narrow down the centroid.
Actually, we can iterate over the path if we know the order. We don't.
However, we can maintain a pair of candidates $(u, v)$ that definitely enclose the centroid.
Initially $(1, v)$ where $v$ is farthest from 1.
Pick midpoint $m$ conceptually? No, we need an actual node.
We can pick a random node $z$ on the path? Or just pick the midpoint of the current interval?
Wait, if we don't know the nodes on the path, we can't pick a specific node "halfway" unless we traverse.
But traversing takes time proportional to distance. Sum of distances could be $O(N^2)$.
Total queries allowed $400,000$. $N=75,000$.
We can visit nodes along the path efficiently?
If we keep track of current node $curr$, and next node $next$ is found? No, we need to detect neighbors.
If we have current node $u$ and we know $v$ is farthest, we can look for a node $w$ that is farther from $u$ than from $u$ to $curr$?
No, simply: To move from $u$ towards $v$, we want a neighbor $w$ of $u$ such that $dist(w, v) < dist(u, v)$. Wait, if $u$ is closer to $v$ than $w$, then $dist(u, v) = dist(u, w) + dist(w, v)$, so $dist(w, v) = dist(u, v) - dist(u, w) < dist(u, v)$.
If $w$ is not on the path, $dist(w, v)$ might be larger? No, triangle inequality. $dist(w, v) \ge dist(u, v) - dist(u, w)$.
Wait, to move towards $v$ from $u$, we should pick $w$ such that $dist(w, v) < dist(u, v)$.
Among neighbors of $u$, there will be one that minimizes this distance.
But we don't know neighbors.
We can test any node $k$ from $1 \dots n$.
If we check all nodes $k$, we find $k$ minimizing $dist(k, v)$?
That would require $O(N)$ checks per step. Too slow.

Wait, looking back at constraints.
Score is quadratic. Base score 100k. Limit 400k.
This is generous enough for $O(N)$ total steps if the constant factor is small.
$N \le 75000$. 5 queries per node is fine.
Can we do $O(N)$ total?
Algorithm:
1. Start with $u=1, v=$ a random node (or 2).
2. Compute all $dist(u, i)$ to find farthest $v$.
   Cost: $N$ queries.
3. Now we know $v$ is an endpoint of a diameter (or path to it).
   Assume centroid $C$ is on path $u \to v$.
   We need to find $C$.
   Consider the set of nodes $S$ such that $dist(S, u) \ge \dots$
   Maybe we can use binary search on the distance values?
   Wait, if $C$ is on the path between $u$ and $v$, and we don't know the intermediate nodes...
   Is it possible to verify if $u$ is "left" of $C$?
   If $u$ is left of $C$, then the component containing $u$ has size $\le n/2$.
   Can we estimate the size of component containing $u$?
   Not directly.
   
   Alternative Idea:
   What if we simply iterate all nodes $i=1 \dots N$?
   Calculate $dist(1, i)$.
   Is the centroid the one that minimizes $\max(dist(i, j))$? No.
   Is it the one that minimizes $\sum dist(i, j)$? No.
   
   Let's revisit the centroid property with distances.
   For the centroid $c$, $\forall x$, $dist(c, x) \le n-1$.
   Is there a property: $c$ is the median of something?
   
   Actually, there is a solution that uses binary search with randomized guesses?
   Or just finding the diameter endpoints is not enough to pinpoint centroid if we don't have subtree sizes.
   BUT, if we treat the whole tree as a "black box", we can determine the centroid by checking if removing a node splits it properly.
   Wait, we can't check removal.
   
   Let's go back to the condition: Centroid is on the path between $u$ and $v$ where $v$ is farthest from $u$.
   This allows us to reduce the problem to a line graph of unknown vertices.
   However, we don't need to identify the vertices on the line. We just need to find one vertex $c$ on this line that is the centroid.
   Notice that for the centroid $c$, any deviation from $c$ along the path either moves towards $u$ or towards $v$.
   If we move away from $c$ towards $u$, we enter a larger component (the one with $> n/2$ nodes)?
   Yes, because $c$ balances the tree. $c$ removes the largest chunk. If we move to a neighbor towards $u$, the new node is inside the component of $u$ (when $c$ removed). That component has size $> n/2$ ?? No.
   The component containing $u$ (when $c$ removed) has size $S_u$. Since $c$ is centroid, $S_u \le n/2$.
   Wait, $S_u$ could be $\le n/2$.
   So moving towards $u$ doesn't immediately imply entering a bad component.
   
   However, consider the following:
   Pick node $1$. Find farthest node $v_1$.
   Centroid $C$ is on path $1 \to v_1$.
   Pick node $2$. Find farthest node $v_2$.
   Centroid $C$ is on path $2 \to v_2$.
   Intersection of path $1 \to v_1$ and $2 \to v_2$?
   Could be a single point or empty (if not intersecting properly).
   But they must share the centroid.
   Actually, if $n$ is small, we can maybe brute force?
   
   Let's check the constraint limits again.
   $Q \le 400,000$. $N \le 75,000$.
   This is basically $5N$.
   We can process each node with a constant number of queries.
   
   Proposed Algorithm (Heuristic/Probabilistic/Randomized):
   Randomly select a node $r$.
   Compute distances to all other nodes? No, too expensive.
   Just check $dist(r, i)$ for $i$ such that ...
   
   Wait, if we simply take node 1 and find the farthest node $v$.
   Then we have a candidate path $P$.
   Can we walk along $P$?
   We start at $u=1$. We want to move towards $v$.
   At current node $curr$, we need to find a neighbor $next$ such that $dist(curr, v) > dist(next, v)$.
   This identifies a neighbor towards $v$.
   Does the centroid lie on the path from $u$ to $v$? Yes, with high probability (actually surely for trees).
   So $C$ lies on path $u \to v$.
   If we traverse the path $u \to v$ node by node, and check the "component size" at each node, we can find $C$.
   How to check component size at $curr$ towards $v$?
   Let $S_{curr\_v}$ be the size of the component containing $v$ when $curr$ is removed.
   If $curr = C$, then all components have size $\le n/2$.
   Specifically, the component containing $v$ has size $\le n/2$.
   If $curr$ is "above" $C$ (closer to $u$), then the component containing $v$ might be large?
   Actually, if we walk from $u$ to $v$, initially at $u$, the component towards $v$ (excluding the direction back to $u$, which doesn't exist yet) contains almost everything except $u$'s attached branches.
   As we walk towards $v$, the size of the "forward" component decreases.
   Once we pass $C$, the size of the forward component will suddenly drop below $n/2$? No, it will stay small.
   Before $C$, the size of forward component is $> n/2$.
   Wait, is this true?
   At centroid $C$, all directions lead to components $\le n/2$.
   Suppose we are at $x$ on the path $u \to v$, and $x$ is closer to $u$ than $C$.
   Then removing $x$ puts $C$ in some component. $C$ is "below" $x$.
   The component containing $C$ includes the rest of the path to $v$ plus other branches.
   Does this component have size $> n/2$?
   Let $Comp(x, \to v)$ be the component containing $v$ when $x$ is removed.
   If $x$ is closer to $u$, $Comp(x, \to v)$ includes $C$ and the bulk of the tree towards $C$'s other branches.
   Usually, the "upper" node $u$ is in a "small" component when $C$ is removed?
   Not necessarily.
   However, we can define a potential.
   
   Correct Logic for walking:
   The centroid $C$ is on the path $u \to v$.
   Consider the function $f(z) = \text{size of component containing } v \text{ if } z \text{ is removed}$.
   If $z=C$, $f(C) \le n/2$.
   If $z$ is closer to $u$, the component containing $v$ must be larger than $f(C)$?
   Let's trace. $C$ separates $u$ and $v$ (and others).
   $Comp(C, \to v)$ is a subset of $Comp(C, \to v)$.
   $z$ is in the part towards $u$. When we remove $z$, we cut off the path to $v$.
   Wait, $z$ is between $u$ and $C$.
   Removing $z$: one component is towards $u$, other towards $C$.
   The component towards $C$ includes $v$.
   This component contains all nodes "below" $z$ (towards $v$).
   Is this size $> n/2$?
   Let's see. $z$ is "closer to root" $u$ than $C$ is.
   The total number of nodes is $N$.
   The centroid $C$ ensures that no single component has $> N/2$.
   However, the component "above" $C$ (containing $u$) might be small or large.
   But if we walk from $u$ towards $v$:
   At $u$, the component towards $v$ is basically the whole tree minus branches of $u$ not leading to $v$.
   Is it guaranteed to be $> N/2$?
   Consider $N=5$. $u=1$. Tree $1-2-3-4-5$.
   $u=1, v=5$. Path 1-2-3-4-5.
   Centroid is 3.
   Walk:
   At 1: Remove 1. Comp with 5 is $\{2,3,4,5\}$, size 4. $4 > 5/2$.
   At 2: Remove 2. Comp with 5 is $\{3,4,5\}$, size 3. $3 > 5/2$.
   At 3: Remove 3. Comp with 5 is $\{4,5\}$, size 2. $2 \le 5/2$.
   At 4: Remove 4. Comp with 5 is $\{5\}$, size 1. $1 \le 5/2$.
   At 5: Remove 5. Comp with 5 is $\emptyset$. Size 0.
   
   Another example:
   $u$ is leaf, $v$ is farthest.
   Suppose $u$ is in a small component when $C$ is removed.
   Example: Star graph. $C$ center. Leaves $L_1, \dots, L_N$.
   $u=L_1$. Farthest is any other leaf $L_k$ ($dist=2$).
   Path $L_1 \to C \to L_2$.
   Walk:
   At $L_1$: Remove $L_1$. Comp with $L_2$ is $\{C, L_3 \dots L_N\}$. Size $N-1$.
   If $N-1 > N/2$, i.e., $N > 2$. True for $N \ge 3$.
   At $C$: Remove $C$. Comp with $L_2$ is $\{L_2\}$. Size 1.
   $1 \le N/2$ (for $N \ge 2$).
   So condition seems to hold:
   While `dist(curr, v)` > 0:
     Find neighbor `next` towards `v`.
     Check if component size towards `v` at `curr` > N/2.
     If yes, move to `next`.
     Else, `curr` is likely the centroid (or passed it?).
     Wait, in star graph example:
     At $L_1$ (node before C): Size > N/2. Move.
     At $C$ (node at C): Size $\le N/2$. Stop.
     
     Is it strictly decreasing?
     In path graph: $1, 2, 3, 4, 5$.
     At 1: Size 4.
     At 2: Size 3.
     At 3: Size 2.
     Condition `Size > 2` holds for 1 and 2. Fails for 3.
     So we stop at 3. 3 is centroid.
     
     What if we overshoot?
     Path $1-2-3-4-5$.
     $u=2, v=4$. Centroid is 3.
     Path $2-3-4$.
     At 2: Remove 2. Comp with 4 is $\{3,4\}$. Size 2. $\le 2.5$.
     Wait, here at 2, size is $\le N/2$.
     If we start at 2, we satisfy condition $\le N/2$ immediately.
     But 2 is NOT the centroid. 3 is.
     So the condition `Size > N/2` is sufficient to move towards $v$, but failing it (`Size <= N/2`) doesn't guarantee $C$ is current node, it just means $C$ is not in the "large component" towards $v$.
     This implies $C$ is "below" or at current node.
     
     So we need to find the FIRST node where `Size <= N/2`.
     Wait, is it guaranteed that there is a transition?
     At $u$ (start), size is usually large.
     As we get closer to $C$, size decreases.
     Once we reach $C$, size is $\le N/2$.
     If we continue to $v$, size continues to decrease.
     So we are looking for the first node on the path $u \to v$ where `ComponentTowardsV(curr) <= N/2`.
     But how to calculate `ComponentTowardsV(curr)`?
     We don't know the tree structure.
     But notice:
     `ComponentTowardsV(curr)` is the set of nodes $x$ such that $dist(curr, x)$ goes through $v$? No, through $C$.
     Actually, `ComponentTowardsV(curr)` = set of nodes $y$ where the simple path from $curr$ to $v$ does not pass through $C$ (assuming $curr \ne C$).
     More precisely, it's the set of nodes in the component containing $v$ when $curr$ is removed.
     
     Crucial observation:
     If we just want to distinguish if `curr` is "before" $C$ or "at/after" $C$.
     If `curr` is before $C$ (towards $u$), then removing `curr` isolates $u$'s side. The component containing $v$ is quite large.
     If `curr` is $C$, component is small.
     If `curr` is after $C$ (towards $v$), component is even smaller.
     
     But how to query `ComponentTowardsV(curr)`?
     We can estimate this using distances?
     Wait, we can't count nodes easily.
     
     However, we don't need the exact size.
     We can observe that if `curr` is before `C`, then `dist(curr, v)` is "far".
     Actually, there is a trick.
     If `curr` is the centroid, then for ALL $k$, $dist(curr, k)$ is relatively small compared to $dist(u, k)$? No.
     
     Let's step back.
     With $N \le 75000$ and $Q \le 400000$, we can afford roughly 5-6 operations per node.
     We can run the process for ALL nodes.
     Wait, if we can check if node $i$ is the centroid with few queries, we can loop $1 \dots N$.
     But checking one node naively takes too many queries.
     
     Is there a randomized check?
     Pick $u, v$. If $u$ is centroid, then $dist(u, v) \approx$ min/max?
     
     Let's look at the "binary search on tree" approach.
     Usually, we maintain a range of nodes $[l, r]$ that contains the centroid.
     Initially, we don't know anything.
     But we can guess that centroid is near the middle of diameter.
     Finding diameter takes $O(N)$ queries (finding furthest from arbitrary node, then furthest from there).
     Let diameter be endpoints $A, B$.
     The centroid lies on the path between $A$ and $B$.
     If we can "walk" from $A$ to $B$, identifying if we passed $C$, we are good.
     Walking strategy:
     Current node $curr = A$. Target $target = B$.
     While `dist(curr, target)` > 0:
        Pick a random node $z$.
        Check relation of $z$ to the path? No.
        
     We can try to find the centroid using the property of distances to ALL nodes.
     Specifically, centroid minimizes $\sum dist(node, i)$.
     We can approximate this sum.
     But we don't need exact sum.
     
     Is it true that the centroid is the node that minimizes the variance of distances?
     
     Let's revisit the constraint. $400,000$ queries is the HIGHEST score threshold.
     Usually, $O(N)$ solutions are expected.
     If we implement the strategy:
     1. Find diameter endpoints $A, B$.
        - Pick random $r$.
        - Find $v$ s.t. $dist(r, v)$ is max.
        - $v$ is endpoint. (Maybe $A$).
        - Pick random $x$.
        - Find $y$ s.t. $dist(x, y)$ is max.
        - $y$ is other endpoint.
        - Repeat until stable?
        Actually, finding the exact diameter takes 2 distance computations per node. $2N$ queries.
        This gives $A, B$.
     2. Now we know the centroid is on path $A-B$.
        Let's verify this.
        In a tree, the centroid lies on the path between any two nodes $u, v$ if and only if...
        Actually, it is known that the centroid is the unique node where the maximum subtree size is minimized.
        Also, the centroid is always on the intersection of all diameters? No.
        But the centroid is on the diameter.
        Proof: The centroid $C$ splits the tree into components of size $\le N/2$. The diameter is the longest path. Any path longer than $N-1$? No.
        If we remove $C$, we break all paths passing through $C$.
        The diameter passes through $C$ if $C$ is not "close" to one end?
        Wait, consider a star graph with center $C$. Diameter is $L_1 - C - L_2$. Passes through $C$.
        Consider a "T" shape. Long vertical bar, short horizontal.
        Centroid is on the vertical bar. Diameter is top of vertical to tip of horizontal.
        Wait, if vertical bar is length 10, horizontal tips at middle are length 2.
        Vertices on vertical: $0, 1, \dots, 10$.
        Tip at 5 (horiz len 2): node 17.
        Distance $17 \to 5$ is 2.
        Diameter path goes $0 \to \dots \to 5 \to 17$? No.
        Diameter is between ends of vertical arm and ends of horizontal arms.
        Ends of vertical: $0$ (bottom), $10$ (top).
        Horizontal attaches at 5. Ends $H_1, H_2$ dist 2 from 5.
        Path $H_1 \to 5 \to H_2$ len 4.
        Path $0 \to 10$ len 10.
        Path $0 \to H_1$: $0 \to 1 \dots \to 5 \to H_1$. Len $5 + 2 = 7$.
        Path $10 \to H_1$: $10 \to \dots \to 5 \to H_1$. Len $5 + 2 = 7$.
        Diameter is $0 \leftrightarrow 10$. Length 10.
        Path: $0 \to 10$.
        Centroid?
        Removing 5:
        Components: $0 \dots 4$ (size 5), $6 \dots 10$ (size 5), $H_1$ (size 1), $H_2$ (size 1).
        Max size 5. $N = 10 + 1 + 2 = 13$. $\lfloor 13/2 \rfloor = 6$.
        So 5 is centroid.
        Diameter path $0 \dots 10$. Contains 5.
        It seems true. The centroid always lies on ANY diameter.
        Actually, the diameter endpoints $A, B$ are such that $dist(A, B)$ is max.
        If $C$ was not on $AB$, then removing $C$ would disconnect the tree. The path $A-B$ must cross $C$. If it doesn't cross $C$, then $A$ and $B$ are in the same component, but $C$ is elsewhere.
        If $C$ is not on path $A-B$, then $A$ and $B$ are in some component $K$ formed by removing $C$. Size $K \ge |A| + |B|$...
        Wait, if $A, B$ are in same component $K$ of removing $C$, then $K$ contains all of path $A-B$.
        So $|K| \ge 2 + dist(A, B)$? No, nodes.
        If $A, B$ are in same component, and $C$ is centroid, then $|K| \le N/2$.
        But $dist(A, B)$ is diameter. Is it possible that diameter lies entirely within a "small" component?
        Only if the rest of the tree is also small?
        But $N$ is large.
        Actually, if $C$ is not on diameter $A-B$, then the component containing $A, B$ has size $\le N/2$.
        But this implies the rest of the tree (components not containing $A, B$) has size $\ge N/2$.
        However, $dist(A, B)$ being the diameter is usually much larger than $N/2$ (in terms of edges, nodes).
        In the worst case, diameter is $O(\sqrt{N})$?
        Anyway, assuming centroid lies on diameter is a standard competitive programming intuition.
        Let's proceed with this assumption.
        
        So we need to find $C$ on path $A \to B$.
        We have established that for any node $u$ on path $A \to B$:
        $C$ splits the path into $A \dots C \dots B$.
        Component containing $B$ when $u$ removed:
        If $u$ is closer to $A$, this component contains $C$ and thus "most" of the tree (specifically the part with size $> N/2$ relative to $C$).
        Wait, earlier I said if $u$ is closer to $A$, component towards $B$ is large.
        Let's verify with the T-graph example.
        $A=0, B=10$. $C=5$.
        $u=0$. Remove 0. Comp towards 10 is $\{1 \dots 10\} \cup \{H_1, H_2\}$.
        Size $10 + 2 = 12$. $N=13$. $12 > 13/2$. Large.
        $u=4$. Remove 4. Comp towards 10 is $\{5 \dots 10\} \cup \{H_1, H_2\}$.
        Size $6 + 2 = 8$. $8 > 13/2$. Large.
        $u=5$. Remove 5. Comp towards 10 is $\{6 \dots 10\} \cup \{H_1, H_2\}$.
        Size $5 + 2 = 7$. $7 > 6.5$? No $N=13, \lfloor 13/2 \rfloor = 6$.
        $7 > 6$. Still Large?
        Wait, my previous calculation for Centroid condition: Max component size $\le \lfloor N/2 \rfloor$.
        Here, removing 5 gives sizes: $5$ (from $0..4$), $5$ (from $6..10$), $1$ ($H_1$), $1$ ($H_2$).
        Wait, why did I think $H_1, H_2$ were connected to 5?
        Yes, they are branches at 5.
        So removing 5 splits into 4 components.
        Component containing 10 is $\{6..10\}$. Size 5.
        Component containing $H_1$ is $\{H_1\}$. Size 1.
        Component towards $0$ is $\{0..4\}$. Size 5.
        So if we check "Component containing $B$ (where $B=10$)", for $u=5$, size is 5.
        $5 \le 6$. OK.
        For $u=4$ (closer to $A$):
        Remove 4.
        Comp containing 10 is $\{5, H_1, H_2\} \cup \{6..10\}$.
        Size $1 + 1 + 5 = 7$.
        $7 > 6$. Large.
        
        So the condition `CompContainingTarget(curr) <= N/2` holds for $C$ and nodes closer to $B$.
        It fails for nodes closer to $A$.
        Therefore, we can binary search for the transition point!
        
        Wait, "nodes closer to $B$" means nodes between $C$ and $B$.
        If we walk from $A$ to $B$, we find the FIRST node satisfying the condition.
        Is this monotonic?
        Path nodes: $p_0=A, p_1, \dots, p_m=B$.
        We saw sizes: 12, 8, 7 (at C?), ...
        At $C=5$, size is 5.
        At $u=6$, remove 6.
        Comp containing 10 is $\{7, \dots, 10\}$. Size 4. $\le 6$.
        So for $u=6$, condition holds.
        So the set of nodes satisfying `CompSize(B) <= N/2` is $\{C, C+1, \dots, B\}$.
        Wait, what about nodes "above" $C$ on the other branches?
        Wait, we restricted our search to nodes ON the path $A-B$.
        We assumed $C$ is on path $A-B$.
        If we can identify the correct path and walk on it, we can find $C$.
        
        Problem: How to verify `CompSize(B)` at a node $u$ on the path?
        We don't know the size of the component.
        But we can check if $dist(u, B)$ changes?
        No.
        However, note that $CompSize(B)$ is roughly proportional to the distance $dist(u, B)$ if the tree was a line.
        In general, $CompSize(B)$ is the number of nodes in the component of $u$ containing $B$ upon removal.
        This is equal to $N - (\text{size of component containing } A \text{ when } u \text{ removed})$.
        (Assuming $A$ is in the other component).
        So we just need to check if `dist(A, B)` is large enough?
        
        Actually, there is a very fast check.
        If we pick a random node $r$.
        We compute $dist(u, r)$.
        This helps in determining the "direction" or "position" in the tree?
        
        Wait, look at the scoring again.
        We have plenty of queries.
        But we can't check all nodes.
        Can we construct the path $A \to B$?
        If we find $A, B$, we still don't know the sequence of nodes on the path.
        Wait, if we have $A$ and $B$, we can traverse the path from $A$ to $B$ by repeatedly finding a neighbor $v$ such that $dist(v, B) < dist(curr, B)$?
        No, that moves away from $B$. We want to move towards $B$.
        So we want neighbor $v$ such that $dist(v, B) = dist(curr, B) - 1$.
        Since we don't know neighbors, we must iterate all $i \in [1, N]$.
        This takes $N$ queries per step. Path length can be $N$. Total $N^2$. Too slow.
        We need a way to traverse without iterating all nodes.
        
        Key realization:
        The distance function defines the geometry.
        If we fix $A$ and $B$, we know $dist(x, A)$ and $dist(x, B)$ for any node $x$.
        $dist(x, A) + dist(x, B) \ge dist(A, B)$. Equality holds iff $x$ is on path $A-B$.
        So, if we find a node $x$ such that $dist(x, A) + dist(x, B) == dist(A, B)$, then $x$ is on the path.
        But finding *all* such nodes takes time.
        However, we only need to walk along the path.
        Actually, we don't even need to find the path nodes explicitly!
        We can just maintain a set of valid paths.
        Or use the probabilistic method.
        If we pick a node $u$ uniformly at random.
        We know the centroid $C$ is on the path between $u$ and $Farthest(u)$.
        This is a stronger statement: "For ANY $u$, $C$ lies on the path $u \to Farthest(u)$."
        If this is true, we don't need $A, B$.
        Just picking one $u$ is enough.
        Let's call $F(u)$ the node maximizing $dist(u, x)$.
        If $F(u)$ is unique, path is unique.
        If there are multiple farthest nodes, say $v_1, v_2$, path might be union? No, in a tree, there is a unique path between $u$ and $F(u)$.
        So if we can find one $u$, we can restrict our search to the path $u \to F(u)$.
        But again, we don't know the path nodes.
        
        Wait, maybe we don't need to walk on the path.
        We can just check a specific node $X$ to see if it is the centroid.
        How?
        We can check if $X$ splits the tree correctly.
        But we don't know the tree structure.
        However, if we query $dist(X, Y)$ for all $Y$, we can build the adjacency list?
        No, $O(N^2)$ queries.
        
        Wait, look at the limit again. 400,000 queries.
        $N=75000$.
        Maybe we can just try to find $C$ using a random walk or binary search on the value space?
        But node labels are not sorted.
        
        Let's rethink.
        Is there a specific property of distances from the centroid?
        For the centroid $C$, the average distance $\frac{1}{N} \sum dist(C, x)$ is minimized.
        For any other node $X$, $\sum dist(X, x) > \sum dist(C, x)$.
        We can approximate this sum.
        If we have a candidate $C$, we can sample a few nodes $Y$ and check $dist(Y, C)$.
        But how to compare sums without computing all?
        
        What if we maintain a candidate node?
        Actually, there is a very simple strategy used in Codeforces for this exact problem type (interactive, distance queries, find centroid).
        The strategy:
        1. Pick a node $x$ arbitrarily (e.g. node 1).
        2. Iterate $k$ from 1 to $N$ (randomly shuffled?)
           Check if $x$ is the centroid.
           Wait, we can't check efficiently.
        
        Let's assume the strategy:
        "Centroid is the node that minimizes the maximum distance to any other node?" No, that's center.
        "Centroid is the node that minimizes sum of distances?" No, that's median/centrum?
        Actually, in trees, the centroid and geometric center are often distinct but correlated.
        
        Wait, the problem statement says "exactly one centroid".
        This simplifies things.
        
        Let's use the randomized approach with verification.
        We pick a node $u$. We find the farthest node $v$ from $u$.
        We claim $C$ is on path $u \to v$.
        Now we want to find $C$ on this path.
        How to traverse the path efficiently?
        We can use a technique where we check midpoints.
        But we can't ask "is the midpoint on the path?".
        Wait, we can ask "distance between $u$ and $v$".
        If we could query "does node $x$ lie on path $u-v$?", we could binary search.
        But we can't.
        
        However, we can check if $x$ is "closer" to $C$ than some other node?
        
        Let's try this logic:
        The centroid $C$ has the property that for any node $z$, the maximum distance from $z$ to any node in the tree is $\ge dist(z, C) + \dots$
        
        Let's use the fact that we have $400,000$ queries.
        $N \le 75,000$.
        We can afford $O(N \log N)$ or even $O(N)$ with a small constant.
        We can simply iterate over all possible nodes $i=1 \dots N$ and check if $i$ is a centroid.
        But checking one node $i$ against all others takes $N$ queries.
        So we need a check that takes $O(1)$ or $O(\log N)$?
        Or we can check a fraction of nodes.
        
        What if we find $A, B$ (endpoints of diameter).
        The centroid lies on $A-B$.
        Also, the centroid lies on the path from $A$ to $B$.
        If we pick a random node $X$ on the path $A-B$?
        How to pick?
        We can maintain a range of indices? No.
        
        Wait, there's a simpler way.
        Pick any node $u$.
        Compute $dist(u, i)$ for all $i$? No.
        Just compute $dist(u, v)$ for a few candidates.
        
        Actually, there's a solution involving just finding the diameter and then picking the node on the diameter that splits it "evenly"?
        The centroid is not the midpoint of the diameter.
        But it is "close".
        The diameter length $L$. The centroid distance from endpoints is roughly $L/2$.
        Specifically, $dist(C, A)$ is determined by sizes.
        We can find $C$ by "walking" on the diameter.
        But we need to walk efficiently.
        
        Let's reconsider the "walking" idea.
        To move from $A$ to $B$, we need to find neighbors.
        Instead of finding neighbors explicitly, we can maintain a set of candidates.
        Or better:
        We can perform a "Binary Search" on the nodes of the tree using distances.
        Let's define a predicate $P(x)$: "Does $x$ lie in the component containing $B$ when $A$ is removed?"
        We can't check this easily.
        
        Wait, if we just want to minimize the sum of distances, we can use random sampling.
        If we have a candidate $c$, and a probe $p$.
        If we can show that moving $c$ towards $p$ reduces sum of distances...
        
        Let's try to simulate a traversal on the diameter.
        1. Find $A$ (arbitrary start -> farthest).
        2. Find $B$ (A -> farthest).
        3. Now we have $A, B$.
        We can now assume the centroid is on the path $A-B$.
        We want to find a node $X$ on this path.
        Let's maintain the current node $curr = A$.
        We want to move to the next node on the path towards $B$.
        Let this neighbor be $next$.
        Condition: $dist(curr, B) = dist(curr, next) + dist(next, B)$.
        And $dist(next, B) < dist(curr, B)$.
        We need to find such a $next$.
        Iterating all $i$ takes $O(N)$. Doing this $O(N)$ times is $O(N^2)$.
        But we can optimize.
        Actually, if we find $A, B$, we know $dist(A, B)$.
        Let $D = dist(A, B)$.
        We can perform a binary search on the distance $D$.
        If we pick a distance $k$, is there a node $X$ such that $dist(A, X) = k$ and $X$ is on path?
        If we can identify a node $X$ that is at distance $k$ from $A$ AND lies on the path to $B$.
        Wait, if $X$ is on path $A-B$, then $dist(X, B) = D - k$.
        So $X$ is on path $A-B$ iff $dist(X, A) + dist(X, B) = D$.
        We can check this!
        Algorithm:
        1. Find $A, B$. (Cost $\approx 2N$ queries).
           - Pick 1. Find farthest $u$.
           - Pick $u$. Find farthest $v$. Set $B=v$.
           - Then find $A$ (maybe repeat from 1? No, just ensure $A, B$ are far).
           Actually, standard diameter finding finds $A$ and $B$.
           Pick $r=1$. Find $u$ maximizing $dist(1, u)$.
           Find $v$ maximizing $dist(u, v)$.
           Then $u, v$ are endpoints. (Or one of them is).
           Let's call them $A=u, B=v$.
           Distance $D = dist(A, B)$.
        2. The centroid $C$ satisfies $dist(A, C) + dist(C, B) = D$.
           Also, for the centroid $C$, all other components have size $\le N/2$.
           We can use a binary search for $dist(A, C)$.
           Let $L=0, R=D$.
           Midpoint $m = (L+R)/2$.
           We want to find if there exists a node at distance $m$ on the path?
           We don't know the node.
           But we can query for ALL nodes $x$:
           Is there an $x$ such that $dist(A, x) = m$?
           If we iterate all $x$, we find if such node exists.
           If we find such node $x$, we check if $dist(x, B) = D-m$.
           (Note: due to floating point/distance, we might find multiple such $x$. Those on path satisfy equality).
           Let's collect all $x$ such that $dist(A, x) + dist(x, B) = D$. These are the nodes on the diameter path.
           This gives us the set of vertices on the path.
           There are at most $D+1 \le N$ such vertices.
           We can find them all by checking all $N$ nodes.
           Wait, this takes $2N$ queries (check $dist(A, x)$ and $dist(B, x)$ for all $x$).
           So far $4N \approx 300,000$ queries.
           This fits within 400,000!
           
           Now we have the explicit list of nodes on the path $A \to B$.
           Let them be $v_0, v_1, \dots, v_D$ (sorted by distance from A).
           $v_0 = A, v_D = B$.
           Now we just need to iterate through this list and find the centroid $C$.
           For each $v_i$ in the list, check if it is the centroid.
           How to check?
           We can just use the definition: max component size $\le N/2$.
           But we don't know component sizes for $v_i$ directly without building adjacency list.
           However, we can check if $v_i$ is a centroid by verifying if there is any component with size $> N/2$.
           How to find the largest component of $v_i$?
           The components of $v_i$ correspond to its neighbors.
           Neighbors of $v_i$ in the path?
           In the path list, $v_i$ is adjacent to $v_{i-1}$ and $v_{i+1}$.
           Are there other neighbors? Yes.
           But those neighbors belong to subtrees hanging off the path.
           The size of the "hanging" subtree containing a neighbor $u$ (not on path) is needed.
           We can find this size.
           For a node $u$ (neighbor of $v_i$ not on path), the component size is the number of nodes in that component.
           Since the graph is a tree, removing $v_i$ makes this component isolated.
           This component contains $u$.
           Any node $z$ in this component has $dist(z, v_i)$ path not passing through $v_i$.
           Also, for any $z$ in this component, $dist(z, A) + dist(z, B)$...
           Actually, if $z$ is in a branch at $v_i$, then the path $z \to A$ passes through $v_i$.
           Thus $dist(z, A) = dist(z, v_i) + dist(v_i, A)$.
           Similarly $dist(z, B) = dist(z, v_i) + dist(v_i, B)$.
           So $dist(z, A) + dist(z, B) = 2 \cdot dist(z, v_i) + D$.
           This distinguishes nodes on the path (where sum $= D$) from nodes in subtrees (sum $> D$).
           Specifically, sum equals $D$ iff on path.
           Sum $< D$ impossible.
           So we can group all nodes by "branch" off the path.
           Grouping:
           Nodes with $dist(x, A) + dist(x, B) == D$ are on path.
           For any node $x$ NOT on path, find which path node $v_i$ is "closest" to $x$.
           This node $v_i$ is the unique node on the path such that $dist(x, A) + dist(x, B) > D$ but closest $v_i$ satisfies something.
           Actually, for any $x$, the projection onto path $A-B$ is a unique node $p(x)$.
           $dist(x, p(x))$ is minimized.
           This projection property holds for trees.
           Moreover, $p(x)$ is the node on the path such that $dist(x, A) + dist(x, B)$ is minimized subject to $x$ projecting?
           Actually, $dist(x, A) + dist(x, B) = dist(p(x), A) + dist(p(x), B) + 2 \cdot dist(x, p(x)) = D + 2 \cdot dist(x, p(x))$.
           So if we know $D$, for any node $x$, we can compute $2 \cdot dist(x, p(x)) = dist(x, A) + dist(x, B) - D$.
           So $dist(x, p(x)) = (dist(x, A) + dist(x, B) - D) / 2$.
           This tells us the distance from $x$ to the path.
           But we still don't know WHICH path node $p(x)$ corresponds to.
           However, we can just sort the path nodes by index in our list $v_0 \dots v_D$.
           For any node $x$, we need to find its parent on the path?
           No, we just need to determine the component sizes for $C$.
           
           Algorithm Refinement:
           1. Find $A, B$ endpoints. Cost $2N$.
              - Pick $r=1$.
              - Loop all $i \in [1, N]$, find $u$ maximizing $dist(r, i)$.
              - Loop all $i \in [1, N]$, find $v$ maximizing $dist(u, i)$.
              - Let $A=r$ (wait, $u$ might be better). Let's use $u$ as $A$.
              - Let $B=v$.
              - Note: Need to store distances? No, just re-query?
              - Storing $N^2$ distances is too much memory.
              - We can recompute.
              - Cost $2N$ queries for finding $A$ (actually 2 rounds of scan).
              - Wait, finding farthest takes $N$ queries. So 2 scans = $2N$.
              - Finding $A$ from 1: 1 scan. $u$ is found.
              - Finding $B$ from $u$: 1 scan. $B$ is found.
              - $D = dist(A, B)$ (1 query).
              - Total so far: $2N + 1$.
           
           2. Identify nodes on path $A \to B$.
              - Iterate $i \in [1, N]$.
              - Compute $s = dist(A, i) + dist(B, i)$.
              - If $s == D$, add $i$ to List $P$.
              - Store $dist(A, i)$ for these nodes to keep order.
              - This takes $2N$ queries (scan with $A$ and $B$).
              - Wait, we can just do 1 pass? No, we need $dist(A, i)$ and $dist(B, i)$.
              - If we already queried $dist(A, i)$ during diameter finding, we can reuse?
              - In step 1, we computed $dist(u, i)$ for all $i$.
              - If $A=u$, we have all $dist(A, i)$.
              - We need $dist(B, i)$. So we query $dist(B, i)$ for all $i$. $N$ queries.
              - Sort nodes by $dist(A, i)$.
              - Total cost: $2N + N = 3N$.
              
           3. Find Centroid on Path $P$.
              - We have ordered list $v_0, v_1, \dots, v_D$.
              - For each $v_k$ in $P$, we need to check if it is centroid.
              - Checking one node $v_k$:
                - We need to know the size of the component attached to $v_k$ in each direction.
                - Directions: Towards $A$, Towards $B$, and subtrees attached to $v_k$.
                - Size towards $A$:
                  - This is the number of nodes in the component containing $A$ when $v_k$ removed.
                  - This component consists of nodes $x$ such that $p(x) \in P$ and $p(x)$ is in the path from $A$ to $v_k$ (exclusive of $v_k$?).
                  - Basically, nodes "above" $v_k$.
                  - If $x$ is in a subtree attached to $v_k$, it's not towards $A$.
                  - Wait, if $x$ projects to $p(x) \in \{v_j\}_{j < k}$, then $x$ is in the "up" component (towards $A$) relative to $v_k$.
                  - Wait, is this correct?
                  - If $x$ projects to $p(x)$, then path from $x$ to $A$ goes through $p(x)$?
                  - Yes.
                  - So, if we know for each node $x$ which path node $p(x)$ it projects to, we can count sizes.
                  - $p(x)$ can be determined by finding $j$ such that $v_j$ is "between" $p(x)$ and something?
                  - Actually, we need to assign each $x \notin P$ to some $v_j \in P$.
                  - How to find $j$ for a specific $x$?
                  - $x$ is assigned to $v_j$ if $dist(x, v_j) < dist(x, v_k)$ for all $k$.
                  - Equivalently, $v_j$ is the node on path minimizing distance.
                  - Since we have the list $P$, we can compute $dist(x, v_j)$ for all $j \in P$?
                  - Path length $D$ can be up to $N$.
                  - Doing this for all $x$ takes $N \times D$ queries. Too slow.
                  
              - Optimization:
                - We can determine $j$ without full distance scan.
                - Observe that for $x$, $dist(x, A)$ and $dist(x, B)$ are known.
                - $p(x)$ satisfies $dist(p(x), A) + dist(p(x), B) = D$.
                - And $dist(x, A) = dist(x, p(x)) + dist(p(x), A)$.
                - So $dist(x, A) = \delta + d_A$.
                - Also $dist(x, B) = \delta + d_B$.
                - Where $\delta = dist(x, p(x))$.
                - $d_A + d_B = D$.
                - So $d_A = dist(x, A) - \delta$.
                - $d_B = dist(x, B) - \delta$.
                - From the list $P$, we know $dist(v_j, A)$.
                - Is there a $v_j$ such that $dist(x, A) - \delta = dist(v_j, A)$?
                - We know $\delta = (dist(x, A) + dist(x, B) - D) / 2$.
                - So $d_A = dist(x, A) - (dist(x, A) + dist(x, B) - D)/2 = (dist(x, A) - dist(x, B) + D) / 2$.
                - We can calculate $d_A$ exactly for any $x$.
                - Then we just need to find $v_j$ such that $dist(v_j, A) == d_A$.
                - Since $P$ is sorted by distance from $A$, we can binary search or use a hash map (but map needs construction).
                - Since $P$ is sorted, we can use `std::lower_bound`.
                - Binary search takes $O(\log D)$ per $x$.
                - Total $O(N \log N)$. With $N=75000$, this is feasible.
                - Queries involved: We need to know $dist(A, x)$ and $dist(B, x)$.
                - Wait, we already computed these for all $x$ in Step 2!
                - So we have $d_A, d_B$ stored for each $x$.
                - So we can compute $d_A^{target}$ for each $x$.
                - Then find $v_j$ corresponding to $d_A^{target}$ in $P$.
                - Assign $x$ to that $v_j$.
                - Count sizes for each $v_j \in P$.
                - Note: Nodes ON the path project to themselves.
                  - For $v_k$, we add size of subtree hanging at $v_k$?
                  - Wait, the size of component "towards $A$" when $v_k$ removed:
                    - Includes $v_0 \dots v_{k-1}$ and all subtrees hanging on $v_0 \dots v_{k-1}$.
                  - The size of component "towards $B$" when $v_k$ removed:
                    - Includes $v_{k+1} \dots v_D$ and all subtrees hanging on $v_{k+1} \dots v_D$.
                  - The "subtree" at $v_k$ (hanging off path) consists of all nodes that project to $v_k$ EXCEPT $v_k$ itself.
                  - So for each $v_k$, we sum the counts.
                
                - Let's formalize:
                  - Initialize `size_on_path[j] = 1` for each $j \in [0, D]$.
                  - For each node $x \notin P$:
                    - Calculate $d_A = (dist(x, A) - dist(x, B) + D) / 2$.
                    - Find $j$ in $P$ such that $dist(A, v_j) == d_A$. (Should be exact match).
                    - Increment `count_at_v[j]`.
                  - Compute component sizes for each $v_k \in P$.
                    - `Up[k]`: Sum of `(count_at_v[j])` for $0 \le j \le k-1$. Plus $k$ nodes $v_0..v_{k-1}$? No, each $v_j$ counts as 1 itself.
                    - Let $S_j$ be total nodes whose projection is $v_j$ (including $v_j$).
                    - Then the component containing $A$ (when $v_k$ removed) corresponds to the projection nodes $v_0 \dots v_k$. Wait.
                    - If we remove $v_k$, the node $v_j$ (with $j < k$) is in the component containing $A$.
                    - So component size is $\sum_{j=0}^{k-1} S_j$?
                    - Wait, if $v_k$ is removed, $v_{k-1}$ becomes disconnected from $v_{k+1}$.
                    - So the component towards $A$ includes all $v_j$ with $j < k$ and their subtrees.
                    - So `Up[k] = \sum_{j=0}^{k-1} S_j`.
                    - `Down[k] = \sum_{j=k+1}^{D} S_j`.
                    - `Other[k] = S_k - 1`. (Subtrees at $v_k$ excluding itself).
                      - Wait, $v_k$ itself is removed.
                      - Subtrees at $v_k$ are nodes $x$ projecting to $v_k$ excluding $v_k$.
                      - So size is $(S_k - 1)$.
                    - Wait, is it possible `Other[k]` contributes to `Up` or `Down`?
                    - No, they are separate components when $v_k$ is removed.
                    - So components sizes are:
                      - $U_k = \sum_{j=0}^{k-1} S_j$ (contains $A$).
                      - $O_k = S_k - 1$ (branches at $v_k$). Wait, $O_k$ is one component size?
                        - Actually, `Other[k]` represents the size of a component attached to $v_k$.
                        - Wait, if $v_k$ has branches, they are disjoint components.
                        - $S_k$ counts ALL nodes in $v_k$'s "cluster" (projection $v_k$).
                        - This cluster includes $v_k$ and all its incident branches.
                        - Removing $v_k$ splits this cluster into several components.
                        - Each branch is a separate component.
                        - But we don't know individual branch sizes! We only know the total mass $S_k$.
                        - However, we need to check if ANY component size $> N/2$.
                        - We know $S_k - 1$ is the size of the "union" of all other branches?
                        - No, $S_k$ is the sum of sizes of all branches + 1 ($v_k$).
                        - If $S_k - 1$ is large, is it a single component? No.
                        - But if $S_k - 1 > N/2$, then there is definitely a component $> N/2$?
                          - No. $S_k - 1$ is sum of sizes of multiple components. Individual might be small.
                          - BUT, $S_k$ comes from the definition of centroid.
                          - Actually, wait.
                          - The projection logic groups nodes by their attachment to the path.
                          - If we attach $x$ to $v_j$, it means $v_j$ is the ONLY node on the path separating $x$ from $A$ and $B$.
                          - Wait, this is only true if $v_j$ is the unique path node closest.
                          - But if $x$ is attached to a branch at $v_j$, then yes.
                          - But what about the "main" components towards $A$ and $B$?
                          - The components towards $A$ and $B$ are the ones formed by removing $v_k$.
                          - Their sizes are exactly determined by the projections?
                          - If we remove $v_k$, any node $x$ with projection $v_j$ ($j < k$) remains in the component with $A$.
                          - Any node $x$ with projection $v_j$ ($j > k$) remains in component with $B$.
                          - What about nodes $x$ projecting to $v_k$?
                            - They are in branches attached to $v_k$.
                            - These form components when $v_k$ is removed.
                            - BUT, we don't know how many branches or their individual sizes.
                            - However, we DO know that $v_k$ is a centroid iff max(size of components) $\le N/2$.
                            - The sizes of the "up" and "down" components are `Sum_{j<k} S_j` and `Sum_{j>k} S_j`.
                            - What about the branches at $v_k$?
                            - Do we know their sizes?
                            - Wait, $S_k$ includes $v_k$ and all nodes in its attached branches.
                            - If we assume $S_k$ is distributed among branches.
                            - If there are NO branches (i.e. degree 2 on path), then $S_k=1$.
                            - If there are branches, $S_k > 1$.
                            - However, if $S_k$ is large, does it mean $v_k$ is not a centroid?
                            - Not necessarily.
                            - Example: Path $A \dots v_k \dots B$.
                            - $v_k$ has a large star centered at $v_k$? No, star is attached at $v_k$.
                            - If $v_k$ has a branch of size $M$, then removing $v_k$ creates a component of size $M$.
                            - If $M > N/2$, $v_k$ is not centroid.
                            - But $S_k$ sums up all nodes in the projection bucket.
                            - $S_k = 1 + \sum \text{size}(branch)$.
                            - If we know $S_k$, we don't know max branch size.
                            - UNLESS all branches are small?
                            - But we can't know.
                            
                            Wait, there is a catch.
                            For the Centroid $C$, the components created by removing $C$ all have size $\le N/2$.
                            If $C$ lies on the path, it means:
                            - Component "Above" (towards A) has size $\le N/2$.
                            - Component "Below" (towards B) has size $\le N/2$.
                            - All other components (branches attached to $C$) have size $\le N/2$.
                            
                            Our calculated $S_k$ is the TOTAL size of the cluster at $v_k$.
                            This $S_k$ is composed of branches.
                            Wait, if $v_k$ is the centroid, then all its immediate branches must have size $\le N/2$.
                            This is trivially satisfied if $S_k - 1$ is small? No.
                            But $S_k$ can be large if many small branches attach.
                            Wait, if $S_k$ is large, it doesn't violate the centroid condition directly (unless one specific branch is huge).
                            But we don't know branch sizes.
                            
                            However, there is a strong property:
                            The Centroid $C$ is unique.
                            If we can verify that $C$ is the unique node where $U_C \le N/2$ and $D_C \le N/2$ and ... ?
                            Wait, do we even need to check branch sizes?
                            Maybe for the Centroid, the branch sizes are implicitly handled by the $U$ and $D$ checks?
                            No.
                            
                            BUT, we can refine the set of candidates.
                            The Centroid MUST satisfy $U_{v_k} \le N/2$ and $D_{v_k} \le N/2$.
                            Why?
                            If we remove $v_k$, and $C=v_k$, then component "above" is the one containing $A$. Its size is $U_{v_k}$.
                            Component "below" is containing $B$. Its size is $D_{v_k}$.
                            Since $C$ is a centroid, BOTH $U_{v_k}$ and $D_{v_k}$ must be $\le N/2$.
                            So we can filter candidates using this condition.
                            Candidates satisfying both:
                            $\sum_{j=0}^{k-1} S_j \le N/2$ AND $\sum_{j=k+1}^{D} S_j \le N/2$.
                            Since $\sum_{all} S_j = N$, the second condition is equivalent to $S_k + \sum_{j=0}^{k-1} S_j \ge N/2$.
                            Wait, $N = U_k + S_k + D_k$.
                            Conditions: $U_k \le N/2$ and $D_k \le N/2$.
                            If we find a unique node satisfying this, great.
                            Is it unique?
                            Consider the path. $U_k$ increases as $k$ increases. $D_k$ decreases.
                            So $U_k \le N/2$ selects a prefix of the path.
                            $D_k \le N/2$ selects a suffix.
                            Intersection is a sub-segment of the path.
                            Any node in this sub-segment satisfies the path-conditions.
                            However, for the true centroid $C$, it also satisfies the branch condition.
                            Is it possible that a non-centroid node in this sub-segment also satisfies path conditions?
                            Yes.
                            Example: Line $1-2-3-4-5$. $N=5$.
                            Path is line. $v_0..v_4$.
                            $S_j = 1$ for all $j$.
                            $U_3 = \sum_{0..2} S_j = 3 > 2.5$. Fail.
                            $U_2 = \sum_{0..1} S_j = 2 \le 2.5$. Pass.
                            $D_2 = \sum_{3..4} S_j = 2 \le 2.5$. Pass.
                            $U_1 = 1$. Pass. $D_1 = 3$. Fail.
                            So only $v_2$ (index 2, node 3) satisfies $U \le N/2, D \le N/2$.
                            It seems on a line graph, this uniquely identifies the centroid.
                            What about a "fat" tree?
                            Suppose path $A-C-B$.
                            $A$ has large subtree of size 2.
                            $B$ has large subtree of size 2.
                            $C$ has small branches.
                            $S_A$ includes subtree at $A$.
                            $S_B$ includes subtree at $B$.
                            $U_C = S_A$. $D_C = S_B$.
                            If $S_A \le N/2$ and $S_B \le N/2$, then $C$ is candidate.
                            If we had another node on path?
                            If path is just $A, C, B$.
                            Then $C$ is the only internal node.
                            
                            Is it possible to have multiple candidates?
                            Suppose path length 3: $A - M - C - B$.
                            $S_A = 1$. $S_M = \text{large}$. $S_C = \text{small}$. $S_B = 1$.
                            Wait, if $S_M$ is large, then $S_M > N/2$ probably?
                            No, $S_M$ is the number of nodes attached to $M$.
                            If $S_M$ is huge, it contributes to "total nodes".
                            Wait, $S_M$ is sum of nodes projected to $M$.
                            If $S_M > N/2$, then $M$ is not centroid.
                            And any node "above" $M$ will have $U > N/2$?
                            $U_A = 0$. $U_M = S_A$. $U_C = S_A + S_M$.
                            If $S_M > N/2$, then $U_C > N/2$, so $C$ fails $U_C \le N/2$.
                            So if there is a "massive" projection on the path, nodes after it will fail $U$ check. Nodes before will fail $D$ check.
                            Thus, there is a unique intersection of intervals where $U \le N/2$ and $D \le N/2$.
                            Since $U_k$ is monotonically increasing and $D_k$ monotonically decreasing.
                            The valid range for $k$ is $[L, R]$.
                            Is the size of this range usually 1?
                            Consider the case where $U_k \approx N/2$ for a range of nodes.
                            E.g. path $1-2-3-4-5$.
                            $N=5$. $N/2 = 2.5$.
                            $U_0 = 0, D_0 = 4$.
                            $U_1 = 1, D_1 = 3$.
                            $U_2 = 2, D_2 = 2$. (Valid).
                            $U_3 = 3, D_3 = 1$.
                            So only index 2 works.
                            What if $N=4$. $N/2 = 2$.
                            Path $1-2-3-4$.
                            $U_1 = 1, D_1 = 2$. (Valid).
                            $U_2 = 2, D_2 = 1$. (Valid).
                            Both 2 and 3 are candidates on path.
                            Centroid is node 2 or 3?
                            Removing 2 gives components sizes 1, 2 (node 3, 4).
                            Wait, removing 2: {1}, {3,4}. Sizes 1, 2. $\le 2$. OK.
                            Removing 3: {4}, {1,2}. Sizes 1, 2. $\le 2$. OK.
                            But problem guarantees EXACTLY ONE CENTROID.
                            Ah! The problem statement: "exactly one centroid".
                            So the ambiguous case (two centroids) never happens.
                            Wait, trees can have 2 centroids?
                            Yes, if $N$ is even and splitting gives two equal halves.
                            But problem says "exactly one centroid".
                            So we don't need to worry about tie-breaking.
                            However, my condition $U \le N/2$ and $D \le N/2$ might still yield multiple candidates if $S_k$ is weird?
                            Actually, if $N$ allows 2 centroids, the "unique" claim prevents that.
                            So, the intersection of $U \le N/2$ and $D \le N/2$ should contain exactly 1 node (the centroid).
                            Because if it contained 2 adjacent nodes $k, k+1$:
                            $U_k \le N/2$ and $U_{k+1} = U_k + S_k$.
                            $D_{k+1} \le N/2$ and $D_k = D_{k+1} + S_{k+1}$.
                            If both satisfy:
                            $U_k \le N/2$.
                            $U_k + S_k \le N/2$.
                            $D_k = N - S_k - S_{k+1} + \dots$?
                            Wait, $U_k + D_k + S_k = N$.
                            $D_k = N - S_k - U_k$.
                            Condition for $k$: $U_k \le N/2$ and $N - S_k - U_k \le N/2 \implies S_k + U_k \ge N/2$.
                            Condition for $k+1$: $U_{k+1} \le N/2$ and $N - S_{k+1} - U_{k+1} \le N/2 \implies S_{k+1} + U_{k+1} \ge N/2$.
                            If both hold, it implies a split.
                            Given the uniqueness guarantee, we can just find the unique $k$ satisfying the condition?
                            Or just check the first valid one?
                            Wait, we can just pick the node that minimizes $\max(U_k, D_k, \text{max_branch\_size})$.
                            But we don't know branch sizes.
                            But we know that for the centroid, $\max(U, D)$ is minimized?
                            Actually, we just rely on the fact that only 1 node satisfies $U \le N/2$ AND $D \le N/2$.
                            Wait, let's check $N=4$ case again.
                            If tree was $1-2-3-4$ (line).
                            Removing 2: {1}, {3,4}. Sizes 1, 2. OK.
                            Removing 3: {4}, {1,2}. Sizes 1, 2. OK.
                            So 2 and 3 are both centroids.
                            But problem guarantees "exactly one centroid".
                            So the input tree will not be a line of length 4?
                            Or maybe 2 and 3 are the SAME centroid? No.
                            So the test cases provided will strictly have 1 centroid.
                            So the condition $U \le N/2$ and $D \le N/2$ will yield exactly 1 node.
                            Wait, is this true?
                            Suppose $U_k \le N/2$ and $D_k \le N/2$.
                            Then $k$ is a valid centroid w.r.t path components.
                            If $k$ is not the centroid (due to branch issue), then $k$ violates branch condition?
                            But we don't check branch condition.
                            So if a non-centroid node satisfies path condition, we might pick it.
                            BUT: if $k$ is not the centroid, does it violate $U \le N/2$ or $D \le N/2$?
                            Suppose $k$ is close to $C$.
                            Then $U_k \le N/2$ is likely true (since $U_C \le N/2$ and $U$ increases as we go away from $C$).
                            Wait, $U_k$ increases as we move away from $C$ towards $A$.
                            $D_k$ decreases as we move away from $C$ towards $A$?
                            Wait, let's trace.
                            Let $k$ increase. $U_k$ increases. $D_k$ decreases.
                            $U_k$ is size of component towards $A$.
                            $D_k$ is size of component towards $B$.
                            If $k$ is the centroid $C$, $U_C \le N/2$ and $D_C \le N/2$.
                            If we move to $k-1$ (towards $A$):
                            $U_{k-1} \le U_C$. So $U_{k-1} \le N/2$ holds.
                            $D_{k-1} = D_C + S_k$.
                            So $D_{k-1}$ might exceed $N/2$.
                            If $S_k$ is large, $D_{k-1} > N/2$.
                            If $S_k$ is small, $D_{k-1}$ might still be $\le N/2$.
                            Then we have multiple candidates $k, k-1$ satisfying path conditions.
                            Which one is centroid?
                            The one with smallest "max component".
                            If $S_k$ is small, maybe $k-1$ is the centroid?
                            But if $k-1$ is centroid, $C=k-1$.
                            Then $U_{k-1} \le N/2$ and $D_{k-1} \le N/2$.
                            Wait, if $C=k-1$ is centroid, then $U_{k-1} \le N/2$ and $D_{k-1} \le N/2$.
                            My derived condition $U \le N/2$ and $D \le N/2$ checks exactly this.
                            So IF the condition filters out non-centroids, we are good.
                            Does it?
                            If $C=k$, then $U_k, D_k$ satisfy.
                            If $k-1$ is not centroid, does it fail?
                            Suppose $C=k$. Then $U_k, D_k \le N/2$.
                            Check $k-1$. $U_{k-1} \le U_k \le N/2$. OK.
                            $D_{k-1} = D_k + S_k$.
                            If $D_{k-1} > N/2$, then $k-1$ fails.
                            If $D_{k-1} \le N/2$, then $k-1$ satisfies path condition.
                            But if $k-1$ satisfies, is it the centroid?
                            If $k-1$ satisfies, then max component towards $A$ and $B$ are balanced.
                            This implies $k-1$ divides the "backbone" correctly.
                            But is it possible $k-1$ is not centroid?
                            Only if some branch of $k-1$ is too large.
                            Wait, $S_k$ is the size of projection bucket $k$.
                            This bucket contains nodes attached to $k$.
                            If $k-1$ is considered, its neighbors on path are $k-2$ and $k$.
                            The component "towards $k$" has size $D_{k-1}$.
                            This component includes $k$ and everything "below" $k$ (including bucket $k$, bucket $k+1$, etc).
                            Wait, $D_{k-1}$ IS the size of component containing $B$ when $k-1$ removed.
                            This component includes $k$.
                            The node $k$ is part of it.
                            But the size of the component containing $k$ (when $k-1$ removed) is exactly $D_{k-1}$.
                            If $D_{k-1} \le N/2$, then this big component is small enough.
                            Inside this component, we have the node $k$.
                            Removing $k-1$ leaves this component intact? No, it's one piece.
                            But the definition of centroid is about components formed by removing $k-1$.
                            One such component is the one containing $B$ (which is $D_{k-1}$).
                            So if $D_{k-1} \le N/2$, then this component is valid.
                            What about the component containing $A$ (size $U_{k-1}$)? Valid if $\le N/2$.
                            What about components attached to $k-1$ (excluding path)?
                            These are formed by removing $k-1$.
                            Wait, my $U, D$ calculations only account for path segments.
                            What about subtrees at $k-1$?
                            These are exactly what is captured by $S_{k-1}$? No, $S_{k-1}$ is bucket size at $k-1$.
                            But $U_{k-1}$ is sum of buckets $j < k-1$.
                            $D_{k-1}$ is sum of buckets $j > k-1$.
                            Wait, the nodes in bucket $k-1$ (excluding $k-1$) are attached to $k-1$.
                            When we remove $k-1$, these nodes become separate components.
                            Their combined size is $S_{k-1} - 1$.
                            But individually they might be small.
                            However, if $S_{k-1} - 1$ is large, does it imply failure?
                            No, unless a single branch is large.
                            BUT, notice that if $k-1$ is a valid centroid candidate (satisfies $U, D \le N/2$), then $k-1$ splits the path nicely.
                            If $k$ was the true centroid, then $U_k, D_k \le N/2$.
                            If $k-1$ also satisfies, it means $U_{k-1} \le N/2$ (ok) and $D_{k-1} \le N/2$.
                            $D_{k-1} = D_k + S_k$.
                            Since $S_k \ge 1$, $D_{k-1} > D_k$.
                            So $D_{k-1}$ might cross the threshold.
                            If $D_k \le N/2$ and $D_{k-1} \le N/2$, then both are candidates.
                            But if $C=k$, then $C$ is the centroid.
                            If $k-1$ also works, then we have ambiguity?
                            Wait, problem guarantees exactly one centroid.
                            So it's impossible that both $k$ and $k-1$ satisfy the path conditions IF $k$ is the centroid.
                            Wait, if $k$ is the centroid, then removing $k$ yields components $\le N/2$.
                            Removing $k-1$ yields component containing $k$ (size $D_{k-1}$).
                            Since $k$ is centroid, $D_{k-1}$ must be $> N/2$?
                            Wait. If $k$ is centroid, then $S_k$ (the component towards $k$ from $k-1$ perspective?) No.
                            Let's check.
                            $k$ is centroid.
                            $D_{k-1}$ is the size of component containing $k$ when $k-1$ is removed.
                            This component contains $k$ and all nodes "below" $k$ (including $S_k, S_{k+1} \dots$).
                            Is it possible this component is $\le N/2$?
                            If yes, then $k-1$ would also be a candidate (since $U_{k-1} \le U_k \le N/2$).
                            If $k-1$ is candidate, and problem guarantees unique centroid, then $k-1$ must NOT be a centroid.
                            So it must fail the branch condition.
                            Meaning some branch of $k-1$ is large.
                            Wait, does $D_{k-1}$ include branches of $k$?
                            Yes.
                            If $S_k$ is large, $D_{k-1}$ will be large.
                            Actually, if $D_{k-1} \le N/2$, then all nodes "below" $k-1$ are small.
                            This includes the whole subtree at $k$.
                            If $S_k$ is small, then $k$ is a valid "attachment" point?
                            But $k$ itself is a centroid.
                            If $S_k$ is small (just $k$ and small branches), then $D_{k-1} \approx D_k$.
                            If $D_k \le N/2$, then $D_{k-1}$ might be $\le N/2$.
                            But if $k-1$ also satisfies, does it imply $k-1$ is a centroid?
                            Not necessarily. It implies $k-1$ splits the "path" well.
                            However, if $k$ is the centroid, then $k$ must be the unique node where max component is minimized?
                            Wait, actually, the condition $U \le N/2$ and $D \le N/2$ is NECESSARY for a centroid.
                            Is it SUFFICIENT?
                            If $U_k \le N/2$ and $D_k \le N/2$, it means $k$ divides the tree into two main parts of size $\le N/2$ via the path $A-B$.
                            The remainder is $S_k - 1$ (subtrees at $k$).
                            If $S_k - 1$ is also $\le N/2$, then $k$ divides the tree into ALL parts $\le N/2$?
                            No, $S_k-1$ is the SUM of subtrees.
                            The maximum of subtrees could be small while sum is small.
                            Or sum is small but one subtree is large? Impossible if sum is small.
                            If $S_k - 1 \le N/2$, then ALL subtrees at $k$ must be $\le S_k - 1 \le N/2$.
                            So the condition $U_k \le N/2$ and $D_k \le N/2$ AND $S_k \le N/2 + 1$ (actually $S_k-1 \le N/2$) would imply $k$ is a centroid.
                            Wait, $S_k$ is the bucket size.
                            Is $S_k$ the total number of nodes in the "cluster" at $k$?
                            Yes.
                            So if $S_k - 1 \le N/2$, then the sum of branch sizes is $\le N/2$.
                            So every branch size is $\le N/2$.
                            So $S_k - 1 \le N/2$ is a necessary condition.
                            Is it checked?
                            My algorithm checks $U_k$ and $D_k$.
                            Does it check $S_k$?
                            If we check all three conditions, we ensure $k$ is a centroid.
                            So, refined strategy:
                            For each candidate $k$ (in path range), check:
                            1. $U_k \le N/2$
                            2. $D_k \le N/2$
                            3. $S_k - 1 \le N/2$
                            If all satisfied, report $k$.
                            
                            Wait, do we really need $S_k - 1 \le N/2$?
                            If $k$ is centroid, $S_k - 1$ is sum of branch sizes.
                            Since each branch size $\le N/2$, their sum can be larger than $N/2$?
                            Yes. If there are two branches of size $N/3$ each, sum is $2N/3$.
                            $2N/3 > N/2$ for $N > 0$.
                            So $S_k - 1$ can be $> N/2$.
                            So checking $S_k - 1 \le N/2$ is NOT required/sufficient.
                            
                            Wait, we are stuck with finding which node satisfies $U \le N/2$ and $D \le N/2$ AND branch condition.
                            But we don't know branch condition directly.
                            HOWEVER, we noticed that if $k$ is centroid, it satisfies $U, D$.
                            Is it possible for a NON-centroid to satisfy $U, D$?
                            Yes.
                            But maybe the NON-centroid violating branch condition will be distinguished?
                            No.
                            Wait, the path $A-B$ is just ONE path.
                            The centroid is on SOME diameter.
                            We assumed $A-B$ is a diameter.
                            If $A-B$ is a diameter, then the centroid MUST be on it.
                            Proof: Centroid splits tree into $\le N/2$.
                            If centroid is not on diameter, then removing it leaves diameter endpoints in the same component (since they are not separated by C).
                            That component has size $> N/2$ (since it contains the diameter path? No, diameter path has length $L$. Number of nodes $\ge L+1$. If $L \ge N/2$, it's large).
                            Is diameter length always $> N/2$?
                            For $N=75000$, diameter can be 2 (star graph).
                            In star graph, diameter is 2. Centroid is center.
                            Diameter endpoints are leaves.
                            Removing center leaves components of size 1.
                            Center is on path between leaves.
                            So yes, centroid is on diameter.
                            Assuming $A, B$ found by 2-radius method are endpoints of a diameter.
                            Then centroid lies on path $A-B$.
                            And among nodes on path $A-B$, only the centroid will satisfy the "local balance" property properly.
                            
                            Is there a case where a node $k \ne C$ on path satisfies $U_k, D_k \le N/2$?
                            Yes, as seen in line graph $N=4$. 2 and 3 both satisfy.
                            But problem says exactly 1 centroid. So this case won't happen.
                            What about $N=6$?
                            Line $1-2-3-4-5-6$. $N=6$. $N/2=3$.
                            Centroids are 3 and 4.
                            Both satisfy $U, D \le 3$.
                            Again, 2 centroids.
                            So if the problem guarantees exactly 1 centroid, the "intersection of intervals" will likely contain only 1 node.
                            Let's assume this is safe.
                            Wait, is there any case with exactly 1 centroid where multiple nodes on path satisfy $U, D \le N/2$?
                            Suppose $N=5$. $N/2=2$.
                            Path $A-C-B$. $S_A=1, S_C=1, S_B=1$.
                            $U_C=1, D_C=1$. OK.
                            $U_A=0, D_A=4$.
                            $U_B=4, D_B=0$.
                            Only $C$ works.
                            
                            Suppose $S_A=2$ (one branch at $A$).
                            $U_C = 2$. $D_C = 1$. OK.
                            $U_A = 0, D_A = 5 - 2 = 3$. Fail ($3 > 2$).
                            So only $C$ works.
                            
                            It seems safe.
                            So finding the unique node satisfying $U \le N/2$ and $D \le N/2$ on the path should identify the centroid.
                            
                            Complexity:
                            1. Find $A, B$. (2 passes, $2N$ queries).
                            2. Build path $P$. (Scan $N$ nodes with $A, B$, sort $P$). ($2N$ queries).
                            3. Filter path nodes.
                               - Precompute cumulative sums of $S_j$.
                               - For each $k \in P$, check $U_k = CumSum[k-1] \le N/2$.
                               - Check $D_k = N - U_k - S_k \le N/2$.
                               - Identify unique $k$.
                            This step involves no queries, just arithmetic on precomputed data.
                            Wait, we need to compute $S_j$.
                            We computed $S_j$ as count of $x$ projecting to $v_j$.
                            We need to do this assignment.
                            Assignment requires binary search on $P$ for each node $x \notin P$.
                            Cost: $N \log N$ operations.
                            Total complexity is fine.
                            Total Queries $\approx 5N \approx 375,000$.
                            This fits within 400,000.
                            
                            Implementation details:
                            - Reading $N$.
                            - Flushing outputs.
                            - Helper function `query(u, v)`.
                            - Logic for finding $A, B$.
                              - Start with $start=1$.
                              - Loop $i=1 \dots n$ to find $farthest$ from $start$. Let it be $u$.
                              - Loop $i=1 \dots n$ to find $farthest$ from $u$. Let it be $v$.
                              - Now $v$ is farthest from $u$.
                              - Find $A=u, B=v$.
                              - Wait, to be sure about diameter, we should check if there's another node?
                              - The standard 2-BFS/DFS is sufficient to find endpoints of diameter in a tree.
                              - Let's stick with $A=u, B=v$.
                              - $D = dist(A, B)$.
                            - Build list $P$.
                              - Store pairs `(dist(A, i), dist(B, i))` for all $i$.
                              - Keep `dist(A, i)` to sort.
                            - Assign counts.
                              - For each $i$, calculate `dA = (dist(A, i) - dist(B, i) + D) / 2`.
                              - Binary search `dA` in `dist_vals` of $P$.
                              - `dist_vals` is vector of distances from $A$ to nodes in $P$.
                              - Use `std::lower_bound`.
                              - If `found`, increment count for that index.
                            - Check condition.
                              - Compute prefix sums of counts.
                              - Find $k$ such that `prefix_count[k] <= N/2` and `N - prefix_count[k] - count[k] <= N/2`.
                              - Output result.
                            
                            Corner case: What if $dist(A, i) + dist(B, i) != D$?
                            Triangle inequality $dist(A, i) + dist(B, i) \ge dist(A, B) = D$.
                            So sum $\ge D$.
                            If sum $> D$, then $i$ is not on path.
                            We only care about nodes where sum $== D$.
                            Wait, what if due to precision (integer arithmetic)? Distances are integers. Exact check is fine.
                            
                            Wait, step 1 finding $A, B$ uses $2N$ queries.
                            If we use the randomized approach (find farthest from random node) to avoid issues with starting node 1?
                            Starting at 1 is fine.
                            However, sometimes finding diameter requires checking multiple candidates?
                            Actually, the standard 2-pass works perfectly for trees.
                            First pass: find $u$ farthest from 1.
                            Second pass: find $v$ farthest from $u$. $u, v$ is diameter endpoints.
                            (Guaranteed for trees).
                            
                            Double check constraints: $N$ up to 75000.
                            Time limit 4s.
                            Vector resizing, sorting, binary search is fast.
                            IO is slow? Use fast IO.
                            `cin.tie(NULL); ios_base::sync_with_stdio(false);`
                            
                            One optimization:
                            We need `dist(A, i)` for all $i$ anyway.
                            We need `dist(B, i)` for all $i$.
                            This is done in step 2.
                            The queries in step 1:
                            Scan from 1: $N$ queries.
                            Scan from $u$: $N$ queries.
                            Total $2N$.
                            Step 2: $2N$ queries (to check $dist(A, i)$ and $dist(B, i)$ for all $i$).
                            Wait, do we need `dist(A, i)` again?
                            We computed it in scan 1. We can cache it!
                            We don't need to re-query.
                            So we only need to query `dist(B, i)` for all $i$. ($N$ queries).
                            So total queries $\approx 2N + N = 3N$.
                            $3 \times 75000 = 225,000$. Very safe.
                            Even safer than 400,000.
                            
                            Wait, is `dist(B, i)` needed for nodes on path? Yes.
                            And to calculate $d_A$.
                            Wait, calculating $d_A$ formula uses `dist(A, i)` and `dist(B, i)`.
                            `dist(A, i)` was queried in scan 1.
                            We must store it.
                            So caching is mandatory.
                            
                            Plan:
                            1. Read $N$.
                            2. $start=1$.
                            3. Vector $DistsA$ size $N+1$. Fill with -1.
                           Loop $i=1 \dots N$: query `? 1 i`. Store in $DistsA[i]$.
                           Find $u$ maximizing $DistsA[i]$.
                           4. Loop $i=1 \dots N$: query `? u i`. Store in $DistsU[i]$.
                           Find $v$ maximizing $DistsU[i]$.
                           $A=u, B=v$.
                           $D = DistsU[v]$. (Note: $DistsU[v]$ is dist from $u$ to $v$).
                           Wait, we didn't compute $dist(A, B)$ yet. It's in $DistsU[v]$? No.
                           Wait, step 4 computes distances from $u$ to all $i$.
                           So $DistsU[A]$ is 0. $DistsU[B]$ is dist(A, B).
                           Wait, variable naming:
                           Let $start=1$.
                           Query $1 \to i$ for all $i$. Store in `dist1`.
                           Find $u$ argmax `dist1`.
                           Query $u \to i$ for all $i$. Store in `distU`.
                           Find $v$ argmax `distU`.
                           $A=u, B=v$.
                           $D = distU[v]$.
                           
                           Now we have `dist1` (which is $dist(A', i)$ where $A'=1$) and `distU` ($dist(A, i)$).
                           Wait, `dist1` is not $dist(A, i)$.
                           We need `dist(A, i)` for all $i$.
                           So we must query `? u i` for all $i$ (which is `distU`).
                           But we need `dist(B, i)`.
                           $B=v$.
                           We haven't computed $dist(B, i)$ for all $i$.
                           So we need 3rd scan?
                           Wait, $B=v$.
                           We have `dist1` (dist from 1).
                           We have `distU` (dist from $A$).
                           We need `dist(B, i)`.
                           We can compute it now.
                           Loop $i=1 \dots N$: query `? B i`. Store in `distB`.
                           
                           So total scans:
                           Scan 1 (1->all): $N$ queries.
                           Scan 2 (A->all): $N$ queries.
                           Scan 3 (B->all): $N$ queries.
                           Total $3N$. Safe.
                           
                           Then process.
                           Construct $P$:
                           Filter $i$ where $distU[i] + distB[i] == D$.
                           Sort $P$ by $distU[i]$.
                           Compute `cnt` array.
                           Prefix sums.
                           Check condition.
                           Report result.
                           
                           Final check:
                           Input $N$.
                           Interactor format:
                           "? u v"
                           Flush.
                           "! x"
                           Flush.
                           
                           Data structures:
                           `vector<int> dist1(n+1)`, `distA(n+1)`, `distB(n+1)`.
                           `struct Node { int val; int id; }` for sorting.
                           
                           Corner case: $N=3$.
                           $1 \to 2$ (dist 1), $1 \to 3$ (dist 1). $u=2$ or $3$.
                           Say $u=2$.
                           $2 \to 1$ (1), $2 \to 3$ (2? No, 1-2-3 line -> dist 2. Star -> dist 1).
                           Case 1 (Line): 1-2-3.
                           Scan 1 (1->): $d[1]=0, d[2]=1, d[3]=2$. $u=3$.
                           Scan 2 (3->): $d[1]=2, d[2]=1, d[3]=0$. $v=1$.
                           $A=3, B=1$. $D=2$.
                           Scan 3 (1->): same as Scan 1? No, query again. $B=1$. $distB[3]=2, distB[1]=0, distB[2]=1$.
                           Filter:
                           $i=3$: $distU[3]=0, distB[3]=2$. Sum 2 == 2. In P.
                           $i=1$: $distU[1]=2, distB[1]=0$. Sum 2 == 2. In P.
                           $i=2$: $distU[2]=1, distB[2]=1$. Sum 2 == 2. In P.
                           Sorted P: 1, 2, 3 (order by distU).
                           Wait, distU is dist from 3. So order 3 (0), 2 (1), 1 (2).
                           Indices in P: $p_0=3, p_1=2, p_2=1$.
                           Counts:
                           Project 3 -> 3.
                           Project 2 -> 2.
                           Project 1 -> 1.
                           $S = \{3:1, 2:1, 1:1\}$.
                           Cumulative Sums (from end $A$? Wait, A=3, B=1).
                           $U_k$ is component towards $A$ (3).
                           $U_3 = 0$.
                           $U_2 = 1$.
                           $U_1 = 2$.
                           Check $U_k \le N/2=1.5$.
                           $U_3=0 \le 1.5$. $D_3 = 2 - 1 = 1 \le 1.5$. Valid.
                           $U_2=1 \le 1.5$. $D_2 = 2 - 1 - 1 = 0 \le 1.5$. Valid.
                           $U_1=2 > 1.5$. Invalid.
                           Both 3 and 2 valid.
                           Centroid of 1-2-3 is 2.
                           Why 3 failed $D_3$?
                           Wait $D_3$ (component towards B) size?
                           $D_3 = \sum_{j > 3} S_j$? No.
                           My formula for $D_k$ was $N - U_k - S_k$.
                           $S_3 = 1$. $U_3 = 0$. $N-U-S = 3 - 0 - 1 = 2$.
                           $2 \not\le 1.5$. So 3 fails.
                           $S_2 = 1$. $U_2 = 1$. $N-U-S = 3 - 1 - 1 = 1$.
                           $1 \le 1.5$. So 2 passes.
                           $S_1 = 1$. $U_1 = 2$. Fail.
                           So 2 is unique. Correct.
                           
                           Wait, earlier I calculated $D_3=1$.
                           $D_3$ is component containing $B=1$ when 3 removed.
                           Removing 3 from 1-2-3 leaves {1, 2}. Size 2.
                           So $D_3 = 2$.
                           My formula gave 2. Correct.
                           
                           Okay, the logic holds.
                           
                           One detail: Sorting.
                           Sort by `distU` to align with path from $A$ to $B$.
                           Wait, $A$ is $u$. $B$ is $v$.
                           $distU[x]$ is distance from $A$.
                           So sorting by `distU` sorts by proximity to $A$.
                           So index $k$ in sorted list is distance $k$ from $A$ along path.
                           Wait, distance in edges.
                           $p_k$ corresponds to distance $distU[p_k]$ from $A$.
                           Is $p_{k+1}$ necessarily adjacent to $p_k$?
                           Not necessarily adjacent in ID, but adjacent on path.
                           Wait, $p_k$ and $p_{k+1}$ are consecutive nodes on path $A-B$.
                           So yes.
                           $U_{p_k}$ is sum of $S$ for all nodes with dist $< distU[p_k]$.
                           Wait, we just need the set of $S_j$.
                           The condition $U_k \le N/2$ applies to the component "above" $k$ (towards $A$).
                           This component contains all nodes whose projection is on path segment $A \to k$ (exclusive).
                           Wait, projection is unique node $p$.
                           If we remove $p_k$, the component containing $A$ consists of nodes with projection $p_j$ for $j < k$.
                           Wait, is this correct?
                           If $x$ projects to $p_j$ with $j < k$, then path $x-A$ goes through $p_j$.
                           $p_j$ is closer to $A$ than $p_k$.
                           So path $x-p_k$ goes through $p_j$.
                           So yes, $x$ is in the component of $A$.
                           So $U_k = \sum_{j < k} S_j$.
                           Wait, $S_j$ includes $p_j$ itself.
                           Is $p_j$ in the component of $A$?
                           Yes, because $p_j \ne p_k$.
                           Wait, if $j < k$, then $p_j$ is in the "upper" part.
                           So $U_k = \sum_{j=0}^{k-1} S_j$.
                           And $D_k = \sum_{j=k+1}^{M} S_j$.
                           (where $M$ is last index).
                           Wait, check $N-U_k-S_k = \sum_{j=k+1} S_j$.
                           So formula $N - U_k - S_k$ holds.
                           Wait, my manual trace for $N=3$:
                           Sorted P: 3 (dist 0), 2 (dist 1), 1 (dist 2).
                           Indices 0, 1, 2.
                           $p_0=3$. $p_1=2$. $p_2=1$.
                           $S_0=1, S_1=1, S_2=1$.
                           $U_0 = 0$.
                           $U_1 = S_0 = 1$.
                           $U_2 = S_0+S_1 = 2$.
                           Formula check:
                           For $p_1=2$: $U=1$. $D=N-U-S = 3-1-1=1$.
                           Matches manual trace ($D_2$ size 1).
                           For $p_0=3$: $U=0$. $D=3-0-1=2$.
                           Matches manual trace ($D_3$ size 2).
                           
                           Looks solid.
                           
                           Complexity check:
                           Vector ops are fast. $O(N)$ or $O(N \log N)$.
                           Constraints $N=75000$.
                           $N \log N \approx 75000 \times 17 \approx 1.2 \times 10^6$.
                           Operations count is low.
                           Query count $3N = 225000$.
                           Plenty of headroom.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

using namespace std;

// Function to handle interaction
void solve() {
    int n;
    cin >> n;
    
    // Distance arrays
    // Index 1 to n
    vector<int> dist_start(n + 1); 
    vector<int> dist_A(n + 1);
    vector<int> dist_B(n + 1);
    
    // Find initial endpoint u from node 1
    // We scan all nodes from node 1 to find farthest
    // Optimization: if we just pick 1, it's fine.
    int u = 1;
    int max_dist_from_1 = -1;
    // Actually, we need to query.
    // To save queries, we can iterate.
    
    // Scan 1: From 1 to find u
    // We store dist from 1 to all nodes
    for (int i = 1; i <= n; ++i) {
        cout << "? 1 " << i << endl;
    }
    
    // Read responses
    for (int i = 1; i <= n; ++i) {
        cin >> dist_start[i];
    }
    
    // Find u maximizing dist
    u = 1;
    max_dist_from_1 = 0;
    for (int i = 2; i <= n; ++i) {
        if (dist_start[i] > max_dist_from_1) {
            max_dist_from_1 = dist_start[i];
            u = i;
        }
    }
    
    // Scan 2: From u to find v (endpoint of diameter)
    // Store dist from u to all nodes
    for (int i = 1; i <= n; ++i) {
        cout << "? " << u << " " << i << endl;
    }
    
    for (int i = 1; i <= n; ++i) {
        cin >> dist_A[i];
    }
    
    // Find v maximizing dist from u
    // Note: dist_A[u] should be 0
    int v = 1;
    int max_dist_from_u = 0;
    for (int i = 1; i <= n; ++i) {
        if (dist_A[i] > max_dist_from_u) {
            max_dist_from_u = dist_A[i];
            v = i;
        }
    }
    
    // v is the farthest node from u. 
    // The centroid is guaranteed to be on the path between u and v (diameter).
    // Let A = u, B = v.
    int A = u;
    int B = v;
    int D = max_dist_from_u; // Distance between A and B
    
    // Scan 3: From B to compute dist_B for all nodes
    // We can reuse dist_A? No, dist_B is needed.
    // Wait, in step 2 we computed dist_A.
    // We need dist_B for projection logic.
    for (int i = 1; i <= n; ++i) {
        cout << "? " << B << " " << i << endl;
    }
    
    for (int i = 1; i <= n; ++i) {
        cin >> dist_B[i];
    }
    
    // Collect nodes on the path A -> B
    // Condition: dist_A[i] + dist_B[i] == D
    struct PathNode {
        int id;
        int dist_A_val; // distance from A
        // We might need dist_B_val but it's derivable or stored
    };
    
    vector<PathNode> P;
    P.reserve(n);
    
    for (int i = 1; i <= n; ++i) {
        if (dist_A[i] + dist_B[i] == D) {
            P.push_back({i, dist_A[i]});
        }
    }
    
    // Sort P by distance from A
    sort(P.begin(), P.end(), [](const PathNode& a, const PathNode& b) {
        return a.dist_A_val < b.dist_A_val;
    });
    
    int m = P.size();
    // Counts of nodes projected to each path node
    // Using vector of size m initialized to 0
    vector<int> counts(m, 0);
    
    // We also need to efficiently find the projection node for each i
    // Since P is sorted by dist_A, and dist_A[i] maps to P[index].dist_A_val
    // For any node i, calculate projection distance dA_proj = (dist_A[i] - dist_B[i] + D) / 2
    // Then find index k such that P[k].dist_A_val == dA_proj
    
    // To speed up, we can store dist_A_vals in a separate array/vector for lower_bound
    vector<int> dist_A_P;
    dist_A_P.reserve(m);
    for (auto &node : P) {
        dist_A_P.push_back(node.dist_A_val);
    }
    
    // Process every node to determine which path node it belongs to
    // And accumulate counts.
    // Note: Nodes NOT on path will be counted towards some path node (their projection).
    // Nodes ON path will be counted towards themselves.
    
    for (int i = 1; i <= n; ++i) {
        int da = dist_A[i];
        int db = dist_B[i];
        // Check if on path
        if (da + db == D) {
            // Find index in P
            // Since P is sorted and P[k].id == i for some k
            // We can use binary search on dist_A_P
            auto it = lower_bound(dist_A_P.begin(), dist_A_P.end(), da);
            if (it != dist_A_P.end() && *it == da) {
                // find index
                int idx = distance(dist_A_P.begin(), it);
                counts[idx]++;
            }
        } else {
            // Not on path
            // Find projection point
            // Distance from A to projection point
            // dist(proj, A) + dist(proj, B) = D
            // dist(i, proj) = (da + db - D) / 2
            // dist(i, A) = dist(i, proj) + dist(proj, A) => da = delta + dpA
            // dpA = da - delta = da - (da + db - D)/2 = (da - db + D) / 2
            int dpA = (da - db + D) / 2;
            auto it = lower_bound(dist_A_P.begin(), dist_A_P.end(), dpA);
            if (it != dist_A_P.end() && *it == dpA) {
                int idx = distance(dist_A_P.begin(), it);
                counts[idx]++;
            }
        }
    }
    
    // Now compute component sizes
    // U[k]: sum of counts of nodes whose projection is on path A -> P[k] (exclusive)
    // Actually U[k] is sum of counts for indices < k
    // D[k]: sum of counts for indices > k
    
    // Check if we have a unique centroid
    // Since problem guarantees exactly one centroid, we can search for it.
    // We look for k such that U[k] <= n/2 and D[k] <= n/2.
    // Wait, my derivation showed that satisfying these two path conditions is likely sufficient 
    // or at least the centroid will satisfy it, and uniqueness implies only one node does.
    // But actually, we should be careful about strictness.
    // Let's just iterate and check.
    
    int ans_node = -1;
    int current_U = 0;
    
    // Prefix sums of counts
    // Actually, U[k] corresponds to sum_{j=0}^{k-1} counts[j]
    // But wait, counts[j] is the number of nodes projecting to P[j].
    // These nodes form the component attached to the path segment ending at P[j]?
    // When removing P[k], the component towards A contains all nodes projecting to P[0]...P[k-1].
    // So yes, U[k] = sum(counts[0...k-1]).
    // D[k] = sum(counts[k+1...m-1]).
    // Note: counts[k] includes P[k] itself.
    // The component size of "towards A" when P[k] removed is exactly sum_{j<k} counts[j].
    // The component size of "towards B" when P[k] removed is sum_{j>k} counts[j].
    // (Because P[j] for j<k is in the component containing A).
    
    for (int k = 0; k < m; ++k) {
        int U = current_U;
        // Compute D
        // Total nodes N = sum(all counts)
        // sum(all counts) = N.
        // D = N - U - counts[k];
        int D_comp = n - U - counts[k];
        
        if (U <= n / 2 && D_comp <= n / 2) {
            ans_node = P[k].id;
        }
        
        current_U += counts[k];
    }
    
    if (ans_node != -1) {
        cout << "! " << ans_node << endl;
    } else {
        // Should not happen given problem constraints and correctness logic
        // Fallback or debug
        cout << "! 1" << endl;
    }
}
```
Double check sorting and indexing.
`dist_A_P` stores `dist_A` values for nodes in `P` sorted.
`counts` array indexed by rank in sorted `P`.
When we query a node $i$ on path (so `dpA == dist_A_P[idx]`), we increment `counts[idx]`.
When we query node not on path, we calculate `dpA` and find matching `idx`.
The logic holds: projection logic assigns nodes to the path node closest to them (on the diameter).
Wait, is the closest path node always the correct one?
Yes, in a tree, projection onto a path is well-defined and unique.
And $dpA = (da - db + D) / 2$ calculates exactly the distance of the projection from $A$.
And `dist_A_P` contains exactly these distances (since $P$ is sorted).
So mapping is correct.

One edge case: Floating point division?
`da - db + D` must be even.
In a tree with integer distances, `da + db` and `D` have same parity (both even or both odd relative to path distance? No.)
Triangle inequality: $da + db = D + 2 \cdot dist(i, proj)$.
So $da + db - D$ is always even.
Thus $dpA$ is integer.

Complexity:
Queries: $3N$.
Time: Sorting $O(N \log N)$, Loop $O(N \log N)$.
Space: $O(N)$.
All good.

```cpp
// Final check of headers and namespaces
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;
// ... code ...
```
The prompt asks for code wrapped in ```cpp.
Also need to ensure flushing is done correctly. `endl` flushes.
Input reading `cin >> n`.
Output formatting `? u v` and `! x`.
Code looks complete.

Wait, check memory usage.
Vectors of size 75000 ints. $3 \times 75000 \times 4$ bytes $\approx 1$ MB. Very low.
Stack depth is minimal.
Execution time dominated by IO and logic, 4 seconds is plenty.

One minor thing:
Is it possible that $da - db + D$ is negative?
$D = da_{min} + db_{min}$.
Actually $D$ is max possible distance.
For any node $i$, $da + db \ge D$.
$dpA = (da - db + D)/2$.
Also $dpA + dpB = D$.
$dpA \ge 0$.
So lower bound check is safe.

Final check on the scoring:
Base score 100k queries.
My solution uses $\approx 225k$ queries (for $N=75k$).
It is well within limits.

```cpp
/*
    Solution for Centroid Guess
    Approach:
    1. Find diameter endpoints A and B using 2 radii BFS logic (implemented via distance queries).
    2. Collect all nodes lying on the path between A and B.
    3. Determine for every node in the tree which point on the path A-B it projects to.
       This effectively partitions the tree nodes into sets attached to path nodes.
    4. Iterate through the path nodes from A to B, accumulating the sizes of subtrees/hanging components.
    5. Identify the unique node on the path that splits the remaining mass (towards A and towards B) 
       both being <= N/2. This node is the centroid.
*/

#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int n;
    if (!(cin >> n)) return 0;

    // Arrays to store distances
    vector<int> dist_start(n + 1);
    vector<int> dist_A(n + 1);
    vector<int> dist_B(n + 1);

    // --- Step 1: Find Endpoint A ---
    // Find node u farthest from node 1
    int u = 1;
    int max_d = -1;
    
    for (int i = 1; i <= n; ++i) {
        cout << "? 1 " << i << "\n";
        cin >> dist_start[i];
    }

    for (int i = 1; i <= n; ++i) {
        if (dist_start[i] > max_d) {
            max_d = dist_start[i];
            u = i;
        }
    }
    
    // --- Step 2: Find Endpoint B (Diameter endpoint) ---
    // Find node v farthest from u
    int v = 1;
    max_d = 0;
    // Re-use buffer dist_A for distance from u
    for (int i = 1; i <= n; ++i) {
        cout << "? " << u << " " << i << "\n";
        cin >> dist_A[i];
    }

    for (int i = 1; i <= n; ++i) {
        if (dist_A[i] > max_d) {
            max_d = dist_A[i];
            v = i;
        }
    }

    int A = u;
    int B = v;
    int diameter_len = dist_A[v]; // Distance between A and B

    // --- Step 3: Collect Nodes on Path A-B ---
    // We need distances from B as well to verify path membership and projection
    for (int i = 1; i <= n; ++i) {
        cout << "? " << B << " " << i << "\n";
        cin >> dist_B[i];
    }

    // Identify nodes on the simple path between A and B
    // A node i is on the path if dist(A, i) + dist(B, i) == dist(A, B)
    // We store them sorted by dist(A, i) to linearize the path
    vector<pair<int, int>> P; // Stores {dist_from_A, node_id}
    P.reserve(n);
    
    for (int i = 1; i <= n; ++i) {
        if (dist_A[i] + dist_B[i] == diameter_len) {
            P.push_back({dist_A[i], i});
        }
    }

    // Sort by distance from A
    sort(P.begin(), P.end());

    // Extract distances for binary search
    vector<int> dist_A_vals(P.size());
    for(int i=0; i<P.size(); ++i) dist_A_vals[i] = P[i].first;

    // Array to store counts of nodes projecting to each path node
    int m = P.size();
    vector<int> proj_counts(m, 0);

    // --- Step 4: Distribute Nodes to Path Projections ---
    for (int i = 1; i <= n; ++i) {
        int da = dist_A[i];
        int db = dist_B[i];
        int dAB = diameter_len;
        
        // Calculate distance from A to the projection of i on the path A-B
        // Property: dist(proj, A) = (dist(i, A) - dist(i, B) + dist(A, B)) / 2
        int dA_proj = (da - db + dAB) / 2;
        
        // Find the path node corresponding to this distance
        // Since dist_A_vals is sorted, use binary search
        auto it = lower_bound(dist_A_vals.begin(), dist_A_vals.end(), dA_proj);
        
        if (it != dist_A_vals.end() && *it == dA_proj) {
            int idx = distance(dist_A_vals.begin(), it);
            proj_counts[idx]++;
        }
    }

    // --- Step 5: Locate Centroid ---
    // Iterate through path nodes and check centroid condition.
    // The centroid must satisfy: 
    // 1. Component size towards A <= n/2
    // 2. Component size towards B <= n/2
    // (These are necessary conditions and sufficient for the unique centroid on the path)

    int ans = -1;
    int current_size_A_side = 0;
    
    for (int k = 0; k < m; ++k) {
        // Component size towards A (excluding current node P[k]) is sum of projections strictly before k
        // This matches current_size_A_side accumulated so far
        
        // Component size towards B (excluding current node P[k]) is N - (size_A_side + size_of_cluster_at_k)
        // Cluster at k is proj_counts[k]
        
        int size_A = current_size_A_side;
        int size_B = n - current_size_A_side - proj_counts[k];
        
        if (size_A <= n / 2 && size_B <= n / 2) {
            ans = P[k].second; // The original node ID
        }
        
        // Update accumulation for next iteration
        current_size_A_side += proj_counts[k];
    }

    if (ans != -1) {
        cout << "! " << ans << "\n";
        cout.flush();
    }

    return 0;
}
```
The logic seems sound.
Constraints check:
$N \le 75000$.
Queries: $3N + O(N)$. $3 \times 75000 = 225000 < 400000$.
Time: $O(N \log N)$. 4s is ample.
Memory: $O(N)$. 512MB is ample.

One tiny thing:
`cout << "? " << B << " " << i << "\n";`
Need to ensure flush. `\n` doesn't flush automatically without sync off/cout tie null.
With `ios_base::sync_with
````

---

## 直接SFT 样本(sample_idx=2,得分 0.0,completion_tokens=1237)

````
The key idea is to maintain a candidate path between two nodes u and v, where the centroid must lie on this path. We'll try to extend this path or find its middle point.

Initially, set any two nodes as u and v (say 1 and 2). Then:
- If dist(u,v) = dist(v,u) + |dist(u,c) - dist(v,c)| for all c, then one node is adjacent to the centroid, so we can use it directly.
- Otherwise, check if there's an adjacent vertex w such that moving from the farther end toward the closer end reduces distance significantly. If dist(farther, closer) > dist(farther, neighbor_of_closer) + distance_gain, move along that direction.
- When extending stops, take the midpoint of the current path. For each node, estimate its "level" relative to the centroid using depth differences weighted by distances to both endpoints. A smaller level suggests being closer to the centroid. Select the minimum level node and output it.
</think>

```cpp
#include <bits/stdc++.h>
using namespace std;

int n;
vector<int> adj[75005];
vector<int> parent_path;

int dist(int u, int v) {
    cout << "? " << u << " " << v << "\n";
    int x; cin >> x;
    return x;
}

void bfs_dist(int start, vector<int> &d) {
    d.assign(n + 1, -1);
    queue<int> q;
    q.push(start);
    d[start] = 0;
    while (!q.empty()) {
        int u = q.front(); q.pop();
        for (int v : adj[u]) {
            if (d[v] == -1) {
                d[v] = d[u] + 1;
                q.push(v);
            }
        }
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    cin >> n;

    // Build full tree by querying all edge lengths
    for (int i = 1; i <= n; ++i) {
        for (int j = i + 1; j <= n; ++j) {
            int d = dist(i, j);
            adj[i].push_back(j);
            adj[j].push_back(i);
        }
    }

    // BFS to compute depths from root (node 1)
    bfs_dist(1, vector<int>(n + 1));

    // Initial candidate pair for centroid search
    int u = 1, v = 2;
    bool done = false;
    
    // First pass: check if either endpoint is adjacent to centroid
    for (int c : adj[u]) if (adj[c].size() >= adj[u].size()) u = c;
    for (int c : adj[v]) if (adj[c].size() >= adj[v].size()) v = c;
    
    bool valid_u = true; for (int c : adj[u]) if (dist(u, c) + dist(c, v) != dist(u, v)) valid_u = false;
    bool valid_v = true; for (int c : adj[v]) if (dist(v, c) + dist(c, u) != dist(v, u)) valid_v = false;
    
    if (valid_u && !valid_v) u = v = u;
    else if (!valid_u && valid_v) v = u;
    else {
        // Extend the path u-v until no better adjacent vertex exists
        while (!done) {
            int du = dist(u, v), dv = dist(v, u);
            int far = (du > dv) ? u : v;
            int clo = (du > dv) ? v : u;
            
            // Try neighboring vertices of far and compare with neighbor of closer
            int best_improve = 0;
            int next_far = far;
            
            for (int nb : adj[far]) {
                int dnb = dist(nb, clo);
                int improve = abs(dnb - du);
                if (improve > best_improve) {
                    best_improve = improve;
                    next_far = nb;
                }
            }
            
            if (best_improve > 0) {
                u = next_far;
                continue;
            }
            
            done = true;
        }
    }
    
    // Compute distance matrix between final u and all candidates
    vector<vector<int>> D(n + 1, vector<int>(n + 1));
    bfs_dist(u, D[u]);
    bfs_dist(v, D[v]);
    
    int best_node = u;
    double best_level = 1e9;
    
    // Evaluate level of every candidate node relative to centroid position
    for (int x = 1; x <= n; ++x) {
        double dx = D[u][x], dy = D[v][x];
        double sumdx = 0, sumdy = 0;
        
        for (int y = 1; y <= n; ++y) {
            sumdx += D[x][y];
            sumdy += D[y][x];
        }
        
        double level = fabs(dy - dx) / (sumdy + sumdx);
        if (level < best_level) {
            best_level = level;
            best_node = x;
        }
    }
    
    cout << "! " << best_node << "\n";
    cout.flush();
    
    return 0;
}
```
````
