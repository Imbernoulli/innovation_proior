# 题 222 — base均分 0.0 vs 直接SFT均分 0.0(掉点 0.0)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=0,得分 0.0,completion_tokens=32768)

````
The user wants a C++ solution for an interactive competitive programming problem called "Hedgehog Graph".

**Problem Constraints & Details:**
- There are $n$ hedgehog graphs, where $n \le 10$.
- Each graph has $10^6$ vertices.
- Structure of a hedgehog graph:
    - Directed graph.
    - Each vertex has exactly one outgoing edge.
    - Contains exactly one directed cycle of length $\ge 3$.
    - No loops (length 1) or cycles of length 2.
    - For every edge $u \to v$, $v$ belongs to the *directed cycle*. This is a crucial part of the definition. Wait, let me re-read carefully.
    - "For every edge e = u -> v in the hedgehog graph, v belongs to the aforementioned single directed cycle."
    - If $v$ always belongs to the cycle, that implies there are no nodes outside the cycle?
    - Let's check the standard "Rho-shaped" graph or functional graph definition. Usually, functional graphs have components consisting of a cycle with trees rooted on the cycle nodes pointing towards it.
    - The problem statement says: "A hedgehog graph is a directed graph where each vertex has exactly one outgoing edge and contains exactly one directed cycle... For every edge e = u -> v ... v belongs to the aforementioned single directed cycle."
    - If $v$ belongs to the cycle for *every* edge $u \to v$, then the target of any edge must be part of the cycle. This means all vertices point directly to vertices on the cycle. Since each vertex has exactly one outgoing edge, this means *all* vertices in the graph are part of the cycle itself.
    - So, the graph is simply a simple directed cycle involving all $10^6$ vertices?
    - Or does "v belongs to the aforementioned single directed cycle" mean something else?
    - Let's re-read: "contains exactly one directed cycle of length at least 3".
    - "For every edge e = u -> v ... v belongs to the aforementioned single directed cycle."
    - If $v$ is the target of $e$, and $v$ must be in the cycle, then every node $u$ points to a node $v$ that is in the cycle. Since every node has an out-degree of 1, the set of targets is the cycle. Thus, the domain is also the cycle.
    - However, if there were nodes not in the cycle, say a tail leading into the cycle, let $u$ be a node on the tail. $next(u)$ would eventually reach the cycle. But if $next(u)$ is in the cycle, does that satisfy the condition?
    - Example: Cycle $1 \to 2 \to 3 \to 1$. Node $4 \to 1$. Here $u=4, v=1$. $v=1$ is in the cycle. Condition satisfied?
    - What about $5 \to 4$? Then $v=4$ must be in the cycle. But we assumed 4 is not.
    - If *every* edge $u \to v$ implies $v \in \text{Cycle}$, then by induction starting from any node, the next node is in the cycle. If a node is on a tail leading to a cycle, its neighbor might be the entry point to the cycle (which is in the cycle).
    - But wait, if a node $w$ is on the tail, it points to some $z$. $z$ must be in the cycle. So every tail must have length 1? i.e., every node on the tail points directly to a cycle node.
    - If $u$ points to $v \in Cycle$, then $u$ is essentially connected to the cycle.
    - BUT, if $u$ is not in the cycle, then $u$ cannot be reached from the cycle (since cycle nodes only point to other cycle nodes).
    - Is it possible to have multiple components? No, "a hedgehog graph" usually implies a single connected component structure in these contexts, but technically $N=10^6$.
    - Actually, looking at similar problems (e.g., Codeforces 868C etc?), the definition of "functional graph" often allows tails.
    - Let's look closely at the text: "contains exactly one directed cycle ... For every edge e = u -> v ... v belongs to the aforementioned single directed cycle."
    - This phrasing is extremely strong. It means the image of the function defined by the edges is exactly the set of vertices on the cycle.
    - If $v$ must be in the cycle for all $u$, then the set of vertices mapping to anything is exactly the set of vertices in the cycle.
    - This implies $V = \text{Cycle}$. Because if there was a vertex $u$ not in the cycle, it maps to some $v$. If $v \in \text{Cycle}$, then $u$ is distinct from $v$. Where does $u$ come from? Some $w \to u$. Then $u$ must be in the cycle. Contradiction.
    - Wait, root cause: In a finite graph where every node has out-degree 1, every component contains exactly one cycle. Nodes can form trees rooted on the cycle nodes (edges directed towards cycle).
    - If $u \to v$, then $v$ is closer to the cycle (distance reduced by 1).
    - If the rule "v belongs to the cycle" applies to ALL edges, then there can be NO nodes whose immediate successor is NOT in the cycle.
    - This means for any node $u$, $next(u)$ is in the cycle.
    - Consider a tree component attached to a cycle node $c$. A node $t$ in the tree (leaf of the tree in terms of edges going away from cycle) has $next(t)$ pointing towards cycle.
    - If $t$ is distance $k > 1$ from cycle, $next(t)$ is at distance $k-1$. If $k-1 > 0$, $next(t)$ is not in cycle. But the problem says $next(t)$ MUST be in cycle.
    - So $k-1$ must be 0. So distance is 1.
    - This implies the graph consists of the cycle nodes, plus potentially some nodes that point directly to the cycle nodes.
    - However, if $u$ is such a node ($u \notin \text{Cycle}, next(u) \in \text{Cycle}$), what points to $u$? Say $w \to u$. Then $u$ must be in cycle. Contradiction.
    - Unless $u$ has no incoming edges (is a source). But in a functional graph on $N$ vertices, sources exist.
    - Let's reconsider the interpretation. Maybe the graph is *just* the cycle?
    - "Kipa has n hedgehog graphs with $10^6$ vertices."
    - "Each vertex is numbered from 1 to $10^6$."
    - If the graph was just a cycle, why mention the query mechanism allowing arbitrary jumps?
    - And why limit $x$ up to $5 \cdot 10^{18}$?
    - If the graph is just a simple cycle, finding the length is equivalent to finding the smallest $x > 0$ such that `query(1, x) == 1`.
    - With cycle length $L$, `query(v, k) == v` iff $k$ is a multiple of $L$.
    - Since we want $L$, binary search works.
    - The constraint $N_{vertices} = 10^6$ suggests the cycle length could be up to $10^6$.
    - Wait, if the graph contains tails (nodes not in cycle), the structure is still "hedgehog".
    - Re-read: "For every edge e = u -> v ... v belongs to the aforementioned single directed cycle."
    - If $u$ is not in the cycle, let $v = next(u)$. Then $v$ must be in the cycle.
    - If $w \to u$, then $u$ must be in the cycle.
    - By reverse induction, if there is a path ending at the cycle not being on the cycle, say $w_1 \to w_2 \to c$ ($c \in Cycle$), then $w_2$ points to $c \in Cycle$, so $w_2 \notin Cycle$ is allowed. But $w_1$ points to $w_2$. Does the condition apply to $w_1 \to w_2$? Yes, "For every edge e=u->v". So $v$ (which is $w_2$) must be in the cycle.
    - If $w_2$ is in the cycle, then $w_1$ points to a cycle node.
    - But if $w_1$ points to $w_2$ ($\in Cycle$), $w_1$ is not in cycle necessarily.
    - What about $z \to w_1$? Then $w_1$ must be in the cycle.
    - If the graph is connected (as implied by "The interactor guarantees that there exists at least one hedgehog graph"), then all nodes are reachable from the cycle? No, edges go $u \to v$. Reachability flows outwards from the cycle.
    - Wait, functional graph components are: Cycle + Trees rooted on cycle (edges directed towards cycle).
    - If $u \to v$, $u$ is parent of $v$ in the tree sense. The flow goes towards the cycle.
    - The condition "For every edge e = u -> v, v belongs to the cycle" means: for ANY $u$, $next(u)$ is in the cycle.
    - This means every node has its successor on the cycle.
    - This means no node has a successor that is NOT on the cycle.
    - In the "Tree towards Cycle" model, nodes on the branches have successors further down the branch until they hit the cycle.
    - If the condition holds for ALL edges, then even the node immediately preceding the cycle node must point to it (so that node's successor is the cycle node).
    - But what about the node preceding THAT node? Its successor must be the first branch node.
    - But the condition says the successor must be IN THE CYCLE.
    - So the first branch node must be in the cycle.
    - This implies there are no branch nodes.
    - Therefore, the ONLY nodes are those in the cycle.
    - If so, the graph IS the cycle. The vertices are $1..10^6$.
    - Why $10^6$ vertices? Just to define the space. The cycle could use all of them or fewer?
    - If the graph is just a cycle of length $L$ involving some subset of vertices, and vertices are $1..10^6$.
    - Wait, if the graph is just a cycle, and vertices are labeled $1..10^6$, do unused vertices exist?
    - "Kipa has n hedgehog graphs with 10^6 vertices."
    - Usually this means the vertex set size is $10^6$.
    - If the condition "v belongs to cycle" holds for all edges, then every vertex $u$ must point to a vertex in the cycle. Since $u$ has out-degree 1, $u$ itself is just a label.
    - If $u$ points to $v \in Cycle$, and $u$ is not in cycle... who points to $u$? Let $w \to u$. Then $u \in Cycle$ (by condition). Contradiction.
    - So if the graph is weakly connected (implied by "hedgehog graph" singular entity per instance?), then there are no tails. All vertices are part of the cycle.
    - Is it possible to have multiple components? "exactly one directed cycle". Usually implies per component.
    - Given the strict wording "A hedgehog graph is...", it defines the object.
    - If the wording implies a single connected component with $10^6$ nodes, and every edge target is in the unique cycle, then all nodes are in the cycle.
    - Thus, the graph is a permutation cycle of length $10^6$? Or maybe length is unknown but $10^6$ is max?
    - "with $10^6$ vertices".
    - If the graph structure was complex, the scoring based on queries would matter more (finding cycle length in rho graph is hard without traversing).
    - Finding cycle length in a pure cycle is easy. Binary Search.
    - Query `? v x` returns `(v + x) % L` adjusted for 1-based indexing. Specifically, result is `v + x` in modular arithmetic relative to the cycle position.
    - We need to find $L$.
    - If the graph is purely a cycle, $L$ divides the number of steps to return to start.
    - Wait, checking connectivity again.
    - If I assume the graph is a simple cycle, does that fit "hedgehog graph"?
    - "contains exactly one directed cycle of length at least 3".
    - If graph is simple cycle, yes.
    - "For every edge e = u -> v, v belongs to the cycle". In a simple cycle, $v$ is always in the cycle. Yes.
    - So a simple cycle is a valid Hedgehog graph.
    - Are there other valid ones? Only if isolated nodes exist (disconnected components)?
    - "exactly one directed cycle". If there are isolated vertices not pointing anywhere? No, "each vertex has exactly one outgoing edge".
    - If there are multiple components, each has a cycle. But "contains exactly one directed cycle". So the whole graph must be one component.
    - So it must be a single connected component.
    - So it must be a single cycle containing all $10^6$ vertices?
    - If it contains all $10^6$ vertices, the cycle length $L$ must be $10^6$.
    - That seems too trivial for an interactive problem with a scoring system based on queries.
    - Let's re-read "Kipa has n hedgehog graphs with 10^6 vertices."
    - Could "with 10^6 vertices" refer to the *range* of IDs, or the count? "Each vertex is numbered from 1 to 10^6".
    - It likely means the universe of vertices is $1..10^6$. The actual vertices present in the graph are the whole set.
    - Perhaps the definition allows $u$ such that $next(u)$ is in cycle, but $u$ itself is not, provided $u$ is not reachable from cycle?
    - If $u \to v$ ($v \in Cycle$), and we ask for pre-images. If $u$ is not in cycle, who points to $u$? Someone $w$. $next(w)=u$. Then $u$ must be in cycle. Contradiction.
    - Unless $u$ has no pre-image. i.e. $u$ is a source.
    - Can we have multiple components where each has exactly one cycle, but total cycles = 1?
    - Yes, one large component with 1 cycle.
    - But we proved earlier that implies $V=Cycle$.
    - Let's check the example interaction.
    - Input: `1`, then `3` (graph 1), `7`, `10`, `1`.
    - Output:
      `? 1 2`
      `? 2 5`
      `? 10 11`
      `! 11`
    - Here $n=1$.
    - Vertex 1, step 2 returns ?
    - Vertex 2, step 5 returns ?
    - Vertex 10, step 11 returns ?
    - Answer 11.
    - If cycle length is 11.
    - Vertices used: 1, 2, ..., 10, 11 (maybe?).
    - Wait, if answer is 11, the cycle length is 11.
    - With $N=10^6$, maybe the cycle doesn't cover all vertices?
    - How? If $u$ is not in cycle, $u \to v \in Cycle$.
    - If $u$ has no incoming edge, this is allowed.
    - Is "hedgehog graph" implying the shape is like a cycle with spines?
    - Spines would be $w \to \dots \to u \to v \in Cycle$.
    - But condition: "$v$ belongs to cycle for every edge $u \to v$".
    - This kills spines of length > 1.
    - So only spines of length 0 (nodes in cycle) or length 1 (sources pointing to cycle).
    - If a source points to a cycle node, it's disconnected from the rest of the graph's "flow".
    - Since "exactly one directed cycle", all components merge into the cycle? No, sources don't merge. Sources are roots. Flow merges.
    - Edges are directed. Flow goes $Source \to \dots \to Cycle$.
    - If there is a spine, say $A \to B \to C$ ($C \in Cycle$).
    - Edge $A \to B$: $B$ must be in cycle.
    - Edge $B \to C$: $C$ must be in cycle.
    - So $B$ must be in cycle. So $B=C$.
    - Then $A \to C$.
    - Edge $X \to A$ (if $A$ has predecessor). Then $A$ must be in cycle.
    - So any node in the "tree" part pointing to a non-cycle node violates the condition.
    - Basically, the set of nodes is partitioned into the Cycle and others. Others must map to Cycle.
    - If $u$ is in Others, $next(u) \in Cycle$.
    - If $w \in Others$ and $next(w)=u$, impossible since $u \in Others$ but must map to Cycle.
    - So nodes in Others must have in-degree 0.
    - If there are nodes with in-degree 0 that are not in cycle, they form isolated components?
    - "exactly one directed cycle".
    - If there are other components, say just a node pointing to nothing? No, out-degree 1.
    - So it's possible to have many components, each being a "tail" feeding a cycle?
    - But "exactly one directed cycle" implies there is only ONE cycle in the entire graph.
    - This means there can only be ONE component that feeds the cycle.
    - Any other component would either contain a cycle (forbidden, as "exactly one") or feed a cycle (then it must be part of the same component).
    - Wait, if $u \in Component 1$, $next(u) \in Component 1$. Connected components are defined by edges.
    - If "hedgehog graph" implies the whole thing is one structure.
    - Given "The interactor guarantees that there exists at least one hedgehog graph", and the constraints.
    - Most likely, the graph is just a single cycle of some length $L$, using a subset of vertices, OR $L=10^6$.
    - Wait, if $L$ can be small, say 3.
    - And other vertices $10^6 - 3$ map to the cycle directly (indegree 0).
    - But how do we know which vertex is in the cycle?
    - Queries give us info about traversal.
    - Query `? v x`: returns $(v+x) \pmod{cycle}$. Wait, indices.
    - Let's denote the cycle permutation $\pi$ restricted to the cycle vertices.
    - If $v$ is on the cycle, $next(v)$ is well defined on the cycle.
    - If $v$ is not on the cycle, $next(v)$ is on the cycle.
    - $next(next(v))$ is on the cycle (since target is on cycle).
    - So within at most 1 step, you enter the cycle? No.
    - If $next(v) \in Cycle$, then after 1 step you are in the cycle.
    - From there, you stay in the cycle forever.
    - So the sequence is $v, c_0, c_1, c_2, \dots$.
    - If $next(v) \in Cycle$, the second term is in the cycle. The third is in the cycle...
    - So for $x \ge 1$, the destination is in the cycle.
    - The value returned by `? v x` depends on whether $v$ is in the cycle or not.
    - But wait, if $v$ is NOT in the cycle, $v \to c_0 \to c_1 \dots$.
    - $Query(v, x)$: move $x$ times.
    - If $x=1$, goes to $c_0$ (in cycle).
    - If $x=2$, goes to $c_1$ (in cycle).
    - Basically, for $x \ge 1$, the result is strictly determined by the cycle behavior relative to the entry point.
    - Specifically, if we identify the cycle nodes, they behave cyclically.
    - But we don't know the labels of the cycle nodes easily without guessing.
    - However, we just need the *length* of the cycle.
    - We can run queries starting from different nodes.
    - If we pick $v=1$.
    - `? 1 1` gives $u = next(1)$.
    - `? 1 2` gives $v = next(u)$.
    - If $1$ is on the cycle, then $next(1)$ is on the cycle, $next(1)$'s next is on the cycle...
    - If $1$ is NOT on the cycle, $next(1)$ is on the cycle.
    - In both cases, for $k \ge 1$, $pos(k)$ (vertex visited at step $k$) lies on the cycle.
    - Wait, if $1$ is not in cycle, $next(1)$ is on cycle.
    - If we treat `res = ? 1 1`, `res` is in cycle.
    - `next(res)` will be in cycle.
    - So `? res x` where `x` is large will trace the cycle.
    - Algorithm idea:
      1. Find a node definitely in the cycle.
         - Pick `v=1`. `res = query(1, 1)`. Since `res` is the target of `1`, it must be in the cycle.
         - Verify: If 1 was on cycle, target is on cycle. If 1 not on cycle, target MUST be on cycle (by problem statement).
         - So `res` is guaranteed to be in the cycle.
         - Let $S = res$. Now we are at a known cycle node.
      2. Find the length of the cycle containing $S$.
         - We know $next(S)$ is the next node.
         - Since we are now strictly on the cycle, the structure is a simple cycle passing through $S$.
         - The cycle length $L$ satisfies: $S = query(S, L)$. Also $S = query(S, 2L), \dots$.
         - We need to find smallest $L \ge 3$.
         - The vertices are $1..10^6$. Max cycle length $10^6$?
         - Actually, the problem states vertices $1..10^6$.
         - If the graph contains ONLY the cycle nodes involved in the query, maybe not all are used?
         - Wait, if there are nodes with indegree 0 not in cycle, they are never visited starting from cycle.
         - So we just need to traverse the cycle.
         - The question is: is it possible that there are multiple disjoint cycles? No, "exactly one directed cycle".
         - So the component containing $S$ is the ONLY cycle.
         - So we just find $L$ such that `query(S, L) == S`.
         - We can binary search for $L$ in range $[3, 10^6]$?
         - Wait, can $L$ be larger than $10^6$? "Each vertex is numbered from 1 to 10^6". Vertices are unique. A cycle is a subset of vertices. Max size is $10^6$.
         - Binary Search is efficient. Range $3$ to $10^6$.
         - Number of steps $\approx \log(10^6) \approx 20$.
         - Total queries per graph: $1 (\text{entry}) + \approx 20 = 21$.
         - Limit 2500. Very safe.
         - Complexity: $O(N \cdot \log(\text{MaxVertices}))$.
    - Wait, is it possible that `query(v, x)` with huge `x` helps?
    - We can use `query(S, 2*L)` to verify?
    - To find $L$ via binary search:
      - We want minimal $L$ such that `query(S, L) == S`.
      - Check mid.
        - `val = query(S, mid)`.
        - If `val == S`, then $mid$ is a multiple of $L$.
          - New search range $[3, mid]$.
        - Else, $L$ does not divide $mid$.
          - But wait, we don't know divisibility. We know `val == S` iff `mid` is a multiple of $L$.
          - If `val != S`, it could mean $L$ does not divide `mid`.
          - But if `val == S`, we reduce upper bound.
          - This finds a value $M$ such that $L | M$. It doesn't find $L$ exactly.
          - However, if we know $S$ is on the cycle, we can compute distances between adjacent nodes?
          - No, that requires knowing $S$ and finding when it repeats.
          - Floyd's Tortoise and Hare is $O(L)$ queries? No, each step is a query? No, Floyd requires traversing the path sequentially. One query gives "jump X".
          - With "jump X", we can simulate doubling or binary lifting.
          - But we just need to find $L$.
          - If we have a node $S$ on the cycle.
          - We know $S \to next(S) \to next(next(S)) \dots$
          - The sequence is periodic with period $L$.
          - $query(S, k) = S$ if and only if $L | k$.
          - We want to find $L$.
          - Is it sufficient to find the greatest common divisor of two values $k_1, k_2$ for which `query(S, k_i) == S`?
          - We don't know such $k$ beforehand.
          - Wait, if `query(S, k) != S`, then $L$ does not divide $k$.
          - If we find a $k$ where `query(S, k) == S`, then $L \le k$.
          - Can we find $L$ exactly?
          - Suppose we perform a random walk?
          - Better: Use the fact that vertices are $1..10^6$.
          - Is it possible $L$ is prime?
          - Let's consider `x` can be up to $5 \cdot 10^{18}$.
          - We can check `query(S, x) == S` for large $x$?
          - If we find a multiple, great.
          - If we pick a random $x$, probability $x$ is multiple of $L$ is low unless $L$ is small.
          - If $L$ is large, we need to construct it.
          - Wait, if we binary search for $L$, the logic `if val == S` means $L | mid$.
          - If `val != S`, it means $L \nmid mid$.
          - This tells us $L \neq mid$.
          - But does it tell us $L > mid$ or $L < mid$? No.
          - Example: Cycle length 3.
            - Test $mid=4$. `query(S, 4)` corresponds to $S+1 \neq S$.
            - Test $mid=3$. `query(S, 3)` corresponds to $S=S$.
            - If we test $mid=6$. `query(S, 6)` is $S$.
          - So we can collect multiples of $L$ by finding hits.
          - But finding $L$ exactly using only divisibility checks is basically finding GCD of found multiples.
          - We can generate multiples.
          - Start with $k=1$. While `query(S, k) != S` increment $k$? Too slow.
          - Start with $k=L$. `query(S, k) == S`.
          - We know $L \le 10^6$.
          - We can check `query(S, 1)`, `query(S, 2)...`? No, $O(L)$ queries is too slow if $L \approx 10^6$.
          - But wait, we can jump.
          - Let's try to find two multiples $A$ and $B$ such that we can compute $\gcd(A, B) = L$.
          - We can find one multiple easily?
          - If we query `? S 10^9`. Result will likely not be $S$.
          - Wait, if $L$ divides $X$, then `query(S, X) == S`.
          - We can pick random $X$ such that `query(S, X) == S`? No, that's hard.
          - However, we can compute the cycle length by iterating.
          - Wait, we can't iterate if $L$ is large.
          - But notice: we have access to `? v x` for any `x`.
          - Consider the property of `? v x`. It computes $next^x(v)$.
          - If we start with $v=1$ (or whatever entry).
          - $u_0 = query(v, 1)$. $u_1 = query(v, 2)$.
          - If $v$ is in cycle, $query(v, k) = v+k \pmod L$.
          - If $v$ is not in cycle, $query(v, 1)$ enters cycle.
          - Wait, earlier I said "If $1$ is not in cycle, `next(1)` is in cycle".
          - Is it guaranteed that `next(1)` is in the cycle?
          - Problem: "For every edge e=u->v ... v belongs to the cycle".
          - Yes.
          - So `query(v, 1)` is ALWAYS a vertex on the cycle.
          - Let $C = query(v, 1)$.
          - Now we are at a known cycle node $C$.
          - For any integer $k$, `query(C, k)` is a vertex on the cycle.
          - Specifically, if we identify the cycle as $C, C', C'', \dots, C^{(L-1)}$, then `query(C, k)` is $C^{(k \pmod L)}$.
          - We need to find $L$.
          - We can compute the order of the permutation element $C$.
          - Since $C \in \text{Cycle}$, the orbit of $C$ is the cycle itself.
          - So `query(C, L) == C`.
          - We want smallest $L$.
          - We can compute $L$ by finding $x$ such that `query(C, x) == C` efficiently?
          - This looks like finding the order of an element in a group, or cycle length.
          - Since we have a query oracle, this is the standard "find period" problem.
          - The maximum cycle length is $10^6$.
          - We can use Pollard's rho algorithm logic? Or simply check divisors?
          - Checking $10^6$ is too much if we do linear scan?
          - Wait, $5 \cdot 10^{18}$ allows very large steps.
          - We can try to determine $L$ by binary search if we can handle the modulo logic.
          - Actually, if we just find *any* $k$ where `query(C, k) == C`, say $K$. Then $L$ divides $K$.
          - If we find another $J$ where `query(C, J) == C`, then $\gcd(K, J)$ is a multiple of $L$.
          - If we take $\gcd$ of two numbers $K, J$ that are multiples of $L$, the result is a multiple of $L$.
          - Ideally, we want the GCD to be exactly $L$ (i.e., coprime ratios).
          - With large random $K, J$, the probability that $\gcd(K/L, J/L) = 1$ is high.
          - How to find *one* $K$ quickly?
          - Pick $x = 10^9$. If `query(C, x) == C`, good. Unlikely if $L$ small.
          - If $L \approx 10^6$, $x$ might be multiple.
          - Wait, the constraint is $Q \le 2500$. We have plenty.
          - Linear scan up to $2 \cdot 10^6$?
          - If $L$ is small, we find it fast.
          - If $L$ is large ($10^6$), linear scan takes $10^6$ queries. Too many ($N=10$ graphs, budget 2500 per graph).
          - $10^6$ queries is OOM/TLE.
          - We need sub-linear.
          - We can binary search for $L$?
          - If we binary search for $L$, we assume $L$ divides mid?
          - No, we need to find smallest $x$ s.t. `query(C, x) == C`.
          - This is finding the generator of the subgroup generated by the shift operator.
          - Wait, we can just check divisibility?
          - Actually, if we have $C$, we know that $C$ corresponds to index $0$ in the cycle.
          - `query(C, k)` corresponds to index $k \pmod L$.
          - If we pick a large random $k$, the chance $k \equiv 0 \pmod L$ is $1/L$.
          - We can try to construct $L$ from bits?
          - Since $L \le 10^6$, we can determine it bit by bit?
          - No, that doesn't work with modulo.
          - Wait, there is a technique called "Baby-step Giant-step" but that's for collisions. Here we know the start.
          - We can use the property: $L$ divides $LCM(a, b)$? No.
          - If we find $K$ such that `query(C, K) == C`, we know $L|K$.
          - If we can find $K$ and $K'$ such that $\gcd(K, K') = L$, we are done.
          - But we don't know $L$.
          - However, note that we can find $L$ by computing the order.
          - We can try powers of 2.
          - Let $K_1 = 2^m$. Check `query(C, K_1)`. If not $C$, then $L \nmid K_1$.
          - This doesn't help finding $L$.
          - Let's re-evaluate the structure.
          - $N \le 10$. Time limit 5s.
          - We can afford $\sim 250$ queries per graph if we want score 100, but even slightly less is fine.
          - Wait, we have 2500 queries.
          - $10^6$ vertices.
          - Is it possible that the cycle length $L$ is small?
          - If we pick $v=1$. $next(1)$ is on cycle.
          - $next(next(1))$ is on cycle.
          - Actually, let's step back.
          - If we have a cycle of length $L$.
          - We know $C = query(v, 1)$.
          - Then $next(C)$ is $query(C, 1)$.
          - We want smallest $x$ such that $query(C, x) = C$.
          - This is exactly finding the order of the element in the cycle permutation.
          - If we can't find $L$ fast, we can just pick a random $x$ large enough?
          - No.
          - Is there a trick?
          - Notice that we are allowed to choose $v$ and $x$.
          - If we choose $v$ such that we land on $C$ at step 1.
          - Wait, can we determine $L$ using the fact that we have $10^6$ vertices but only one cycle?
          - What if we check `? 1 1`, `? 1 2`, `? 1 3`... until we see repetition?
          - Sequence $s_k = query(1, k)$.
          - $s_1 = next(1) \in Cycle$.
          - $s_2 = next(s_1) \in Cycle$.
          - $s_3 = next(s_2) \in Cycle$.
          - ...
          - This sequence enters the cycle at step 1 and stays there.
          - So $s_1, s_2, \dots$ traces the cycle starting from $s_1$.
          - So $s_k = s_j$ implies $k \equiv j \pmod L$ (assuming $k, j \ge 1$).
          - If we simply search for duplicates in $s_1, s_2, \dots, s_{2L}$, we can find $L$.
          - If we don't know $L$, we can store visited vertices in a hashset/set.
          - But wait, we can't check "visited" without remembering.
          - We can pick an arbitrary $u \in Cycle$ (from $s_1$) and compute $L$ by binary searching on the exponent of 2?
          - Let's check if we can calculate $L$ faster.
          - Since $L \le 10^6$, maybe we can just compute $s_k$ for $k=1, 2, \dots$?
          - If $L$ is small, it's fast.
          - If $L$ is large (close to $10^6$), linear scan is too slow.
          - Wait, is it guaranteed that $L$ is not too large? No, "at least 3". Could be $10^6$.
          - Let's look at the constraints again.
          - $n \le 10$.
          - 2500 queries per graph.
          - If $L$ is around $10^6$, 2500 queries is way too few for linear scan.
          - We MUST use binary search or similar.
          - Can we binary search for $L$?
          - Let $f(x) = query(C, x)$. We want smallest $x$ such that $f(x) = C$.
          - Note that $f(x)$ is periodic with period $L$.
          - If we guess $M$.
          - If $M$ is a multiple of $L$, $f(M) = C$.
          - If $M$ is not a multiple, $f(M) \neq C$.
          - This boolean function is true for $x \in \{L, 2L, 3L, \dots\}$.
          - We want to find the smallest positive integer $x$ for which this is true.
          - We can check $x=M$. If true, we know $L|M$, so $L \le M$.
          - If false, we know $L \nmid M$.
          - This doesn't give a tight inequality like $L > M/2$.
          - It just eliminates $L$ as a divisor.
          - Wait, if we find a specific $K$ such that $f(K) = C$, we know $L$ divides $K$.
          - So we just need to find *one* $K$ that is a multiple of $L$.
          - And ideally, a $K$ that is small?
          - If we pick random $x$, probability $L|x$ is $1/L$. If $L=10^6$, prob $10^{-6}$. We would need $10^6$ attempts to get a match. Impossible.
          - BUT, we don't need a random $x$. We can construct a multiple.
          - Wait, $L$ is unknown.
          - Let's re-read "Hedgehog Graph".
          - Is it possible the graph is constructed differently?
          - "v belongs to the cycle".
          - Maybe I should verify my assumption that $L$ is small or special.
          - Actually, if $N$ is small (10 graphs), maybe there is a probabilistic approach?
          - If we pick two nodes $u, w$ on the cycle?
          - We know $C = query(1, 1)$ is on the cycle.
          - We can find $L$ by checking $query(C, x) == C$.
          - If we can check $x$ such that $x$ is a multiple of $L$.
          - How?
          - If we assume $L$ is not too small, maybe we can compute $L$ from bits?
          - Try to recover $L$ by testing $L \& 1$?
          - Or maybe simply: $L$ is a divisor of some large number?
          - Wait, if we query `? C (LargeNumber)`, we get some result.
          - If LargeNumber is not multiple of $L$, we don't learn much.
          - Wait, we can compute $L$ by computing $\gcd(x, y)$ where $x, y$ are known periods?
          - We can produce a period by finding $k$ where `query(C, k) == C`.
          - How to guarantee such $k$?
          - Ah, $10^6$ is not that large. But 2500 queries is tight.
          - Wait, maybe the cycle length is bounded differently? No.
          - What if we use the query `? v x` to effectively multiply elements in the group?
          - If we interpret vertices as elements of a group (additive mod $L$).
          - $next(v)$ adds 1.
          - $query(v, x)$ adds $x$.
          - So $query(v, x) = v + x \pmod L$.
          - We want $x \equiv 0 \pmod L$.
          - If we pick $x = \Phi(L) = L-1$? No, we don't know $L$.
          - However, we can determine $L$ if we know the order of a random element?
          - Wait, $L$ is the cycle length. Every element in the cycle generates the full cycle?
          - Yes, because it is a single cycle. It is a transposition of order $L$.
          - So any node in the cycle has order $L$.
          - So if we can find $L$, we are good.
          - Is it possible to solve this using the discrete log problem idea? No.
          - Is there a property of the graph that helps?
          - "Hedgehog". Maybe multiple cycles? No "exactly one".
          - "Directed graph".
          - Let's think about the scoring.
          - It penalizes usage.
          - So we must be efficient.
          - The binary search idea:
          - We can find *an* $L$ candidate using `gcd` method.
          - We need two integers $A, B$ such that $L|A$ and $L|B$ and $\gcd(A, B) = L$.
          - We can find $A$?
          - Suppose we pick random $u \in [1, 10^{18}]$.
          - We check if $query(C, u) == C$.
          - If yes, $L|u$.
          - Probability is $1/L$. Bad if $L$ is large.
          - BUT wait.
          - We have $n \le 10$.
          - Can we use the property that $v$ can be any vertex?
          - Maybe we can find $L$ by querying `? 1 1` (enters cycle), then `? (result) 1` (moves on cycle).
          - The path from $1$ is $v \to c_0 \to c_1 \dots$.
          - If $L$ is large, the cycle covers most vertices?
          - Actually, if $L \approx 10^6$, we can just run BFS/DFS from $C$?
          - No, we only have forward queries.
          - If we can traverse $C \to C' \to C''$, we just need to detect the loop.
          - We can use Floyd's cycle-finding algorithm?
          - But standard Floyd needs "next" operation (cost 1 step per iteration).
          - Here, "query(v, x)" costs 1 query but gives $x$ steps.
          - This is actually better than unit steps if we can skip.
          - But to detect cycle with Floyd's, we need two pointers $Tortoise, Hare$.
          - Step: $T \to next(T)$, $H \to next(next(H))$.
          - Here: $T \to query(T, 1)$, $H \to query(T, 2)$?
          - If we move $T$ and $H$ simultaneously, we can detect collision.
          - But detecting collision requires comparing values. `query` returns values. We can compare.
          - Algorithm:
            - Let $v = query(1, 1)$. (First cycle node).
            - $x = query(1, 1)$, $y = query(query(1, 1), 1)$.
            - Move $x$ and $y$.
            - Eventually $x=y$.
            - But $y$ is ahead.
            - Let's formalize:
              - $cur1 = v$.
              - $cur2 = v$.
              - We want to find $L$.
              - Actually, finding $L$ is easier if we can jump by $L/2$ or something.
              - Wait, finding $L$ with $N \approx 10^6$ vertices and limited queries.
              - Can we perform a "Doubling" approach?
              - $x = v$.
              - $dist = 1$.
              - While $query(x, dist) \neq v$:
                - $dist = dist + dist$? No, if we double, we skip over the cycle start? No, we are ON the cycle.
                - Wait, if we are on the cycle, `query(x, d)` moves $d$ steps along the cycle.
                - If `query(x, d) == x`, then $d$ is a multiple of $L$.
                - We want to find $L$.
                - If we maintain a current distance $D$ which is NOT a multiple of $L$ (initially 1).
                - We want to check if $2D$ is a multiple? No.
                - We know $L$ is a period.
                - Let's use the property of the cycle.
                - We want to find the smallest $L \ge 3$ such that $L$ divides $k$ where `query(v, k) == v`.
                - Wait, I keep circling back to finding a multiple.
                - Let's flip the problem.
                - We have a black box function $P(x) = next(x)$.
                - We know $P(P(...))$ is periodic.
                - We can compute $P^k(x)$ efficiently via binary exponentiation of steps?
                - $P^k(x) = query(x, k)$.
                - We want to find order of $P$ acting on $v$.
                - This is exactly finding the multiplicative order in a group (cyclic group of size $L$).
                - The group is $Z_L$. We know we are in $Z_L$.
                - We want to find $L$.
                - But we don't know the modulus.
                - We can check if $P^k(x) = x$.
                - This checks if $k \equiv 0 \pmod L$.
                - We can find $L$ by finding two numbers $A, B$ such that $A \equiv B \pmod L$ but we don't know $L$.
                - Wait, if $P^A(x) == P^B(x)$, then $A \equiv B \pmod L$.
                - So $L | (A - B)$.
                - So we need to find $A, B$ such that $A > B$ (positive diff) and $P^A(x) == P^B(x)$.
                - This gives a candidate multiple $M = A - B$.
                - We want $L$ to be exactly $M$ (or a factor).
                - How to ensure $L$ is derived correctly?
                - If we can find $A$ such that $A$ is small and $P^A(x) == x$ (i.e. $L=A$), that's best.
                - If we find $A$ such that $P^A(x) == x$, then $L$ divides $A$.
                - We can repeat to find smaller multiple.
                - Start with $A=1$.
                - If $P^1(x) == x$, $L=1$ (invalid, min 3).
                - We can increase $A$.
                - Instead of increasing by 1, we can use binary lifting.
                - We want to find the *smallest* $A$.
                - We can check if $P^A(x) == x$.
                - If yes, $L$ divides $A$.
                - If no, $L$ does not divide $A$.
                - This doesn't help narrow down $L$ range like binary search on interval.
                - Example: Target $L=5$.
                  - Check 8: No ($8 \% 5 = 3$).
                  - Check 12: No ($12 \% 5 = 2$).
                  - Check 5: Yes.
                  - Check 10: Yes.
                - We want to avoid checking all numbers.
                - BUT! We have the constraint: "For every edge e = u -> v ... v belongs to the cycle".
                - This implies $next(u)$ is in cycle for all $u$.
                - So starting from ANY node $u$, we enter the cycle at $next(u)$.
                - So $next(u)$ is a fixed cycle node.
                - Let $u=1$. $v = next(1)$.
                - $v$ is on the cycle.
                - Then we just analyze the cycle starting from $v$.
                - We want to find $L$ for the permutation starting at $v$.
                - The permutation is a simple cycle.
                - So $v \to v+1 \to \dots \to v+L-1 \to v$.
                - We just need $L$.
                - We know $L \in [3, 10^6]$.
                - Wait, if $L$ is unknown, we can't use binary search.
                - Is there any other information?
                - "The interactor is adaptive".
                - "You must flush output".
                - Scoring depends on Q.
                - Wait, if $L$ can be $10^6$, checking `query(v, L) == v` requires $L$ known.
                - Wait! There is a randomized approach for order finding (Pollard's rho style) but that finds factors of order.
                - Here order is $L$.
                - If we can find two collisions $P^a(x) = P^b(y)$?
                - Wait, if we just iterate $x=1, 2, \dots$.
                - If $L$ is large, we fail time limit.
                - Is it possible that $L$ is small?
                - No, $L$ can be $10^6$.
                - Wait, I missed something.
                - The graph has $10^6$ vertices.
                - If the graph is just the cycle, then $L$ must be $\le 10^6$.
                - If $L$ is close to $10^6$, binary search won't work directly because we can't check "is $mid$ a multiple".
                - Wait, can we?
                - If we can check `query(v, mid) == v`, that is checking `mid % L == 0`.
                - If we check for $mid = 2^K$.
                - We want smallest $L$.
                - Let's try to find the smallest power of 2, $2^k$, such that `query(v, 2^k) == v`.
                - Let this be $2^k$.
                - Then $L$ divides $2^k$. So $L = 2^j$.
                - But $L$ is not necessarily a power of 2.
                - So this only works if $L$ is power of 2.
                - However, we can do a generalized GCD method.
                - Randomized algorithm to find $L$ given an oracle for `check(k)`: "does L divide k?"
                - No, we can't implement `check(k)` easily because we need `query(v, k) == v`.
                - Wait, if we pick two random integers $A, B$, and find $K = \gcd(A, B)$? No.
                - Wait, we have $N \le 10$ graphs.
                - The time limit is generous (5s).
                - 2500 queries per graph is generous if we only do ~100 queries per graph.
                - Wait, if $L$ is large, we have to find it.
                - Let's revisit the property: $v$ points to cycle.
                - Let $u = query(1, 1)$. $u$ is on cycle.
                - $next(u)$ is on cycle.
                - If we just compute $query(u, k)$ for $k=1, 2, \dots$ and stop when we hit $u$.
                - If $L$ is large, this is slow.
                - Can we speed this up?
                - Yes, using baby-step giant-step.
                - We want to find smallest $L$ such that $next^L(u) = u$.
                - We can write $L = i \cdot m + j$.
                - This is for solving $g^x = h$. Not applicable directly.
                - But finding cycle length is the classic application of Baby-step Giant-step (BSGS).
                - Here we have a sequence $u, next(u), next(next(u)), \dots$.
                - We want to find the first index $L$ where $sequence[L] = u$.
                - BSGS finds the length of the cycle in a functional graph.
                - Standard implementation uses $O(\sqrt{L})$ time/memory.
                - With $L \le 10^6$, $\sqrt{L} = 1000$.
                - Queries required $\approx 2 \sqrt{L} = 2000$.
                - This fits within 2500!
                - Memory limit 1024MB. We can store a map/array of size 1000.
                - Wait, BSGS usually stores `map` of visited states to step index.
                - Here states are vertex IDs.
                - Map size: $\sqrt{L}$.
                - Steps: $\sqrt{L}$ giant steps + $\sqrt{L}$ baby steps?
                - Actually, we can just store `step -> vertex`.
                - We perform $m = \lceil \sqrt{L} \rceil \approx 1000$ baby steps?
                - Baby steps: Compute $seq[0], seq[1], \dots, seq[m-1]$. Store in map.
                - Giant steps: Compute $seq[k \cdot m]$ for $k=1 \dots m$. Check if $seq[k \cdot m]$ is in map.
                - If match, $seq[j] = seq[k \cdot m]$ implies $k \cdot m - j = L$ (modulo something? No, equality of vertices implies $L$ divides diff).
                - Since it's a pure cycle starting at $u$, $seq[x] = u \implies x=L$.
                - So if $seq[j] == seq[k \cdot m]$, then $L | (km - j)$.
                - Since we want smallest $L \ge 1$ (actually $\ge 3$), we want minimal positive solution.
                - With $m$ chosen properly, the first match usually gives $L$.
                - Specifically, if $seq[j] == seq[k \cdot m]$, then $u$ appears at step $j$ and $km$. So $L$ divides $km-j$.
                - If we ensure $km - j$ is minimal, we get $L$.
                - Since we know $u$ is the start of our observation, and $u$ appears again at $L$.
                - Wait, does $u$ appear earlier? No, $u$ is the entry to cycle.
                - Wait, is $u$ guaranteed to be the "start" of the cycle traversal in a sense?
                - $u$ is on the cycle. The sequence $u, next(u), \dots$ visits cycle nodes.
                - Since it is a simple cycle of length $L$, $u$ appears at indices $0, L, 2L, \dots$.
                - So $seq[x] == seq[y] \iff x \equiv y \pmod L$.
                - If we find $seq[j] == seq[k \cdot m]$ for some $0 \le j < m$ and $k \ge 0$, then $L | (km - j)$.
                - If we pick $m$ roughly $\sqrt{L_{max}}$.
                - Then $km - j$ will be a multiple of $L$.
                - If we pick $m \approx 1000$, we might miss small $L$ if $km-j$ is large.
                - But we want to minimize queries.
                - $m \approx \sqrt{10^6} = 1000$.
                - Total steps $\approx 1000$.
                - Query cost: each step is 1 query?
                - Wait, `query(v, x)` moves $x$ steps.
                - To do baby step of 1, we call `query(u, 1)`.
                - To do giant step of $m$, we call `query(curr, m)`.
                - This reduces complexity.
                - Baby steps:
                  - Store `map[vertex_id] = index`.
                  - $v_0 = u$. `map[u] = 0`.
                  - $v_1 = query(u, 1)$. `map[v_1] = 1`.
                  - ...
                  - $v_m = query(u, m)$? No.
                  - Baby steps should be computed using unit steps?
                  - $v_i = query(u, i)$ for $i=0 \dots m$.
                  - Cost: $m$ queries.
                - Giant steps:
                  - Current position $curr = query(u, 0)$ (which is $u$).
                  - Loop $k = 1 \dots m$:
                    - $curr = query(curr, m)$.
                    - Check if $curr$ in map.
                    - If found at index $j$, candidate $L = k \cdot m - j$.
                    - Return candidate.
                - Total queries: $2m = 2000$.
                - Fits perfectly in 2500.
                - Max $L$ is $10^6$. $\sqrt{L} = 1000$.
                - $2000$ queries.
                - If $L$ is small, we finish early.
                - But we can't check $L$ exactly without BSGS.
                - Actually, BSGS works for finding $x$ in $a^x = b$.
                - Here we want $x$ such that $P^x(u) = u$.
                - Wait, we know $u$ is in the cycle.
                - So we want smallest $x > 0$ s.t. $next^x(u) = u$.
                - This is exactly what BSGS solves.
                - We map $next^i(u)$ for $i \in [0, m-1]$.
                - Then check $next^{k \cdot m}(u)$.
                - If $next^{k \cdot m}(u) == next^j(u)$, then $next^{km - j}(u) == u$.
                - So $L$ divides $km - j$.
                - We want smallest such value.
                - Usually, with random $m$ or optimal $m$, this yields $L$ or a multiple.
                - If we just take the first match, say $diff = km - j$, it might be $L, 2L, \dots$.
                - But since $km - j < (m+1)m - 0 \approx m^2 \approx L$, it will be $L$.
                - (Assuming $m \approx \sqrt{L_{max}}$).
                - If $L$ is actually smaller than $m^2$, and $L$ divides $km-j$, we might get $L$ or $2L$ etc?
                - Actually, we want to return $L$.
                - If we find a match giving $d = km - j$, then $d$ is a multiple of $L$.
                - We can try to reduce $d$ by dividing by GCD with a verified small multiple?
                - Or simply: The cycle is $u \to \dots \to u$.
                - If we check $query(u, d) == u$, then $d$ is a multiple of $L$.
                - We need to ensure $d=L$.
                - Since $d < m^2$ and $m \approx \sqrt{L_{max}} \ge \sqrt{L}$.
                - If $L_{max} = 10^6$, $m=1000$, $m^2 = 10^6$.
                - So $d$ is in range $[1, 10^6]$.
                - If $d$ is a multiple of $L$ and $d < m^2$.
                - If we choose $m$ such that $m^2 \approx 10^6$.
                - If $L \le 10^6$, then $d$ could be $L$.
                - Is it possible $d$ is a multiple $2L$?
                - Yes.
                - Example: $L=3$. $m=2$.
                  - Map: $0 \to u$.
                  - $km$ steps: $2, 4$.
                  - $k=1 (2): query(u, 2)$. If $L=3$, $next^2(u) \neq u$.
                  - $k=2 (4): next^4(u) = next(u)$. Is $next(u)$ in map?
                  - Map has $next(u)$ at index 1. Match!
                  - $d = 4 - 1 = 3$. Correct.
                  - Wait, this worked.
                  - Another example $L=5$. $m=3$.
                  - Baby: $0, 1, 2$. Map $\{u:0, n1:1, n2:2\}$.
                  - Giant:
                    - $k=1 (3): next^3(u) = n3$. Not in map (0,1,2).
                    - $k=2 (6): next^6(u) = n1$ (since $6 \equiv 1 \pmod 5$).
                    - In map at index 1.
                    - $d = 6 - 1 = 5$. Correct.
                - It seems this strategy finds $L$ directly.
                - Why?
                - We are looking for smallest non-negative $j, k$ such that $next^j(u) = next^{km}(u)$.
                - This implies $km - j$ is a multiple of $L$.
                - Since we search $k$ starting from 1 upwards.
                - And we check against $j < m$.
                - The value $val = km - j$.
                - Since $0 \le j < m$ and $k \ge 1$, we have $km - j > 0$.
                - Also max val $\approx m^2$.
                - We want to find minimal $val$ divisible by $L$? No, minimal $val$ encountered.
                - Wait, BSGS ensures we check all combinations $km - j$ in range $(-m, \approx m^2)$.
                - Since $L$ is in this range, and $L$ is the minimal period.
                - Will we encounter a smaller multiple $L, 2L \dots$?
                - We check $k=1$ (value $m, m-1, \dots, 1$ roughly? No).
                - The value checked is $k \cdot m - j$.
                - We iterate $k=1, 2, \dots$.
                - First match gives some $V = km - j$.
                - Is it guaranteed that $V=L$?
                - $next^V(u) = u$. So $L | V$.
                - Could $V$ be $2L$?
                - Yes, if $L$ divides $km-j$ but $L$ is smaller.
                - However, we just need to output the cycle length.
                - If we get $V$ which is a multiple of $L$, say $2L$.
                - We need to divide by 2 until we find exact $L$.
                - How?
                - We can just verify $query(u, V/2) == u$.
                - If true, $L | V/2$. Repeat.
                - How many divisions? $V \le 10^6$.
                - This is cheap.
                - Wait, we don't know $L$. We don't know if $V=2L$ is valid.
                - We can test `query(u, V/2)`.
                - Wait, if `query(u, V/2) != u`, then $L$ does not divide $V/2$.
                - So $L$ might be $V$ or a factor of $V$ not dividing $V/2$ (like if $V$ had a square factor? No, $V=kL$, if $V=2L$, $2L/2 = L$, divides).
                - Actually, we just divide $V$ repeatedly by 2 as long as `query(u, V/d) == u`?
                - If `query(u, V/d) == u`, then $L | (V/d)$.
                - If `query(u, V/d) != u`, then $L$ does NOT divide $V/d$.
                - So we can find the largest divisor of $V$ that is a multiple of $L$.
                - Wait, $L$ divides $V$. If we divide $V$ by factors $p$, we remove $p$ from $V$.
                - If $V = L \cdot Y$, and we can check if $V/p$ is multiple of $L$.
                - We can find $L$ by taking $GCD(V, \text{some random } k \cdot L)$?
                - No, simpler:
                - If $V$ is a multiple of $L$.
                - Try to reduce $V$.
                - Is $V/L$ small?
                - Actually, with BSGS, $V$ is usually $L$.
                - Because $km - j$ spans a range where $L$ is likely to hit first.
                - Let's check worst case.
                - $m \approx \sqrt{L}$. Range $1 \dots m^2$.
                - If $L$ is large, $L$ is near $m^2$.
                - Then $V$ is likely $L$.
                - If $L$ is small, $L \ll m$.
                - Then $L$ divides many $km-j$.
                - E.g. $L=3, m=1000$.
                - $km - j = 3, 6, 9 \dots$
                - We want the smallest $>0$ satisfying `query(u, val) == u`.
                - Wait, BSGS finds *a* solution to $x \equiv j \pmod m$? No.
                - We check intersection of sets.
                - $Set1 = \{ u, n(u), \dots, n^{m-1}(u) \}$.
                - $Set2 = \{ n^m(u), n^{2m}(u), \dots \}$.
                - We find collision $u^{km} = u^j$.
                - This gives $n^{km-j}(u) = u$.
                - So $L | (km-j)$.
                - If we iterate $k$ from 1.
                - The values $km-j$ we check are not monotonic?
                - For $k=1$, we check $m-j$. Values $1, \dots, m$.
                - For $k=2$, we check $2m-j$. Values $m+1, \dots, 2m$.
                - So we check integers in increasing order $1 \dots m^2$.
                - So the FIRST collision found will correspond to the SMALLEST integer $V \in [1, m^2]$ such that $next^V(u)=u$.
                - Which is exactly $L$.
                - Provided $m^2 \ge L$.
                - We set $m = \lceil \sqrt{10^6} \rceil = 1000$.
                - Then $m^2 = 10^6$.
                - If $L \le 10^6$, then $L$ is in the search range.
                - The first match gives $L$.
                - Perfect!
                - So the algorithm is:
                  1. Read $n$.
                  2. Loop $n$ times:
                     a. Pick vertex $v=1$.
                     b. Call `u = query(1, 1)`.
                        - Note: $u$ is guaranteed to be on the cycle.
                     c. Set $m = 1000$. (Since $\sqrt{10^6}=1000$).
                     d. Create a map `vis` mapping `vertex -> index`.
                     e. For $i = 0 \dots m$:
                        - `vis[u] = i`
                        - `u = query(u, 1)`? Wait.
                        - Wait, Baby steps are $next^i(u_{start})$.
                        - Let's fix variables.
                        - Start cycle node $S$.
                        - Baby steps: Store positions $S, next(S), \dots, next^{m}(S)$.
                          - Indices $0$ to $m$.
                          - Need $m$ queries.
                          - But we can optimize?
                          - We need to store $next^i(S)$.
                          - Let $curr = S$.
                          - `vis[curr] = 0`.
                          - Loop $i$ from 1 to $m$:
                            - `curr = query(curr, 1)`?
                            - Wait, `query` with 1 step is just `next(curr)`.
                            - Cost: $m$ queries for baby steps.
                         f. Giant steps:
                            - `curr_giant = query(S, 0)` which is $S$?
                            - No, we want $next^{km}(S)$.
                            - Let `curr_val = query(S, m)`. (Cost 1 query).
                            - Loop $k$ from 1 to $m$:
                              - Check if `curr_val` in `vis`.
                              - If found at index $j$:
                                - Found match! $L = k \cdot m - j$.
                                - Output `L`.
                                - Flush and read response.
                                - If correct, print `! L`.
                                - Break.
                              - `curr_val = query(curr_val, m)`. (Move to $next^{(k+1)m}$).
                      g. After loop, if not found (unlikely given bounds), fallback?
                     h. Wait, we need to ensure $S$ is on cycle.
                        - My previous deduction: `query(1, 1)` is on cycle.
                        - Let's verify this strictly.
                        - Condition: "For every edge e = u -> v, v belongs to cycle".
                        - Edge $1 \to next(1)$. So $next(1)$ is on cycle.
                        - Yes.
                        - So $S = query(1, 1)$ is correct.
                        - Wait, what if $1$ IS on cycle? Then $next(1)$ is on cycle. Still holds.
                        - What if $next(1)$ IS on cycle?
                        - So yes, $S$ is safe.
                  3. Wait, what if $query(1, 1)$ gives the same as $query(1, 0)$ (which is 1)?
                     - If $1$ is on cycle, $next(1) \neq 1$ (no self loops, length $\ge 3$).
                     - So $next(1) \neq 1$.
                     - So $S \neq 1$.
                     - It's possible $next(1)$ is on cycle.
                     - Wait, if $1$ is not on cycle, $next(1)$ is on cycle.
                     - If $1$ is on cycle, $next(1)$ is on cycle.
                     - So $S$ is always on cycle.
                     - The cycle starts at $S$ in our traversal logic.
                     - Wait, does $next(S)$ equal the node following $S$ in cycle? Yes.
                     - So we are traversing the cycle.
                     - Baby steps cover $0 \dots m$.
                     - Giant steps cover $m \dots 2m^2$?
                       - Wait, $k$ goes up to $m$. Max index $km - 0 = m^2$.
                       - So we check values up to $m^2 = 10^6$.
                     - So this covers the full range of possible $L$.
                     - One detail: $m=1000$ gives $m^2=10^6$.
                     - If $L=10^6$.
                     - $m=1000$.
                     - Baby steps: $0, \dots, 1000$.
                     - Giant steps: $1000, 2000, \dots, 10^6$.
                     - Collision at $km - j$.
                     - We want to ensure we find $L=10^6$.
                     - At $k=1$, check $1000$. Not found (unless $L=1000$).
                     - ...
                     - At $k=1000$, check $1000000$.
                     - `curr_val` becomes $next^{10^6}(S) = S$.
                     - Is $S$ in `vis`? Yes, at $j=0$.
                     - $L = 1000 \cdot 1000 - 0 = 10^6$.
                     - Correct.

    - Implementation details:
      - `std::unordered_map` or `std::map`? `map` is safer/cleaner, $N=10$, operations 2000. Time negligible.
      - `map<long long, int> vis;`
      - Vertices up to $10^6$.
      - Query parameter $x$ can be large ($5 \cdot 10^{18}$).
      - In giant steps, we need to accumulate position.
      - $curr\_val = query(S, m)$.
      - Loop:
        - Check `curr_val` in map.
        - Update `curr_val = query(curr_val, m)`.
        - Be careful: if $m$ is large, we might exceed `long long`? No, $m=1000$. $x$ up to $1000 \times m \approx 10^6$ or accumulated.
        - Wait, `query` expects $x \le 5 \cdot 10^{18}$.
        - My code passes $m=1000$. Safe.
        - When updating `curr_val`, we query it.
        - The logic `curr_val = query(curr_val, m)` effectively does `next^{m}(curr_val)`.
        - So after $k$ iterations, `curr_val` is at `next^{k*m}(S)`.
        - Wait, initial state before loop $k=1$:
          - We need to check $next^{1 \cdot m}(S)$.
          - Initial `curr_val` should be $next^m(S)$.
          - So:
            `current_node = query(S, m);`
            `for k = 1 to m:`
               `if map.find(current_node) != map.end(): return k*m - map[current_node];`
               `current_node = query(current_node, m);`
        - Wait, is `query(curr_node, m)` valid if `current_node` is the result of `query(..., m)`?
        - Yes.
        - Optimization: `query(curr, m)` where $m=1000$.
        - We might do $m$ queries for baby steps + $m$ for giant steps.
        - Total $2m = 2000$ queries.
        - Plus 1 query for $S = query(1, 1)$.
        - Total 2001. Within 2500.
        - What if $m$ is optimized?
        - $m$ should be roughly $\sqrt{L}$. Since max $L=10^6$, $m=1000$ is good.
        - Actually, we can dynamically set $m$? No need.
        - Corner case: $L < m$?
          - Baby steps will catch it?
          - If $L < m$, then $S, next(S), \dots$ will have collision within baby steps.
          - Wait, `vis` stores `index`. If $next^i(S) = next^j(S)$, collision detected during baby phase.
          - My algorithm checks collisions between Baby and Giant phases.
          - Does it handle internal Baby collisions?
          - $next^i(S) = next^j(S)$ implies $L | (i-j)$.
          - If $0 \le j < i \le m$.
          - If we find collision inside baby steps, we can detect it.
          - If $i=j$, ignore.
          - If $i \neq j$, $L = i-j$ (or factor).
          - Wait, we want $L$. $next^{i-j}(S) = S$. So $L$ divides $i-j$.
          - Since we check all pairs, we'd find smallest.
          - But we can't easily check all pairs.
          - Just check map existence?
          - If `map[next^i(S)]` exists (let's say it was put there at step $j$), then collision!
          - Then $L$ divides $i-j$.
          - Since we want smallest $L$, we should update candidate?
          - But wait, if $L$ is small, $i-j$ could be large? No, minimal positive difference.
          - If we process $i$ from 1 to $m$.
          - If `next^i(S)` is already in map at $j$, then $L$ divides $i-j$.
          - Is it possible $L > i-j$? Yes.
          - But we know $L \le i-j$.
          - So $L \le i-j < m$.
          - So we can return $i-j$ as a candidate?
          - Wait, we need to be sure $i-j = L$.
          - We know $next^{i-j}(S) = S$.
          - So $L$ divides $i-j$.
          - If $L < m$, then we find a multiple.
          - We can just run binary search on $i-j$ to find exact $L$ later?
          - Or just rely on Giant steps.
          - Giant steps search range includes $1 \dots m^2$.
          - If $L < m$, then $L$ is a multiple of $L$.
          - Is $L$ represented as $k \cdot m - j$?
          - $L$ is small. $m$ is 1000.
          - If $L=3$. $m=1000$.
          - We might find $1000 - 997 = 3$.
          - Yes.
          - Does the collision detection logic `if map has curr_node` handle this?
          - If $L < m$.
          - Baby steps: $S, n(S), n^2(S)$. $n^L(S) = S$.
          - At step $L$ (where $L \le m$), we insert $S$ into map?
          - Wait, at step 0 we inserted $S$.
          - At step $L$, we see $S$.
          - Collision with $j=0$.
          - Value $L - 0 = L$.
          - So we found $L$ during baby steps.
          - We can handle this inside the baby loop too.
          - Algorithm refined:
            - Baby Phase ($i=1 \dots m$):
              - $curr = query(curr, 1)$.
              - If `curr` in map:
                - `len = i - map[curr]`.
                - If `len >= 3`, candidate `len`.
                - We can stop and refine `len` later? Or just assume `len` is answer?
                - Wait, if `len` is a multiple of true $L$.
                - True $L \le len$.
                - We can return `len`?
                - Wait, if $L=3, len=6$. $next^6(S)=S$.
                - Then we found 6.
                - But we can check `query(S, 3)`. If $S \neq next^3(S)$?
                - Wait, we can't easily check `len` vs `len/2` efficiently inside loop without extra logic.
                - Actually, if we just proceed to Giant steps, Giant steps will also catch it.
                - Because if $L=3$, Giant step $k=1$ (val 1000) -> $n^{1000}(S) = n^{333 \cdot 3 + 1} = n(S)$.
                - Map has $n(S)$ at index 1.
                - Collision $1000 - 1 = 999$. Multiple of 3.
                - $k=2$ (2000) -> $n(2000)(S) = n(2)(S)$. Index 2.
                - Collision $2000-2 = 1998$.
                - We want smallest.
                - Wait, BSGS finds the collision corresponding to the *first* giant step $k$.
                - $k \cdot m - j$.
                - If $L=3$, $m=1000$.
                - $km - j$ will be $999, 1998 \dots$. All multiples of 3.
                - But none is $L$.
                - So we MUST reduce.
                - If we get a candidate $V$, we compute $V / \gcd(V, \text{something})$?
                - Or just check divisors.
                - Since $V \le 10^6$.
                - We can factorize $V$ and test divisors?
                - Factorization of $10^6$ is fast.
                - Test each divisor $d$ of $V$ (down to $L \ge 3$).
                - Check `query(S, d) == S`.
                - The first one that matches is $L$?
                - No, $L$ is the smallest.
                - Wait, if $V$ is a multiple of $L$, then $V = c \cdot L$.
                - Any divisor of $V$ might be a multiple of $L$? No.
                - If $L | V$, then for any $d$ such that $L | d$, `query(S, d) == S`.
                - For $d$ such that $L \nmid d$, `query(S, d) != S` (since $S$ is on cycle).
                - So we want the smallest divisor $d$ of $V$ such that `query(S, d) == S` AND $d$ is not just a random divisor.
                - Wait, if $L | V$, then $V$ is a multiple.
                - The true $L$ is the minimal positive integer such that `query(S, L) == S`.
                - If we find a $V$ (from collision) where `query(S, V) == S` (guaranteed), then $L$ divides $V$.
                - We need to find the minimal divisor of $V$ that satisfies the property.
                - Wait, if $L$ divides $V$, then $L$ is a divisor.
                - If we find all divisors of $V$, sort them ascending, check if `query(S, d) == S`.
                - The first match is $L$.
                - Divisor enumeration of $10^6$ takes time? $10^6$ has at most 240 divisors. Fast.
                - So:
                  1. Run BSGS to find $V$. (Actually, we might get a multiple).
                  2. Collect $V$.
                  3. Factorize $V$.
                  4. Generate divisors.
                  5. Sort.
                  6. Iterate divisors $d$ (if $d < 3$ skip).
                     - Check `query(S, d) == S`.
                     - If true, answer is $d$.
                     - Print `! d`.
                  7. Wait, if $V$ is very large (e.g. $2m^2$), this works.
                  8. But $V$ comes from $km-j$. Max $V \approx 10^6$.
                  9. What if no collision found?
                  10. Then $L > m^2 = 10^6$. Impossible since $L \le 10^6$.
                  11. So we always find a collision.
      - Wait, optimizing step 2.
      - We don't need to fully factorize.
      - If we have $V$ as a candidate.
      - We know $L | V$.
      - We want $L$.
      - If we check `query(S, V/2) == S`, we check divisibility by 2.
      - If `query(S, V/3) == S`?
      - We can just divide $V$ by gcd with something?
      - No, simplest is:
      - We found a collision $km - j = V$.
      - We know `query(S, V) == S`.
      - We know $L \le V$.
      - Actually, is it possible that we find $V$ such that $L \ll V$ but we can't distinguish?
      - Yes.
      - So we must test.
      - Since we have 2500 queries.
      - Checking $V$ is 1 query.
      - Testing all divisors?
      - Divisors of $10^6$ is small.
      - But if $V \approx 10^6$, factoring might be costly?
      - Trial division up to $\sqrt{V} = 1000$.
      - $1000$ ops per graph. Cheap.
      - Generate divisors from prime factors.
      - Check each divisor with query.
      - Worst case: $V=10^6$, many divisors.
      - Checking each divisor takes 1 query.
      - If we have 500 divisors? (Highly composite numbers). $720720$ has 240.
      - If we test all 240 divisors, it's 240 queries.
      - Safe.
      - Wait, if we check `query(S, d) == S`, if true, then $L$ divides $d$.
      - If we check divisors in descending order.
      - The first one that satisfies the condition might not be $L$?
      - No, if $L | d$, then `query(S, d) == S`.
      - So if we find *any* $d$ in divisors of $V$ such that `query(S, d) == S`, it implies $L | d$.
      - We want smallest such $d$.
      - So we should check divisors in *ascending* order.
      - Smallest $d$ satisfying `query(S, d) == S` is $L$.
      - So:
        1. Find $V$ using BSGS.
        2. Factor $V$ to get all divisors.
        3. Sort divisors ascending.
        4. Iterate $d \in divisors$:
             if `query(S, d) == S`:
                 return $d$ as cycle length.
                 break.
      - Wait, checking `query(S, d) == S` assumes $S$ is on cycle. Yes.
      - Wait, do we need to check `d < 3`?
      - Problem says cycle length $\ge 3$.
      - If we find $d=1$ or $2$, it contradicts problem statement, so ignore or something is wrong.
      - Since problem guarantees $\ge 3$.
      - Wait, is it possible $L > V$?
      - No, $L | V$ and $L \ge 1$, so $L \le V$.
      - Wait, did I guarantee $L | V$?
      - Yes, because $next^V(S) = S$ is the condition derived from collision.
      - So $L$ divides $V$.
      - So $L$ is a divisor of $V$.
      - So this approach is robust.
      - Wait, $V$ calculation.
      - BSGS gives $V = km - j$.
      - We must verify `query(S, V) == S`? It should be true.
      - But if $V$ comes from `query(S, km)` matching `vis[S]` (which is 0), then `km` is a multiple of $L$. So `query(S, km) == S`.
      - Wait, `vis` contains `map[next^i(S)]`.
      - So if `curr_val` (which is $next^{km}(S)$) is in `vis` at $j$, it means $next^{km}(S) = next^j(S)$.
      - Implies $next^{km-j}(S) = S$.
      - So $V = km-j$ is indeed a multiple of $L$.
      - So this holds.
      - Complexity:
        - Baby steps: $m$ queries.
        - Giant steps: $m$ queries.
        - Factoring: $O(\sqrt{V}) \approx 1000$ ops.
        - Divisors: few hundred.
        - Verification queries: few hundred.
        - Total queries $\approx 2m + \#divisors$.
        - $2000 + 240 \approx 2240$.
        - Limit 2500.
        - Seems acceptable.
        - Wait, can we optimize checking?
        - We can check if `query(S, V) == S` (1 query).
        - If yes, check $V/2$, $V/3$...
        - We can prune.
        - Actually, $V$ is the result of $km-j$.
        - If $V$ is large, we might have tested many candidates in BSGS?
        - BSGS just finds *one* collision.
        - It doesn't check minimality.
        - Is there a way to find $L$ without testing all divisors?
        - We could have picked $m$ such that it detects small $L$.
        - If $L$ is small, say $L < m$, then $V$ will likely be $L$ (collision with $j=i-L$).
        - In that case $V=L$, divisors check is fast.
        - If $L$ is large, $V \approx L$, divisors few.
        - The bad case is $L$ is small but we missed the collision in baby steps (if we didn't check baby collisions) and got a large $V$?
        - Wait, if $L < m$, baby steps will definitely collide.
        - In my refined loop:
          ```cpp
          // Inside baby loop
          if (vis.count(curr)) {
              int diff = i - vis[curr];
              // diff is a multiple of L
              // We can just collect candidates?
              // Or just return diff?
              // If we return diff, we need to check if it is L.
              // We can check query(S, diff/2) == S?
              // If yes, divide.
              // Actually, just collecting V and running divisor check later is cleaner.
          }
          ```
        - But wait, if $L$ is small, we will get small $V$ (e.g. $L$).
        - Then divisor check is trivial ($V=L$).
        - The problematic case is when $L$ is somewhat larger than $m$?
        - If $L \approx 1000$, $m=1000$.
        - Then $L$ could be in the middle.
        - Anyway, 2500 queries is quite high.
        - $m=1000$ is fine.
        - Wait, what if $m$ is smaller?
        - If we choose $m = \sqrt{10^6} = 1000$.
        - Then we are safe.
        - Actually, can we set $m = 100000$? Then $\sqrt{V}$ grows?
        - $m$ determines number of queries for step. $2m$ queries.
        - We want $m^2 \ge 10^6$ to cover full range.
        - Minimizing $m$ minimizes queries.
        - So $m=1000$ is optimal.

    - Refinement:
      - Reading $n$.
      - Loop $n$ times.
      - In each graph, reset logic.
      - Interaction:
        - `cin >> n`.
        - while(n--) {
          - Find cycle.
          - Output `! ans`.
          - read status.
          - if status -1 exit.
          - if status 1 continue (wait, status 1 means correct, then start next? No, if status 1, we processed this graph. Next graph starts processing immediately? No, input format says "After that, read a single integer... 1 if correct. You should immediately start processing the next graph". Wait, does the jury provide the status?
          - "read a single integer which is either: 1, if the answer is correct."
          - But does the jury provide the status for *every* graph?
          - Or just for the last?
          - "Once you have determined the length of the cycle s, output: ! s"
          - "After that, read a single integer..."
          - "Failure to handle this properly may result in unexpected behavior."
          - This implies we need to process graphs one by one.
          - However, the problem says "Your task is to help Kipa determine the length of the directed cycle for EACH hedgehog graph."
          - It implies sequential processing.
          - "n graphs".
          - So for each graph, we interact, output answer, get feedback.
          - If feedback -1, exit.
          - If feedback 1, continue to next graph.
          - Wait, "read a single integer... 1 ... You should immediately start processing the next graph".
          - So loop $n$ times.
          - If we get 1, we proceed to next graph (which asks for new inputs? No, next graph data? The problem description is a bit ambiguous about input for multiple graphs.
          - "n graphs". Usually input for $n$ graphs is NOT provided upfront.
          - The example shows:
            Input:
            1
            3
            7
            10
            1
            Wait, `3`, `7`, `10`, `1`.
            These look like query answers?
            The example says:
            Example Input:
            1
            3
            7
            10
            1
            
            Example Output:
            ? 1 2
            ? 2 5
            ? 10 11
            ! 11
            
            Then presumably reading 1, 1, 1?
            The example output ends at `! 11`.
            Wait, "Example Output" usually includes only what program writes.
            The "Input" block includes responses from Jury?
            Yes, typically.
            Line 1: `1` (n).
            Lines 2+: Responses to queries.
            Response to `? 1 2` is `3`.
            Response to `? 2 5` is `7`.
            Response to `? 10 11` is `10`.
            After `! 11`, response is `1`.
            And that's it (since n=1).
            
            So, structure:
            - Read `n`.
            - For `k = 0` to `n-1`:
                - Solve graph.
                - Output `! ans`.
                - Read response `ok`.
                - If `ok == 1` and `k != n-1`: (Wait, do we need to check?)
                - Actually, if `ok == 1`, we continue. If `ok == -1`, we exit 0.
                - Wait, "finish your program with exit code 0".
                - The problem says: "After that, read a single integer ... 1 ... or -1".
                - If 1: start next graph.
                - If -1: exit 0.
                - If n is reached, just exit?
                - Wait, after the last graph, we also output `!` and get 1.
                - Do we need to process the response? Probably.
                - If we get `1`, we loop. If `n` done, loop finishes.
                
            - One subtle point: Is there a "new graph" setup?
            - The interactor maintains the state.
            - We just loop $n$ times.
            - In each iteration, we do queries and output answer.
            
            - What if the graph changes between outputs?
            - "The interactor is adaptive."
            - It's just a stream of queries and answers.
            - The state persists.
            - So yes, standard interactive loop.

    - Data types:
      - Queries return vertex ID.
      - Vertex ID fits in `int`.
      - $x$ fits in `long long`.
      - `m` fits in `int`.
      - `n` fits in `int`.
      
    - Includes: `iostream`, `vector`, `map`, `algorithm`, `cmath`.
    - Using fast IO? `ios_base::sync_with_stdio(false); cin.tie(NULL);`

    - Logic check:
      - Is `query(u, 1)` always safe?
      - `query(v, x)` where `x >= 1`.
      - `m=1000` calls `query(..., 1000)`.
      - `vis` map key type `int` (or `long long`).
      - `ans` type `int`.
      
    - Corner case: `n` can be up to 10.
    - Total time limit 5s.
    - Max queries: $10 \times (2000 + 240) \approx 22400$.
    - Wait, limit 2500 per graph?
    - The problem statement: "For each graph, the program can ask the following query at most 2500 times".
    - Ah! 2500 PER GRAPH.
    - My analysis used 2500 total? No, I analyzed 2000 per graph.
    - With $N=10$, total queries would be 20000?
    - Wait, time limit 5s for TOTAL execution.
    - But 2500 queries per graph is the constraint.
    - So per graph I am safe.
    - Wait, is 2500 per graph or total?
    - "For each graph, the program can ask the following query at most 2500 times".
    - Yes, per graph.
    - So 2000 queries per graph is safe.
    - Wait, I estimated 2000 queries for BSGS.
    - Plus verification steps.
    - If I add verification steps (checking divisors), it increases query count significantly.
    - Divisors of $10^6$ can be up to 240.
    - $2000 + 240 = 2240 < 2500$.
    - It's tight but okay.
    - Wait, can I optimize the verification?
    - Yes!
    - If I get $V$ from BSGS.
    - I know $V = k \cdot m - j$.
    - I also know that $m=1000$.
    - If $L$ is large, $V$ will be $L$ (mostly).
    - If $L$ is small, $V$ could be a multiple.
    - Wait, if $L$ is small (e.g. 3), and $m=1000$.
    - Baby steps will detect collision at $L$.
    - `vis` map stores $0 \to S, 1 \to n(S), \dots$.
    - At $i=L$, `curr` is $S$.
    - Collision with $0$.
    - $diff = L - 0 = L$.
    - So for small $L$, $V$ will be exactly $L$.
    - For large $L$ (e.g. $L \approx 10^6$), $V$ will be $L$.
    - When does $V$ become a multiple $c \cdot L$ where $c > 1$?
    - When $L \cdot c \approx m^2$.
    - Wait, if $L=3$, $V=999$? No.
    - If $L=3$. $m=1000$.
    - Baby steps: $S$ at 0, $n(S)$ at 1, $n^2(S)$ at 2, $n^3(S)=S$ at 3.
    - Collision at $i=3$, $j=0$. $V = 3$.
    - It returns 3.
    - What if $L$ is prime, e.g. 7. $V=7$.
    - What if $L$ is not found in Baby steps?
    - This happens if $L > m$.
    - Since $m=1000$.
    - If $L > 1000$.
    - Baby steps $0 \dots 1000$ do not wrap around (unless $L \le 1000$).
    - So no collision in Baby steps.
    - Giant steps starts checking $1000, 2000, \dots$.
    - First collision occurs at $k \cdot m - j = V$.
    - Since $L > m$, $V$ must be at least $L$.
    - If $L$ is in $(m, m^2]$.
    - Since $L$ is unique in the cycle traversal, and we check $km-j$.
    - $km-j$ iterates through $1 \dots m^2$.
    - The *first* time we hit a multiple of $L$ is when $km-j = L$.
    - Because $L$ is the smallest positive multiple.
    - And we check values in increasing order of magnitude roughly.
    - Wait, $km-j$ ranges:
      - $k=1$: $1 \dots m$.
      - $k=2$: $m+1 \dots 2m$.
      - ...
    - Yes, we check in increasing order.
    - So we will find $L$ exactly as the FIRST match!
    - Because $L$ is the smallest positive integer such that `query(S, L) == S`.
    - Our search sequence $km-j$ generates every integer in $1 \dots m^2$?
    - Wait, no. It generates $m-j$.
    - We need to check ALL integers $1 \dots m^2$.
    - But we only check those of form $km-j$.
    - Wait, $j$ ranges $0 \dots m-1$.
    - So for a fixed $k$, we check $km, km-1, \dots, km-(m-1)$.
    - As $k$ increases, the gaps fill up.
    - Range covered: $1 \dots m^2$.
    - Are there holes?
    - $k=1$: $1, \dots, m$ (values $1000, 999, \dots, 1$).
    - $k=2$: $2000, 1999, \dots, 1001$.
    - Yes! The intervals $[km-m+1, km]$ cover $[1, m^2]$ completely without overlap.
    - Wait, union is $[1, m^2]$.
    - So we ARE checking EVERY integer in $[1, m^2]$.
    - Thus, the FIRST collision we find (smallest $V$) is EXACTLY $L$.
    - Proof: $next^V(S) = S$. We want smallest $V$.
    - We check $V \in [1, m^2]$.
    - Since $L \le m^2$ (if $L > 10^6$, we are doomed, but $L \le 10^6$).
    - We will encounter $V=L$ eventually.
    - Since we iterate $k$ upwards and inner check decreases index, the order of checking is:
      - $k=1$: checks $1, 2, \dots, m$ (roughly, depending on implementation order).
      - Wait, if we check `map[current]`.
      - `current` corresponds to $next^{km}(S)$.
      - We look up $j$ such that $next^j(S) == next^{km}(S)$.
      - $km - j$ is the candidate.
      - For $k=1$, $curr = next^m(S)$. Check if in map.
      - If $curr == next^0(S)$ (i.e. $S$), then $V = m$.
      - If $curr == next^1(S)$, then $V = m-1$.
      - ...
      - If $curr == next^{m-1}(S)$, then $V = 1$.
      - So for $k=1$, we cover $1 \dots m$.
      - For $k=2$, we cover $m+1 \dots 2m$.
      - Wait, we don't explicitly iterate $j$.
      - `map` lookup gives the $j$.
      - So yes, we find the minimal $V$ in the range.
      - So no need to check divisors.
      - Just returning the $V$ found is sufficient!
      - The only concern: What if $L > m^2$?
      - $m=1000 \implies m^2 = 10^6$.
      - $L \le 10^6$.
      - So $L \le m^2$.
      - So $L$ will definitely be found.
      - And since we scan $1 \dots m^2$ (implicitly via $k, j$), we find $L$.
      - Wait, do we scan $1 \dots m^2$ strictly?
      - Yes, because $km - j$ for $k=1 \dots m, j=0 \dots m-1$ produces every integer in $[1, m^2]$.
      - The value $V = km - j$ is what we test.
      - If `query(S, V) == S` (implied by collision), then $L | V$.
      - But since we are finding the smallest $V$ among ALL $V \in [1, m^2]$ that satisfy the collision condition?
      - The collision condition is $next^V(S) = S$.
      - The smallest positive integer satisfying this is $L$.
      - So we will definitely find $L$.
      - And since we check $V=1$, then $2$, then $3$... (effectively), the first one is $L$.
      - So no divisor check needed.
      - This is extremely important. It cuts query count from 2240 to 2001.
      - This is very safe.

    - Double check: does `map[current]` return the smallest $j$?
      - We insert $S, n(S), \dots$ into map.
      - If a collision occurs, $S$ is already in map.
      - The stored index is the first occurrence (smaller $j$).
      - So $j$ is minimal.
      - Then $V = km - j$ is maximal for that specific $km$.
      - Wait. We want to minimize $V$.
      - If we have multiple collisions for same $k$?
      - `map` only stores one value per node (since $j$ increases).
      - Actually, if `curr` appeared multiple times, we store the first time.
      - So $j$ is minimal.
      - But does minimizing $j$ maximize or minimize $V$?
      - $V = km - j$. To minimize $V$, we want to maximize $j$.
      - Wait.
      - If $next^{km}(S) = next^j(S)$.
      - This means $next^{km-j}(S) = S$.
      - So $L | (km-j)$.
      - We want to find smallest $x$ such that $L|x$.
      - We know $km-j$ is such an $x$.
      - If we find a collision, we found a multiple.
      - We want to know if we find the *smallest* multiple in the range.
      - The range of checked multiples is dense?
      - We check all integers $V \in [1, m^2]$.
      - For a specific $V$, if $next^V(S) = S$, then $next^{km}(S) == next^j(S)$ where $km \approx V$ and $j \approx 0$ (relative to the period).
      - Actually, let's trace for $V=L$.
      - We want to find $L$.
      - $L$ is in $[1, m^2]$.
      - Write $L = k \cdot m - j$.
      - Then $next^{km}(S) = next^j(S)$.
      - This implies collision at $k$.
      - Will this collision be caught?
      - We check $k$ from 1 upwards.
      - For $k=1$, we check $V \in [1, m]$. If $L \le m$, we find it (as $V=L$).
      - For $k=2$, we check $V \in [m+1, 2m]$.
      - If $L$ falls in this range, we find it.
      - Generally, if $L \in [ (k-1)m+1, km ]$.
      - Then $L = km - j$ where $j = km - L$.
      - Since $0 \le j < m$, this maps to the check.
      - Will `map[curr]` contain $j$?
      - `curr` is $next^{km}(S)$.
      - Since $L = km-j$, $next^{km}(S) = next^{km-j+j}(S) = next^j(S)$.
      - So `curr` is $next^j(S)$.
      - Since $0 \le j < m$ (because $L > (k-1)m \implies km - L < m$).
      - Wait. $L > (k-1)m \implies L \ge (k-1)m + 1$.
      - $j = km - L$.
      - $j \le km - ((k-1)m + 1) = m - 1$.
      - And $j \ge km - m^2$ (if $L$ large).
      - If $L \le m^2$, then $j$ can be negative?
      - No, we need $j$ to be in map range $[0, m]$.
      - The collision is detected if $next^{km}(S)$ equals some $next^j(S)$ with $0 \le j \le m$.
      - Does $next^j(S)$ exist for the required $j$?
      - We have inserted $next^x(S)$ for $x \in [0, m]$.
      - So we need $0 \le j \le m$.
      - Is $j = km - L$ in $[0, m]$?
      - $L \ge 3$.
      - We need $km - m \le j \le km$.
      - We have $j \in [0, m]$.
      - So we need intersection of $[km - L, km]$ and $[0, m]$?
      - Wait.
      - The condition $L = km - j$ implies $j = km - L$.
      - For $next^{km}(S) = next^j(S)$ to hold, $j$ must be valid index.
      - Valid indices in map are $0 \dots m$ (stored).
      - Actually, I stored up to $m$.
      - So if $0 \le km - L \le m$, we detect it.
      - $km - L \ge 0 \implies L \le km$.
      - $km - L \le m \implies L \ge km - m$.
      - So if $L \in [km - m, km]$.
      - Wait, this interval is $[(k-1)m, km]$.
      - But map only stores $0 \dots m$.
      - So we only detect if $L \in [(k-1)m, km]$?
      - Wait, $j$ must be in $[0, m]$.
      - So if $L$ is in $[(k-1)m, km]$, then $j = km - L \in [0, m]$.
      - Wait, $(k-1)m \le L \le km$.
      - Then $km - km \le j \le km - (k-1)m \implies 0 \le j \le m$.
      - Yes.
      - So for any $L$, we can express it as $L = km - j$ with appropriate $k, j$.
      - Specifically $k = \lfloor (L+m)/m \rfloor$?
      - Wait. $j = km - L$. We want $0 \le j \le m$.
      - $0 \le km - L \implies km \ge L \implies k \ge L/m$.
      - $km - L \le m \implies km \le L+m \implies k \le (L+m)/m$.
      - So there is always such $k$.
      - Example $L=3, m=1000$.
      - $k=1$: range $[0, 1000]$. $L \in [0, 1000]$.
      - $j = 1000 - 3 = 997$.
      - Map contains index 997?
      - We inserted $0 \dots m$. Yes.
      - So $next^{1000}(S)$ will be compared with $next^{997}(S)$.
      - Collision found. $V = 1000 - 997 = 3$.
      - Since we scan $k$ from 1 upwards.
      - For $k=1$, we check `curr` (which corresponds to $next^{1000}(S)$) against map.
      - `curr` matches `next^997(S)`.
      - Returns $V = 1000 - 997 = 3$.
      - Wait, does it return 3?
      - `map` stores `next^j`.
      - If $j=997$ is stored.
      - Yes.
      - But does scanning $k=1$ allow us to check $V=1 \dots 1000$?
      - For $k=1$, $curr = next^{1000}(S)$.
      - It checks if $curr \in \{ next^0(S), \dots, next^{1000}(S) \}$.
      - If $curr = next^{1000}(S)$ (i.e. $j=1000$), then $L=0$. (Not possible).
      - If $curr = next^{999}(S)$, then $V = 1000 - 999 = 1$.
      - If $curr = next^{997}(S)$, then $V = 3$.
      - It checks existence.
      - If multiple collisions occur for $k=1$?
      - E.g. if $L=2$. $curr = next^{1000}(S) = next^{500}(S)$? No, $next^{1000}(S) = next^{1000 \pmod 2}(S) = S$.
      - So $curr=S$. $S$ is at index 0.
      - Collision $1000 - 0 = 1000$.
      - Wait, if $L=2$, then $next^{1000}(S) = S$.
      - Collision found with $j=0$. $V=1000$.
      - But $L=2$. $V=1000$ is multiple.
      - Did we miss $L=2$?
      - For $k=1$, we only check if `curr` is in map.
      - If `curr` is in map at $j$, we find *one* collision.
      - We don't know if $L$ is smaller.
      - Wait, earlier I thought we check $1 \dots m$.
      - For $L=2, m=1000$.
      - We check $k=1$. $curr = S$. Matches $j=0$.
      - $V=1000$.
      - We return 1000.
      - WRONG.
      - We should have found $L=2$.
      - Why?
      - Because $L$ corresponds to period 2.
      - $L$ is not necessarily the first collision.
      - We need to find the *smallest* $L$.
      - $V=1000$ is a multiple of 2.
      - We MUST check if $V$ is minimal.
      - Since we scan $k=1$, we find $V \approx 1000$.
      - We don't check small $V$.
      - The issue is that the map lookup `vis[curr]` retrieves $j$.
      - If we have multiple $j$'s (but map stores only 1).
      - Here $j=0$ is the smallest index for $curr$.
      - But $curr$ corresponds to $S$ (index 0) via step $1000$.
      - Collision is $1000 - 0 = 1000$.
      - Is there any other $j$?
      - $next^{1000}(S)$ is $S$.
      - Map has $S$ at 0.
      - So we get 1000.
      - If $L=2$, we want to find 2.
      - Can we find it?
      - If $L=2$. $next^2(S) = S$.
      - We insert $S, n(S), n^2(S)=S$ into map?
      - If we insert $next^x(S)$ for $x=0 \dots m$.
      - If $S$ appears twice, map keeps first.
      - So $next^2(S)$ collides with $next^0(S)$ (stored).
      - Wait, inside baby loop, we check collision.
      - If `vis.count(curr)` is true inside baby loop ($x < m$):
      - Then we found a collision with $j < x$.
      - $V = x - j$.
      - Since $x < m$, $V < m$.
      - If $L=2$, then $next^2(S)=S$.
      - At $x=2$, `curr` is $S$. $j=0$.
      - $V = 2 - 0 = 2$.
      - We found $L=2$ inside baby loop!
      - So if $L < m$, we find it.
      - If $L > m$.
      - Then baby loop finishes with no collision.
      - Then giant loop starts.
      - $k=1$. $curr = next^{1000}(S)$.
      - If $L > 1000$.
      - Then $next^{1000}(S) \neq S$.
      - We check if $curr$ in map.
      - If $L$ is prime (e.g. 1001).
      - Then $next^{1000}(S)$ is some node $U$.
      - $U = next^{1000}(S)$.
      - Is $U$ in map? Map contains $next^j(S)$.
      - Since $L=1001$, $next^{1000}(S) \neq next^j(S)$ for $0 \le j < 1001$ (except trivially? No, $L=1001$, so $next^{1000}(S)$ is distinct from $next^0(S), \dots, next^{1000}(S)$? No $next^{1000}(S)$ is $next^{1000}(S)$).
      - Wait, $next^j(S)$ are distinct for $0 \le j < 1001$.
      - Map size is 1000.
      - So $next^{1000}(S)$ is likely not in map?
      - Actually $next^{1000}(S)$ is in map if $j=1000$ is stored?
      - Yes, if we stored up to $m$.
      - If we stored up to $m=1000$, then $next^{1000}(S)$ is in map at index 1000.
      - Wait, in my logic "Store $0 \dots m$".
      - If we store at $x=m$, we add $next^m(S)$.
      - Then check $k=1$. $curr = next^m(S)$.
      - It is in map. $j=m$.
      - $V = m - m = 0$.
      - Wait, $k \cdot m - j$ should be positive.
      - So we should exclude $j=k \cdot m$? Or rather check $j < k \cdot m$.
      - Wait, collision implies $next^{km}(S) = next^j(S)$.
      - If $j=km$, $next^0(S) = S$.
      - If we stored $up to m$.
      - $curr = next^m(S)$. $j=m$.
      - $V = 1000 - 1000 = 0$. Ignore.
      - But we need to find $L > m$.
      - We need a collision with $j < m$?
      - If $L > m$.
      - Then $next^m(S)$ is distinct from $next^0 \dots next^{m-1}$.
      - (Unless $m$ is multiple of $L$, impossible).
      - So $curr$ not found for $j < m$.
      - Then $k=2$. $curr = next^{2000}(S)$.
      - Check if in map.
      - If $L=1001$.
      - $2000 \pmod{1001} = 2000 - 1001 = 999$.
      - $curr = next^{999}(S)$.
      - Map contains $next^{999}(S)$ at index 999.
      - Found!
      - $j=999$.
      - $V = 2000 - 999 = 1001$.
      - Correct.
      - So for $L > m$, we find $L$ in giant steps.
      - What about $L < m$?
      - We find it in baby steps.
      - What about $L$ small multiple?
      - If $L=2$. $m=1000$.
      - Baby steps: at $x=2$, $curr=S$. $S$ in map at 0.
      - $V = 2$. Found.
      - If $L=1001$. Giant steps find 1001.
      - Wait, is there a case where we find a multiple instead of $L$?
      - If $L < m$, we find $V=L$.
      - If $L > m$, we find $V=L$ (since $L \le m^2$).
      - Wait, what if $L$ is small but $L > m$? Impossible.
      - What if $L$ is small but we missed it in baby steps?
      - Baby steps covers $0 \dots m$.
      - If $L \le m$, $next^L(S) = S$ is detected.
      - So $V=L$ found.
      - If $L > m$.
      - Then $next^L(S)=S$.
      - Giant steps check $km$.
      - Since $L \le m^2$, there exists $k$ such that $L \in [(k-1)m, km]$.
      - Then $next^{km}(S) = next^{km \pmod L}(S)$.
      - Let $rem = km \pmod L$.
      - If $rem < m$, then $next^{rem}(S)$ is in map.
      - Wait, we need $km - j = L$?
      - If $rem < m$, does $km \equiv j \pmod L$ imply $L | (km-j)$?
      - Yes.
      - But we want $km-j = L$.
      - If $rem = j$. Then $km - j = km - (km \pmod L)$.
      - This is the largest multiple of $L$ less than or equal to $km$.
      - Let this be $L \cdot q$.
      - If $q=1$, we find $L$.
      - If $q > 1$, we find a multiple.
      - When is $q > 1$?
      - When $km$ is large, specifically $km \ge 2L$.
      - If we find collision at $k$ such that $km \approx 2L$.
      - Is it possible that we skipped $k$ where $km \approx L$?
      - For $L > m$, the interval where $km \ge L$ starts at $k=1$ (since $m < L$).
      - Wait, $m < L$. So $1 \cdot m < L$.
      - So $k=1$ corresponds to $km = m$.
      - If $L > m$, $m$ is not a multiple of $L$ (unless $m=0$, no).
      - So no collision with $k=1$ yielding $L$ (since $L > m$).
      - Actually, for $k=1$, we check $curr = next^m(S)$.
      - $next^m(S)$ is $next^{m}(S)$.
      - $next^{km-j}(S) = S$.
      - $L | (m-j)$.
      - If $L > m$, then $m-j$ can't be a non-zero multiple of $L$.
      - So $m-j$ must be 0. $j=m$.
      - Then $V=0$.
      - So $k=1$ yields nothing relevant (only $j=m$).
      - Then $k=2$. $curr = next^{2m}(S)$.
      - $next^{2m}(S) = next^{2m \pmod L}(S)$.
      - We want to find $j$ such that $2m - j = L$ (ideally).
      - This requires $2m \equiv j \pmod L$.
      - Since $j \in [0, m]$, we need $2m \pmod L \in [0, m]$.
      - Let $r = 2m \pmod L$.
      - We need $r \le m$.
      - If this holds, we find $L$ (if $q=1$).
      - Or we find $q \cdot L = 2m - r$.
      - If $q=1$, we find $L$.
      - If $q > 1$, we find multiple.
      - But wait.
      - If $L \in (m, 2m]$.
      - Then $2m \ge L$.
      - $2m = 1 \cdot L + (2m - L)$.
      - So $r = 2m - L$.
      - Is $r \le m$?
      - $2m - L \le m \iff m \le L$.
      - Yes, since we assumed $L > m$.
      - And $r \ge 0$ obviously.
      - So if $L \in (m, 2m]$, we find collision at $k=2$ with $V = 2m - r = 2m - (2m-L) = L$.
      - So we find $L$.
      - What if $L \in (2m, 3m]$?
      - Then $k=3$. $curr = next^{3m}(S)$.
      - $3m = L + (3m - L)$.
      - Wait, $3m - L$ is remainder? No.
      - $3m \pmod L = 3m - L$.
      - Is $3m - L \le m$?
      - $3m - L \le m \iff 2m \le L$.
      - Yes.
      - So generally, if $L \in ((k-1)m, km]$, we find it at step $k$ with $V=L$.
      - Wait, is it always true that $L \le km$?
      - Yes.
      - Is it always true that $km - L < m$?
      - $km - L < m \iff (k-1)m < L$.
      - Yes.
      - So we always find $V=L$ exactly!
      - The condition is $0 \le km - L \le m$.
      - Which simplifies to $(k-1)m < L \le km$.
      - This covers all integers $L > 0$.
      - So YES, we always find $L$ exactly.
      - So no divisor check needed.
      - The only "risk" is if $L$ was not unique? No.
      - Or if $m=1000$ logic is flawed for some edge cases.
      - $L=1, 2$ handled by baby.
      - $L$ large handled by giant.
      - So algorithm is solid.
      - Complexity: $2m = 2000$ queries.
      - Time limit 5s.
      - Space limit 1GB.

    - One final check: $L \ge 3$.
      - If $L < 3$, the logic still works but answer would be invalid.
      - Problem says $L \ge 3$.
      - So we are good.
      - Also need to handle "hedgehog graph" property.
      - "Contains exactly one directed cycle of length at least 3".
      - "v belongs to the cycle".
      - We established $next(1)$ is in cycle.
      - Is it possible `query(1, 1)` leads to a tail that enters the cycle?
      - No, because if $u \to v$ and $v$ is in cycle, then $u$ points directly to cycle.
      - If $u$ was on a tail of length $>1$, say $z \to u \to v$.
      - Then $u$ must point to $v \in Cycle$.
      - But $z \to u$ means $u \in Cycle$ (condition says $v$ for edge $z \to u$ must be in cycle).
      - So $u \in Cycle$.
      - So no tails.
      - So $next(1)$ is definitely in cycle.
      - Wait, what if $next(1) = 1$?
      - Length $\ge 3$ implies no self-loop.
      - So $next(1) \neq 1$.
      - So $S \neq 1$.
      - Also $S$ is unique.
      - Everything seems correct.

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <cmath>

using namespace std;

// Global query function interface
void query(int v, long long x);
int read_query_response(); // placeholder for interactor
// We'll handle interaction in main loop

int main() {
    int n;
    if (!(cin >> n)) return 0;
    
    // Process n graphs
    for (int i = 0; i < n; ++i) {
        // 1. Find a vertex guaranteed to be on the cycle.
        // According to problem: for every edge u->v, v is in cycle.
        // So next(v) is always in cycle.
        // Let's start from vertex 1.
        // query(1, 1) moves from 1 to next(1).
        // next(1) is in the cycle.
        
        // Wait, we cannot access the interactor directly here.
        // We must implement the interaction logic.
        // But wait, standard CP templates usually put IO in functions.
        // Since it's an interactive problem, we print then read.
        
        int s;
        cout << "? 1 1" << endl; // Flush automatically? 
                                 // The problem asks to flush output. endl forces flush usually, or tie(NULL)
                                 // In C++, endl flushes stdout.
        
        cin >> s;
        // s is the vertex next(1). It is on the cycle.
        
        // 2. Run BSGS-like logic to find L.
        // We want smallest L >= 3 such that query(s, L) == s.
        // Max vertices = 10^6. m = sqrt(10^6) = 1000.
        int m = 1000;
        
        map<int, int> vis;
        int curr = s;
        vis[s] = 0;
        
        // Baby steps: compute s, next(s), ..., next^(m)(s) -- wait, actually we need to store positions.
        // But since we query, we need to track 'steps'.
        // Let's store vertex -> index.
        // But we can't iterate 1 by 1 if we want to save queries?
        // Wait, m=1000. Iterating 1 by 1 takes 1000 queries. Acceptable.
        
        // Wait, earlier analysis:
        // Baby steps loop i from 1 to m.
        // At each step we query(curr, 1) ?
        // No, query(curr, 1) is moving 1 step.
        // If we do this m times, we incur m queries.
        // Giant steps also m queries. Total 2m = 2000.
        // This is fine.
        
        // Let's optimize slightly:
        // In baby steps, we just traverse.
        // Store `vis[next_step] = i`.
        // Note: `next_step` is the vertex we arrive at after i steps.
        // Initially `vis[s] = 0`.
        // For i = 1 to m:
        //   s_next = query(s_curr, 1);
        //   vis[s_next] = i;
        //   s_curr = s_next;
        
        // Wait, query(u, 1) takes 1 query.
        // This is expensive if m is large.
        // But m=1000 is fixed.
        // Can we combine queries?
        // Maybe not needed.
        
        // Giant steps:
        // curr_giant = query(s_start, m).
        // For k = 1 to m:
        //   if vis contains curr_giant:
        //      L = k*m - vis[curr_giant];
        //      output L.
        //      break
        //   curr_giant = query(curr_giant, m).
        
        // Wait, we need to start with s.
        // So we need to run queries.
        
        // Is query(v, 1) efficient?
        // Yes.
        
        // Refined Plan:
        // 1. Get cycle_entry = query(1, 1).
        // 2. Baby steps:
        //    vis[cycle_entry] = 0.
        //    curr_cycle = cycle_entry.
        //    for i=1 to m:
        //       next_c = query(curr_cycle, 1);
        //       vis[next_c] = i;
        //       curr_cycle = next_c;
        //    Note: query(curr_cycle, 1) advances.
        //    Wait, is `query` idempotent? Yes.
        //    Wait, inside loop `curr_cycle` updates.
        //    So we make `m` queries.
        
        // 3. Giant steps:
        //    curr_giant = query(cycle_entry, m). // Moves m steps from entry
        //    // We need to check if curr_giant was visited.
        //    // The map has entries for 0..m.
        //    // Wait, if m <= 1000, map covers 0..1000.
        //    // If curr_giant is s (visited at 0), then m - 0 = m.
        //    // If curr_giant is s_next (visited at 1), then m - 1.
        //    // We check if collision.
        //    // If found, candidate L = k*m - vis[curr].
        //    // Iterate k.
        
        // 4. Output L.
        
        // Wait, the problem is interactive. We must not buffer or assume non-interactive.
        // Also, flushing.
        
        // Implementation details.
    }
}
```
One correction:
In Baby steps, we need to query `next(curr)`.
My loop:
`curr = query(curr, 1)`.
This performs 1 step.
We do this $m$ times.
Map stores `next(i)` at index `i`.
So `vis[curr] = i`.
Total $m$ queries.

In Giant steps, we need `query(start, m)`.
Then `query(curr, m)`.
Total $m$ queries.

Overall $2m = 2000$.
This fits.

However, `query` arguments:
$v$ is vertex index. $x$ is steps.
$v$ up to $10^6$. $x$ up to $5 \cdot 10^{18}$.
$m=1000$ is fine.

Double check the logic:
We assume cycle length $L \le 10^6$.
Is it possible cycle length is $> 10^6$?
Problem says: "Kipa has n hedgehog graphs with $10^6$ vertices".
Usually means the number of vertices is $10^6$.
Cycle length is bounded by number of vertices.
So $L \le 10^6$.
My logic holds.

Another optimization:
Can we use `query(start, i)` to get `next^i(start)` directly?
Yes.
Instead of `query(curr, 1)` repeated, we can do `query(start, i)`?
No, `query(start, i)` takes 1 query.
But `query(curr, 1)` also takes 1 query.
And we are moving state.
If we use `query(start, i)` for $i=1 \dots m$:
At step $i$, we call `res = query(start, i)`. `vis[res] = i`.
This saves tracking `curr` manually, but costs the same (1 query per $i$).
But wait, `query(start, i)` is not just jumping 1 step from previous.
It is jumping $i$ steps from `start`.
If we do this, we need to store `vis[res] = i`.
Then Giant steps: `curr = query(start, m)`.
If `vis
````

---

## 直接SFT 样本(sample_idx=0,得分 0.0,completion_tokens=877)

````
I need to find the cycle length in a hedgehog graph.

A hedgehog graph has:
- Each vertex has exactly one outgoing edge (functional graph)
- Contains exactly one directed cycle of length ≥ 3
- For every edge u -> v, vertex v belongs to the single directed cycle

Key property: Every edge leads to the unique cycle. The first time we return to a previously visited vertex during traversal, we've completed the cycle. Since every edge leads directly into the cycle (the problem states "v belongs to the aforementioned single directed cycle" for every edge), we can efficiently find it.

The challenge is querying x up to 5 * 10^18 times is too slow. We need to use modular arithmetic with Fermat's Little Theorem.

Let's denote:
- next(u) as going forward from vertex u
- After n queries starting from vertex u: query(u, n) = next^(n)(u) - applying next n times
- The cycle has length c, so next^c(v) = v for any v on the cycle

Using Fermat's Little Theorem: if gcd(n, m) = 1, then n^(m-1) ≡ 1 (mod m)

So:
query(u, n × c + r) = query(u, r) when n is large enough and n mod c ≠ 0

Strategy:
1. Start at vertex v and move forward until we hit a repeated vertex w (w = next(w))
2. This repeated vertex w lies on the cycle, since every path leads to it
3. Count how many vertices are between v and w on the path
4. Let d be the number of edges from w to w on the cycle; this is what we're looking for
5. If we know the cycle contains at least 3 vertices, then w cannot be an isolated fixed point or part of a 2-cycle
6. Querying w multiple times and averaging should let us recover d
</think>

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
using namespace std;

typedef long long ll;

ll q(ll v, ll x) {
    cout << "? " << v << " " << x << endl;
    ll ans; cin >> ans;
    return ans;
}

int main() {
    int n; cin >> n;
    
    while (n--) {
        ll v = 1, seen_v[1000001], cnt = 0, p = v;
        
        while (!seen_v[p]) {
            cnt++;
            seen_v[p] = 1;
            p = q(p, 1);
            if (cnt > 2500) break;
        }
        
        ll t = cnt;
        ll k = 0;
        
        while (t) {
            if (t & 1) k ^= seen_v[p];
            p = q(p, 1);
            t >>= 1;
            cnt++;
        }
        
        for (auto e : seen_v) e = 0;
        cnt = 0; p = 1;
        
        while (!seen_v[p]) {
            cnt++;
            seen_v[p] = 1;
            p = q(p, 1);
            if (cnt > 2500) break;
        }
        
        t = cnt;
        k = 0;
        
        while (t) {
            if (t & 1) k ^= seen_v[p];
            p = q(p, 1);
            t >>= 1;
            cnt++;
        }
        
        for (auto e : seen_v) e = 0;
        
        cout << "! " << __builtin_popcountll(k ^ seen_v[p]) << endl;
        
        ll x; cin >> x;
        if (x == -1) exit(0);
    }
    return 0;
}
```
````
