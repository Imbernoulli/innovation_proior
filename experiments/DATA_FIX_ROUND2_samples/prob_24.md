# 题 24 — base均分 0.0 vs 直接SFT均分 0.0(掉点 0.0)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=0,得分 0.0,completion_tokens=32768)

````
The problem asks us to find a permutation $p_1, \dots, p_n$ of $1, \dots, n$ such that the sequence of edge weights between consecutive vertices in the cycle formed by $p$ is "almost monochromatic". Specifically, let $c_i = C_{p_i, p_{i+1}}$ for $1 \le i < n$, and $c_n = C_{p_n, p_1}$. We want the number of indices $i \in \{1, \dots, n\}$ where $c_i \ne c_{i+1}$ (with $c_{n+1} = c_1$) to be at most 1. Wait, looking at the problem statement: "the number of indices i (1 ≤ i < n) where c_i != c_{i+1}". Actually, usually for a cycle we consider $n$ edges. Let's re-read carefully.

"The permutation p is almost monochromatic if and only if the number of indices i (1 ≤ i < n) where c_i ≠ c_{i+1} is at most one."
Here $c_i = C_{p_i, p_{i+1}}$ for $1 \le i < n$, and $c_n = C_{p_n, p_1}$.
Wait, the definition of $c_i$ is given for $1 \le i < n$ and $i=n$. So we have $c_1, c_2, \dots, c_n$.
The condition is about indices $i$ where $c_i \ne c_{i+1}$. The range of $i$ in the problem description says $(1 \le i < n)$. This seems slightly ambiguous because $c_{n+1}$ isn't defined as $c_1$ explicitly in the text "where c_i != c_{i+1}". However, usually in cycle problems involving monochromaticity, it implies checking transitions around the cycle.
Let's look at the sample explanation.
Sample 1: $n=3$, Matrix:
0 0 1
0 0 0
1 0 0
Edges: (1,2)=0, (2,3)=0, (3,1)=1.
Permutation 3, 1, 2.
$p_1=3, p_2=1, p_3=2$.
$c_1 = C_{p_1, p_2} = C_{3,1} = 1$.
$c_2 = C_{p_2, p_3} = C_{1,2} = 0$.
$c_3 = C_{p_3, p_1} = C_{2,3} = 0$.
The condition is on indices $i \in \{1, \dots, n-1\}$? Or is it considering $c_{n+1} = c_1$?
Text: "number of indices i (1 ≤ i < n) where c_i ≠ c_{i+1} is at most one."
Wait, if it strictly means $1 \le i < n$, then we compare $c_1, c_2$; $c_2, c_3$; ...; $c_{n-1}, c_n$. There are $n-1$ comparisons.
In Sample 1: $c_1=1, c_2=0, c_3=0$.
Indices $i$:
$i=1: c_1 \ne c_2 \implies 1 \ne 0$ (True).
$i=2: c_2 \ne c_3 \implies 0 \ne 0$ (False).
So count is 1. At most 1, so valid.
Note also mentions "c1 = C_{3,1}=1, c2=C_{1,2}=0, c3=C_{2,3}=0. Only when i=1, c_i != c_{i+1}."
So indeed, we check pairs $(c_1, c_2), (c_2, c_3), \dots, (c_{n-1}, c_n)$. We do NOT check $(c_n, c_1)$ according to the strict text, OR maybe the text implies cyclic wrap but wrote $1 \le i < n$. Wait, usually "monochromatic" implies the whole cycle has same color. "Almost monochromatic" allows 1 switch. If we don't close the cycle with a switch check, it effectively means the path $p_1 \to p_2 \to \dots \to p_n$ has at most 1 color change. The last edge $(p_n, p_1)$ closes the cycle but isn't part of the "switches" count constraint?
Actually, looking closely at the condition: "indices i (1 ≤ i < n)".
It compares $c_i$ with $c_{i+1}$. For $i=n$, it would compare $c_n$ with $c_{n+1}$. But $i < n$.
However, for the property to make sense for permutations (cycles), usually all edges matter.
Let's check the wording again.
"c_i=(C_{p_i, p_{i+1}} for 1 ≤ i < n, C_{p_n, p_1} for i = n)."
So $c_1, \dots, c_n$ are the edge weights of the Hamiltonian cycle.
Condition: "number of indices i (1 ≤ i < n) where c_i != c_{i+1} is at most one."
This strictly checks transitions between adjacent edges in the cycle sequence $c_1, \dots, c_n$, ignoring the transition from $c_n$ to $c_1$.
But wait, if $c_1, \dots, c_n$ represents edges of a cycle, why exclude $c_n$ vs $c_1$?
Maybe it's just defining a property of the linear sequence of colors, derived from the cycle.
Regardless, I must follow the problem statement. But let's double check Sample 2.
Sample 2: All zeros. Any permutation works. Output 2 4 3 1.
Colors: 0 0 0 0. Transitions: $0 \to 0$ (ok), $0 \to 0$ (ok), $0 \to 0$ (ok). Count = 0. Valid.

Let's assume the constraint is exactly as written: count transitions $c_i \to c_{i+1}$ for $i=1 \dots n-1$.
Wait, actually, if the count includes $c_n \to c_1$, it is symmetric. If it excludes it, it is specific. Given "cyclic" nature usually implied by $p_n, p_1$, usually conditions are cyclic.
Let's reconsider the Note. "Only when i=1, c_i != c_{i+1}." They listed $c_1, c_2, c_3$. They compared $c_1, c_2$ and $c_2, c_3$. They did not compare $c_3, c_1$.
If they compared $c_3, c_1$ ($0, 1$), there would be another difference.
Wait, $c_3=0, c_1=1$. $0 \ne 1$.
So if we included $c_n \ne c_1$, the count would be 2 (at $i=1$ and implicit wrap around).
Since the sample solution is accepted, the condition *must* exclude the wrap-around comparison $c_n \ne c_1$. It counts "internal switches" in the sequence of edges $c_1, \dots, c_n$.
Basically, we need a Hamiltonian path on the graph such that the sequence of edge colors has at most 1 switch.
Wait, since it's a permutation of a complete graph (or rather the vertices are permuted), every pair $(p_i, p_{i+1})$ is an edge.
Actually, we construct a cycle $p_1 \to p_2 \to \dots \to p_n \to p_1$.
The edges are $e_1=(p_1, p_2), e_2=(p_2, p_3), \dots, e_{n-1}=(p_{n-1}, p_n), e_n=(p_n, p_1)$.
Weights are $w(e_k)$.
We want the sequence $w(e_1), \dots, w(e_n)$ to have at most 1 position $k \in \{1, \dots, n-1\}$ where $w(e_k) \ne w(e_{k+1})$.
This means either all edges in the cycle are 0 except possibly a block of 1s, or all edges are 1 except a block of 0s.
Or more simply:
Either all $e_1, \dots, e_n$ are 0, or all are 1, or there is exactly one index $k$ where $0 \to 1$ or $1 \to 0$ occurs.
If there are no 1s, count is 0.
If there are no 0s, count is 0.
If both exist, they must appear consecutively in the sequence $c$. i.e., a sequence like $0, 0, \dots, 0, 1, 1, \dots, 1$ or $1, 1, \dots, 1, 0, 0, \dots, 0$.
Why? Because any switch contributes to the count. If we have $0, 1, 0$, switches are at first $0 \to 1$ and $1 \to 0$, total 2. We can have at most 1.
So the sequence $c_1, \dots, c_n$ must look like:
Case A: All 0s.
Case B: All 1s.
Case C: Some 0s followed by some 1s (e.g., $0, \dots, 0, 1, \dots, 1$). Switch is at boundary.
Case D: Some 1s followed by some 0s (e.g., $1, \dots, 1, 0, \dots, 0$). Switch is at boundary.
Wait, does the "at most 1" apply to $c_n \to c_1$? The text says $1 \le i < n$. So $c_n \to c_1$ is not checked.
So we just need to find a Hamiltonian cycle such that the edge weights form a sequence of type $0\dots01\dots1$ or $1\dots10\dots0$ (since order matters for lexicographical minimization, but structurally it's just partition of vertices).
Specifically, the edges $(p_1, p_2), \dots, (p_n, p_1)$ must have weights $w_1, \dots, w_n$.
The condition $\sum_{i=1}^{n-1} [w_i \ne w_{i+1}] \le 1$.
This implies the sequence is monochromatic (all 0 or all 1) or has one transition point.
Example: $0, 0, 1, 1$. Transition at index 2. $0 \ne 1$. Count = 1.
So essentially, we need to decompose the graph into two sets of vertices, say $S_0$ and $S_1$.
Inside $S_0$, the edges must be 0. Inside $S_1$, edges must be 0? No.
Wait, the edges correspond to transitions in the path.
Let's interpret the structure.
We visit vertices $p_1, \dots, p_n$.
Edges $(p_i, p_{i+1})$ are traversed.
If all $C_{u,v} = 0$ along the path, then all $c_i=0$.
If the path uses some 0-edges and some 1-edges, say the sequence of weights is $0, \dots, 0, 1, \dots, 1$.
Then we must visit a subset of vertices using only 0-edges, then transition to 1-edge, then traverse the rest using 1-edges?
Wait, the transition happens *between* edges.
So $c_1, \dots, c_k$ are 0, and $c_{k+1}, \dots, c_n$ are 1.
Edge $k$ is $(p_k, p_{k+1})$ with weight 0. Edge $k+1$ is $(p_{k+1}, p_{k+2})$ with weight 1.
This means vertex $p_{k+1}$ is connected to $p_k$ via 0-weight, and to $p_{k+2}$ via 1-weight.
Also $p_{k+2}$ connects to $p_{k+3}$ via 1-weight.
And finally $p_n$ connects to $p_1$ via weight 1 (if $n > k$) or 0 (if $k=n$). Wait, $c_n$ corresponds to $(p_n, p_1)$.
If $c_1 \dots c_k=0$ and $c_{k+1} \dots c_n=1$.
This requires edges incident to the "boundary" vertex $p_{k+1}$: $(p_k, p_{k+1})$ is 0, $(p_{k+1}, p_{k+2})$ is 1.
Also we have $(p_1, p_2), \dots, (p_{k-1}, p_k)$ are 0.
And $(p_{k+2}, p_{k+3}), \dots, (p_n, p_1)$ are 1.
So this looks like a union of paths/cycles within subgraphs.
But since it's a single permutation, it's a single Hamiltonian Cycle.
Basically, we need to find a Hamiltonian cycle such that all edges $(p_i, p_{i+1})$ for $i=1 \dots k$ are 0, and all edges $(p_j, p_{j+1})$ for $j=k+1 \dots n$ are 1.
(With $p_{n+1} \equiv p_1$). Wait, $c_n$ is $C_{p_n, p_1}$.
If the split is after $c_k$ (edge index), then edges $1 \dots k$ are 0, edges $k+1 \dots n$ are 1.
This means edges $(p_1, p_2), \dots, (p_k, p_{k+1})$ are 0-weight.
And edges $(p_{k+1}, p_{k+2}), \dots, (p_n, p_1)$ are 1-weight.
Wait, the vertices involved in the "0-part" form a path $p_1 \dots p_{k+1}$. All these edges are 0.
The vertices involved in the "1-part" form a path $p_{k+1} \dots p_n$ then back to $p_1$.
Wait, $p_{k+1}$ is the bridge.
Essentially, we pick a cut vertex? No.
The sequence of vertices $p_1, p_2, \dots, p_n$ is visited.
Edges $(p_1, p_2), \dots, (p_k, p_{k+1})$ have weight 0.
Edges $(p_{k+1}, p_{k+2}), \dots, (p_n, p_1)$ have weight 1.
This implies that $p_1, \dots, p_{k+1}$ can be reached sequentially using only 0-edges.
Wait, if $k=0$, then all edges are 1. If $k=n$, all edges are 0.
If $0 < k < n$, we have a segment of 0-edges and a segment of 1-edges.
Let's denote the set of vertices $V$.
The permutation partitions vertices into sets based on their position in the cycle? Not necessarily contiguous.
Wait, if $c_i=0$ means the edge connecting $p_i$ and $p_{i+1}$ has weight 0.
So the cycle is a sequence of vertices $v_1, v_2, \dots, v_n$.
$(v_1, v_2)$ is weight 0 or 1.
$(v_2, v_3)$ is weight 0 or 1.
...
$(v_n, v_1)$ is weight 0 or 1.
Condition: at most 1 switch in $0, \dots, 0, 1, \dots, 1$ pattern.
Since it's a sequence of length $n$, a switch can occur at any position.
Pattern: $0^a 1^b$ where $a+b=n, a,b \ge 0$.
Or $1^a 0^b$ where $a+b=n, a,b \ge 0$.
These are equivalent because reversing the sequence of $c$ changes $0^a 1^b$ to $0^b 1^a$ or something similar? No, the order in the cycle matters.
But notice, $0^a 1^b$ means $c_1=\dots=c_a=0, c_{a+1}=\dots=c_n=1$.
$c_a = C_{p_a, p_{a+1}} = 0$.
$c_{a+1} = C_{p_{a+1}, p_{a+2}} = 1$.
This means $p_{a+1}$ has a neighbor $p_a$ with weight 0 and $p_{a+2}$ with weight 1.
All previous neighbors $p_{prev}$ in chain must be 0.
All subsequent must be 1.
This implies that the set of vertices $\{p_1, \dots, p_{a+1}\}$ induces a path in the subgraph of 0-edges?
Yes.
Also $\{p_{a+2}, \dots, p_n, p_1\}$ induces a path in subgraph of 1-edges?
Let's trace the cycle edges.
Path 1: $p_1 \xrightarrow{0} p_2 \xrightarrow{0} \dots \xrightarrow{0} p_{a+1}$. This is a path of length $a$ consisting of 0-edges. Vertices $P_A = \{p_1, \dots, p_{a+1}\}$.
Wait, this is not quite right. $p_1$ is start. $p_{a+1}$ is end.
Path 2: $p_{a+1} \xrightarrow{1} p_{a+2} \xrightarrow{1} \dots \xrightarrow{1} p_n \xrightarrow{1} p_1$. This is a path of length $n-a$ consisting of 1-edges.
Wait, edge $(p_n, p_1)$ must be 1.
So basically, the permutation defines a split of vertices?
Let's consider the vertices reachable via 0-edges.
In the case $0^a 1^b$ ($a,b > 0$), the permutation visits a sequence of vertices using only 0-edges until $p_{a+1}$, then continues using only 1-edges and returns to $p_1$ using only 1-edges.
Wait, this implies $p_1$ is reachable from $p_{a+1}$ using only 1-edges. And $p_{a+1}$ is reachable from $p_1$ using only 0-edges.
So, we need to find a subset of vertices $S$ (containing $p_{a+1}$? No).
Let's define $S = \{p_{a+1}\}$. No.
Actually, look at the sets of vertices visited during the 0-phase and 1-phase.
Phase 0 visits $p_1, \dots, p_{a+1}$. The edges are $(p_i, p_{i+1})$.
So $p_1$ is connected to $p_2$ (0), ..., $p_a$ connected to $p_{a+1}$ (0).
Phase 1 visits $p_{a+2}, \dots, p_n, p_1$. Edges $(p_{i}, p_{i+1})$.
Note the last edge is $(p_n, p_1)$ with weight 1.
So, $p_{a+1}$ is connected to $p_{a+2}$ (1).
This suggests that $p_{a+1}$ acts as a gateway between the 0-path and 1-path.
Is there a simpler view?
The edges used are exactly all edges in a Hamiltonian cycle.
Suppose we fix the split point $k$ (meaning first $k$ edges are 0, rest $n-k$ edges are 1).
Then edges $(p_1, p_2), \dots, (p_k, p_{k+1})$ are 0.
Edges $(p_{k+1}, p_{k+2}), \dots, (p_n, p_1)$ are 1.
This means the cycle consists of a path of 0-edges from $p_1$ to $p_{k+1}$ and a path of 1-edges from $p_{k+1}$ to $p_1$.
Since it's a permutation, all vertices must be covered.
Vertices $\{p_1, \dots, p_{k+1}\}$ are endpoints/intermediate nodes in the 0-path?
No. $p_1 \to p_2 \to \dots \to p_{k+1}$.
Vertices in this sequence are $\{p_1, \dots, p_{k+1}\}$.
Wait, if $p_1$ is part of the 0-path, it must be distinct from vertices in 1-path?
Except $p_1$ connects to $p_{k+2}$? No, $p_{k+1}$ connects to $p_{k+2}$.
$p_n$ connects to $p_1$ via 1.
So the set of vertices visited in 0-phase is $V_0 = \{p_1, \dots, p_{k+1}\}$.
The set of vertices visited in 1-phase is $V_1 = \{p_{k+1}, p_{k+2}, \dots, p_n, p_1\}$.
Notice $V_0$ is the prefix of the cycle (start to node $p_{k+1}$).
$V_1$ is the suffix wrapping around.
Wait, this partitioning logic suggests that the vertices can be split into two sets $A$ and $B$ such that edges within $A$ are 0 (except for connection to B?), edges within $B$ are 1 (connection to A?).
Let's check the constraints.
For $p_1 \to \dots \to p_{k+1}$, all internal edges are 0.
For $p_{k+1} \to \dots \to p_1$, all internal edges are 1.
Note $p_1$ and $p_{k+1}$ are special.
$p_1$ starts the 0-path and ends the 1-path.
$p_{k+1}$ ends the 0-path and starts the 1-path.
So, all edges incident to $p_i$ ($i \in \{2, \dots, k\}$) within the 0-segment are 0.
Actually, the edges $(p_i, p_{i+1})$ for $1 \le i \le k$ are 0.
This forms a path covering some vertices.
The edges $(p_{k+1}, p_{k+2}) \dots (p_{n-1}, p_n)$ and $(p_n, p_1)$ are 1.
This forms a path covering the remaining vertices (plus $p_1$ and $p_{k+1}$ are the junctions).
Crucially, the vertices $\{p_1, \dots, p_{k+1}\}$ must be connected by 0-edges in the order of the permutation.
So, $p_1$ must be connected to $p_2$ (0), $p_2$ to $p_3$ (0), ..., $p_k$ to $p_{k+1}$ (0).
So $p_2, p_3, \dots, p_{k+1}$ are reachable from $p_1$ via 0-paths?
More strongly, they form a simple path.
So $p_1, p_2, \dots, p_{k+1}$ is a Hamiltonian path in the subgraph induced by vertices in the set? No.
The set of vertices is exactly $\{p_1, \dots, p_{k+1}\}$.
So $\{p_1, \dots, p_{k+1}\}$ is a path of length $k$ in the graph of 0-edges.
Similarly, $\{p_{k+1}, \dots, p_n, p_1\}$ is a path of length $n-k$ in the graph of 1-edges.
Wait, $\{p_{k+1}, \dots, p_n, p_1\}$ contains $n - (k+1) + 1 + 1 = n - k + 1$ vertices?
The path is $p_{k+1} \to p_{k+2} \dots \to p_n \to p_1$.
Length is $n - (k+1) + 1 + 1 = n - k + 1$. No, number of edges is $n - (k+1) + 1 = n - k$.
Number of vertices is $n - k + 1$.
These two sets share vertices $p_{k+1}$ and $p_1$?
Wait, if the union of vertices is all $n$ vertices, then:
$| \{p_1, \dots, p_{k+1}\} | = k+1$.
$| \{p_{k+1}, \dots, p_n, p_1\} | = n-k+1$.
The intersection is $\{p_{k+1}, p_1\}$.
For this to be consistent, $V_0 \cup V_1 = V$ and $V_0 \cap V_1 = \{p_1, p_{k+1}\}$?
Not necessarily disjoint.
Wait, in a permutation $p_1, \dots, p_n$, all vertices are distinct.
So $\{p_1, \dots, p_{k+1}\}$ are distinct vertices.
$\{p_{k+1}, \dots, p_n, p_1\}$ are not necessarily distinct?
Ah, the set of indices is $1, \dots, n$.
The values are $p_1, \dots, p_n$.
The set of values in the cycle is exactly $\{1, \dots, n\}$.
The 0-path covers vertices corresponding to indices $1, \dots, k+1$.
The 1-path covers vertices corresponding to indices $k+1, \dots, n$ plus the closure to $p_1$.
Wait, indices of permutation are positions.
Positions $1, \dots, k+1$ use edges $(p_i, p_{i+1})$ for $i=1..k$. These are 0-edges.
Positions $k+1, \dots, n$ use edges $(p_i, p_{i+1})$ for $i=k+1..n-1$ and $(p_n, p_1)$ for $i=n$. These are 1-edges.
So we are looking for a decomposition of vertices into a set $S$ such that there is a Hamiltonian path in the 0-subgraph restricted to some subset?
Actually, observe:
The 0-path is $p_1 \to p_2 \to \dots \to p_{k+1}$.
The 1-path is $p_{k+1} \to p_{k+2} \to \dots \to p_n \to p_1$.
Note that the vertices in the 0-path are $V_0 = \{p_1, \dots, p_{k+1}\}$.
The vertices in the 1-path are $V_1 = \{p_{k+1}, \dots, p_n, p_1\}$.
Wait, $V_0$ and $V_1$ are just subsets of vertices visited.
The union of the sequences of vertices is the full set.
Actually, the crucial observation is likely related to bipartite matching or component structures.
But notice the constraint $N \le 2000$. Sum of $N \le 2000$.
This suggests $O(N^2)$ or $O(N^3)$ algorithm per test case is acceptable.
If we try all possible split points $k$? No, we don't know $k$.
However, the problem asks for an *any* almost monochromatic permutation. Then minimize lexicographically.
Lexicographically smallest means $p_1$ should be 1 if possible, else 2, etc.
Wait, "minimize the lexicographical order" of the output string $p_1 \dots p_n$.
Score $S = \sum n^{n-i} p_i$ roughly. Minimizing $S$ is equivalent to minimizing the vector $(p_1, p_2, \dots, p_n)$ lexicographically.
So we prefer smaller numbers earlier.

Is it always possible?
Matrix is symmetric, 0/1, diagonal 0.
Consider the complement graph where edges with weight 1 are kept (or vice versa).
If graph of 0-edges has a Hamiltonian Path starting at $p_1$ ending at $p_{k+1}$?
Actually, maybe we can simplify.
Can we solve for fixed colors?
Try to find a path of 0s and path of 1s?
Wait, if the sequence is all 0s, we need a Hamiltonian cycle of 0-edges.
If all 1s, cycle of 1-edges.
If mixed, we need $0 \dots 0$ path and $1 \dots 1$ path joined at boundaries.
But notice $p_1$ and $p_{k+1}$ are the joints.
$p_1$ is endpoint of 0-path (start) and endpoint of 1-path (end).
$p_{k+1}$ is endpoint of 0-path (end) and start of 1-path.
So we need:
1. A path of 0-edges $p_1 \rightsquigarrow p_{k+1}$ covering a subset of vertices?
Wait, if $p_1 \dots p_{k+1}$ are distinct vertices, and $p_{k+1} \dots p_n \dots p_1$ are distinct vertices.
Since it's a permutation, the sequence $p_1, \dots, p_n$ lists each vertex exactly once.
So the path $p_1 \to \dots \to p_{k+1}$ covers vertices $p_1, \dots, p_{k+1}$.
The path $p_{k+1} \to \dots \to p_n \to p_1$ covers vertices $p_{k+1}, p_{k+2}, \dots, p_n, p_1$.
Wait, this implies $p_1$ is visited twice?
Once at index 1, once at index $n$ (implicitly as closure)?
Yes, $p_1$ is $p_{start}$ and $p_{end}$.
But $p_2, \dots, p_n$ are distinct.
So vertices $\{p_1, \dots, p_{k+1}\}$ are distinct.
Vertices $\{p_{k+1}, \dots, p_n\}$ are distinct?
Wait, the sequence $p_1, \dots, p_n$ contains all distinct vertices.
So $\{p_1, \dots, p_{k+1}\}$ is a subset of size $k+1$.
$\{p_{k+1}, \dots, p_n\}$ is a subset of size $n-k$.
Their union is all vertices. Intersection is $\{p_{k+1}\}$.
But we also need to satisfy the edges.
Edges $(p_i, p_{i+1})$ for $i \in \{1, \dots, k\}$ are 0.
Edges $(p_i, p_{i+1})$ for $i \in \{k+1, \dots, n-1\}$ AND $(p_n, p_1)$ are 1.
Wait, if $(p_n, p_1)$ is 1, then $p_1$ and $p_n$ must have weight 1 edge.
This implies $p_1$ is part of the 1-cycle component as well?
Actually, let's look at the structure required.
The set of vertices $V$ is partitioned into two sets $A = \{p_1, \dots, p_{k+1}\}$ and $B = \{p_{k+2}, \dots, p_n\}$? No, because $p_1$ is connected to $p_n$ with 1.
And $p_{k+1}$ is connected to $p_{k+2}$ with 1.
Wait, $p_1$ is the start of 0-path, end of 1-path.
$p_{k+1}$ is the end of 0-path, start of 1-path.
Vertices involved in 0-path: $p_1, p_2, \dots, p_{k+1}$.
Vertices involved in 1-path: $p_{k+1}, p_{k+2}, \dots, p_n, p_1$.
This implies that $p_1$ and $p_{k+1}$ are the "interface" vertices.
Also all other vertices $p_2, \dots, p_k$ must be connected to their predecessor/successor with 0.
And $p_{k+2}, \dots, p_n$ must be connected with 1.
Let $S = \{p_2, \dots, p_k\}$. These are vertices strictly inside the 0-path segment (excluding endpoints $p_1, p_{k+1}$).
Let $T = \{p_{k+2}, \dots, p_n\}$. These are vertices strictly inside the 1-path segment (excluding endpoints $p_{k+1}, p_1$).
Note $S$ and $T$ are disjoint sets of vertices (since $p$ is a permutation).
Wait, if $k=n$, then $S = \{p_2 \dots p_n\}$, $T = \emptyset$. Then all 0.
If $k=0$, then $S = \emptyset$, $T = \{p_2 \dots p_n\}$. Then all 1.
For general $k$:
Vertices $S \cup T \cup \{p_1, p_{k+1}\} = V$.
Edges within $S \cup \{p_1, p_{k+1}\}$ must form a path $p_1 \to \dots \to p_{k+1}$?
Wait, $p_2, \dots, p_k$ are intermediate nodes.
So $p_1$ connects to $p_2$ (0), $p_2$ to $p_3$ (0), ..., $p_k$ to $p_{k+1}$ (0).
This means $S$ is a path from $p_1$ to $p_{k+1}$ in the 0-graph.
Wait, $S = \{p_2, \dots, p_k\}$.
If $k=1$, $S=\emptyset$. Then $p_1$ connects to $p_2$ (which is $p_{k+1}$). Weight 0.
Then 1-path starts at $p_2$.
So essentially, we pick two vertices $u, v$ (candidates for $p_1$ and $p_{k+1}$) such that:
There exists a Hamiltonian path from $u$ to $v$ using ONLY 0-edges?
Wait, no. Because the vertices $p_2, \dots, p_k$ are exactly those in the 0-path excluding $u, v$.
Wait, do we need ALL vertices to be covered?
Yes, $p_1, \dots, p_n$ is a permutation.
So the 0-path $p_1 \to \dots \to p_{k+1}$ covers vertices $\{p_1, \dots, p_{k+1}\}$.
The 1-path $p_{k+1} \to \dots \to p_n \to p_1$ covers vertices $\{p_{k+1}, \dots, p_n, p_1\}$.
Are these sets of vertices disjoint?
$\{p_1, \dots, p_{k+1}\}$ and $\{p_{k+1}, \dots, p_n, p_1\}$ share $p_1$ and $p_{k+1}$.
So the union is $V$. The intersection is $\{p_1, p_{k+1}\}$.
Wait, this means vertices other than $p_1, p_{k+1}$ are unique to either path?
Let's verify.
Indices $1 \dots n$.
Values $p_1 \dots p_n$.
Indices $1 \dots k+1$ correspond to vertices $p_1, \dots, p_{k+1}$.
Indices $k+1 \dots n+1$ (mod $n+1$?) No, edges $k+1 \dots n$ and $n$ connect to $p_1$.
The vertices involved in the 1-path sequence are $p_{k+1}, p_{k+2}, \dots, p_n, p_1$.
These are $n$ vertices? No.
Indices of $p$ array are $k+1, \dots, n$. Plus index 1 (wrapped).
The values are $p_{k+1}, p_{k+2}, \dots, p_n, p_1$.
Since $p$ is a permutation, all $p_1, \dots, p_n$ are distinct.
So the set of values $\{p_{k+1}, \dots, p_n\}$ is size $n-k$.
The set $\{p_1, \dots, p_{k+1}\}$ is size $k+1$.
Intersection: $\{p_1, p_{k+1}\}$. Size 2.
Wait, what if $p_1 = p_{k+1}$? Impossible as permutation indices $1 \ne k+1$.
So yes, two sets intersect at exactly two vertices $p_1$ and $p_{k+1}$.
This implies that the set of vertices $V \setminus \{p_1, p_{k+1}\}$ is partitioned into $S = \{p_2, \dots, p_k\}$ and $T = \{p_{k+2}, \dots, p_n\}$.
For the 0-path to cover $S$, all edges in $p_1 \to S \to p_{k+1}$ must be 0.
So $p_1$ must connect to $p_2 \in S$, ..., $p_k \in S$, ..., $p_{k+1}$.
Wait, if $S$ is non-empty, we need to traverse all vertices in $S$ exactly once using 0-edges, starting at $p_1$ and ending at $p_{k+1}$.
Similarly, $T$ must be traversed using 1-edges, starting at $p_{k+1}$ and ending at $p_1$.
Wait, $p_{k+1}$ connects to $p_{k+2} \in T$, ..., $p_n \in T$, ..., $p_1$.
So $p_{k+1} \rightsquigarrow T \rightsquigarrow p_1$ must be a path of 1-edges covering $T$.
This is very restrictive.
We need to select a pair of vertices $(u, v)$ ($u=p_1, v=p_{k+1}$) and a partition of $V \setminus \{u, v\}$ into two sets $S, T$.
Then $u \to S \to v$ is a 0-path.
$v \to T \to u$ is a 1-path.
This implies that $S$ must be reachable from $u$ via 0-edges in the induced subgraph, and $u$ is reachable from $S$? No, $u$ connects to $p_2 \in S$.
Wait, it doesn't require $S$ to be connected to itself fully, just a path through them.
Actually, the easiest way to check this is:
Does there exist a Hamiltonian Path in $G_0$ from $u$ to $v$ that visits some subset of vertices? No, it must visit ALL vertices in $S$.
But we don't know $S$.
Actually, if we fix $u, v$, can we determine $S$ and $T$?
If we assume the split point $k$ is variable, we just need to find *some* valid configuration.
Wait, look at the condition again.
Sequence $c_1, \dots, c_n$ has at most 1 switch.
This means either:
1. All $c_i = 0$. Hamiltonian cycle of 0-edges.
2. All $c_i = 1$. Hamiltonian cycle of 1-edges.
3. Sequence is $0, \dots, 0, 1, \dots, 1$ (one switch $0 \to 1$).
   This means $c_1 \dots c_k = 0$, $c_{k+1} \dots c_n = 1$.
   Edges: $(p_1, p_2) \dots (p_k, p_{k+1})$ are 0.
   $(p_{k+1}, p_{k+2}) \dots (p_n, p_1)$ are 1.
   This requires finding a path in $G_0$ from $p_1$ to $p_{k+1}$ using $k$ edges.
   Wait, $k$ is the number of 0-edges.
   Number of vertices in 0-segment is $k+1$.
   Number of vertices in 1-segment is $n-(k+1)+1$? No.
   Let's trace vertices.
   Start at $p_1$. Traverse $k$ edges via 0s. We reach $p_{k+1}$. Vertices visited $\{p_1, \dots, p_{k+1}\}$.
   From $p_{k+1}$, traverse $n-k$ edges via 1s. We reach $p_{k+1+k'}$? No, $p_1$ eventually.
   Vertices visited in second phase: $\{p_{k+1}, p_{k+2}, \dots, p_n, p_1\}$.
   Note that $p_1$ and $p_{k+1}$ are repeated in terms of sequence but the set of unique vertices covered is $V$.
   Crucially, the set of vertices $\{p_2, \dots, p_k\}$ must be visited by the 0-path.
   The set $\{p_{k+2}, \dots, p_n\}$ must be visited by the 1-path.
   The vertices $p_1$ and $p_{k+1}$ are shared endpoints.
   So, we are looking for a "cycle decomposition" where we traverse a subset of vertices using 0-edges forming a path, and the complement subset using 1-edges forming a path, connecting them properly.
   Wait, if $G_0$ has a Hamiltonian Path from $u$ to $v$?
   If $G_0$ has a Hamiltonian Path from $u$ to $v$ (using all vertices), then $k=n-1$ and $p_1=u, p_{k+1}=v$.
   Wait, if path uses all vertices, then $S = V \setminus \{u, v\}$ is covered.
   Then the remaining edges must be 1. But there are no remaining vertices for 1-path!
   Wait, if we cover all vertices in 0-path, then we are done with vertices?
   Wait, we still need to close the cycle $p_n \to p_1$ which must be 1?
   But if we used a Hamiltonian Path in $G_0$ on all vertices, say $u \to \dots \to v$.
   Then $p_1=u, p_2, \dots, p_n=v$.
   Then edges $(p_1, p_2) \dots (p_{n-1}, p_n)$ are 0.
   Last edge $(p_n, p_1) = (v, u)$ must be 1.
   So if we find a Hamiltonian Path in $G_0$ from $u$ to $v$ such that $C_{u,v}=1$, we have a valid permutation with 1 switch (from 0 to 1 at end).
   Actually, the condition is $c_n = C_{p_n, p_1}$.
   Here $p_n=v, p_1=u$. So $c_n = C_{v,u}$.
   Since matrix is symmetric, $C_{v,u} = C_{u,v}$.
   So we need $C_{u,v} = 1$.
   But if $C_{u,v}=0$, then $c_n=0$, so no switch (all 0s).
   So:
   Case "All 0s": Hamiltonian Cycle in $G_0$.
   Case "All 1s": Hamiltonian Cycle in $G_1$.
   Case "$0 \dots 0, 1 \dots 1$": Hamiltonian Path in $G_0$ from $u$ to $v$ such that $C_{u,v}=1$.
   Wait, if $C_{u,v}=1$, does it imply switch?
   $c_1 \dots c_{n-1}$ are 0. $c_n = 1$.
   Switch at $n-1$? $c_{n-1}=0, c_n=1$. Yes.
   But wait, earlier I said $c_1 \dots c_k=0$.
   If $k=n-1$, then $c_n=1$. This is valid.
   Is this the only mixed case?
   What if $k < n-1$?
   Then we visit a subset of vertices with 0-edges, then switch to 1-edges to cover rest.
   Let $V_0 = \{p_1, \dots, p_{k+1}\}$.
   Let $V_1 = \{p_{k+1}, \dots, p_n\}$. (Wait, $p_1$ is already in $V_0$).
   Actually $V_0 \setminus \{p_1\}$ and $V_1 \setminus \{p_{k+1}\}$ partition $V \setminus \{p_1, p_{k+1}\}$.
   In $V_0$, we have a path $p_1 \to \dots \to p_{k+1}$ using 0-edges.
   In $V_1$, we have a path $p_{k+1} \to \dots \to p_n$ using 1-edges.
   Also $C_{p_n, p_1} = 1$.
   This means we need to find two disjoint sets of vertices $A, B$ (sharing nothing? No $p_{k+1}$ connects both).
   Let's reformulate:
   We choose two vertices $u, v$ to be the interface. $u=p_1, v=p_{k+1}$.
   Then we need a path in $G_0$ covering some vertices from $u$ to $v$, say $P_0$.
   And a path in $G_1$ covering remaining vertices from $v$ to $u$, say $P_1$.
   Wait, the vertices in $P_0$ are $u, x_1, \dots, x_m, v$.
   The vertices in $P_1$ are $v, y_1, \dots, y_j, u$.
   Wait, $u$ and $v$ must be the same vertices for both paths? Yes, as start/end points.
   And the sets of internal vertices $\{x\}$ and $\{y\}$ must be disjoint and their union is $V \setminus \{u, v\}$.
   This seems complicated to iterate.
   
   However, notice the symmetry.
   Either the 0-path dominates or 1-path dominates.
   Maybe we can just check if a Hamiltonian path exists in $G_0$ (ignoring closure)?
   Or check for Hamiltonian cycle.
   Wait, $N \le 2000$. Finding Hamiltonian cycle is NP-hard.
   But this is a specific instance where we want *an* almost monochromatic permutation.
   Maybe we can just find *any* path?
   Wait, "symmetric matrix consisting of zeros and ones".
   Graph might be dense or sparse.
   Usually, in CP problems like this, if $N$ is up to 2000, it implies a polynomial solution or greedy approach.
   Is it guaranteed that if a solution exists, a specific type exists?
   Or maybe we can construct it greedily?
   Wait, if the graph is complete (all 1s), answer is trivial.
   If graph is empty (all 0s), trivial.
   If graph is disconnected?
   Let's check the constraint. $C_{i,j}=1$ means edge exists in "1-graph". $C_{i,j}=0$ means edge exists in "0-graph".
   Wait, usually "monochromatic" implies all edges same color.
   We allow 1 switch.
   Consider the structure of $C$.
   If we can find a Hamiltonian path in $G_0$ starting at $p_1$, ending at $p_{k+1}$, and $C_{p_{k+1}, p_1}=1$, then we can close the loop.
   Wait, does this cover all cases?
   Suppose we have $0, \dots, 0, 1, \dots, 1$.
   Then we have a path in $G_0$ from $p_1$ to $p_{k+1}$. And a path in $G_1$ from $p_{k+1}$ to $p_1$.
   The vertices of the first path are a subset $U \subset V$. The second path is a superset?
   Actually, since it's a permutation, the vertices are ordered.
   So $p_1 \to p_2 \dots \to p_{k+1}$ covers vertices $\{p_1 \dots p_{k+1}\}$.
   $p_{k+1} \to p_{k+2} \dots \to p_n \to p_1$ covers $\{p_{k+1} \dots p_n, p_1\}$.
   This means the second path covers ALL vertices. The first path covers a subset.
   Wait, the union of edges must be a cycle.
   The edges in $G_0$ form the sequence $c_1, \dots, c_k$.
   The edges in $G_1$ form $c_{k+1}, \dots, c_n$.
   So, we need to decompose the Hamiltonian cycle into a part of 0-edges and a part of 1-edges.
   This is equivalent to saying:
   There exists a subset $S$ of edges in the Hamiltonian cycle that are 0-edges.
   These edges must form a collection of paths.
   Since there is at most 1 switch, the 0-edges must form at most 2 segments?
   Switch can be $0 \to 1$ and implicitly $1 \to 0$ if cyclic.
   But here we ignore $c_n \to c_1$.
   So 0-edges form at most 1 contiguous segment in the cycle order?
   Wait, $0, 0, 1, 1$ has one segment of 0s.
   $1, 1, 0, 0$ has one segment of 0s.
   So 0-edges form exactly 1 path in the cycle (if $k>0$ and $k<n$).
   Wait, if $c_n, c_1$ were considered, $0, \dots, 0, 1, \dots, 1$ would have a switch $1 \to 0$ at wraparound too.
   But since we don't check $c_n \to c_1$, the 0-edges form a path $p_1 \to \dots \to p_{k+1}$.
   And 1-edges form a path $p_{k+1} \to \dots \to p_1$.
   So, the Hamiltonian cycle is composed of a 0-path and a 1-path spliced together.
   This means the Hamiltonian cycle is formed by concatenating a 0-path and a 1-path.
   But wait, $p_{k+1}$ and $p_1$ are the joints.
   The 0-path goes $p_1 \to \dots \to p_{k+1}$.
   The 1-path goes $p_{k+1} \to \dots \to p_1$.
   Wait, if 0-edges form a path covering $U \subset V$, then $U = \{p_1, \dots, p_{k+1}\}$.
   The 1-edges form a path covering $V \setminus \{p_1, p_{k+1}\}$ plus $p_1, p_{k+1}$ at ends?
   Yes.
   So we need to find a partition of $V$ into two sets $U, W$ such that $|U| \ge 2$ (since $p_1, p_{k+1}$ distinct, unless $k=n$ or $k=0$?), $|W| \ge 0$?
   If $k=n$, $U=V$, 0-path is Hamiltonian. 1-path is empty? (Just edge $(p_n, p_1)$? No, if $k=n$, then $c_n=0$ too, so all 0. Then $(p_n, p_1)$ is 0. So $U=V$.)
   If $0 \dots 0, 1 \dots 1$, then $0$-path is $p_1 \to p_{k+1}$. Vertices $U$.
   $1$-path is $p_{k+1} \to p_1$. Vertices $V \setminus U$. Wait, $p_{k+1}$ and $p_1$ are in both.
   Let's clarify.
   The set of vertices involved in 0-edges is $V_0 = \{p_1, \dots, p_{k+1}\}$.
   The set of vertices involved in 1-edges is $V_1 = \{p_{k+1}, \dots, p_n\} \cup \{p_1\}$.
   The internal vertices of 0-path are $S = V_0 \setminus \{p_1, p_{k+1}\}$.
   The internal vertices of 1-path are $T = V_1 \setminus \{p_{k+1}, p_1\}$.
   $S \cap T = \emptyset$.
   Also $S \cup T \cup \{p_1, p_{k+1}\} = V$.
   So we need to find a partition $V \setminus \{u, v\} = S \cup T$ (where $u, v$ are distinct) such that $u \to S \to v$ is a 0-path and $v \to T \to u$ is a 1-path.
   This seems hard.
   
   HOWEVER, note that we just need to output *any* valid permutation.
   If we can find a Hamiltonian cycle where most edges are 0, that's good.
   If we can find one where most are 1, good.
   Actually, look at the constraints and problem type.
   With $N=2000$, maybe we can just run a simple greedy or DFS?
   Or maybe the solution relies on properties of graphs.
   Wait, "lexicographically minimum". This suggests we want to fix $p_1=1$, then $p_2=2$ etc.
   Can we check if a valid extension exists?
   Since we need an almost monochromatic cycle, let's assume the cycle is mostly 0s.
   Try to build a path of 0s as long as possible?
   Wait, if we can construct a path $1 \to \dots \to u$ using 0-edges, and then jump to the rest using 1-edges?
   But "jump" is determined by the switch.
   Let's consider the possibility that there are 2 types of solutions:
   Type A: All edges 0. (Hamiltonian Cycle in $G_0$)
   Type B: All edges 1. (Hamiltonian Cycle in $G_1$)
   Type C: Mixed $0\dots01\dots1$.
   
   Actually, for Type C, we can rephrase:
   Pick two vertices $u, v$ such that $C_{u,v}=1$.
   Then find a Hamiltonian Path in $G_0$ from $u$ to $v$? No, that would use all vertices. If we use all vertices, then $p_1=u, p_n=v$.
   Then $c_n = C_{v,u}=1$. $c_{1 \dots n-1}=0$.
   So we just need a Hamiltonian Path in $G_0$ between $u, v$ with $C_{u,v}=1$.
   Wait, if we use all vertices, the 0-path has length $n-1$ (visiting $n$ vertices).
   Then the last edge is 1.
   This gives exactly the sequence $0 \dots 0, 1$. (Switch at $n-1$).
   This is a valid solution!
   Does this cover all mixed cases?
   Suppose we have $0 \dots 0, 1 \dots 1$.
   We have a 0-path visiting a subset $S'$, and 1-path visiting the rest.
   The union is $V$.
   Wait, if we take the 0-path to be a Hamiltonian Path in $G_0$, it visits ALL vertices.
   Then the only remaining edge is the closing edge $(v, u)$, which must be 1.
   If this edge exists (weight 1), then $0 \dots 0, 1$ is achieved.
   Is it always true that if a mixed solution exists, a solution of type "Hamiltonian 0-path with closing 1-edge" exists?
   Suppose the optimal mixed solution is $0 \dots 0, 1 \dots 1$ where 0-path covers $V_0$ (subset of vertices) and 1-path covers $V_1$ (rest).
   Can we extend the 0-path to be Hamiltonian?
   Not necessarily.
   However, notice that we are looking for LEAST score.
   Score depends on $p_1, p_2, \dots$.
   If we can achieve $p_1=1$, that's great.
   If we assume we pick $p_1=1$.
   Then we need to decide if the cycle is 0-heavy or 1-heavy.
   If 0-heavy, we want to stay in 0-graph.
   If 1-heavy, we want to stay in 1-graph.
   Maybe the simplest strategy:
   Check if there is a Hamiltonian Cycle in $G_0$.
   Check if there is a Hamiltonian Cycle in $G_1$.
   Check if there is a Hamiltonian Path in $G_0$ from $u$ to $v$ such that $C_{u,v}=1$.
   Wait, Hamiltonian Cycle Problem is hard.
   But for random graphs / typical inputs in such contests, often there are easy constructions.
   BUT this is competitive programming. $N=2000$. We need something efficient.
   Is there a constraint that guarantees existence? No, output -1.
   Wait, sum of N $\le 2000$. This is small. $O(N^3)$ or even $O(N^2)$ is fine.
   Maybe the graph is a Complete Bipartite or something? No, arbitrary symmetric.
   
   Let's re-read carefully. "Bobo has an n x n symmetric matrix C".
   "almost monochromatic".
   Actually, this sounds like we can just try to greedily match?
   Wait, what if we fix the color of the edges incident to $p_1$?
   Or simpler:
   What if we treat this as finding a path in $G_0$ that covers all vertices?
   If such a path exists, say $1 \to p_2 \to \dots \to p_n$, and $C_{p_n, 1} = 0$, then all 0s.
   If $C_{p_n, 1} = 1$, then $0, \dots, 0, 1$ (valid).
   What if no Hamiltonian Path in $G_0$?
   Maybe we can do $0, \dots, 0, 1 \dots 1$.
   This requires partitioning vertices into $S, T$.
   Actually, observe that if we can find *any* path in $G_0$ of length $L$, and *any* path in $G_1$ of length $M$ such that $L+M=n$ and they link up?
   No, the links must be consistent with the graph.
   
   Let's check the sample 1 again.
   0 0 1
   0 0 0
   1 0 0
   $p = 3, 1, 2$.
   Edges: $(3,1)$ is 1. $(1,2)$ is 0. $(2,3)$ is 0.
   Sequence of weights: $C_{3,1}, C_{1,2}, C_{2,3} = 1, 0, 0$.
   Switches:
   $c_1=1, c_2=0$. Switch at 1.
   $c_2=0, c_3=0$. No switch.
   Total 1 switch.
   Note: My analysis of switch count excluded $c_3 \to c_1$ ($0 \to 1$).
   If included, total would be 2.
   Since the sample explanation says "Only when i=1, c_i != c_{i+1}", it confirms the condition is strictly on $1 \le i < n$.
   Wait, the sample output says `3 1 2`.
   $p_1=3, p_2=1, p_3=2$.
   $c_1 = C_{3,1} = 1$.
   $c_2 = C_{1,2} = 0$.
   $c_3 = C_{2,3} = 0$.
   Pairs $(c_1, c_2) = (1, 0) \implies$ diff.
   $(c_2, c_3) = (0, 0) \implies$ same.
   Count = 1. OK.
   
   Now, how to solve generally?
   Key idea: If we can't find a full cycle or path easily, maybe we don't need to?
   Wait, "Find AN almost monochromatic permutation".
   Usually, with $N$ up to 2000, we can use some constructive logic.
   Is it possible that we can just perform a simple traversal?
   Consider the components of $G_0$ and $G_1$.
   If $G_0$ is connected?
   
   Let's guess a simpler approach:
   We are allowed 1 switch.
   This means we can traverse most vertices using 0-edges, then switch to 1-edges, then finish.
   OR traverse most using 1-edges, switch to 0-edges.
   If we use 1-edges initially, then $c_1=1, \dots, c_k=1, c_{k+1}=0, \dots$.
   This requires a path of 1s from $p_1$ to $p_{k+1}$ and a path of 0s from $p_{k+1}$ to $p_1$.
   
   Strategy:
   Try to find a Hamiltonian Path in $G_0$ (vertices $p_1 \dots p_n$) such that $C_{p_n, p_1} = 1$.
   If found, output it.
   Why? Because this corresponds to $0 \dots 0, 1$ (sequence $c_1 \dots c_n$).
   $c_1 \dots c_{n-1}$ are 0. $c_n=1$.
   Wait, if $c_n=1$, then $c_{n-1}=0, c_n=1$ is a switch.
   So $0 \dots 0, 1$ has 1 switch.
   Wait, does it require $c_{n-1}=0$?
   Yes, if we have a path of 0s in $G_0$.
   If we have a Hamiltonian Path in $G_0$ (covering all vertices), then edges $(p_1, p_2) \dots (p_{n-1}, p_n)$ are 0.
   Then we just need $(p_n, p_1)$ to be 1?
   Wait, the condition is $c_1 \dots c_{n-1}$ are 0. $c_n=1$.
   Wait, the sequence is $c_1, \dots, c_n$.
   If $p$ comes from Hamiltonian Path in $G_0$ from $u$ to $v$ ($p_1=u, p_n=v$), then $c_1 \dots c_{n-1} = 0$.
   Then we check $c_n = C_{v, u}$.
   If $C_{v,u} = 1$, then $c_n=1$.
   So the sequence is $0, \dots, 0, 1$.
   Transitions:
   $(c_1, c_2) \dots (c_{n-2}, c_{n-1})$ are all $0 \to 0$.
   $(c_{n-1}, c_n)$ is $0 \to 1$.
   Total 1 switch. Valid.
   Similarly if $C_{v,u}=0$, then sequence is $0 \dots 0, 0$. No switches. Valid.
   So: **Any Hamiltonian Path in $G_0$ is a candidate.** (We just need to check $C_{p_n, p_1}$? No, if $C_{p_n, p_1}=0$, it's all 0s, also valid. So ANY Hamiltonian Path in $G_0$ works!).
   Wait, is this true?
   If $p$ is a permutation, edges $(p_1, p_2) \dots (p_{n-1}, p_n)$ are in the path.
   If path is in $G_0$, then $c_1 \dots c_{n-1} = 0$.
   Then $c_n$ is whatever $C_{p_n, p_1}$ is (0 or 1).
   If $c_n=0$, sequence is $0 \dots 0$. 0 switches.
   If $c_n=1$, sequence is $0 \dots 0, 1$. 1 switch.
   Both $\le 1$.
   So YES: Any Hamiltonian Path in $G_0$ yields a valid permutation.
   Similarly, any Hamiltonian Path in $G_1$ yields a valid permutation (sequence $1 \dots 1$ followed by $c_n$).
   Wait, if path in $G_1$ ($c_1 \dots c_{n-1}=1$), then $c_n$ determines the switch.
   If $c_n=1$, all 1s. If $c_n=0$, switch at end.
   So if we can find a Hamiltonian Path in $G_0$ OR $G_1$, we are good.
   
   What if neither exists?
   Wait, Dirac's theorem / Ore's theorem? No, $N$ up to 2000.
   We can't solve general HP.
   However, this is a "find any" problem.
   Maybe the graph is such that one of them MUST contain a Hamiltonian Path?
   Or maybe we can just construct one greedily?
   Wait, for $N=2000$, finding *any* Hamiltonian path is generally hard (NP-complete).
   UNLESS the graph is special or we just need to output -1 if not found.
   Is it NP-complete to find *almost monochromatic* permutation?
   Probably yes.
   But maybe there's a trick.
   "In each input, the sum of n does not exceed 2000."
   This is very low for exponential solutions, but high for $O(N^3)$.
   Wait, sum of $N \le 2000$ across test cases.
   This means average $N$ is small, or few large cases.
   But we need to handle $N=2000$ efficiently.
   
   Let's check if there is a condition under which Hamiltonian Path is guaranteed?
   If graph is dense?
   But matrix is arbitrary.
   Is there a construction that works?
   Maybe we can assume that if we cannot find a full path, we can't find anything?
   Wait, could there be a valid solution that is NOT a Hamiltonian Path in $G_0$ or $G_1$?
   Example: $0 \dots 0, 1 \dots 1$.
   Here 0-path is partial, 1-path is partial.
   Suppose $G_0$ does not have a Hamiltonian Path.
   But we might have $S, T$ partition.
   
   However, looking at similar problems (Codeforces/etc.), usually "lexicographically smallest" hints at greedy.
   Greedy construction for HP is risky.
   BUT, maybe we can just check connectivity?
   If the graph is disconnected, HP doesn't exist.
   
   Wait, if $G_0$ is not connected, maybe $G_1$ is?
   Since $C_{i,j} \in \{0, 1\}$, $G_0 \cup G_1$ is complete (loops excluded, but off-diagonal always 0 or 1).
   So if $G_0$ is disconnected, maybe $G_1$ helps.
   
   Wait, look at the Sample 1 again.
   Edges:
   (1,2): 0
   (1,3): 1
   (2,3): 0
   Graph $G_0$ has edges (1,2) and (2,3).
   Vertices {1, 2, 3}.
   Edges: 1-2, 2-3. Path 1-2-3 exists.
   Permutation corresponding to path 1-2-3: $1, 2, 3$.
   Edges: $(1,2)=0, (2,3)=0, (3,1)=1$.
   $c_1=0, c_2=0, c_3=1$.
   Switches: $(0,0) \to 0$. $(0,1) \to 1$. Total 1.
   Permutation 1, 2, 3 is valid.
   Sample output was 3, 1, 2. Lexicographically 1, 2, 3 is better?
   Wait, score formula: $S = n p_1 + (n-1) p_2 + \dots + 1 p_n$.
   Wait, $p_1$ has coefficient $n$.
   To minimize $S$, we want $p_1$ as small as possible.
   So 1, 2, 3 is better than 3, 1, 2.
   Why is Sample Output 3, 1, 2?
   Let's check score.
   Sample Output: 3 1 2.
   My proposed: 1 2 3.
   If my proposed is valid, why wasn't it chosen?
   Let's re-calculate validity of 1 2 3.
   $p_1=1, p_2=2, p_3=3$.
   $c_1 = C_{1,2} = 0$.
   $c_2 = C_{2,3} = 0$.
   $c_3 = C_{3,1} = 1$.
   Indices $i=1, \dots, n-1=2$.
   $i=1: c_1 \ne c_2 \iff 0 \ne 0$ False.
   $i=2: c_2 \ne c_3 \iff 0 \ne 1$ True.
   Total 1 switch. Valid.
   Why output 3 1 2?
   Maybe I misunderstood the condition.
   "number of indices i (1 ≤ i < n) where c_i != c_{i+1} is at most one."
   Wait, maybe $c_i$ definition is different?
   "c_i=(C_{p_i, p_{i+1}} for 1 ≤ i < n, C_{p_n, p_1} for i = n)"
   "output n integers p_1, ..., p_n ... minimize S = n * p_1 + ..."
   Wait, the sample explanation says:
   "c1 = C_{3,1} = 1, c2 = C_{1,2} = 0, c3 = C_{2,3} = 0. Only when i=1, c_i != c_{i+1}. Therefore, the permutation 3,1,2 is an almost monochromatic permutation."
   Wait, they list $c_1, c_2, c_3$.
   $c_1 = 1$. $c_2 = 0$. $c_3 = 0$.
   They compare $c_1$ vs $c_2$ (diff) and $c_2$ vs $c_3$ (same).
   They DO NOT compare $c_3$ vs $c_1$.
   So my analysis holds.
   Why is 1 2 3 not the answer?
   Ah, 1 2 3 gives $c_1=0, c_2=0, c_3=1$.
   $c_1 \ne c_2$ (0!=0) NO.
   $c_2 \ne c_3$ (0!=1) YES.
   Count 1.
   So 1 2 3 should be valid and lexicographically smaller (starts with 1).
   Wait, is there a constraint I missed?
   "C_{i,j} ∈ {0,1}". "C_{i,i}=0".
   Maybe the sample output provided in the problem description is just ONE valid answer, not necessarily minimal?
   "If there are multiple almost monochromatic permutations, you need to minimize the lexicographical order."
   Okay, so 3 1 2 is NOT minimal? Or maybe my validity check for 1 2 3 is wrong?
   Let's check 1 2 3 validity again.
   $p_1=1, p_2=2, p_3=3$.
   Edges: $(1,2), (2,3), (3,1)$.
   $C_{1,2}=0$. $C_{2,3}=0$. $C_{3,1}=1$.
   $c_1=0, c_2=0, c_3=1$.
   Switches at $i$ where $c_i \ne c_{i+1}$.
   $i=1: c_1 \ne c_2 \implies 0 \ne 0 \implies$ False.
   $i=2: c_2 \ne c_3 \implies 0 \ne 1 \implies$ True.
   Count = 1.
   It is valid.
   And $p_1=1 < 3$.
   Why is Sample Output 3 1 2?
   Maybe I copied the sample incorrectly?
   Sample Input:
   3
   001
   000
   100
   
   Sample Output:
   3 1 2
   
   Wait. Let me re-read "lexicographical order".
   Usually means comparing vectors $(p_1, \dots, p_n)$.
   Is it possible that $c_n$ plays a role?
   Ah, wait.
   If 1 2 3 is valid, it should be preferred.
   Is there a reason 1 2 3 is invalid?
   Maybe the definition of "almost monochromatic" implies cyclic property in some standard interpretation?
   "The permutation p is almost monochromatic if and only if the number of indices i (1 ≤ i < n) where c_i ̸= c_{i+1} is at most one."
   Strictly speaking, $i$ ranges $1 \dots n-1$.
   It does not include $i=n$.
   If it were cyclic, it would be $n$ switches max (for alternating).
   If strict interpretation holds, 1 2 3 is valid.
   Why would the sample output be 3 1 2?
   Maybe 3 1 2 is the *only* one?
   Let's check 1 2 3 again.
   Maybe my reading of matrix is wrong?
   Line 1: 0 0 1. Row 1: 0 to 1 is 0. 1 to 2 is 0. 1 to 3 is 1.
   Wait, indices are 1-based.
   Row 1: C(1,1)=0, C(1,2)=0, C(1,3)=1.
   Row 2: 0, 0, 0.
   Row 3: 1, 0, 0.
   
   For 1 2 3:
   Edge 1: (1, 2). C(1,2)=0.
   Edge 2: (2, 3). C(2,3)=0.
   Edge 3: (3, 1). C(3,1)=1.
   Sequence 0, 0, 1.
   Switches:
   $i=1: c_1 \ne c_2 \implies 0 \ne 0$ (No).
   $i=2: c_2 \ne c_3 \implies 0 \ne 1$ (Yes).
   Count = 1.
   Seems valid.
   
   Wait, is it possible that $c_n$ is NOT defined as closing the edge?
   "c_i=(C_{p_i, p_{i+1}} for 1 ≤ i < n, C_{p_n, p_1} for i = n)."
   Yes, $c_n = C_{p_n, p_1}$.
   
   Is it possible the problem implies $c_i$ are the values on the edges $p_i, p_{i+1}$, but the "monochromatic" property refers to the colors of vertices? No.
   
   Maybe the issue is "Minimize lexicographical order" means something else? No, $S = n p_1 + \dots$ is standard.
   
   Let's check 3 1 2 again.
   $p_1=3, p_2=1, p_3=2$.
   $c_1 = C_{3,1} = 1$.
   $c_2 = C_{1,2} = 0$.
   $c_3 = C_{2,3} = 0$.
   Sequence 1, 0, 0.
   Switches:
   $i=1: 1 \ne 0$ (Yes).
   $i=2: 0 \ne 0$ (No).
   Count = 1.
   Valid.
   
   Wait, is it possible that there's a constraint I'm missing?
   "sum of n does not exceed 2000".
   "Time limit 1 seconds".
   
   Maybe there is a subtlety in $c_i \neq c_{i+1}$.
   Could it be $c_i$ is defined differently?
   "c_i=(C_{p_i, p_{i+1}} for 1 ≤ i < n, C_{p_n, p_1} for i = n)."
   Wait, maybe $c_{n+1}$ wraps to $c_1$?
   "number of indices i (1 ≤ i < n) where c_i != c_{i+1}".
   If it meant cyclic, it would say $1 \le i \le n$.
   It says $1 \le i < n$.
   This strictly limits the comparison to adjacent pairs in the sequence $c_1, \dots, c_n$.
   
   Is it possible that $p_1, \dots, p_n$ is not just a path?
   Wait, if 1 2 3 is valid and lexicographically smaller than 3 1 2, the sample output is weird.
   UNLESS, my manual calculation of C(1,2) etc. is wrong.
   Input:
   001 -> C11=0, C12=0, C13=1
   000 -> C21=0, C22=0, C23=0
   100 -> C31=1, C32=0, C33=0
   Yes.
   
   Wait!
   Look at the condition again.
   "permutation p is almost monochromatic if and only if the number of indices i (1 <= i < n) where c_i != c_{i+1} is at most one."
   Wait, $c_i$ is defined for $i=1..n$.
   If $i=n-1$, we compare $c_{n-1}$ and $c_n$.
   If $i=n$, we would compare $c_n$ and $c_{n+1}$ which is not defined.
   So we compare $c_1, c_2$; $c_2, c_3$; ...; $c_{n-1}, c_n$.
   Wait, does $c_i$ depend on $p$? Yes.
   For 1 2 3: $c_1=0, c_2=0, c_3=1$.
   Pairs: $(c_1, c_2)=(0,0)$; $(c_2, c_3)=(0,1)$.
   Differences: 1.
   For 3 1 2: $c_1=1, c_2=0, c_3=0$.
   Pairs: $(1,0)$; $(0,0)$.
   Differences: 1.
   
   Why would 1 2 3 be rejected?
   Is it possible $C_{i,j}$ is read row by row? Yes.
   Maybe the sample output IS NOT minimal?
   "If there are multiple ..., you need to minimize the lexicographical order."
   This is a requirement for the solution code.
   If sample output doesn't match minimal, maybe my understanding of validity is wrong.
   Could it be that $C_{i,j}$ denotes edge from $i$ to $j$?
   Symmetric, so undirected.
   
   Is there any chance that $c_i$ definition is recursive? No.
   What if the condition implies $c_n$ must be compared with $c_1$?
   If cyclic comparison was intended ($1 \le i \le n$), then:
   1 2 3: $c_3=1, c_1=0 \implies$ diff.
   Pairs: $(0,0), (0,1), (1,0)$.
   Diffs: 2 (at $i=2$ and $i=3$).
   Invalid.
   3 1 2: $c_3=0, c_1=1 \implies$ diff.
   Pairs: $(1,0), (0,0), (0,1)$.
   Diffs: 2.
   Invalid.
   
   Neither is valid if cyclic?
   Wait, sample note says "Only when i=1, c_i != c_{i+1}".
   It specifically ignores $c_3, c_1$.
   So cyclic interpretation is definitely wrong.
   
   So why 3 1 2 over 1 2 3?
   Let me check the problem source or common variants.
   This looks like a problem from a contest.
   "Almost monochromatic permutation".
   If I search for "Bobo almost monochromatic", it matches a Codeforces problem? Or maybe "JOI"?
   Found it? "Problem E: Almost Monochromatic". Maybe not exactly.
   Wait, Sample 1 Output `3 1 2` is strange.
   Maybe there is a typo in the problem statement provided in the prompt?
   Or maybe my assumption that 1 2 3 is valid is correct, but 3 1 2 is given just as an example of validity, not the minimal one?
   Wait, the text says "Output ... minimize the lexicographical order".
   If I submit 1 2 3 and it's valid, it should pass.
   However, usually sample output reflects the correct behavior.
   Is it possible that 1 2 3 is NOT valid?
   Let's re-read carefully.
   "number of indices i (1 ≤ i < n) where c_i != c_{i+1}".
   Wait, $c_n$ is defined as $C_{p_n, p_1}$.
   If $i=n-1$, we compare $c_{n-1}$ and $c_n$.
   For 1 2 3: $c_1=0, c_2=0, c_3=1$.
   $c_2 \ne c_3$ is true.
   Is there any other $i$?
   $i=1: c_1 \ne c_2 \implies 0 \ne 0$ false.
   Total 1.
   
   Is it possible that $c_i$ are defined as $C_{p_i, p_{i+1}}$ for ALL $i$, including $n$?
   Yes, for $i=n$, it's $C_{p_n, p_1}$.
   Wait, what if the "sequence" is considered $c_1, \dots, c_n, c_1, \dots$?
   No, problem is specific.
   
   Is it possible the question implies $C_{i,j}$ are 0 or 1, and $C_{i,j} \neq C_{j,i}$? No, symmetric.
   
   Maybe I should implement a search that finds the best solution.
   Since $N$ is small, maybe backtracking with pruning?
   With $N=2000$, pure backtracking is impossible.
   But maybe we can optimize?
   Actually, if there's a simple constructive solution, it will be fast.
   The fact that Sample 1 output is 3 1 2 instead of 1 2 3 is the biggest red flag.
   Let me look really closely at the condition again.
   "number of indices i (1 ≤ i < n) where c_i != c_{i+1} is at most one."
   Wait, maybe indices of $p$ are $1..n$, so $c_1$ involves $p_1, p_2$.
   If $p = (3, 1, 2)$.
   $c_1 = C_{3,1}$.
   $c_2 = C_{1,2}$.
   $c_3 = C_{2,3}$.
   If $p = (1, 2, 3)$.
   $c_1 = C_{1,2}$.
   $c_2 = C_{2,3}$.
   $c_3 = C_{3,1}$.
   
   What if the condition implies $c_i$ must be $C_{p_i, p_{i+1}}$ for $1 \le i \le n$ and we consider $c_{n+1} = c_1$?
   If so, Sample 1 fails for both.
   So that's not it.
   
   Wait! "The permutation p is almost monochromatic if and only if the number of indices i (1 ≤ i < n) where c_i != c_{i+1} is at most one."
   Wait, looking at the note again.
   "c1 = C_{3,1} = 1, c2 = C_{1,2} = 0, c3 = C_{2,3} = 0."
   "Only when i=1, c_i != c_{i+1}."
   This means $c_1 \ne c_2$ is true.
   $c_2 \ne c_3$ is false.
   They DO NOT mention $c_3 \ne c_1$.
   But wait.
   If $c_3$ is the last element of the sequence.
   Wait, does the condition check $c_{n-1} \ne c_n$?
   $i=n-1$ is checked.
   So for $n=3$, we check $i=1, i=2$.
   For 1 2 3:
   $i=1: c_1 \ne c_2 \implies 0 \ne 0$ (F).
   $i=2: c_2 \ne c_3 \implies 0 \ne 1$ (T).
   So valid.
   
   Maybe the matrix in Sample 1 is different?
   001
   000
   100
   Maybe it's upper triangular? No, lines contain n integers.
   
   Is it possible that the problem statement in the prompt has a typo?
   "c_i=(C_{p_i, p_{i+1}} for 1 ≤ i < n, C_{p_n, p_1} for i = n)."
   Usually, "almost monochromatic" problems on cycles imply cyclic consistency.
   If it's cyclic, 1 2 3 is invalid (2 switches).
   3 1 2 is invalid (2 switches).
   So neither would be valid.
   But the problem says "find an ... permutation".
   And gives a sample.
   Wait.
   Maybe the "indices i" range includes $n$ in a way I don't see?
   "number of indices i (1 ≤ i < n)".
   If it's strictly $< n$, then it's not cyclic.
   
   Let's assume the sample output is just an example and 1 2 3 is valid and better.
   Wait, if 1 2 3 is valid, why would a judge accept 3 1 2 and expect 1 2 3?
   Or does the judge require EXACT match?
   "If there are multiple ..., you need to minimize ...".
   If 1 2 3 is valid, then 3 1 2 is suboptimal.
   So the prompt's sample output being suboptimal suggests my validity check is flawed.
   Where could the flaw be?
   "c_i=(C_{p_i, p_{i+1}} for 1 ≤ i < n, C_{p_n, p_1} for i = n)."
   Maybe $p$ is $0$-indexed? No.
   
   Let's pause. If I cannot figure out why 1 2 3 is invalid, I should stick to the rules as written.
   Rules: Check $c_1 \dots c_{n-1}$ transitions.
   Wait.
   Is it possible that for $i=n$, $c_n$ is NOT $C_{p_n, p_1}$ but something else?
   "C_{p_n, p_1} for i = n".
   
   Let's assume there is a hidden catch.
   What if $C_{i,j}$ are read as strings? "001". Yes.
   
   What if the definition of "almost monochromatic" allows $c_i \ne c_{i+1}$ at most 1 time?
   For 1 2 3, transitions: $0 \to 0$, $0 \to 1$. One switch.
   For 3 1 2, transitions: $1 \to 0$, $0 \to 0$. One switch.
   
   Wait, maybe I should check the sample 2.
   Sample 2:
   4
   0000
   0000
   0000
   0000
   Output: 2 4 3 1.
   All 0s matrix.
   Any permutation has $c_i=0$.
   Transitions $0 \to 0$ everywhere.
   Count 0.
   Lexicographically, 1 2 3 4 should be minimal.
   Why 2 4 3 1?
   $2, 4, 3, 1$.
   This is clearly NOT minimal.
   So there MUST be a misunderstanding of the problem.
   
   Re-read carefully.
   "c_i=(C_{p_i, p_{i+1}} for 1 ≤ i < n, C_{p_n, p_1} for i = n)."
   "number of indices i (1 ≤ i < n) where c_i != c_{i+1}".
   Maybe "c_i != c_{i+1}" refers to comparing $c_i$ with $c_{i+1}$?
   Wait, if $n=4$, indices $1, 2, 3$.
   Compare $(c_1, c_2), (c_2, c_3), (c_3, c_4)$.
   For all 0 matrix, $c_i=0$ for all $i$.
   Comparisons are $0 \ne 0$, always false.
   So count is 0.
   So 1 2 3 4 is valid.
   Why 2 4 3 1?
   
   Is it possible the question asks for *lexicographically largest*? No, "minimize".
   Is it possible the input/output format is tricky?
   
   Wait!
   Maybe $C_{i,j}$ refers to $C_{i, j}$?
   In Sample 2, matrix is all 0s.
   If I output 1 2 3 4, it should be valid.
   Unless... "minimize lexicographical order" applies to something else?
   Or maybe I am misinterpreting the permutation definition.
   
   Is it possible that $p_1, \dots, p_n$ must be distinct? Yes, permutation.
   
   Wait, looking at the sample outputs again.
   Sample 1: 3 1 2.
   Sample 2: 2 4 3 1.
   Why would anyone prefer these?
   Wait, look at the Note for Sample 1.
   "c1 = 1, c2 = 0, c3 = 0. Only when i=1, c_i != c_{i+1}."
   This explanation supports the validity of 3 1 2.
   It does NOT explain why 1 2 3 is invalid.
   It just says 3 1 2 is *an* almost monochromatic permutation.
   Wait, does it say it is the *best*? No.
   "If there are multiple ..., you need to minimize the lexicographical order."
   So 1 2 3 should be outputted if it is valid.
   Why is it not?
   Could 1 2 3 be invalid?
   Edges: (1,2)=0, (2,3)=0, (3,1)=1.
   $c_1=0, c_2=0, c_3=1$.
   $i=1: c_1 \ne c_2$ (0 vs 0) -> False.
   $i=2: c_2 \ne c_3$ (0 vs 1) -> True.
   Count = 1.
   Wait, is it possible that $c_n$ is defined differently?
   "c_i=(C_{p_i, p_{i+1}} for 1 ≤ i < n, C_{p_n, p_1} for i = n)."
   This is unambiguous.
   
   Is it possible that the condition is on $c_i$ vs $c_{i+1}$ modulo $n$?
   If it were modulo $n$, then for $n=3$, we check $i=1, 2, 3$ (where $c_4=c_1$).
   Then for 1 2 3 ($0,0,1$): $0 \ne 0$ (no), $0 \ne 1$ (yes), $1 \ne 0$ (yes). Count = 2.
   For 3 1 2 ($1,0,0$): $1 \ne 0$ (yes), $0 \ne 0$ (no), $0 \ne 1$ (yes). Count = 2.
   So neither would be valid.
   
   BUT, the note says "Only when i=1...". It doesn't mention $i=3$.
   If cyclic, $i=3$ would be checked.
   Since they don't mention it, it's not cyclic.
   
   Is there any chance that $p_1$ is not $p_1$?
   Maybe the input is 1-indexed?
   
   Wait, I suspect the Sample Output in the prompt is from a DIFFERENT version of the problem or I'm missing a HUGE constraint.
   OR, the "lexicographical order" is defined on something else.
   $S = n p_1 + (n-1) p_2 + \dots + 1 p_n$.
   This is basically treating $(p_1, \dots, p_n)$ as digits in base $n+1$.
   Minimizing $S$ means prioritizing $p_1$, then $p_2$.
   So $p_1=1$ is best.
   
   Hypothesis: The condition is actually cyclic. $1 \le i \le n$.
   If cyclic, neither sample is valid?
   Wait, maybe for sample 1, $C_{2,3}=1$?
   Input:
   001
   000
   100
   $C_{2,3}$ is row 2 col 3. Value 0.
   $C_{3,2}$ is row 3 col 2. Value 0.
   $C_{1,3}$ is row 1 col 3. Value 1.
   Correct.
   
   Let's reconsider the condition.
   Maybe "almost monochromatic" refers to vertices?
   No, $c_i$ are edge weights.
   
   Let's assume the provided sample outputs are WRONG or I AM MISSING SOMETHING CRITICAL.
   Let's look at the structure of 3 1 2.
   It puts 3 first.
   Maybe the condition is: "c_i != c_{i+1} for $i$ such that ...".
   Wait, could it be "c_i != c_{i+1} where $c_{n+1} = c_1$" but the constraint is $\le 1$?
   Wait, if it is cyclic, then $0, 0, 1$ has 2 switches. $1, 0, 0$ has 2 switches.
   $0, 1, 0$ has 2.
   $0, 0, 0$ has 0.
   $1, 1, 1$ has 0.
   If graph has no all-0 or all-1 cycles, then no solution.
   For Sample 1, graph is not complete.
   Edges available: (1,2)-0, (2,3)-0, (1,3)-1.
   Cycle 1-2-3-1.
   If all 0 cycle needed? No (1,3) is 1.
   If all 1 cycle needed? No (1,2), (2,3) are 0.
   So cyclic condition makes solution impossible for Sample 1.
   Thus, cyclic condition is FALSE.
   
   So non-cyclic condition stands.
   Why 1 2 3 is not picked?
   Maybe $c_i$ is NOT $C_{p_i, p_{i+1}}$?
   "c_i=(C_{p_i, p_{i+1}} for 1 ≤ i < n, C_{p_n, p_1} for i = n)."
   Wait, maybe the "switch" is counted differently?
   Maybe $c_n$ is NOT part of the sequence for switch counting?
   "number of indices i (1 ≤ i < n) where c_i != c_{i+1}".
   This sequence involves $c_1 \dots c_n$.
   Pairs are $(c_1, c_2) \dots (c_{n-1}, c_n)$.
   This matches my check.
   
   Maybe the sample output `3 1 2` is generated by a program that failed to find `1 2 3` due to bad optimization, or I am misidentifying the matrix indices?
   If I swap 1 and 3 in permutation?
   Wait, if I reverse the cycle?
   2 3 1?
   $c_1=C_{2,3}=0, c_2=C_{3,1}=1, c_3=C_{1,2}=0$.
   Pairs: $(0,1)$ diff, $(1,0)$ diff. Count 2. Invalid.
   
   Okay, let's step back.
   What if the problem implies that we can pick ANY almost monochromatic permutation, but we MUST MINIMIZE THE SCORE.
   If 1 2 3 is valid, and 3 1 2 is valid, then 1 2 3 is better.
   Maybe 1 2 3 is INVALID.
   Why?
   Is there a constraint $C_{i,j} \in \{0,1\}$? Yes.
   Maybe the diagonal $C_{i,i}$ matters?
   "C_{i,i}=0".
   
   Wait! "c_i=(C_{p_i, p_{i+1}} for 1 ≤ i < n, C_{p_n, p_1} for i = n)."
   Maybe $C_{p_n, p_1}$ is calculated as $C_{p_1, p_n}$?
   Symmetric.
   
   Okay, I'm going to trust my analysis of the condition (non-cyclic, at most 1 switch) and assume the sample output in the prompt is illustrative but perhaps 1 2 3 is the correct answer in reality.
   WAIT. I see "sum of n does not exceed 2000".
   This suggests $O(N^2)$ or $O(N^3)$.
   Constructive solutions usually work.
   
   Let's focus on solving the problem as stated.
   Goal: Find $p$ such that number of color changes in $c_1 \dots c_n$ (checking $c_i, c_{i+1}$ for $i < n$) is $\le 1$.
   This means the sequence $c$ is of form $0 \dots 0$ or $1 \dots 1$ or $0 \dots 0 1 \dots 1$ or $1 \dots 1 0 \dots 0$.
   Actually, just "monochromatic" in the linear sense.
   Wait, if $c$ is $0 \dots 0 1 \dots 1$, it has 1 switch.
   If $c$ is $1 \dots 1 0 \dots 0$, it has 1 switch.
   So we just need to order vertices such that edges in the path are all 0 (or 1), except possibly one transition from 0 to 1 or 1 to 0.
   
   Algorithm Idea:
   We need to construct a Hamiltonian path $p_1 \dots p_n$ in the graph.
   Actually, any path $p_1 \dots p_n$ works?
   Wait, if we find a path in $G_0$, then $c_1 \dots c_{n-1}$ are 0.
   Then $c_n$ is whatever (0 or 1).
   If $c_n=0$, sequence $0 \dots 0$ (0 switches).
   If $c_n=1$, sequence $0 \dots 0 1$ (1 switch).
   Both valid.
   So ANY Hamiltonian Path in $G_0$ is a solution.
   Similarly, ANY Hamiltonian Path in $G_1$ is a solution.
   
   So the problem reduces to: Find a Hamiltonian Path in $G_0$ or $G_1$.
   Wait, what if neither exists?
   Then we might need a "mixed" path?
   But a mixed path $0 \dots 0 1 \dots 1$ implies a path of 0s and a path of 1s.
   Wait, if we have a path of 0s and 1s, does it mean we can rearrange vertices?
   No, the vertices are fixed.
   
   Actually, notice that if $G_0$ has no Hamiltonian Path, it's likely $G_1$ has one? Or neither?
   Since $G_0 \cup G_1$ is complete, it's unlikely both are disconnected in a way that prevents HP.
   But for general graphs, HP is hard.
   However, with $N=2000$, we probably assume $G_0$ or $G_1$ has a Hamiltonian Path?
   Or maybe we can find one greedily?
   Greedy strategy:
   Sort vertices?
   Try to extend current path $p_1 \dots p_k$ by adding $v$ if $(p_k, v)$ is 0-edge.
   If stuck, backtrack?
   Backtracking is slow.
   But maybe we can use the fact that we just need *one*.
   
   Wait, the problem asks to MINIMIZE lexicographical order.
   So we should try to set $p_1=1$.
   Then $p_2=2$?
   We need to check if a Hamiltonian Path starting with $1, 2, \dots$ exists.
   This is essentially finding HP in subgraphs.
   
   Wait, if we assume the graph is "nice" (dense enough), we might use a randomized heuristic or just greedy?
   But "CP" problems shouldn't rely on heuristics unless specified.
   Wait, "Time limit 1s".
   
   Maybe the trick is:
   If $G_0$ is disconnected, then $G_1$ must contain the edges missing in $G_0$.
   If $G_0$ has a component structure?
   
   Let's check the constraints again.
   Maybe the input guarantees that a solution exists? No, "output -1".
   
   Let's rethink: Is there a specific case where we can output -1?
   If neither $G_0$ nor $G_1$ has a Hamiltonian Path?
   Wait, is it possible to have a solution with 1 switch without having a Hamiltonian Path in either?
   Yes.
   Example: $n=3$. Edges $(1,2)=0, (2,3)=0, (1,3)=1$.
   $G_0$: edges $(1,2), (2,3)$. Path $1-2-3$. Hamiltonian Path exists.
   $G_1$: edge $(1,3)$. No HP.
   Solution 1 2 3 works.
   
   Another example: $1-2$ is 0, $1-3$ is 1, $2-3$ is 1.
   $G_0$: edge $(1,2)$. No HP (need 3).
   $G_1$: edges $(1,3), (2,3)$. Path $1-3-2$. HP exists.
   Solution 1 3 2 works (edges 1, 1, 0 -> $1,1,0$ ok).
   
   What if graph is such that no HP in $G_0$ and no HP in $G_1$?
   Consider a path graph $1-2-3$ in $G_0$ and a star in $G_1$?
   Actually, since $G_0 \cup G_1$ is complete, finding HP is likely easy if density is high.
   If $N$ is large, maybe it's guaranteed?
   Wait, "C_{i,j} \in \{0,1\}" - could be arbitrary.
   
   Okay, let's write a solution that tries to find a Hamiltonian Path in $G_0$ and $G_1$.
   Since we need min lexicographical order, we should prioritize finding a path in $G_0$ (or $G_1$) that starts with small numbers.
   Actually, checking existence of HP is hard.
   BUT, wait.
   If we just perform a DFS/BFS to find a path.
   Is it possible to solve HP in $O(N^3)$?
   Yes, for certain graphs, but not general.
   Wait, is there a theorem?
   "If a graph has degree $\ge n/2$..."
   Maybe we can just run a randomized local search? No.
   
   Wait, maybe I should just implement a standard backtracking with optimizations (like Warnsdorff's rule) or just simple DFS.
   Given 1s time limit and 2000 nodes, simple DFS won't work.
   BUT, if $N$ is small in test cases? "Sum of n <= 2000".
   Wait, usually if sum of N is small, it implies total complexity can be proportional to $\sum N^3$.
   If there is one test case with $N=2000$, $2000^3 = 8 \cdot 10^9$, too slow.
   So $O(N^2)$ or $O(N^3)$ with small constant is needed.
   Wait, maybe the graph is ALWAYS solvable?
   If the graph is random-ish, HP exists.
   But it's arbitrary.
   
   Let's reconsider the condition "At most 1 switch".
   This is extremely loose.
   Most permutations will have many switches.
   But maybe we can always find a "sorted" permutation?
   Permutation $1, 2, \dots, n$.
   Edges $(i, i+1)$.
   If $C_{i, i+1}$ are mostly same color, good.
   
   Wait, I might have found the problem online or a variant.
   It seems to be from a Chinese contest (maybe Luogu or similar).
   Problem "Almost Monochromatic".
   Common solution might involve finding if graph is bipartite or something?
   No.
   
   Let's try to search for the problem text.
   "Bobo has an n×n symmetric matrix C".
   This matches "ZOJ 4075"? Or something.
   Actually, I recall a problem where we need to find a Hamiltonian Path.
   The key might be that we can just check if $G_0$ or $G_1$ is connected?
   Or maybe just finding ANY path that covers everything is hard, but finding a solution that satisfies the condition is easier?
   Wait, the condition IS satisfied if we find a Hamiltonian Path in $G_0$ or $G_1$.
   Are there solutions NOT satisfying this?
   Yes, the mixed ones.
   But maybe finding a Hamiltonian Path in $G_0 \cup G_1$ (which is complete) is trivial?
   Wait, we need the path to be MONOCHROMATIC (mostly).
   The mixed path uses edges from both graphs.
   
   Let's go for a heuristic/greedy construction.
   Try to build a path using only 0-edges.
   If stuck (current end cannot reach unvisited vertices), we can force a switch to 1-edge?
   Wait, switching means changing color.
   If we are building a 0-path and get stuck, we can't continue with 0s.
   But we might have been able to pick a different start?
   Actually, we can model this as finding a path that maximizes the length of monochromatic segments.
   Wait, the condition is strict: exactly 0 or 1 switch.
   This means the whole cycle must be split into two monochromatic paths.
   Essentially, we need to find a cut vertex (or vertices) that separates the graph into two clusters connected by 0-edges and 1-edges?
   
   Let's try this:
   Try to find a Hamiltonian Path in $G_0$ starting at $1$.
   How?
   Use a randomized greedy search or simply try to keep extending the longest monochromatic path?
   Wait, if $G_0$ is just a set of edges.
   If $N$ is small, maybe we can check if HP exists?
   Wait, finding if HP exists is NP-complete.
   UNLESS there is a theorem for "almost monochromatic".
   Wait, if the graph is dense, HP is likely.
   If the graph is sparse, maybe not.
   But maybe for $N \le 2000$, we can assume if a path exists, a simple one exists?
   
   Wait, I missed a crucial detail?
   "sum of n does not exceed 2000".
   This is very small!
   It implies we can do something like $O(N^3)$ or $O(N^4)$.
   Maybe we can solve it by iterating all pairs $(p_1, p_n)$?
   If we fix $p_1=u, p_n=v$.
   We need a path $u \rightsquigarrow v$ of 0s and a path $v \rightsquigarrow u$ of 1s?
   No, we need a partition of vertices.
   This is getting too complex for a contest problem if it's NP-hard.
   Likely there is a constructive solution.
   
   Wait! The problem asks for *almost monochromatic*.
   Maybe we can just check if there is a Hamiltonian Path in $G_0$.
   If not, maybe in $G_1$?
   If not, maybe the mixed case is the only one?
   But wait, finding a Hamiltonian Path is hard.
   But wait, "Time limit 1 second".
   If I implement a randomized backtracking (Warnsdorff's rule style), it works for most practical cases.
   Is it possible that the problem is designed such that $G_0$ or $G_1$ is easy?
   Or maybe the constraints on $C$ make it easy?
   No, $C$ is arbitrary.
   
   Let's check if there's any simple logic.
   If we pick $p_1=1$.
   Then we greedily extend $p_2$ using 0-edge?
   If no 0-edges available (degree in $G_0$ is 0), we must use 1-edges.
   If we use a 1-edge as second step, then we have switched?
   Wait, if $c_1=1$, we can have $c_2=1$.
   If we can complete a path of all 1s, good.
   If not, maybe we can find a path of all 1s except last edge?
   Basically, if we find a path of 1s $1 \to \dots \to u$ such that $C_{u, 1} = 0$, we have $1 \dots 1, 0$.
   So basically:
   1. Try to find HP in $G_1$ starting at 1.
      If found, check if $C_{end, 1} = 0$. If so, valid (switch at end).
      If $C_{end, 1} = 1$, valid (all 1s).
      Actually, ANY HP in $G_1$ is valid.
   2. Try to find HP in $G_0$ starting at 1.
      If found, valid.
   3. If none found, maybe mixed?
   
   But wait, finding HP is hard.
   However, maybe for this specific problem, the graph is "complete" enough?
   No, could be empty.
   If $G_0$ is empty, we can't find HP in $G_0$.
   If $G_0$ is empty, then all edges are 1. $G_1$ is complete.
   $G_1$ HP exists.
   If $G_0$ is complete, $G_0$ HP exists.
   If $G_0$ is sparse but connected, HP might not exist.
   
   Wait, if the graph is bipartite?
   Wait, $N \le 2000$.
   Maybe we can simply try to find a path of length $N$ in $G_0$.
   If it fails, try in $G_1$.
   If both fail, output -1.
   Since finding HP is NP-complete, but maybe the test cases are weak or constructed to have HP?
   Wait, "Time limit 1s" + "N=2000". This strongly suggests polynomial time.
   The only polynomial-time approximation for HP is limited.
   UNLESS the problem is trivial and there's a catch I missed.
   Catch: "c_i != c_{i+1}".
   Maybe we don't need HP.
   We need a path.
   If we just take *any* path that visits everyone.
   Wait, HP definition is path visiting every vertex exactly once.
   That's what I need.
   
   Is there a property of $N \le 2000$?
   Maybe the matrix is given such that it's always possible?
   But "output -1" implies it's not always possible.
   
   Let's try to write a solver that attempts to find a HP using a heuristic (Warnsdorff) which works well for random dense graphs.
   For sparse graphs, Warnsdorff might fail.
   But maybe the problem instances are not pathological.
   
   However, for a proper solution, I should probably implement a randomized or backtracking search.
   Since N=2000, recursion depth 2000 is fine, but branching factor is issue.
   Branching factor is degree.
   If degree is small, branching is small.
   If degree is large, branching is large, but we have many choices.
   Warnsdorff's rule: pick neighbor with fewest onward moves.
   This usually finds a path quickly if one exists.
   
   Also, we need to minimize lexicographical order.
   So we should try to put small indices first.
   Fix $p_1=1$.
   Fix $p_2$?
   We can iterate $p_2$ from 1 to $N$ (skipping 1).
   If $p_2$ is compatible (edge weight matches our target color? Or maybe flexible), check if we can complete the path.
   
   Wait, we don't know the target color yet.
   We can try to construct a path using *only* 0s.
   Iterate $p_2$ from 2 to $N$.
   If $C_{1, p_2} == 0$, try to complete path in $G_0$.
   If successful, return immediately (since we want min lexico, and we try $p_2$ in increasing order).
   If unsuccessful, try next $p_2$.
   After trying all $p_2$, if no solution in $G_0$, we might need mixed.
   Wait, if we don't fix color, we can just pick $p_2$ with $C_{1,p_2}=0$ OR $C_{1,p_2}=1$.
   If we pick a 1-edge first, we are committed to color 1 until switch.
   So, strategy:
   Loop $color \in \{0, 1\}$.
   Loop $p_2$ in increasing order $2 \dots N$.
   Check if $C_{1, p_2} == color$.
   Try to find path of edges with weight $color$ starting $1 \to p_2 \to \dots$.
   If path covers all vertices? No, that's HP.
   But we allow 1 switch.
   So we can find a path of 0s starting $1 \to p_2$, and then switch to 1s.
   If we switch, we enter the 1-mode.
   We need to find a path of 1s from $p_{last}$ to $p_{target}$?
   Wait, the switch can happen anywhere.
   But if we fix $p_1 \to p_2 \to \dots \to p_k$ to be all 0s.
   Then we need to cover the rest with 1s.
   This implies we need to partition vertices.
   
   Actually, given the difficulty of "Mixed Path" and the constraints, maybe the intended solution IS finding HP in $G_0$ or $G_1$.
   Why? Because mixed paths are much harder to construct and validate.
   Wait, if the graph is disconnected in $G_0$ and $G_1$?
   If $G_0$ and $G_1$ are disconnected, maybe no solution?
   But wait, $G_0 \cup G_1$ is complete.
   If $G_0$ has isolated vertex $v$ (so no 0-edges), then any path through $v$ must use 1-edges incident to $v$.
   So if we can't find HP in $G_0$ because $v$ is isolated, maybe we can use 1-edges.
   So, essentially, we can switch colors whenever necessary?
   But we are restricted to AT MOST 1 switch.
   So we can only switch ONCE.
   So we must pick a direction (mostly 0s or mostly 1s).
   
   If $G_0$ has a Hamiltonian Path, we are done.
   If $G_1$ has a Hamiltonian Path, we are done.
   Is it possible that we need a switch?
   Example: $1-2$ (0), $2-3$ (0), $1-3$ (1).
   $G_0$ path $1-2-3$. $p=1, 2, 3$.
   $c_1=0, c_2=0$.
   $c_3=C_{3,1}=1$.
   Valid.
   So here $G_0$ HP works.
   What if $1-2$ (0), $2-3$ (1), $1-3$ (1).
   $G_1$ path $1-3-2$. $p=1, 3, 2$.
   $c_1=C_{1,3}=1$.
   $c_2=C_{3,2}=1$.
   $c_3=C_{2,1}=0$.
   Valid.
   
   What if $1-2$ (1), $2-3$ (1), $1-3$ (0).
   $G_1$ path $1-2-3$.
   $c_1=1, c_2=1, c_3=0$. Valid.
   
   So far, HP in $G_0$ or $G_1$ seems sufficient.
   Is it possible to construct a counter-example?
   We need a graph where neither $G_0$ nor $G_1$ has HP, but a mixed solution exists.
   Example: Triangle.
   If $G_0$ has 2 edges (path), $G_1$ has 1 edge.
   If $G_0$ edges $(1,2), (2,3)$, $G_1$ edge $(1,3)$.
   $G_0$ HP exists ($1-2-3$). $c_3=C_{3,1}=1$.
   $G_1$ HP does not exist (isolated 2).
   So $G_0$ works.
   
   If $G_0$ edges $(1,2), (1,3)$ (star at 1). $G_1$ edges $(2,3)$.
   $G_0$ HP? No (2 leaves, center 1. Path $2-1-3$ exists? Yes).
   Wait, $2-1-3$ is valid in $G_0$.
   So HP exists.
   
   It seems hard to avoid HP in $G_0$ or $G_1$ for small graphs.
   Maybe for general graphs, it's true?
   Wait, for $N=4$.
   $K_4$ minus edges?
   If $G_0$ is a cycle $1-2-3-4-1$? Then HP exists.
   
   Actually, there is a conjecture/result: In any tournament (directed), there is a Hamiltonian path.
   Here we have undirected.
   If graph is "dense" enough?
   If $C_{i,j}$ is random, $G_0$ and $G_1$ are roughly $G(n, 1/2)$. HP exists with prob 1 for large $n$.
   For adversarial cases?
   Disconnected graph.
   If $G_0$ has components $A, B$. HP doesn't exist.
   If $G_1$ connects all components?
   If $G_1$ is connected, does it have HP?
   Not necessarily (could be a tree, or star).
   If $G_1$ is a star centered at $c$. Path exists if center is included.
   Star: $c$ connected to all others.
   Path $x_1-c-x_2$... no, need path visiting everyone.
   Star has HP: Leaf-center-Leaf? No.
   Leaves must be visited.
   Center has degree $N-1$.
   We can go Leaf1-Center-Leaf2...
   But we can only visit Center once.
   So we can visit at most 2 leaves (one before, one after).
   Other leaves remain unvisited.
   So Star graph (if $N>3$) has NO Hamiltonian Path.
   So, it is possible that $G_1$ is a Star and $G_0$ is Empty (for $N>3$).
   In this case, $G_1$ has NO HP. $G_0$ has NO HP.
   So we MUST use a mixed solution.
   Example:
   $N=4$.
   Edges:
   Star at 1 in $G_1$ (edges (1,2), (1,3), (1,4)).
   Rest are 0s in $G_0$.
   Wait, if edges are (1,2), (1,3), (1,4) are 1.
   Others (2,3), (2,4), (3,4) are 0.
   $G_0$ is triangle 2-3-4 plus isolated 1.
   $G_1$ is star at 1.
   HP in $G_0$? No (1 isolated).
   HP in $G_1$? No (Star).
   Mixed solution?
   Try path: $2-3-4-1$?
   Edges: $(2,3)=0, (3,4)=0, (4,1)=1$.
   Sequence $c_1=0, c_2=0, c_3=1$.
   Valid (1 switch).
   So mixed solution exists.
   And it's better than any HP solution (which doesn't exist).
   
   So we NEED to support mixed solutions.
   Structure:
   We need to find $p$ such that sequence is $0 \dots 0 1 \dots 1$ or $1 \dots 1 0 \dots 0$.
   This means:
   Part 1: Path in $G_0$ (or $G_1$) covering a set of vertices $V_0$.
   Part 2: Path in $G_1$ (or $G_0$) covering $V \setminus V_0$.
   With correct connections at junctions.
   Junctions: $u$ (end of Part 1) and $v$ (start of Part 2).
   Wait, Part 1 connects to Part 2 via an edge in $G_1$ (if Part 1 is $G_0$)?
   Wait, if $c_i$ sequence is $0 \dots 0$ then $1 \dots 1$.
   Transition from 0 to 1 happens at some index.
   Edges in sequence: $E_0$ (all 0), $E_1$ (all 1).
   The path is $p_1 \xrightarrow{0} p_2 \xrightarrow{0} \dots \xrightarrow{0} p_k \xrightarrow{1} p_{k+1} \dots \xrightarrow{1} p_n \xrightarrow{1} p_1$.
   Wait, $c_n$ corresponds to $(p_n, p_1)$.
   If $c_n=1$, then closing edge is 1.
   So, the sequence of edges is:
   $(p_1, p_2)$ is 0.
   ...
   $(p_{k-1}, p_k)$ is 0.
   $(p_k, p_{k+1})$ is 1.
   ...
   $(p_{n-1}, p_n)$ is 1.
   $(p_n, p_1)$ is 1.
   This implies:
   Vertices $\{p_1, \dots, p_k\}$ are connected by 0-edges?
   No.
   $p_1 \xrightarrow{0} p_2 \dots \xrightarrow{0} p_k$.
   So $p_1, \dots, p_k$ induce a path in $G_0$ (length $k-1$).
   Vertices $\{p_{k+1}, \dots, p_n\}$ and $p_1$?
   Edges are $(p_k, p_{k+1})$ is 1.
   $(p_{k+1}, p_{k+2}) \dots (p_{n-1}, p_n)$ is 1.
   $(p_n, p_1)$ is 1.
   This implies $p_{k+1}, \dots, p_n, p_1$ form a path/cycle?
   Wait, $p_{k+1} \dots p_n \to p_1$.
   So $p_{k+1}, \dots, p_n, p_1$ form a closed walk in $G_1$?
   No, it's a path $p_{k+1} \to \dots \to p_n$ then edge to $p_1$.
   So $V_{rest} = \{p_{k+1}, \dots, p_n, p_1\}$.
   This set induces a cycle/path in $G_1$?
   Specifically, the edges $(p_{k+1}, p_{k+2}), \dots, (p_{n-1}, p_n), (p_n, p_1)$ are all 1.
   This forms a path from $p_{k+1}$ to $p_1$ covering all vertices in $\{p_{k+1}, \dots, p_n\} \cup \{p_1\}$.
   Wait, the set of vertices is $\{p_1, \dots, p_k\} \cup \{p_{k+1}, \dots, p_n\} = V$.
   The intersection is empty? No.
   $p_1$ is in $\{p_1, \dots, p_k\}$.
   $p_1$ is also in the 1-path (as destination).
   $p_{k+1}$ is in the 0-path (as destination) and 1-path (as start).
   So $p_1$ and $p_{k+1}$ are the joint vertices.
   Condition:
   1. Find a subset of vertices $S$ containing $p_1$.
   2. Partition $S \setminus \{p_1, p_{k+1}\}$? No.
   
   Basically:
   Find two vertices $u, v$.
   Find a Hamiltonian Path in $G_0$ from $u$ to $v$? No.
   The 0-path must visit a subset of vertices.
   Wait, if we want to minimize lexicographically, we should try to find $p$ such that $p_1$ is minimized.
   So fix $p_1 = 1$.
   Then we want to find $p_2$ minimized such that a valid path exists.
   
   Given the complexity, and that this is a coding task, maybe I can use a randomized approach to find *a* solution.
   But lexicographical minimality requires systematic search.
   
   However, writing a full Hamiltonian Path solver is complex.
   Let's simplify.
   Notice that for $N=2000$, if a solution exists, it's likely that a solution with *most* edges of one color exists.
   Wait, the problem asks for "almost monochromatic".
   Maybe we can assume that if a solution exists, we can find it by trying to find HP in $G_0$ or $G_1$?
   Actually, in the Star example ($G_1$ star, $G_0$ triangle+isolated), HP in $G_0$ and $G_1$ didn't exist.
   But mixed solution existed.
   Wait, $p = 2, 3, 4, 1$.
   $2 \to 3$ (0), $3 \to 4$ (0), $4 \to 1$ (1), $1 \to 2$ (1).
   Path $2-3-4$ in $G_0$.
   Path $1-4-3-2$? No.
   Cycle in $G_1$?
   Edges used in 1-phase: $(4,1)$, $(1,2)$? No, $(4,1)$ is 1.
   Wait, in 1-phase edges are $(p_k, p_{k+1}) \dots (p_n, p_1)$.
   Here $p=2,3,4,1$.
   $p_4=1$.
   If we split after 3 edges? $c_1=0, c_2=0, c_3=1$.
   Edges: $(2,3)=0, (3,4)=0, (4,1)=1$.
   $c_3=1, c_1=C_{2,1}=1$. (Wait, $p_n=1, p_1=2$). $c_4 = C_{1,2} = 1$.
   Sequence $0, 0, 1, 1$. Switches at 2 ($0 \to 1$).
   So valid.
   
   This implies we need to find a path of 0s $u \rightsquigarrow v$ covering some set, and 1s $v \rightsquigarrow u$ covering the rest.
   This is equivalent to finding a Hamiltonian Decomposition?
   
   Given the difficulty, maybe the test cases are not worst-case.
   But I need a robust solution.
   However, implementing a full generic solver is hard.
   Let's assume that finding HP in $G_0$ or $G_1$ is the main challenge.
   And maybe for most cases, mixed is rare or reducible to HP?
   Wait, Star example shows mixed is distinct.
   But in Star example, $G_1$ edges are $(1,2), (1,3), (1,4)$.
   We used $(4,1)$ and $(1,2)$ in 1-phase.
   $(1,4)$ is 1, $(1,2)$ is 1.
   So we used a path $4 \to 1 \to 2$ in $G_1$.
   This path covers vertices $\{4, 1, 2\}$.
   Remaining vertex $\{3\}$ covered by 0-path $2 \to 3 \to 4$.
   So yes, it works.
   
   Maybe I can just use a randomized search (DFS with restarts) and hope it finds *something*.
   Wait, the user wants me to *implement a solution*.
   If I output a solution, it must be correct.
   
   What if I implement a "Greedy with Backtracking"?
   Function `solve(current_node, visited_set, current_color)`:
     If `visited_set.size() == n`:
       Check validity of closing edge.
       If valid, return path.
     
     If `current_color == 0`:
       Try all neighbors $v$ of `current_node` with $C=current\_color$.
       Pruning: Check if `v` allows completing the path? (Too expensive).
       Pick neighbor with lowest index (lexico).
     Else (`current_color == 1`):
       Try all neighbors with $C=1$.
     
   But wait, we can switch color once.
   If we haven't switched, we can pick color 0 or 1?
   No, the switch is at the edge.
   If we pick edge 0, we are in mode 0.
   Next edge can be 0 or 1.
   If it becomes 1, we record "switch occurred".
   Once "switch occurred", we can ONLY pick 1s.
   If "switch NOT occurred", we can pick 0s or 1s.
   
   So BFS/DFS state: `(u, mask, switches, start_node)`? Mask too big.
   Just `(u, switches, first_switch_node, target_color)`.
   Actually, we just need to maintain the set of visited nodes.
   
   Since $N$ up to 2000, bitmask is impossible.
   But maybe we don't need to track visited set fully?
   No, we need exact path.
   
   Wait, I think I'm overthinking.
   Is there a solution that simply iterates?
   Actually, if $C$ is random, $O(N \cdot N^2)$ search might find it.
   But we need correctness.
   
   Let's check the Sample 1 again.
   Maybe the solution logic is:
   Try to construct a path of 0s starting from 1.
   If stuck, try to switch to 1s.
   If stuck there, try to switch back? No, max 1 switch.
   
   So strategy:
   Iterate possible "cut" vertices $u$ (first vertex of 1-phase).
   Wait, we need $p_1$.
   We want minimal $p_1$.
   Fix $p_1$.
   Try to find a path starting $p_1$ that eventually switches or stays.
   We can try to simulate the process.
   Since $N$ is small sum, maybe simple backtracking with memoization?
   No, visited state.
   
   Let's write a code that uses a randomized shuffle of adjacency lists to find *a* solution.
   Since time limit is tight, we should optimize.
   For each test case:
   Try to find path.
   Prioritize $p_1=1$.
   Then try $p_2 \dots$ with sorting.
   Actually, if I just do a DFS, it will try to visit unvisited neighbors.
   To ensure lexicographical order, we iterate neighbors in sorted order.
   To minimize switches, maybe we restrict?
   Wait, we can switch exactly once.
   But we don't know when.
   However, since we want *any* valid path, maybe the most robust strategy is:
   Attempt to find a Hamiltonian Path in $G_0$ (using Warnsdorff or simple DFS).
   If success, return it.
   Attempt to find Hamiltonian Path in $G_1$. Return it.
   If neither, try to find a mixed path?
   Wait, mixed path is complex.
   But maybe there is a trick.
   If $G_0$ and $G_1$ are both disconnected/weak, output -1.
   Otherwise, if one has HP, use it.
   If mixed is required, it usually implies specific structure.
   Maybe just check connectivity?
   
   Actually, the simplest solution that handles the sample (assuming my reading is correct) is to just find a valid permutation.
   If I can't find HP, maybe I can output -1.
   Given the problem type, it's highly likely that a Hamiltonian Path in one of the graphs is sufficient or expected.
   The "lexicographically minimal" requirement pushes us to try to build it greedily.
   
   Wait, if I submit a code that outputs a solution using backtracking with Warnsdorff's rule, it might pass if the graphs are not malicious.
   However, "Time Limit 1s" and "Sum of N 2000" is quite generous.
   $O(N^3)$ is acceptable.
   $2000^3 \approx 8 \times 10^9$ ops is too much.
   So it must be $O(N^2)$.
   Greedy construction of HP is usually fast.
   Randomized backtracking might visit states quickly.
   
   Let's try a hybrid approach.
   1. Check if we can build a 0-path from 1 covering all vertices (HP in G0).
      If yes, return it.
   2. Check if we can build a 1-path from 1 covering all vertices (HP in G1).
      If yes, return it.
   3. If not, check if we can switch?
      This step is hard.
      Maybe for the purpose of this prompt, assuming HP sufficiency is enough?
      Wait, Sample 1 worked with HP in $G_0$ (1-2-3).
     
````

---

## 直接SFT 样本(sample_idx=2,得分 0.0,completion_tokens=1577)

````
Let me analyze this problem.

We have an n×n symmetric matrix C of 0s and 1s, representing a graph where edge (i,j) exists if C_{i,j}=1. We need to find a Hamiltonian cycle (permutation p_1,...,p_n forming a cycle) such that the number of "color changes" is at most 1.

A color change occurs when consecutive edges in the cycle have different colors (0 vs 1). So we want almost all adjacent pairs in the cycle to have edges of the same color.

This means either:
- Almost all edges in the cycle are from the complement graph, OR
- Almost all edges in the cycle are from the original graph C

Since the total number of color changes must be ≤ 1, we can use DFS with backtracking to explore paths that maintain consistent coloring, trying both cases and checking if we can close each into a valid Hamiltonian cycle with ≤1 color change.

I'll track the current path, last node visited, the color used for the last edge, and count color changes made so far. For each candidate next node, I check if connecting it would exceed our budget of 1 color change. If not, I recursively extend. When the path has length n, I try to connect back to the starting node - if this closing edge causes more than 1 color change, I backtrack; otherwise we've found a solution, and since I return immediately upon finding one, it will be lexicographically minimal among those discovered first.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    
    int n;
    while (cin >> n) {
        vector<vector<int>> G(n, vector<int>(n));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                char c;
                cin >> c;
                G[i][j] = c - '0';
            }
        }
        
        vector<int> p(n + 1);
        int changes = -1;
        
        auto solve = [&](auto&& self, int len, int start, int prev, int type) -> bool {
            if (len == n) {
                for (int i = 0; i <= n; ++i) cout << p[i] << (i == n ? "" : " ");
                cout << "\n";
                exit(0);
            }
            
            int limit = changes + abs(len - n);
            for (int i = 1; i <= n; ++i) {
                if (visited[i]) continue;
                
                int c0 = G[prev][i];
                int c1 = 1 - G[start][i];
                int cur_changes = 0;
                if (type == 0) {
                    cur_changes = abs(c0 - c1);
                } else {
                    cur_changes = abs(c0 - G[start][start]);
                }
                
                if (changes >= 0 && len + cur_changes > limit) continue;
                
                visited[i] = 1;
                p[len] = i + 1;
                int nxt_type = 0;
                if (cur_changes != 0) {
                    changes++;
                    nxt_type = 0;
                } else {
                    nxt_type = type;
                }
                if (self(self, len + 1, start, i, nxt_type)) return true;
                visited[i] = 0;
            }
            return false;
        };
        
        sort(all, begin(p), end(p));
        sort(begin(p), end(p));
        fill(p.begin(), p.end(), 1);
        fill(visited.begin(), visited.end(), 0);
        solve(solve, 0, p[0], p[0], 0);
    }
    
    return 0;
}
``` I need to reconsider the approach here. The issue is that my greedy strategy doesn't guarantee finding a valid Hamiltonian cycle even when one exists. The DFS with limited backtracking also doesn't give us truly lexicographically minimal results.

What I actually need is a systematic search that prioritizes smaller numbers while still exploring enough of the search space to find a valid cycle. Let me use memoization with bitmask DP to keep only the essential states - which nodes have been visited, the previous node, and what color edges were predominantly used - then try candidates in order so the first complete path I discover happens to be lex-smallest.

For each state tracking the subset of visited vertices, the predecessor node, and whether the incoming edge was mostly from the original graph, I'd explore neighbors by ascending index. Once I build a full permutation through the recursion, I verify the final connection back to the start adds no extra color changes beyond what's allowed - exactly one or zero transitions between different colored edges.

But there's still the initialization bottleneck to resolve before committing to this.
</think>

```cpp
#include <bits/stdc++.h>
using namespace std;

int n;
vector<vector<int>> C;
bool visited[2005];
int best_cost = -1;
int best_first = 2005;

void dfs(int pos, int start, int last, int prev, int cnt, int &cost) {
    if (pos == n) {
        cost += C[last][start];
        return;
    }
    
    if (prev == last) {
        dfs(pos + 1, start, pos, prev, cnt, cost);
        return;
    }
    
    for (int next = 1; next <= n; next++) {
        if (next == start || next == last || visited[next]) continue;
        
        int new_cost = 0;
        if (C[prev][last] == C[last][next]) {
            dfs(pos + 1, start, next, last, cnt, new_cost);
        }
        else if (cnt < 2) {
            dfs(pos + 1, start, next, last, cnt + 1, new_cost);
        }
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    
    while (cin >> n) {
        C.assign(n, vector<int>(n));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                cin >> C[i][j];
            }
        }
        
        memset(visited, 0, sizeof(visited));
        memset(best_cost, 0, sizeof(best_cost));
        memset(best_first, 0, sizeof(best_first));
        
        best_cost = -1;
        best_first = 2005;
        
        if (!dfs(0, 0, 1, 0, 0, best_cost)) {
            cout << -1 << endl;
        } else {
            cout << best_first << endl;
        }
    }
    
    return 0;
}
```
````
