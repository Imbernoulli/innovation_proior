# FrontierCS：Tree distance（交互式竞赛题） — SFT（q35 a100 method 全参微调）

> **本文件是完整、未截断的真实评测记录。** 模型类别：`sft`。
> 评测：FrontierCS（thinking 口径，竞赛算法编程，官方评测器抽最长 ```cpp 块编译跑测试点给连续部分分）。
> 题目内部编号 `problem_idx=2` / 题库目录 `algorithmic/problems/10` / 第 `sample_idx=0` 个采样（共 5 采样/题）。
>
> **本采样得分（FCS score，0–100 连续部分分）：`0.0`**　|　生成 token 数：`32768`　|　error：`None`

---

## 1. 完整输入（喂给模型的真实 prompt）

模型实际收到的 user prompt = 一段固定的「C++ 竞赛求解」system 引导 + 下面这道题的 `statement.txt` 原文 + `Generate solution code:`。题面原文如下（逐字照抄题库）：

```text
Problem: Tree distance

Time limit: 2 second

Memory limit: 512 MB

This is an interactive problem.

Little Cyan Fish has a weighted tree of n verticies generated in the following way:
- First, generate a random labeled tree: uniformly randomly select from all nn−2 possible labeled trees.
- Then, independently assign random integer edge weights in the range [1, K] to each edge, where K is a hidden parameter.

You cannot directly observe the structure of the tree or the edge weights, but Little Cyan Fish grants 
you a superpower: querying! Each time, you can query the distance between two vertices. Specifically, you
can choose two vertices u, v (1 \leq u, v \leq n, u \neq v), and we will tell you the distance between these two
vertices (i.e., the sum of the edge weights on the simple path connecting these two vertices).

Now, Little Cyan Fish wants you to determine all the edges and their weights within queries as less as you can. 

Scoring

The score is calculated based on the linear formula determined by the range [5n, Z]:
- Score = 100 * (Z - Q) / (Z - 5n)


Interaction Protocol

Each test case contains multiple sets of test data. First, you need to read an integer T (1 \leq T \leq 10^4)
indicating the number of data sets.
For each set of test data, you first need to read an integer n (1 \leq n \leq 10^5).
Next, the interaction process begins. To
make a query, you need to output a line “? u v” (1 \leq u, v \leq n, u \neq v), describing a query. Then, you need
to read the result from standard input.
To provide your answer, you need to output “! u_1 v_1 w_1 u_2 v_2 w_2··· u_{n−1} v_{n−1} w_{n−1}”. You can output
these edges in any order. The output of the answer will not count towards the n * n / 3 query limit. After you
output the answer, you need to immediately read the next set of test data or terminate your program.
After outputting a query, do not forget to output a newline character and flush the output stream.
To do this, you can use fflush(stdout) or cout.flush() in C++, System.out.flush() in Java,
flush(output) in Pascal, or stdout.flush() in Python.
It is guaranteed that 1 \leq K \leq 10^4, and the sum of all n in the test data does not exceed 10^5.

In this problem, it is guaranteed that the interaction library is non-adaptive. That is, the shape of the
tree and the edge weights are determined before the interaction process. They will not change with your
queries.

Example input:
2
3

3

4

7

4

3

7

2

4

5

9

Example Output:


? 1 2

? 2 3

? 1 3

! 1 2 3 2 3 4

? 1 2

? 2 3

? 2 4

? 1 3

? 1 4

? 3 4

! 1 2 3 1 3 4 2 4 2
```

---

## 2. 完整输出（模型的真实生成，逐字照抄，未截断）

> 注：FCS 评测开启 thinking。下面是模型 `text` 字段的**全文**（包含推理与最终代码）。

````text
The problem asks us to determine the structure and weights of a hidden weighted tree. We are given $n$ vertices, and we can query the distance between any two vertices. The edges have weights in the range $[1, K]$. We need to output all $n-1$ edges with their weights. The total number of queries should be as small as possible. The sum of $n$ over test cases is $O(N)$, but the query budget is roughly $5n$.

Let's start with $n=2$. It's trivial, one edge, weight is dist(1, 2).

For general $n$, let's consider what information one query gives us. Query $(u, v)$ gives $\sum_{e \in Path(u,v)} w(e) = d(u, v)$.
If we fix a root, say vertex 1, and find distances from 1 to all other vertices, we get $d(1, i)$.
Can we recover the individual edge weights?
Consider an arbitrary vertex $i$. If we know for every neighbor $j$ of $i$, both the edge weight $w(i,j)$ and $d(1, j)$, then:
$d(1, i) + w(i, j) - d(1, j)$ ... this depends on whether the path from 1 to $j$ goes through $i$.
Actually, if we pick a spanning tree rooted at 1, say $P_1$. For any node $v$, there is a unique parent $p(v)$ such that removing $(v, p(v))$ would disconnect the tree (since it's a spanning tree of the whole graph, actually, it IS the whole graph being solved). Wait, the whole thing is a tree. So fixing a root makes every node except root have exactly one "parent". Let's call this the BFS tree rooted at 1. Since the underlying graph is already a tree, the BFS tree consists of ALL edges connected to each component towards the root? No, BFS just defines parent pointers. In a general graph, the spanning tree might not contain all edges, but here the graph IS a tree. So the parent pointer $p(v)$ corresponds to a specific edge $(v, p(v))$.
So, if we can identify $d(1, v)$ for all $v$ (which requires $N-1$ queries), we still don't know which specific edge connects $v$ to its ancestor. We know $d(1, v)$ is the distance along the actual edges, but we assumed a tree structure formed by parent pointers.
Actually, let's look at the relation:
$d(1, v) = d(1, p(v)) + w(p(v), v)$ holds if the shortest path from 1 to $v$ goes through $p(v)$. But since the whole graph is a tree, there is only ONE path. So ANY path from 1 to $v$ goes through the neighbors of $v$ on the path. But wait, in the whole graph, there is no choice. If we pick *any* vertex $i$ as the root, and for *every* other vertex $v$, we identify a parent $p(v)$... we are essentially picking a direction for the edges away from root. This orientation gives us a directed tree.
If we orient all edges away from root 1, then for every $v \neq 1$, there is a unique incoming edge $(p(v), v)$. Then clearly $d(1, v) = d(1, p(v)) + w(p(v), v)$. So $w(p(v), v) = d(1, v) - d(1, p(v))$.
This means, IF we know the "parent" relationship, we can compute all weights with just $N-1$ queries ($d(1, i)$ for all $i$).
The problem is identifying the parents. A simple BFS or DFS traversal works ONLY if we know the graph structure. Here we DON'T.
However, we DO know it is a tree.
In a tree, for any vertex $v$ (not root), let $S_v$ be the set of neighbors. One of them is on the path to the root, and the others are "children".
Key Property:
$d(1, u) + d(1, v) - d(u, v) = 2 \times (\text{distance from } 1 \text{ to } LCA(u, v))$.
Or simply, consider the median property or similar geometric properties on trees?
No, let's use the property of the single edge incident to the root.
We want to identify the neighbor of 1 that leads to a child. Let $x$ be a neighbor of 1. Then $w(1, x) = d(1, x)$. Also for any descendant $v$ of $x$ (child of 1), $d(1, v) > d(1, x)$.
Wait, knowing $d(1, v)$ doesn't tell us if $v$ is in the subtree of a specific neighbor of 1, because we don't know the structure.
Actually, the simplest approach for recovering the structure of a tree given all-pairs distances (or subset thereof) is the following:
Find a vertex $c$ that minimizes $\max_i d(c, i)$ (the center) or something similar? Or maybe just finding the diameter.
Let's try finding a diameter first. Pick arbitrary $u$, find $v$ maximizing $d(u, v)$. $v$ is an endpoint of a diameter. Find $w$ maximizing $d(v, w)$. Path $v-w$ is a diameter.
Finding endpoints takes $2N$ queries if we do BFS/DFS naively. Can we do better?
With $d(u, v)$ queries, finding a farthest node takes $N-1$ queries if you scan all? No, we want $O(N)$ or close.
Actually, maybe we can do it faster?
Let's re-examine the constraint. $\sum n \le 10^5$. Queries allowed roughly $5N$.
Is finding the diameter fast enough?
Let's try a randomized approach.
Algorithm Idea 1:
1. Pick a random vertex $r$.
2. Run BFS-like search using $d(r, \cdot)$? No.
3. Just run a simple heuristic:
   Maintain a "working set" of candidate endpoints of a diameter. Initially {random vertex}.
   While size < 2:
     Take current candidates, expand?
   
Actually, there is a known algorithm to find a diameter of a tree given distance oracle:
1. Pick arbitrary $u$. Find $x$ s.t. $d(u, x)$ is maximized. ($N-1$ queries worst case?)
2. From $x$, find $y$ s.t. $d(x, y)$ is maximized. This gives diameter length $D = d(x, y)$.
3. The true endpoints of a diameter are some $a, b$ such that $d(a, b) = D$.
4. Any third vertex $z$ satisfies $d(z, a) + d(z, b) = D$ ? No, triangle inequality equality implies $a-z-b$ is a geodesic. If $z$ lies on the diameter, yes. If not, $d(z, a) + d(z, b) = d(a, b)$ is false. Actually $d(z, a) + d(z, b) \ge d(a, b) = D$.
   Equality holds iff $z$ is on the simple path between $a$ and $b$.
5. To identify the path between $a$ and $b$: iterate over all nodes $i$. Check if $i$ lies on path $a-b$.
   Condition: $d(i, a) + d(i, b) = D$.
   If so, insert into path list. Since $a, b$ are adjacent to this path (conceptually), and it is a simple line, sorting these by distance from $a$ reconstructs the path.
   
Wait, step 4 check cost is $N$ queries. Step 2 cost is $N$ queries. Total queries $\approx 2N$.
Step 1 also costs $N$. So finding $a, b$ costs $2N$. Total $3N$.
After finding diameter $a-b$, how do we build the rest of the tree?
This only gives us the backbone. What about branches?
But wait, if we know a diameter $a-b$, does it help with anything else? Not immediately obvious for the whole tree.
And we used $\approx 3N$ queries just to find one diameter. We might not even have time to solve the whole tree.
Is there a way to find *all* edges efficiently?

Let's rethink. We just need to assign a parent $p(v)$ to each vertex $v$.
Then $w(parent(v), v) = d(1, v) - d(1, p(v))$.
So the main difficulty is identifying the parent $p(v)$ for each $v$ (if root is 1).
How to distinguish the parent among neighbors?
Let's assume we fixed a root, say 1. And we have identified the diameter $a-b$ of the tree.
The diameter must go through the root 1? Maybe. If it does, the root splits the diameter?
Actually, let's generalize. Suppose we pick a random node $r$. We perform a search from $r$.
What if we maintain a set of "candidate children" for a node?
Let's restart the structural observation.
If we have a root $r$ and for every node $v$, we pick a neighbor $p(v)$ such that we believe $v$ is a direct descendant of $p(v)$, and this forms a valid tree rooted at $r$, then we are good.
We can construct such a candidate tree greedily.
Start with root $r$. $p(r) = r$ (self-loop or empty).
Iterate $k = 1 \dots n-1$. Pick an unvisited node $u$. Look at all its neighbors.
One of the neighbors will be the "correct" parent (closer to $r$ in tree distance).
Specifically, let $v \in Neighbors(u)$. Then:
$d(1, u) < d(1, v) + w(v, u)$ if path from $1$ to $u$ DOES NOT pass through $v$.
$d(1, u) = d(1, v) + w(v, u)$ if path from $1$ to $u$ passes through $v$.
Wait, the formula I wrote earlier was $d(1, v) = d(1, p(v)) + w(p(v), v)$. This means $d(1, p(v)) < d(1, v)$ is strictly true since weights $> 0$.
So if we have a candidate parent $v$ for $u$, it MUST satisfy: $d(1, u) > d(1, v)$ AND $d(1, u) = d(1, v) + w(v, u)$ ?
Wait, this assumes we know the weight $w(v, u)$. But we only have $d(1, \cdot)$ values. We can calculate weights along the reconstructed tree.
If we tentatively pick $v$ as parent of $u$, we define $w(v, u) = d(1, u) - d(1, v)$.
For this to be consistent, it must hold that $w(v, u) \ge 1$ (given edge weights $\ge 1$).
Also, since weights are positive, $d(1, u) > d(1, v)$.
So, a necessary condition for $v$ being the parent of $u$ is $d(1, v) < d(1, u)$.
This reduces the number of candidates significantly!
Only neighbors of $u$ that are CLOSER to the root than $u$ can be parents.
So for a leaf in our constructed tree (deepest node), we check its neighbors. Most neighbors might be further down, or we haven't visited them yet.
Actually, we process nodes in increasing order of $d(1, i)$.
Let's sort vertices by $d(1, i)$. Let this order be $v_1, v_2, \dots, v_n$ where $v_1$ is closest to root. (Assume $d(1, 1)=0$).
Now consider $v_k$. We need to assign it a parent $p(v_k)$ from $\{v_1, \dots, v_{k-1}\} \cap Neighbors(v_k)$.
Since $p(v_k)$ must have been visited already (dist smaller), checking neighbors among previously visited nodes is correct.
Does there exist a $v_j$ ($j < k$) that is a neighbor of $v_k$ satisfying $d(1, v_k) = d(1, v_j) + w(v_j, v_k)$?
Actually, in the original tree, there is EXACTLY one neighbor $p(v_k)$ of $v_k$ such that path $1 \to v_k$ goes through $p(v_k)$. And thus $p(v_k)$ is closer to 1 than $v_k$. All other neighbors are in subtrees attached at $v_k$, so they are further from 1.
Therefore, $p(v_k)$ is the UNIQUE neighbor of $v_k$ which has $d(1, p(v_k)) < d(1, v_k)$ AND $d(1, v_k) - d(1, p(v_k)) = w(p(v_k), v_k)$.
Wait, the second part relies on the fact that if $p(v_k)$ is the correct parent, the difference equals the edge weight.
But we don't know $w$. However, we know ALL edges have weight $\in [1, K]$.
So, we check all neighbors $v_j$ of $v_k$ with $j < k$.
There is exactly one such neighbor that acts as a parent. Let's call it the "real parent".
For the real parent, $w = d(1, v_k) - d(1, v_j)$. We just check if this calculated weight is a valid integer $\in [1, K]$.
For an incorrect parent $v_{wrong}$, we calculate $w' = d(1, v_k) - d(1, v_{wrong})$. Is it guaranteed to fail?
Not necessarily. It's possible $d(1, v_k) - d(1, v_{wrong}) = K$, and $v_{wrong}$ happens to be a child. Or maybe $d(1, v_{wrong})$ is large, making $w'$ negative.
But $d(1, v_{wrong}) > d(1, v_k)$ is impossible because we sorted by distance. Wait. If $v_{wrong}$ is a child of $v_k$, then $d(1, v_{wrong}) > d(1, v_k)$. So it won't appear in the candidates (candidates are $j < k$, so smaller indices, smaller distances).
Ah! So ALL incorrect neighbors in the candidate set $\{v_1, \dots, v_{k-1}\}$ are NOT children. They could be siblings or cousins (nodes in the same depth level or higher).
Could they be the *grandparent*? Yes. Could they be the parent? Yes.
So the candidates are $\{ \text{neighbors of } v_k \text{ visited so far} \}$.
Among these, exactly one is the actual parent (directly on the path to root). Others are in other branches.
For any OTHER neighbor $x$, the path from root to $v_k$ does NOT go through $x$. The path goes $1 \to \dots \to LCA(x, v_k) \to \dots \to v_k$.
Then $d(1, v_k) < d(1, x) + w(\text{edge into } x \text{'s subtree}) + \dots$
Wait, simpler: $d(1, v_k)$ is distance from 1. $d(1, x)$ is distance from 1.
If $x$ is not the parent, $x$ is not closer to 1 via the parent-link.
Wait, if $x$ is a sibling, $LCA(x, v_k) = P$ (their parent).
$d(1, x) = d(1, P) + w(P, x)$. $d(1, v_k) = d(1, P) + w(P, v_k)$.
Then $|d(1, x) - d(1, v_k)| = |w(P, x) - w(P, v_k)|$.
The proposed weight $W = d(1, v_k) - d(1, x) = w(P, v_k) - w(P, x)$.
Since $W$ would be calculated as $d(1, v_k) - d(1, x)$, if $W$ comes out to be equal to some valid weight $K \le 10^4$, we can't easily distinguish it from the REAL parent unless we have more info.
But wait, $x$ is a child of $P$. So $x$ is in a subtree below $P$. $v_k$ is a different child of $P$.
Is it possible to misidentify?
Suppose we have $1 - 100 \to x$ and $1 - 100 \to v_k$. $x$ and $v_k$ are siblings.
Distance $1-x$ is $100+w_x$. Distance $1-v_k$ is $100+w_{vk}$.
Sorted order puts the closer one first. Say $w_x < w_{vk}$. Then $x$ is visited before $v_k$.
When considering $v_k$, $x$ is a candidate parent.
Calculated weight $w' = d(1, v_k) - d(1, x) = w_{vk} - w_x$.
Real weight of edge $(P, v_k)$ is $w_{vk}$.
Can $w_{vk} - w_x = w_{vk}$? Only if $w_x = 0$. Impossible ($w \ge 1$).
Can $w_{vk} - w_x = w_{xy}$ for some other edge? No, this logic is getting complex.
Basically, the issue is distinguishing the real parent from another node that happened to be processed earlier.
However, note that if we just check the condition $w = d(1, v_k) - d(1, x) \ge 1$, it's satisfied.
BUT, if we simply pick ANY candidate, we form a spanning tree. If it's not THE tree, can we detect it?
The key insight:
If the computed weights form a valid tree where every edge has weight $W$ and satisfies $d(1, v) = \sum W$ along path, then we found the right tree?
Wait, multiple spanning trees (BFS vs DFS) could theoretically yield consistent sets of weights if the geometry permits.
However, we proved earlier that the parent MUST be the neighbor closer to the root in the ACTUAL tree.
So the set of potential parents is $\{ \text{neighbors } x \mid d(1, x) < d(1, v_k) \}$.
Why does sorting by $d(1, i)$ work? Because true parents always have smaller distance.
The problem remains: picking the WRONG one (which is smaller distance) yields a wrong weight.
Example: Real parent $P$ with $w(P, v_k) = 2$. Fake candidate $S$ (sibling) with $w(S) = 5$ and $w(P, S) = 3$.
$d(1, P) = D$. $d(1, v_k) = D+2$. $d(1, S) = D+3$.
Candidate is $S$. Weight computed = $(D+2) - (D+3) = -1$. Invalid.
Another example: $P$ with $w(P, v_k) = 2$. $S$ with $w(S) = 5$, $w(P, S) = 7$.
$d(1, P) = D$. $d(1, v_k) = D+2$. $d(1, S) = D+7$.
Candidate $S$. Computed weight $(D+2) - (D+7) = -5$. Invalid.
It seems computed weights between "non-parent ancestors" often turn out negative because they go UP the wrong branch too far.
Wait, let's look at the case where $S$ is closer to root than $v_k$.
This implies $d(1, S) < d(1, v_k)$.
Since $S$ is not the parent, the path $1 \to S$ does not end at $v_k$ directly. $v_k$ is in a subtree off $S$? Or they diverge somewhere above $S$?
If $v_k$ is in $S$'s subtree, then $S$ MUST be the parent (ancestor). And we'd select it.
But $S$ is NOT a parent. So $v_k$ is not in $S$'s subtree (rooted at $S$).
Thus $S$ and $v_k$ belong to disjoint subtrees attached to $LCA(S, v_k)$, or one is attached to the other?
If $S$ is attached to $LCA$, and $v_k$ is attached to $LCA$, and neither is an ancestor of the other.
Path goes $LCA \to S$ and $LCA \to \dots \to v_k$.
Since $S$ is a valid node visited earlier, $d(1, S) < d(1, v_k)$.
So $S$ must be closer to $LCA$ than $v_k$ is to $LCA$.
$d(1, v_k) = d(1, LCA) + w(LCA, \dots, v_k)$.
$d(1, S) = d(1, LCA) + w(LCA, \dots, S)$.
Calculated $W = d(1, v_k) - d(1, S) = w(LCA, \dots, v_k) - w(LCA, \dots, S)$.
This could be positive! E.g. $w(LCA, \dots, v_k) = 10, w(LCA, \dots, S) = 1$. $W = 9$.
And it could happen that 9 matches a valid weight.
If we blindly accept $S$ as parent, we get an invalid tree (because cycle or mismatch with real path later?).
But actually, if we make a mistake, can we correct it?
Or, does the "randomness" of weights prevent this collision?
$K$ is small ($10^4$)? No, problem says weights are in range $[1, K]$ but we don't know $K$. Wait, the problem description says "$K$ is a hidden parameter", but guarantees $1 \le K \le 10^4$. So weights are SMALL integers.
With small integer weights, collisions are LIKELY. So the simple greedy strategy "pick closest available neighbor with valid positive distance difference" is UNSAFE. A sibling could masquerade as a parent with a perfectly valid edge weight calculation.
We need a stronger condition.

Let's revisit the distance constraints.
Let the true path from root to $v_k$ be $r=v_0, v_1, \dots, v_m = v_k$.
For each node $v_i$ in the tree, let $p(v_i) = v_{i-1}$.
We have $d(1, v_i) = \sum_{j=1}^i w(v_{j-1}, v_j)$.
We are processing nodes by increasing $d(1, u)$.
When at $v_k$, we look for a neighbor $u$ with $d(1, u) < d(1, v_k)$.
If $u$ is the true parent, $w(u, v_k) = d(1, v_k) - d(1, u)$.
Let's denote this value as $D_u$.
The claim is: **A valid assignment of parents is uniquely determined.**
Wait, that's not helpful yet.
Let's consider the pair of vertices $(u, v)$ on the TRUE path.
They are separated by exactly one edge $e$. $w(u, v) = d(1, v) - d(1, u)$ (assuming ordered correctly).
Wait, NO. Only if they are ADJACENT on the true path.
If they are separated by distance $L$ (number of edges), $d(1, v) - d(1, u)$ is the SUM of weights of edges between them, not the weight of one edge.
So we can't directly verify adjacency just from 1-distance.
However, we are given $d(1, x)$ for ALL $x$.
Wait, if we have $d(1, x)$, we know $d(1, x) + d(1, y) - d(x, y) = 2 d(1, LCA(x, y))$.
So if we can query $d(u, v)$, we can find $d(1, LCA)$.
This suggests: if we knew the diameter path (or a specific path), we could locate everyone.
Can we find the diameter cheaper?
There is a probabilistic method to find a diameter edge in linear time.
But wait, the standard algorithm is deterministic $O(N)$ or $O(N \log N)$ queries?
Wait, with random queries, we can do this:
To find diameter:
1. Sample random edge? No.
2. There's a result that the diameter can be found in expected linear time with limited queries.
Actually, there's a very nice trick.
Let $x, y$ be the endpoints of the diameter. For ANY vertex $z$, $d(x, z) + d(y, z) = d(x, y) \iff z$ is on the diameter.
Proof:
On the path $x-y$, distances sum up to $L$.
For $z$ NOT on path, $d(z, x) + d(z, y) = d(z, x) + d(z, y) > d(x, y) = L$ (triangle inequality, strict because tree, not passing through).
Actually, it's equality only if $z$ is on the simple path.
So we just need to find TWO vertices $x, y$ such that $d(x, y)$ is maximized and equals the maximum possible distance.
Wait, "maximized" is hard without knowing all pairs.
But the property $d(x, z) + d(y, z) = d(x, y)$ is powerful. It tells us if $z$ is on the segment $xy$.
So the strategy:
1. Pick two vertices randomly, say $a, b$? No, diameter endpoints aren't random.
2. BUT, we can FIND them.
   Algorithm:
   Initialize $BestPair = (\text{rand}, \text{rand})$. $L = 0$.
   While not converged:
     Try to find a better pair?
   
   Let's refine the "find diameter" idea.
   We want to maximize $d(u, v)$.
   Start with any node $u$. Find farthest $v$. Then farthest from $v$ is diameter.
   BUT, the intermediate "farthest" might not be an endpoint of the LONGEST diameter.
   Example: Star graph, center connected to leaves with weights. Random leaf might lead to a leaf that isn't max-weight leaf?
   No, if weights are uniform, it finds an endpoint.
   The issue with finding diameter endpoints is robustness.
   BUT notice: we don't need a DIAMETER. We just need ONE PATH.
   Actually, ANY path of the tree can be used to locate the rest?
   If we have a path $p_1 - p_2 - \dots - p_k$.
   For any vertex $z$, consider the projection onto this path. Let $proj(z)$ be the vertex on the path nearest to $z$.
   If we knew $proj(z)$, we could connect $z$ to it.
   How to find $proj(z)$?
   We know $proj(z)$ minimizes $d(p_i, z)$. (Wait, metric projection).
   Actually, the projection $u$ of $z$ onto a simple path $v-w$ is defined such that $u$ is the vertex on the path minimizing $d(z, u)$, and $d(z, v) = d(z, u) + d(u, v)$ (meaning $u$ is on the path between $z$ and $v$?? No).
   Correct definition of projection $u$ of $z$ on path $v \to w$: $u$ is the unique vertex on the path such that $z$ is NOT in the subtree of any neighbor of $u$ (relative to the tree rooted away from the path), OR $d(z, v) + d(u, w) = d(v, u) + d(w, z)$? No.
   Simpler property: $d(z, v) + d(z, w) = d(v, w) \iff z$ is on the path $v-w$.
   If $z$ is NOT on the path $v-w$, then the path from $z$ to the path $v-w$ meets it at a unique point $u$.
   Then $d(z, v) + d(z, w) - d(v, w) = 2 \cdot d(z, u)$.
   Thus $d(z, u) = \frac{d(z, v) + d(z, w) - d(v, w)}{2}$.
   So if we pick ANY long path, and for any $z$, check distances to its ends $v, w$, we can find the point $u$ on the path closest to $z$.
   If $z=u$ is found, then $z$ belongs to the path. We keep it.
   But if $z$ is off-path, calculating $u$ requires querying $d(z, v)$ and $d(z, w)$. That's cheap (2 queries).
   Once we know $u$ (the attachment point of $z$ to the current path), we just add edge $(z, u)$ with weight $d(z, u)$!
   This solves everything. We just need to find ONE simple path in the tree.
   
   Is it true that ANY simple path suffices?
   Suppose the path is just a single edge (two vertices).
   Pick any two nodes $u, v$. Path is $u-v$.
   Iterate all other nodes $z$.
   Calculate $w = \frac{d(z, u) + d(z, v) - d(u, v)}{2}$.
   This gives distance from $z$ to the PATH.
   Wait. Does it give us WHICH vertex on the path?
   The formula gives distance to the LINE segment.
   If the path has length 1, the set of vertices on the path is $\{u, v\}$.
   So we find that $z$ projects to either $u$ or $v$ (or midpoint edge? Edge midpoints don't exist).
   Since we are in a discrete graph, if $z$ projects to $u$, then $d(z, v) = d(z, u) + d(u, v)$. If to $v$, $d(z, u) = d(z, v) + d(u, v)$.
   Check: does $d(z, v) \ge d(z, u) + d(u, v)$ hold?
   If yes, $u$ is the attachment point. Edge $(u, z)$ with weight $d(z, u)$.
   If $d(z, u) \ge d(z, v) + d(u, v)$, $v$ is attachment.
   If neither, then $z$ is "between" $u$ and $v$ on the path. This is IMPOSSIBLE if path is just $u-v$ and $z$ is not on it, because that would mean $u-v$ is part of the path to $z$, so $z$ projects to one of them.
   Basically, triangle inequality: $d(u, z) + d(z, v) = d(u, v)$.
   Case 1: $d(u, z) + d(z, v) = d(u, v)$. $z$ is on the edge $uv$ (so path is $u-z-v$). But $u, v$ were picked arbitrarily from the whole graph.
   Wait. If we pick ANY two nodes $u, v$, they might NOT be connected by an edge. There is a path $u=x_0, \dots, x_m=v$.
   Then $z$ could attach anywhere on this path.
   The formula $\frac{d(z, u) + d(z, v) - d(u, v)}{2}$ gives the distance from $z$ to the PATH. Let this be $D_z$.
   So $d(z, \text{path}) = D_z$.
   Now we need to identify WHICH node on the path attaches to $z$.
   Let the path be $x_0, \dots, x_m$.
   We want to find $k$ such that $x_k$ is the projection.
   This looks like binary search. $x_k$ divides the path into two parts.
   Check midpoint $m_{mid}$. Compare $d(z, x_k)$.
   Actually, since we know the sequence of vertices is collinear (distances are additive), we can just walk along the path from one end until $d(z, \cdot)$ stops decreasing.
   Cost to reconstruct this one path:
   1. Guess an initial path. Say, start BFS from arbitrary $r$. Keep track of "best seen distance".
      Wait, we need to identify the path vertices.
      Algorithm:
      Pick arbitrary $r$. Perform "Search":
      Current vertex $curr = r$. Max dist $M = d(r, r) = 0$.
      Best vertex $y = r$.
      While true:
        Query neighbors? No, don't know them.
        Query $d(curr, y)$? No, don't know distance to $curr$.
        We need to traverse the graph. We can't.
        
      Okay, the "search" phase needs to discover edges.
      But we can't discover edges without querying them.
      This circularity is bad.
      
      Let's reconsider the random edge selection.
      Suppose we pick an edge $(u, v)$ uniformly at random.
      Probability that $(u, v)$ is a diameter edge? Low.
      What is a useful random object?
      Two random vertices?
      Prob that 2 random vertices form a valid path?
      In a tree, almost all pairs $(u, v)$ define a simple path. The whole graph IS the union of simple paths between all pairs.
      Every vertex $z$ belongs to SOME simple path.
      So if we pick a pair $u, v$ such that the true path $P_{uv}$ contains ALL vertices... then we are done. But $P_{uv}$ rarely does.
      So we can't rely on one path.
      
      Wait, what if we find a path containing *half* the vertices? Or find the diameter?
      Finding the diameter allows us to locate everything.
      We established earlier: finding diameter can be done with $O(N)$ queries using a heuristic.
      Let's analyze the probability/correctness of the heuristic.
      Pick $u$. Find farthest $x$. $d(u, x)$.
      Find farthest from $x$ to $y$. $d(x, y)$ is likely diameter.
      Why? $y$ is an endpoint.
      In random graphs/trees, a random node has non-trivial path to farthest.
      With $N$ queries, we can find a very long path.
      Specifically:
      1. Pick $x$ randomly.
      2. Walk to farthest neighbor $y$? We don't know neighbors.
         Ah, we can't walk.
         Wait, to "Walk to farthest", we need neighbors.
         How to get neighbors? Query ALL distances from $x$? $N$ queries.
         Let's say we do this once.
         Get $D(i) = d(x, i)$ for all $i$.
         Find $u$ such that $D(u)$ is maximal. $u$ is far from $x$.
         This is a valid step.
         Now we have a "long" path starting at $x$ going through $u$.
         Now query distances from $u$. Find $v$ far from $u$.
         Path $x-u-v$ is likely a good candidate for a diameter or a central long path.
         Wait, actually, just finding ANY valid path covering most nodes is sufficient?
         Yes, finding one valid path covering a significant fraction is enough to bootstrap.
         But wait, if we don't cover a node, we just project it onto the path we HAVE.
         
      Let's formalize this.
      We need to collect all vertices into the set of "covered" vertices.
      Set $C = \{u\}$ initially (where $u$ is chosen).
      We maintain a "current path" inside the discovered subgraph? No, we discover vertices one by one.
      Let's use the property: any vertex $v$ is connected to the set $C$ by a unique path.
      If we have a "core" (a connected subgraph/tree) and for any new node $v$, we find its attachment point on the core.
      
      Algorithm:
      Let $Core = \{ \text{random vertex } r \}$. (Initially singleton).
      While $|Core| < n$:
          Pick $v \notin Core$.
          We want to find which vertex in $Core$ is adjacent to $v$ (or connected via a path through Core that we can resolve).
          Wait, we can't find which node is adjacent.
          
          Instead, let's go back to the Diameter idea.
          We need ONE long path.
          Let's try to find a path $P$ that spans the tree.
          Heuristic:
          1. Pick $u$ random. Find $x$ maximizing $d(u, x)$. (Query $N$ times).
          2. From $x$, find $y$ maximizing $d(x, y)$. (Query $N$ times).
             Path $x-y$ is likely near the diameter.
             If it covers all nodes, great.
             If not, we missed some branches off the diameter.
             
          BUT, we can improve this!
          After finding the approximate diameter endpoints $x, y$ and length $L=d(x, y)$, we can locate EVERY vertex.
          Recall: $d(z, x) + d(z, y) = L \implies z$ is on diameter $x-y$.
          This identifies all nodes ON the diameter.
          What about nodes OFF the diameter?
          Let $z$ be off-diameter. Its projection $u$ onto $x-y$ satisfies $d(z, u) = (d(z, x) + d(z, y) - L)/2$.
          And importantly, $u$ is the attachment point of $z$.
          If we know the vertices of the diameter (sorted order), say $p_1=p_1, \dots, p_k=y$.
          The attachment point $u$ must be some $p_j$.
          Which one?
          Since the distance $d(z, p_j)$ decreases as $p_j$ approaches $u$ along the line $x-y$.
          So we can just check the distance $d(z, p_1), d(z, p_2), \dots, d(z, p_k)$? That's $O(k)$ queries per off-node. Too slow ($O(N^2)$).
          We need $O(\log N)$ or $O(1)$.
          Binary Search!
          Check midpoint $m$ of current range on the diameter.
          Is $d(z, m) > d(z, x_{left})$? etc.
          Comparing distances tells us which side the projection lies.
          Actually, simply comparing $d(z, a)$ and $d(z, b)$ helps?
          Triangle inequality: $d(z, a) + d(a, b) \ge d(z, b)$.
          If $d(z, a) < d(z, b)$, then $z$ is "closer" to $a$ (on the left side relative to $a, b$).
          Wait, $a, b$ are endpoints of a subsegment of the diameter.
          $a=p_i, b=p_{i+m}$.
          If projection is in left half $[a, m]$, then $d(z, a) + d(z, m) \ge d(a, m)$? No.
          $d(z, b) = d(z, m) + d(m, b)$. (By definition of projection $m$ being closer to $z$ than $b$).
          So $d(z, b) > d(z, m) + d(m, b) = d(z, b)$? No.
          Let's check carefully.
          Path $a \dots m \dots b$. Projection $u$.
          If $u$ is to the left of midpoint (closer to $a$): $d(z, a) + d(a, b) = d(z, b)$ is FALSE.
          The projection $u$ minimizes $d(z, x)$ for $x$ on line.
          If $u$ is closer to $a$, then $d(z, a) < d(z, m)$.
          Actually, we can just compare $d(z, p_{left})$ and $d(z, p_{right})$?
          Consider segment $p_L \dots p_R$. Midpoint $p_M$.
          If projection is in $[p_L, p_M]$, then $d(z, p_L) + d(p_L, p_M) \le d(z, p_R)$? No.
          It's much simpler: The function $f(t) = d(z, p_t)$ is convex on the line. It reaches minimum at projection.
          So if $d(z, p_M) \le d(z, p_L)$ and $d(z, p_M) \le d(z, p_R)$, we are close.
          Actually, we can check the sign of $d(z, p_L) - d(z, p_M)$.
          Wait, we need the exact location.
          Actually, the condition $d(z, x) + d(x, y) - d(z, y)$ determines position?
          The formula $d(z, u) = (d(z, x) + d(z, y) - d(x, y))/2$ gives distance $D_{off}$ from the line.
          Wait, $d(x, y)$ is $L$ (constant for whole diameter).
          So $D_{off} = (d(z, x) + d(z, y) - L)/2$.
          Does knowing $D_{off}$ help with BS? No.
          Wait, $u$ is the projection. $u$ lies on the segment.
          Is there a condition using only $d(z, x)$ and $d(z, y)$?
          On the path $x-y$, $d(z, p_k) = \min(d(z, x)+k \cdot \Delta, d(z, y)+(len-k)\cdot \Delta)$?
          No, edge weights vary.
          
          Let's step back. Do we REALLY need the exact path first?
          Maybe we can grow a "tree of vertices" dynamically.
          Algorithm:
          1. Find a random vertex $r$.
          2. Sort all vertices by distance from $r$: $v_1=r, v_2, \dots, v_n$.
          3. For each $v_i$, find its "true parent" $p(v_i)$.
             $p(v_i)$ is a neighbor of $v_i$ such that $d(r, p(v_i)) < d(r, v_i)$ and the edge $(p(v_i), v_i)$ exists.
             If we just pick the neighbor with smallest $d(r, \cdot)$ among neighbors, do we risk error?
             Yes, discussed earlier (siblings).
             
          Let's try to combine the two ideas.
          Using the random vertex root $r$ and sorting vertices, we reduce the search for parents.
          The problem is the ambiguity of "closest available neighbor".
          However, notice:
          If we simply take ANY neighbor $u$ that appears closer to $r$ than $v_i$, and we tentatively assume it's the parent, we generate a tree.
          We assign weight $W = d(r, v_i) - d(r, u)$.
          Then for ALL future descendants of $v_i$, the distances $d(r, \cdot)$ will be consistent with THIS new path?
          Wait, no. If we picked the wrong parent, the distance to root will be computed wrongly for $v_i$'s subtree.
          But we computed distances FROM the ROOT using queries. Those are CONSTANT facts.
          If we change the structure, the path weights MUST ADD UP TO $d(r, \cdot)$.
          If we picked a wrong parent $u$, we implicitly claimed there is a path of weight $W$ connecting $u$ to $v_i$.
          But since $u$ is a neighbor of $v_i$ in the original tree, the edge $uv_i$ HAS weight $W$.
          So if $u$ is indeed a neighbor in the original graph, we never made a mistake about the weight of the edge $u \to v_i$.
          We just decided the path goes THROUGH $u$.
          But $v_i$ might have a child $c$ such that the path to root goes $v_i \to c$.
          But if $u$ is a neighbor and $u$ is closer to $r$, then $u$ is on the path to $r$. $v_i$ is a child of $u$.
          If we mistakenly pick $u'$ (another neighbor closer to $r$) as parent.
          Wait, can $v_i$ have TWO neighbors closer to $r$?
          Only if $r$ is not the root, or if the tree branches back?
          But $r$ is a specific vertex. In a tree, the path to $r$ is unique.
          So the set of neighbors of $v_i$ closer to $r$ is a subset of the ancestors.
          Exactly ONE of them is the direct parent $p(v_i)$.
          The others are "ancestors of the parent", i.e., $gpp(v_i), pp(v_i) \dots$.
          So, we want to pick the DIRECT PARENT.
          We need to pick the neighbor that is IMMEDIATELY adjacent on the path to $r$.
          Let the neighbors closer to $r$ be $u_1, u_2, \dots, u_m$.
          These are $p(v_i)$, and some nodes in the path to $r$ (above $p(v_i)$).
          Let the distances be $d(r, u_1) < d(r, u_2) < \dots < d(r, u_m)$.
          Note that $d(r, u_{next}) = d(r, u_{prev}) + w(\text{edges between})$.
          So the gap between consecutive candidates in the sorted list reflects the edge weight $w(p, c)$.
          Specifically, if $u_k$ is the parent $p(v_i)$, then $d(r, u_k)$ is minimal among those "below" the parent?
          No, $p(v_i)$ is the closest ancestor.
          Wait, the neighbors of $v_i$ in the tree are: $p(v_i)$ (parent), and $c_1, c_2 \dots$ (children).
          Children have $d(r, c_j) > d(r, v_i)$.
          Parent has $d(r, p(v_i)) < d(r, v_i)$.
          Are there ANY other neighbors of $v_i$ closer to $r$?
          NO. All other neighbors are children (further from $r$).
          So $p(v_i)$ is the UNIQUE neighbor of $v_i$ that has $d(r, p(v_i)) < d(r, v_i)$.
          WAIT. This contradicts my earlier worry!
          Earlier I worried about siblings. A sibling $s$ shares a parent.
          $d(r, s) < d(r, v_i)$ is possible.
          If $s$ is a neighbor of $v_i$... then $s$ is a CHILD?
          No, siblings are NOT neighbors.
          Neighbors of $v_i$ are ONLY $p(v_i)$ and children.
          Among neighbors, only one is closer to $r$: the parent $p(v_i)$.
          Wait. Let me double check this fundamental graph theory fact.
          Let $G$ be a tree rooted at $r$.
          Fix a node $u$. Its neighbors are $Parent(u)$ and $Children(u)$.
          By definition, edge $(u, v)$ exists.
          If $v$ is a child, path $r \to u \to v$. So $d(r, v) > d(r, u)$.
          If $v$ is parent, path $r \to \dots \to v \to u$. So $d(r, v) < d(r, u)$.
          Is it possible for a node to be BOTH a neighbor and a non-descendant/non-parent with smaller distance?
          No, a neighbor is EITHER parent OR child.
          Wait. This applies to the TREE STRUCTURE.
          The question is: Does the graph allow edges that are NOT parent-child relations in the rooted view?
          Since the graph IS a tree rooted at $r$, every edge connects a node to its parent or child.
          Therefore, for any $u$, the set of neighbors is partitioned into $\{p(u)\}$ and $\{children\}$.
          Thus, among ALL neighbors of $u$, there is EXACTLY ONE neighbor $p(u)$ with $d(r, p(u)) < d(r, u)$.
          All other neighbors have $d(r, neighbor) > d(r, u)$.
          This simplifies EVERYTHING immensely.
          
          So the algorithm is extremely simple:
          1. Fix root $r$ (say vertex 1).
          2. Compute distances $D[i] = d(r, i)$ for all $i$. (Needs $N-1$ queries? Can we optimize?)
             Wait, computing ALL pairwise distances is impossible.
             We can't run BFS/DFS.
             But wait, we only need to check $d(r, v)$ for all $v$.
             Is it really $N-1$ queries to check ALL nodes against $r$?
             Yes, if we query every $v$ separately.
             Total $N-1$ queries.
             Wait, if we just query every pair, it's too slow.
             BUT, if we can efficiently find the farthest node?
             
             Re-evaluating the complexity.
             Query limit $Z \approx 5N$.
             We need $d(r, i)$ for all $i$.
             If we just query all $N-1$ pairs $(1, i)$, we spend $O(N)$ queries. That fits in budget!
             Wait. $N \le 10^5$. Queries allowed $5 \cdot 10^5$.
             $N-1$ is well within limits.
             Wait, is it really that easy?
             Yes. Just query $(1, i)$ for all $i$. Store results.
             Then sort nodes by distance $D[i]$.
             Iterate sorted nodes. For node $u$, look at neighbors.
             Identify parent as unique neighbor with $d(r, p(u)) < d(r, u)$.
             Wait. We need to CHECK neighbors.
             How do we check neighbors without knowing the graph?
             We can't query "is $x$ a neighbor?" without knowing the full graph.
             Ah, the "check neighbors" part is where I got stuck before.
             We need to iterate over the actual neighbors of $u$.
             But we don't have the adjacency list.
             So we CANNOT identify the parent by inspecting the neighbors.
             
             We need to FIND the parent among the set of ALL nodes $\{v : D[v] < D[u]\}$.
             The parent is the one with $w = d(r, u) - d(r, p(u))$? No, we calculate $w$.
             The parent is the one $p(u)$ such that edge $(p(u), u)$ exists.
             If $(p(u), u)$ exists, then $D[u] = D[p(u)] + w$.
             Since we don't know which edges exist, this is hard.
             
             Wait. We found the parent is the *only* neighbor closer to root.
             But if we don't know the neighbors, how can we distinguish it from other closer nodes (like grandparent)?
             Grandparent is ALSO closer to root.
             Grandparent is NOT a neighbor. Parent IS a neighbor.
             So we cannot simply check "which closer node is the parent" because they all look the same (closer to root).
             Unless... we can verify if $p(u)$ is a neighbor of $u$.
             If we had a way to check edges...
             
             Is there a shortcut?
             Let's use the random pair idea again.
             If we have a random vertex $x$, we can estimate its distance to a far node $y$.
             Suppose we run BFS on the *reconstructed* tree.
             Initially we have nothing.
             Let's add vertices one by one.
             Add first vertex $v_1$ (random).
             Then add $v_2$. Query $(v_1, v_2)$.
             Check edge existence.
             If we can't check edges...
             
             Let's go back to the Diameter approach.
             If we find the diameter path $x \to y$, we have a line of vertices.
             This gives us a reference frame.
             We can locate any vertex $z$ relative to this line.
             If $z$ is connected to a vertex $p$ on the line, then $z$ is a child of $p$.
             But what if $p$ is already a child of some other node on the line?
             Then we just attach $z$ to the existing chain.
             Wait. The diameter path itself consists of edges.
             We need to find ALL edges.
             If we have the full path $x \to \dots \to y$, we have recovered $k$ edges.
             What about the other nodes?
             Every node $z$ attaches to this path at exactly one point $proj(z)$.
             If we find $proj(z)$ and confirm the edge $(proj(z), z)$ has weight $d(proj(z), z)$, we are done!
             So, we just need to find the diameter path.
             If the diameter path contains ALL vertices, perfect.
             If it contains only a SUBSET, then the vertices NOT on the diameter must be attached to the diameter path.
             The attachment point $proj(z)$ is on the diameter path.
             But wait. How does $proj(z)$ relate to the edges?
             $proj(z)$ is a node. $z$ is a node. We connect $z$ to $proj(z)$?
             YES. Since $z$ is not on the diameter path, its unique path to the diameter path enters at $proj(z)$.
             Since the graph is a tree, there is only one path from $z$ to the diameter.
             The FIRST vertex on the diameter encountered is $proj(z)$.
             So the edge connecting $z$ to the rest of the tree (excluding the part already on diameter) is $(proj(z), z)$.
             And its weight is $d(z, proj(z))$.
             Wait. This is valid ONLY IF the diameter path $x-y$ was already fully resolved.
             If we only identified a set of vertices forming the diameter (by ordering them by distance from ends), we have the SEQUENCE of vertices.
             So we effectively have the diameter edges $(p_1, p_2), (p_2, p_3) \dots$
             The remaining nodes $z \notin \{p_i\}$ are all leaves attached to some $p_j$.
             Wait. Are they LEAVES?
             If $z$ attaches to $p_j$, then $z$ becomes a child of $p_j$.
             But $z$ could have children of its own!
             So $z$ is part of a subtree hanging off the diameter.
             If we handle $z$ now, we add $z$ to the tree.
             Can we apply the same logic recursively?
             We found a diameter of the ORIGINAL tree.
             If we remove the diameter path from the tree (remove edges, but keep the path as a spine?), the remaining components are attached at the path nodes.
             For a component attached at $p_j$, let's pick a representative.
             Or, simpler:
             Just repeat the "Project onto current tree" procedure.
             We need to pick a reference path/tree that is "dense".
             The diameter is usually quite dense.
             Algorithm:
             1. Randomly pick two vertices $x, y$.
             2. If we could ensure the path $x-y$ is long, good.
             3. Instead of assuming diameter, just pick $x$ and $y$ such that $d(x, y)$ is MAXIMIZED?
                That's expensive.
             4. Heuristic:
                Find $x$ (random). Find farthest $u$. Find farthest from $u$, call $v$.
                Path $u-v$ is a candidate.
                Project ALL vertices onto $u-v$.
                Some vertices fall ONTO $u-v$ (dist to segment is 0).
                Some vertices fall OFF.
                Vertices falling ONTO are covered. Vertices falling OFF form subtrees.
                To solve the off-subtrees, pick a random one, treat it as the NEW root/diameter base?
                No, simply recurse.
                For the set of uncovered vertices, pick one, find farthest, etc?
                
                Wait, let's refine the heuristic.
                Actually, if we just have one diameter path, do we cover enough?
                Maybe not enough.
                However, we can REPEAT.
                But finding a diameter takes $O(N)$ queries. Doing it $N$ times is $O(N^2)$.
                We need fewer repeats.
                How many? Expected constant?
                Yes, a diameter path is quite special. It likely hits the center of the tree.
                Branches hanging off a diameter path are short?
                Not necessarily. But statistically yes.
                
             Let's implement the recursive heuristic.
             Function `solve(uncovered_nodes)`:
               If empty return.
               Pick $x$ randomly from uncovered.
               Find $y$ maximizing $d(x, y)$ among uncovered? No, globally.
               Wait, distance oracle is global.
               Finding farthest takes $N$ queries.
               We should reuse distances.
               
               Better:
               We have a set of nodes covered by the current diameter. Let's just keep track of the diameter path (as a sorted list of vertices).
               Wait, we don't have the diameter edges sorted.
               Actually, we can find the diameter by scanning.
               Scan $i = 1 \dots N$: update best path if $d(best\_end1, i) > current\_diameter$?
               Wait, we need to be careful not to query redundant things.
               If we scan, for each node, we need distances to diameter endpoints?
               Yes, $d(x, u)$ for each $x$.
               
             Revised Plan:
             1. Find a random vertex $r$.
             2. Run "BFS-like" query scan to find the farthest node $u$ from $r$.
                We need to avoid querying every single node individually if possible.
                But $d(r, v)$ for all $v$ is exactly what we wanted to compute anyway!
                And we only pay $N-1$ queries for it.
                So:
                Query $d(r, v)$ for all $v$. (Cost: $N-1$ queries? Or $N$ queries including $r$?)
                Sort vertices by $d(r, v)$.
                Let sorted list be $L$. $L_1=r$.
                Candidate diameter starts at $r$ and goes to some $v$.
                Wait, if $L$ is sorted by distance from $r$, does it contain the diameter endpoints?
                Usually the last element $L_N$ is an endpoint of a diameter starting at $r$?
                Yes, $u = L_N$ is a valid endpoint of a diameter starting at $r$. (Maximizing $d(r, u)$).
                Let's pick $u = L_N$.
                 Next, we need the OTHER endpoint of the diameter.
                It is farthest from $u$.
                So we find $v$ maximizing $d(u, v)$ among all vertices.
                We need $d(u, v)$ for all $v$.
                (Wait, we can't just query $N$ more times? Total $2N$).
                Budget allows $5N$. So finding diameter $r \to u \to v$ uses $\approx 2N$.
                Remaining budget $3N$. We have a lot.
                The path is $u \to \dots \to v$.
                The vertices on this path are those $z$ with $d(z, u) + d(z, v) = d(u, v)$.
                Let this set be $PathNodes$.
                We need to sort these to reconstruct the edges.
                The projection of $z$ onto $u-v$ gives the distance $d(z, u)$.
                Wait, if $z$ is on the diameter, $d(z, u) + d(z, v) = d(u, v)$ means it is a geodesic.
                Since we know $d(u, v)$ and $d(u, z)$ and $d(v, z)$, we know $z$ is between.
                BUT we don't know the ORDER.
                However, we know $z$ is on the line between $u$ and $v$.
                $d(u, z)$ determines position.
                If $d(u, z) = d(u, p_k)$ for some path node $p_k$, then $z$ is $p_k$.
                So, sort $PathNodes$ by $d(u, z)$.
                This sorts them along the diameter.
                Edges are $(p_k, p_{k+1})$ with weight $d(u, p_{k+1}) - d(u, p_k)$?
                Wait, this calculates EDGE WEIGHTS based on distance from $u$.
                Yes! $w(p_k, p_{k+1}) = d(u, p_{k+1}) - d(u, p_k)$.
                Is this ALWAYS the edge weight?
                If $p_k, p_{k+1}$ are adjacent vertices on the path, then yes, since path is unique.
                So this recovers the FULL diameter and its weights. Cost $\approx 2N$.
                
                Now, what about nodes NOT on diameter?
                For each node $z \notin PathNodes$:
                  Compute $u_z = (d(z, u) + d(z, v) - d(u, v))/2$. This is the distance to the line.
                  Wait, $u_z$ is just a number (scalar). Where is the attachment point?
                  We need to find $p_k$ on the sorted path such that $p_k$ is the parent.
                  Condition: $d(z, p_k) = u_z + d(z, u_{path})$? No.
                  $d(z, p_k) = u_z + d(path, p_k)$?
                  Let's verify.
                  $z$ connects to line at $p_k$.
                  $d(z, u) = d(z, p_k) + d(p_k, u)$.
                  $d(z, v) = d(z, p_k) + d(p_k, v)$.
                  Sum: $d(z, u) + d(z, v) = 2 d(z, p_k) + d(u, v)$.
                  So $d(z, p_k) = (d(z, u) + d(z, v) - d(u, v))/2$.
                  Let this computed value be $H_z$.
                  This is the length of the "hanging edge" (actually the leg).
                  So we need to find $p_k$ such that $d(z, p_k) = H_z$.
                  Since $p_k$ is on the line, $d(z, p_k) = \sqrt{H_z^2 + d(path, p_k)^2}$? No, tree metric is not Euclidean.
                  It's just distance.
                  However, $d(z, p_k) = d(z, \text{projection}) + d(\text{projection}, p_k)$?
                  Wait. $p_k$ IS the projection. So $d(z, p_k) = H_z$.
                  So we just need to find $p_k$ on the path such that $d(z, p_k) = H_z$.
                  But there could be multiple nodes on the path.
                  Does $d(z, p_k)$ increase monotonically as we move away from projection?
                  Yes, in a tree. Moving away from projection increases distance from $z$.
                  So we have a target value $H_z$. We have a list of values $val_k = d(z, p_k)$ for $p_k \in PathNodes$.
                  We want to find the index $k$ such that $val_k = H_z$ (or rather, the vertex on the path).
                  Note: it is possible that the "projection" lands on an EDGE $(p_k, p_{k+1})$ exactly in the middle, or at the midpoint.
                  Then $z$ attaches to the edge, not a vertex.
                  In that case, we have $z$ connected to both $p_k$ and $p_{k+1}$?
                  No, if it attaches to edge midpoint, we introduce a virtual node.
                  Or, check if $val_k + val_{k+1} = d(p_k, p_{k+1})$? No.
                  If $val_k = val_{k+1}$, it is equidistant (projection is on edge).
                  Then $z$ attaches to BOTH $p_k$ and $p_{k+1}$ with weight $val_k$?
                  Wait, if it attaches to an edge, $p_k - z - p_{k+1}$ is not a path. $z$ is off the edge.
                  Wait, the edge has weight $W = d(p_k, p_{k+1})$.
                  If projection is internal, let dist to $p_k$ be $d_1$, to $p_{k+1}$ be $d_2$.
                  Then $d(z, p_k) = \text{dist}(z, p_k)$.
                  Triangle inequality: $d(z, p_k) \le d(z, m) + d(m, p_k)$.
                  If projection is $m$ on edge.
                  Then $d(z, p_k) = d(z, m) + d(m, p_k)$.
                  And $d(z, p_{k+1}) = d(z, m) + d(m, p_{k+1})$.
                  So $d(z, p_k) + d(z, p_{k+1}) = 2 d(z, m) + W = 2 H_z + W$.
                  This is consistent.
                  But we assumed $z$ is attached to the diameter.
                  If it attaches to an edge, it's not attached to a vertex.
                  Wait, does $d(z, p_k)$ help?
                  If we simply add $z$ as a neighbor to $p_k$ with weight $d(z, p_k)$, are we safe?
                  If $z$ projects to edge $(p_k, p_{k+1})$, adding it to $p_k$ creates a cycle or mess.
                  However, observe:
                  $d(z, p_k) = (d(z, u) + d(z, v) - d(u, v))/2 + d(z, p_k)$ ?? No.
                  Formula was $H_z = d(z, p_k)$.
                  So if $z$ projects to $p_k$, then $d(z, p_k) = H_z$.
                  If $z$ projects to edge, $d(z, p_k) \ge H_z$.
                  Wait, if projection is edge, $H_z$ (dist to line) is min dist.
                  $H_z$ is dist to the infinite line? No, segment.
                  So $d(z, p_k)$ for the true projection vertex (if it was a vertex) would be $H_z$.
                  Since we have discrete vertices, let's look at the sequence of distances $D(p_k) = d(z, p_k)$.
                  It decreases then increases. Min is at projection.
                  So we can Find-Minimum!
                  Use binary search on the sorted path $P$.
                  Find the segment $[p_k, p_{k+1}]$ that minimizes $d(z, x)$.
                  Actually, since $p_k$ are sorted by $d(u, p_k)$, they are ordered on the diameter.
                  The function $d(z, \cdot)$ is convex.
                  Find minimum $M$.
                  If $M$ occurs at a vertex $p_k$, then $z$ connects to $p_k$ with weight $d(z, p_k)$.
                  If minimum occurs at an edge $[p_k, p_{k+1}]$ (value $D(p_k) == D(p_{k+1})$), then $z$ attaches to the edge.
                  Then we have two possibilities: connect to $p_k$ or $p_{k+1}$?
                  If we connect to $p_k$ with weight $W$, we imply $w(p_k, z) = W$.
                  But true connection is to edge.
                  Does connecting to $p_k$ work?
                  No, it would put $z$ at a wrong spot relative to $p_{k+1}$.
                  However, note that if it attaches to edge, $d(z, p_k) = d(z, p_{k+1})$.
                  Also $d(z, p_k) = H_z + d(mid, p_k)$.
                  Is there a vertex $x$ such that we can attach it?
                  Actually, maybe we can just append $z$ to the END of the diameter?
                  No, it's somewhere in the middle.
                  Wait, we don't have to find the projection exactly to the edge midpoint.
                  We can add $z$ to the tree.
                  We want to minimize queries.
                  If $z$ projects to edge $(p_k, p_{k+1})$, then $z$ is adjacent to the edge? No, connected to it.
                  Is it possible that $z$ attaches to the edge? Yes.
                  BUT, in the solution format, we output edges $(u, v)$.
                  If $z$ connects to the edge $(p_k, p_{k+1})$, we can create a NEW node? No, we only have $n$ nodes.
                  We must connect $z$ to an EXISTING node.
                  If we connect $z$ to $p_k$, we force a cycle or modify the topology incorrectly UNLESS the topology ALLOWS it.
                  Actually, if $z$ connects to the edge, the path from $z$ to $p_{k+1}$ is $z - \dots - p_k - p_{k+1}$.
                  If we just output $(z, p_k)$, we skip the node on the edge.
                  But there are NO intermediate nodes between $p_k, p_{k+1}$.
                  So $z$ must be connected to $p_k$ OR $p_{k+1}$?
                  If it connects to the edge (internally), we have a violation.
                  Wait. Does a tree allow a node to connect internally to an edge?
                  That edge $(p_k, p_{k+1})$ must have a weight $W$.
                  If $z$ connects to it, we must break that edge? No, edges are primitive.
                  This means my assumption "project to a vertex" was restrictive.
                  But we established $d(z, p_k)$ decreases then increases.
                  If the min is on an edge, then $d(z, p_k) > H_z$.
                  Wait, does this matter?
                  If $d(z, p_k) = d(z, p_{k+1})$, and this is the minimum.
                  Then $z$ is connected to both? No.
                  In a tree, $z$ has one path to the set $\{p_k, p_{k+1}\}$.
                  If it enters the edge, it hits $p_k$ or $p_{k+1}$? No.
                  The edge is $(p_k, p_{k+1})$. The path from $z$ hits the edge AT a point.
                  Since nodes are discrete, the "first hit" is a node.
                  So $z$ is adjacent to a node $p_k$ or $p_{k+1}$?
                  Not necessarily. It could hit the EDGE.
                  Wait. If $z$ is adjacent to an edge, does that mean it shares a node?
                  In graph terms, "adjacent to edge" usually means sharing a vertex.
                  But "connected to an edge" means entering the segment.
                  If the segment $(p_k, p_{k+1})$ represents the ONLY path between $p_k$ and $p_{k+1}$, then $z$ must attach to it.
                  But attaching to the interior of an edge means splitting the edge.
                  Splitting the edge means inserting a node $x$ on the edge.
                  BUT we are GIVEN $n$ nodes. We cannot invent new nodes.
                  This implies the configuration "node attached to edge midpoint" is IMPOSSIBLE in a valid labeled tree of $n$ nodes?
                  Wait, why?
                  Consider a star graph. Center 1, leaves 2, 3. Edge (1, 2) weight 2. Edge (1, 3) weight 2.
                  Node $z$ attached to edge (1, 2) with weight 1?
                  Then $1-z-2$. Dist 1-2 is 2.
                  This creates a path $1-z-2$ instead of $1-2$.
                  But $1-2$ was an edge in the input tree.
                  So this situation can't happen.
                  The input tree is simple. Edges connect nodes directly.
                  You cannot "split" an edge.
                  Therefore, every node $z$ not on the diameter path MUST attach to the diameter path at a NODE.
                  So $proj(z)$ MUST be a vertex.
                  So the case "min on edge" never happens?
                  Wait. $d(z, p_k)$ is convex. Minima on a convex function on discrete domain...
                  If $d(z, p_k) == d(z, p_{k+1})$, is it possible?
                  Yes. Symmetry.
                  e.g. $p_k - 5 - p_{k+1}$. $z$ is above the midpoint. $d(z, p_k) = d(z, p_{k+1})$.
                  Then $z$ is NOT attached to an edge midpoint (because that would split the edge).
                  $z$ must be attached to $p_k$ (weight 5)? Or $p_{k+1}$ (weight 5)?
                  If attached to $p_k$, path is $z - p_k - p_{k+1}$. $d(z, p_{k+1}) = 5+5=10$. But calculated $d(z, p_{k+1})$ from data is 5. Contradiction.
                  So symmetry means we attach to NEITHER?
                  Wait. If $z$ is attached to $p_k$ with weight $w$, and $d(z, p_{k+1})$ is derived as distance in original tree.
                  Original tree has edge $(p_k, p_{k+1})$. $z$ is somewhere else.
                  Wait, $z$ is a vertex in the graph.
                  $p_k, p_{k+1}$ are neighbors on the diameter.
                  So $p_k - p_{k+1}$ is an edge.
                  $z$ is a node attached to this edge? No, $z$ is a node attached to the *structure*.
                  If $z$ is attached to $p_k$, it shares edge $(z, p_k)$.
                  It's physically impossible for $z$ to be "above the edge midpoint" and symmetric unless the tree contains a diamond shape? No, tree.
                  Wait. $d(z, p_k)$ is distance in the tree.
                  $d(z, p_{k+1})$ is distance.
                  If $z$ is attached to $p_k$ with weight $w$, then $d(z, p_{k+1}) = w + d(p_k, p_{k+1})$.
                  Then $d(z, p_k) = w$.
                  So $d(z, p_{k+1}) = d(z, p_k) + d(p_k, p_{k+1})$.
                  This implies $d(z, p_k) < d(z, p_{k+1})$ STRICTLY (since edge weight $> 0$).
                  So the minimum is STRICTLY at the parent.
                  So $d(z, p_k) < d(z, p_{k+1})$ is required.
                  So if we find $p_k$ minimizing $d(z, p_j)$ over all $j$, we just need to ensure we picked the right one.
                  The function is strictly convex (V-shape), so the minimum is unique.
                  Wait, is it strictly convex?
                  $d(z, x)$ behaves like distance on a tree. Yes.
                  So just finding the unique minimizer $p_k$ on the diameter path is sufficient.
                  
             So, the refined plan is solid.
             
             Complexity Analysis:
             1. Query all $d(r, i)$: $N-1$ queries.
             2. Find $u$ (farthest from $r$): Already done, it's the node with max $d(r, \cdot)$.
             3. Query all $d(u, i)$: $N-1$ queries.
             4. Find $v$ (farthest from $u$): Node with max $d(u, \cdot)$.
             Total $2N$ queries.
             This finds diameter endpoints $u, v$.
             Cost of verifying diameter length? 0, we just use lengths.
             Wait, step 4 requires checking all $N$.
             We did check all $d(r, i)$ and all $d(u, i)$.
             So yes, 2 scans.
             
             Now, handling OFF-path nodes.
             For each $z \notin PathNodes$:
               Need $d(z, u)$ and $d(z, v)$.
               We have these!
               Wait.
               When finding $u$, we queried $d(r, \cdot)$. This gave us distances to $u$'s start.
               Wait, no.
               $d(r, x)$ is distance from $r$.
               We need distances to $u$ and $v$.
               We queried $d(u, x)$ in step 3. So we have $d(u, x)$.
               We need $d(v, x)$ for all $x$.
               We can compute $d(v, x)$?
               If $x$ is on diameter, $d(v, x) = D_{diam} - d(u, x)$.
               If $x$ is NOT on diameter, we need to query $d(v, x)$.
               Number of off-diameter nodes can be up to $N$.
               So we might need $N$ more queries.
               Total so far: $2N + N = 3N$.
               For each such $z$, we query $d(v, z)$.
               Then compute position on diameter.
               Wait.
               We need $d(v, z)$ to verify if $z$ is on diameter?
               $z$ is on diameter iff $d(u, z) + d(z, v) = d(u, v)$.
               So we query $d(v, z)$ only if it's NOT verified.
               Wait, we need $d(v, z)$ for ALL $z$ to perform the binary search on the diameter.
               Yes, we need $d(z, v)$ to evaluate $d(z, p_k)$.
               Wait, do we really need $d(z, p_k)$?
               We need to find $p_k$ minimizing $d(z, p_k)$.
               To evaluate $d(z, p_k)$, we need to calculate distance from $z$ to $p_k$.
               We have $d(u, p_k)$ (computed on diameter). We have $d(u, z)$ (from step 3).
               If we knew $d(z, v)$...
               Wait. $d(z, p_k)$ is unknown.
               Can we express $d(z, p_k)$ in terms of knowns?
               $d(z, p_k) = \sqrt{...}$ No.
               But on a tree, $d(a, b) = \min(d(a, x)+d(b, x)-d(x, \cdot))$.
               We know $d(u, z)$. We know $d(u, p_k)$ (along diameter).
               If $z$ attaches to $p_k$, then $d(z, p_k)$ is the answer.
               Is it $d(u, z) + d(u, p_k)$? No.
               Wait. $d(u, z)$ path goes $u \to p_k \to z$.
               Is $u \to p_k$ part of the path?
               If $p_k$ is the attachment point of $z$ to the diameter, then yes, the path from $u$ to $z$ goes through $p_k$.
               Because $z$ is outside diameter, attached at $p_k$.
               $u$ is on diameter. The path $u \to z$ enters the subtree at $p_k$ and leaves at $u$?
               Wait, $u$ is an endpoint. So $u \to z$ passes through the whole diameter segment up to $p_k$.
               Yes! $d(u, z) = d(u, p_k) + d(p_k, z)$.
               SO, $d(p_k, z) = d(u, z) - d(u, p_k)$.
               This is AMAZING.
               It means we don't need $d(v, z)$!
               We just need $d(u, z)$ (already computed) and $d(u, p_k)$ (computed on diameter).
               Then the edge weight $w(p_k, z)$ is exactly $d(u, z) - d(u, p_k)$.
               Wait. This relies on $p_k$ being the attachment point.
               But we don't know $p_k$ yet! We are trying to FIND it.
               So for a FIXED $z$, $d(z, \text{attachment}) = d(u, z) - d(u, \text{attachment})$.
               But we want to find the attachment point.
               Let $f(x) = d(u, z) - d(u, x)$ for $x \in PathNodes$.
               We want to maximize $f(x)$?
               Since $d(u, x)$ is monotonic on diameter (increasing as we go from $u$ to $v$).
               $d(u, z) - d(u, x)$ is MAXIMIZED when $d(u, x)$ is MINIMIZED.
               Minimizing $d(u, x)$ corresponds to picking $x$ CLOSEST to $u$.
               Is the attachment point the one closest to $u$?
               No. The attachment point is the projection.
               Is the distance $d(u, \text{proj})$ minimized?
               Wait. $d(u, z)$ is fixed.
               $d(u, z) = d(u, \text{proj}) + d(\text{proj}, z)$.
               So $d(u, \text{proj}) = d(u, z) - d(\text{proj}, z)$.
               So maximizing $d(\text{proj}, z)$ is equivalent to MINIMIZING $d(u, \text{proj})$?
               No. We want to find $x$ such that $x=\text{proj}$.
               We have the equation $d(u, z) = d(u, x) + d(x, z)$.
               Rearranged: $d(x, z) = d(u, z) - d(u, x)$.
               We want to find $x$ that satisfies this relation with $d(x, z) \ge 1$?
               But this doesn't guarantee $x$ is the projection.
               Any $x$ satisfies $d(x, z) \le d(u, z) - d(u, x)$? No.
               $d(u, z) \le d(u, x) + d(x, z)$ (Triangle Inequality).
               So $d(x, z) \ge d(u, z) - d(u, x)$ is always true.
               Equality holds iff $u-x-z$ is a path (collinear).
               So $x$ is the correct parent iff $d(u, x) + d(x, z) = d(u, z)$ is FALSE?
               No.
               If $x$ is the parent, $u$ is on diameter, $z$ attached at $x$.
               The path $u \to z$ goes through $x$.
               So $u \to x \to z$ IS a path.
               So $d(u, x) + d(x, z) = d(u, z)$.
               So $d(x, z) = d(u, z) - d(u, x)$ is the CORRECT relation.
               However, for other nodes $x$ on the diameter (further from $u$), does this hold?
               Let $y$ be another node on diameter.
               Then $u \to y$ path contains $u \to x$ path.
               $y$ is further from $u$.
               $d(u, y) = d(u, x) + d(x, y)$.
               Check relation: $d(u, y) - d(y, z)$?
               We need $d(y, z)$.
               Is $d(y, z) = d(u, z) - d(u, y)$?
               $d(u, y) = d(u, x) + d(x, y)$.
               RHS: $d(u, z) - d(u, x) - d(x, y)$.
               Substitute $d(u, z) = d(u, x) + d(x, z)$.
               RHS $= d(x, z) - d(x, y)$.
               Is $d(y, z) = d(x, z) - d(x, y)$?
               This implies $d(x, z) = d(x, y) + d(y, z)$.
               This means $x \to y \to z$ is a path.
               But $z$ is attached at $x$. $y$ is further down diameter (away from $x$ in direction $u$? No, away from $u$ means further from $u$).
               If $x$ is parent of $z$, and $y$ is further on diameter from $u$ than $x$, then $y$ is in the subtree attached to $x$ (downwards).
               Then $z \to x \to y$ is path.
               So $d(z, y) = d(z, x) + d(x, y)$.
               So $d(x, y) + d(y, z) = d(x, y) + d(z, x) + d(x, y)$? No.
               $d(y, z) = d(y, x) + d(x, z)$.
               So $d(y, z) - d(x, z) = d(y, x) > 0$.
               So $d(y, z) > d(x, z)$.
               So the distance $d(x, z)$ is strictly DECREASED as we move AWAY from $x$ towards $u$.
               Wait.
               Let's check values.
               At $x$ (parent): $val(x) = d(u, z) - d(u, x)$. This equals $d(x, z)$. Positive.
               At $y$ (further from $u$ than $x$):
               Wait. $y$ is further from $u$. So $d(u, y) > d(u, x)$.
               Let's rewrite $val(y) = d(u, z) - d(u, y)$.
               Since $d(u, y) > d(u, x)$, $val(y) < val(x)$.
               But is $val(y)$ related to $d(y, z)$?
               $val(y) = d(x, z) - d(x, y)$.
               So $val(y) = d(x, z) - d(x, y)$.
               If $y$ is below $x$, $z \to x \to y$ is path.
               $d(y, z) = d(y, x) + d(x, z) = d(x, y) + d(x, z)$.
               So $val(y) = d(x, z) - d(x, y) \neq d(y, z)$.
               So for any candidate $y$ on diameter (that is NOT the parent), $d(y, z)$ is NOT equal to $d(u, z) - d(u, y)$.
               Instead $d(y, z) > d(u, z) - d(u, y)$?
               $d(y, z) = d(x, z) + d(x, y)$.
               $d(u, z) - d(u, y) = d(x, z) - d(x, y)$.
               Clearly $d(x, z) + d(x, y) \ge d(x, z) - d(x, y)$.
               So $d(y, z) \ge val(y)$.
               Wait, we want to verify parentage.
               We want to find $x$ such that $d(x, z) = d(u, z) - d(u, x)$.
               For any $y$ (further from $u$), $d(y, z) = d(x, z) + d(x, y) = (d(u, z) - d(u, x)) + d(x, y) = d(u, z) - (d(u, x) - d(x, y)) = d(u, z) - d(u, y)$.
               WAIT.
               $d(u, y) = d(u, x) + d(x, y)$. So $d(x, y) = d(u, y) - d(u, x)$.
               So $d(y, z) = (d(u, z) - d(u, x)) + d(u, y) - d(u, x)$.
               $d(y, z) = d(u, z) + d(u, y) - 2d(u, x)$.
               And the "calculated" value $val(y) = d(u, z) - d(u, y)$.
               So is $d(y, z) = val(y)$?
               $d(u, z) + d(u, y) - 2d(u, x) = d(u, z) - d(u, y)$?
               $2d(u, y) = 2d(u, x) \implies d(u, y) = d(u, x)$.
               Since $y$ is distinct and further, this is impossible.
               So $d(y, z) \neq val(y)$.
               Therefore, the parent is the UNIQUE node $x$ on the diameter such that $d(x, z) = d(u, z) - d(u, x)$.
               Wait. How do we find it efficiently?
               We need to check $d(x, z) = d(u, z) - d(u, x)$ for $x \in PathNodes$?
               This requires querying $d(x, z)$ for each $x$?
               That is $O(Diameter \times OffNodes)$. Slow.
               We need $O(1)$ or $O(\log N)$ check.
               Can we deduce the parent without checking edges?
               The condition $d(x, z) = d(u, z) - d(u, x)$ rearranges to $d(u, x) + d(x, z) = d(u, z)$.
               This means $x$ lies on the path from $u$ to $z$.
               Since $z$ is off the diameter, the path from $u$ (endpoint) to $z$ must hit the diameter at the attachment point $p_k$ and then exit at $z$.
               So $x$ must be on the path $u \to z$.
               Since $x$ is also on diameter, $x$ must be the FIRST node on diameter encountered on the path $u \to z$.
               So $x$ is the intersection of path $u \to z$ and path $u \to v$ (the diameter).
               Intersection of two paths from a common source $u$ is the segment from $u$ to the branching point.
               The branching point is exactly $p_k$.
               So $p_k$ is the unique point on the diameter such that $z$ is reachable from $u$ by going through $p_k$?
               Wait. Every $z$ is reachable from $u$. The path enters the diameter at $u$ (start).
               Wait. The path is unique.
               $u \to z$ goes $u \to p_k \to z$ (since $z$ branches off at $p_k$).
               So $p_k$ lies on the segment $u-z$.
               But wait. The intersection of $u-z$ and diameter ($u \to v$) is the entire segment $u \to p_k$.
               So we can't distinguish which vertex on $u-p_k$ is the parent.
               BUT, $p_k$ is a NODE on the diameter.
               We want the node $p_k$ such that edge $(p_k, z)$ exists.
               This is the LAST vertex of the intersection segment before leaving the diameter?
               Yes, it is the furthest vertex from $u$ that lies on $u-z$.
               So we want to find $p_k$ such that $d(u, p_k)$ is MAXIMIZED subject to $p_k$ being on $u-z$.
               When is $p_k$ on $u-z$?
               When $d(u, p_k) + d(p_k, z) = d(u, z)$? No.
               When $d(u, p_k) + d(p_k, z) = d(u, z)$ is true.
               So we want to maximize $d(u, p_k)$ such that $d(u, p_k) + d(p_k, z) = d(u, z)$.
               Wait, this is circular.
               Actually, just checking if $p_k$ is on $u-z$ requires checking the path.
               However, notice the formula: $d(u, z) = d(u, p_k) + d(p_k, z)$.
               This holds iff $p_k$ is on $u-z$.
               Wait, does it hold for any $p_j$ "before" $p_k$?
               If $p_j$ is closer to $u$ than $p_k$. Then $p_k$ is after $p_j$ on diameter.
               Since $p_j$ is before the branch, the path $u \to z$ goes $u \to \dots \to p_j \to \dots \to p_k \to z$.
               So $p_j$ is definitely on $u-z$.
               So for ALL $p_j$ in the segment $u-p_k$, the equation holds.
               This includes the whole path!
               This is useless. It confirms $p_k$ is a descendant of $p_j$.
               
               BUT, we have another condition!
               $p_k$ is the parent of $z$.
               So $z$ is a DIRECT child of $p_k$.
               This implies $d(u, z) = d(u, p_k) + w(p_k, z)$.
               Also, for any descendant of $z$ (let's call it $desc$), $d(u, desc) = d(u, p_k) + w(p_k, z) + d(desc, z)$? No, dist along path.
               Basically, $p_k$ is the node such that $z$ connects directly.
               What distinguishes the DIRECT parent from an ancestor?
               Ancestor $p_j$ is connected by path of edges.
               Direct parent is connected by SINGLE edge.
               So $w(p_k, z)$ should be a valid edge weight.
               Wait, $w(p_k, z)$ is a variable.
               The distinction must come from the fact that $z$ is NOT a child of $p_j$ ($j < k$).
               This is true for all $j$.
               
               Let's look at the "Projection" idea again.
               $d(z, u) = d(z, p_k) + d(p_k, u)$.
               Wait, this assumes $u$ is on the path $z \to p_k$.
               But $u$ is the farthest point from $z$.
               Wait, $u$ is just one endpoint.
               We found the diameter $u-v$.
               For any $z$, $proj(z)$ is the point on diameter closest to $z$.
               Let this be $p_k$.
               We need to identify $p_k$.
               The property is: $d(z, x)$ decreases as $x$ approaches $p_k$ from both sides, and increases as it moves away.
               $d(z, p_k) \le d(z, p_j)$ for all $j$.
               So $p_k$ is the ARGMINIMUM of $d(z, \cdot)$ on the diameter.
               Since $p_k$ is the projection, the value is exactly the "height" of $z$ above diameter.
               If $d(z, p_k)$ is minimized, then $d(z, p_k) \le d(z, p_j)$.
               Wait. This means $d(z, p_k)$ is the SMALLEST possible distance from $z$ to any point on diameter.
               Is it computable?
               Yes, if we knew $d(z, p_k)$. But we don't.
               But wait.
               If we assume the parent is $p_k$ and assign edge $(p_k, z)$ with weight $d(u, z) - d(u, p_k)$.
               Does this weight correspond to the actual edge weight $w(p_k, z)$?
               Yes, because $u \to p_k \to z$ is a path of length $d(u, p_k) + w(p_k, z)$.
               And $d(u, z) = d(u, p_k) + w(p_k, z)$ holds because $p_k$ is the projection (intersection of path).
               So, the formula $w = d(u, z) - d(u, p_k)$ is correct.
               BUT, we need to verify that $p_k$ IS the projection.
               Wait. If we pick ANY $x$ on diameter (say far from $p_k$) and use it, we assign weight $d(u, z) - d(u, x)$.
               This value $W$ might be huge (negative if $x$ is past $z$).
               Since edge weights are $\ge 1$, if $W < 1$, $x$ cannot be parent.
               If $W \ge 1$, $x$ is a candidate.
               If we have a candidate parent $x$ on diameter (ancestor), then $w$ calculated this way is a valid integer?
               Not necessarily.
               $W = d(u, z) - d(u, x)$.
               If $x$ is an ancestor, $u \to x \to \dots \to p_k \to z$.
               Then $d(u, z) = d(u, x) + d(x, p_k) + w(p_k, z)$.
               So $W = d(x, p_k) + w(p_k, z)$.
               This is the sum of weights of the path segment $x \to p_k \to z$.
               So $W \ge w(p_k, z)$.
               Since $d(x, p_k) \ge 1$ and $w \ge 1$, $W$ is strictly greater than the actual edge weight if $x$ is an ancestor.
               So the actual edge weight is strictly smaller than the calculated weight for an ancestor.
               Wait. We want to distinguish $p_k$ from $x$.
               If we pick the candidate $x$ with the SMALLEST valid positive weight?
               Or is there a stronger condition?
               If $x$ is the parent, $W$ is the edge weight.
               If $x$ is an ancestor, $W$ is path weight $\ge 2$ (since path length $\ge 2$ edges? No, could be 0 distance if $x=p_k$).
               If $x=p_k$, $W = w(p_k, z) \ge 1$.
               If $x$ is strict ancestor, $W = d(x, p_k) + w(p_k, z) \ge 1 + 1 = 2$.
               Wait, $d(x, p_k)$ is number of edges.
               So $W \ge 1$.
               This doesn't distinguish. $W$ for parent could be anything $\ge 1$.
               
               Wait. Let's use the $d(v, z)$ idea.
               We established that $d(z, p_k) = (d(u, z) + d(z, v) - D_{diam})/2$.
               This is independent of $p_k$. Let this be $H_z$.
               For the true parent $p_k$, $d(z, p_k) = H_z$.
               So we need to find $x \in PathNodes$ such that $d(z, x) = H_z$.
               Wait, do we have $d(z, x)$?
               No, we don't want to query $d(z, x)$.
               However, notice the condition:
               $d(z, p_k)$ is the DISTANCE from $z$ to the line.
               And $d(u, z) = d(u, p_k) + d(z, p_k)$.
               So $d(z, p_k) = d(u, z) - d(u, p_k)$.
               Combining these:
               $H_z = d(u, z) - d(u, p_k)$.
               This implies $p_k$ is a node where $d(u, p_k) = d(u, z) - H_z$.
               Since $H_z$ is constant for $z$ (assuming we computed it), and $d(u, z)$ is known,
               the term $RHS = d(u, z) - H_z$ is a CONSTANT number!
               Let $TargetDist = d(u, z) - H_z$.
               So we are looking for a node $p_k$ on the diameter such that $d(u, p_k) = TargetDist$.
               But $p_k$ is on the diameter, so $d(u, p_k)$ is simply the path length along diameter.
               Is $TargetDist$ a value present on the diameter?
               Yes, if the diameter path is continuous with respect to distance sums.
               So we just need to find the node on the diameter whose cumulative distance from $u$ equals $TargetDist$.
               Wait, is it that simple?
               Why did we think this was hard?
               Let's verify $TargetDist = d(u, z) - H_z$.
               $H_z = (d(u, z) + d(z, v) - d(u, v))/2$.
               $TargetDist = d(u, z) - \frac{d(u, z) + d(z, v) - d(u, v)}{2} = \frac{2d(u, z) - d(u, z) - d(z, v) + d(u, v)}{2} = \frac{d(u, z) - d(z, v) + d(u, v)}{2}$.
               Wait.
               Also $d(z, u) + d(z, v) = d(z, p_k) + d(p_k, u) + d(z, p_k) + d(p_k, v) = d(z, p_k) + d(u, v) + d(z, p_k)$.
               So $d(u, v) = d(u, p_k) + d(p_k, v)$.
               $d(z, u) = d(z, p_k) + d(p_k, u)$.
               $d(z, v) = d(z, p_k) + d(p_k, v)$.
               Sum: $d(z, u) + d(z, v) = 2d(z, p_k) + d(u, v)$. (Used before).
               Subtract: $d(z, u) - d(z, v) = d(p_k, u) - d(p_k, v)$.
               Add equations: $d(z, u) - d(z, v) + d(u, v) = d(p_k, u) - d(p_k, v) + d(p_k, u) + d(p_k, v) = 2d(p_k, u)$.
               So $2d(u, p_k) = d(u, z) - d(z, v) + d(u, v)$.
               So $d(u, p_k) = \frac{d(u, z) - d(z, v) + d(u, v)}{2}$.
               This matches $TargetDist$ derived above.
               So, YES!
               The distance of the parent from $u$ along the diameter is EXACTLY determined by $d(u, z), d(z, v), d(u, v)$.
               This is incredible.
               It means we can locate the parent by calculating a single distance scalar, provided the diameter path preserves the additivity of distances.
               Since the diameter consists of a sequence of vertices $u=p_0, \dots, p_m=v$,
               and $p_k$ is a vertex on it, $d(u, p_k)$ IS the distance along the path.
               We need to find $k$ such that the accumulated distance from $u$ to $p_k$ equals this target.
               But wait. The edges have varying weights.
               We have the vertices $p_k$. We have the weights $w(p_{i}, p_{i+1})$.
               We can compute prefix sums of weights: $S_k = \sum_{i=0}^{k-1} w(p_i, p_{i+1})$.
               Then $d(u, p_k)$ SHOULD BE EQUAL TO $S_k$ (since $u$ is the root of diameter).
               So, we just need to match the target value with the precomputed prefix sums!
               
               So the full algorithm:
               1. Pick random $r$. Compute $d(r, x)$ for all $x$. (Sort $x$ by dist? No, just store).
               2. Pick $u = \arg\max_x d(r, x)$.
               3. Compute $d(u, x)$ for all $x$.
               4. Pick $v = \arg\max_x d(u, x)$.
               5. Determine diameter path $u \to v$.
                  The vertices on diameter are those $z$ such that $d(u, z) + d(z, v) = d(u, v)$.
                  Filter vertices to get $PathNodes$. Sort them by $d(u, \cdot)$.
                  Since $u$ is fixed, sorting by $d(u, \cdot)$ is equivalent to sorting by index if we built it from BFS/DFS on diameter, but here we have implicit path.
                  Actually, to build the path, we need to know the order.
                  Wait. We filtered $PathNodes$. We have the list.
                  But we don't know the order $u \to v$ among them.
                  We sorted them by $d(u, \cdot)$. That sorts them along the diameter!
                  So $p_0=u, p_1, \dots, p_m=v$.
                  Calculate edge weights $w_i = d(u, p_{i+1}) - d(u, p_i)$ (or $d(u, p_{i+1}) - S_i$).
                  Wait, we need $d(u, p_i)$. We have it from step 3.
                  So we have the full diameter with edge weights.
               6. For each remaining node $z$:
                  Calculate $TargetDist = (d(u, z) - d(z, v) + d(u, v))/2$.
                  Note: $d(z, v)$ is NOT in step 3 (which was $d(u, \cdot)$).
                  Wait, do we need $d(z, v)$?
                  Formula used $d(z, v)$.
                  Can we avoid querying it?
                  Wait. Is $d(u, z)$ alone enough?
                  No, $d(u, p_k)$ depends on $d(u, z) - d(z, v)$.
                  Wait, $d(u, p_k) = d(u, z) - d(z, v) + d(p_k, v) + d(p_k, u)$? No.
                  $d(u, p_k) + d(p_k, v) = d(u, v)$.
                  We have $d(u, z)$ from step 3.
                  We need $d(z, v)$.
                  Can we find $d(z, v)$ cheaper than querying?
                  Maybe not. We need $d(z, v)$ to place $z$.
                  Wait. Do we really need $d(z, v)$?
                  If we assume the structure of the tree...
                  Actually, for the parent identification:
                  We established $d(u, p_k) = d(u, z) - d(z, v) + d(u, v)$? No.
                  The derivation was $2 d(u, p_k) = d(u, z) - d(z, v) + d(u, v)$.
                  Yes, $d(z, v)$ is needed.
                  So we query $d(z, v)$.
                  Is this $O(N)$?
                  Yes, 1 query per off-path node.
                  Wait, off-path nodes count can be $N$.
                  So total queries:
                  Scan $r$: $N$ queries.
                  Scan $u$: $N$ queries.
                  Scan $v$: $N$ queries (for each off-node).
                  Total $\approx 3N$.
                  This fits the budget ($5N$).
               
               So the missing link is step 6: how to find $z$'s parent $p_k$ on diameter using $TargetDist$.
               We need to find $k$ such that $d(u, p_k) = TargetDist$.
               We have vertices $p_k$ with known $d(u, p_k)$.
               Wait. Are the $d(u, p_k)$ values distinct?
               Edges have positive weights, so yes, strictly increasing.
               So $d(u, p_k)$ values map bijectively to vertices $p_k$.
               Wait. This assumes that $d(u, p_k)$ computed from the original queries equals the path distance on the diameter.
               Does $d(u, p_k)$ (query result) equal $\sum$ weights along diameter?
               $u$ is a vertex. $p_k$ is a vertex on the diameter path.
               The simple path between $u$ and $p_k$ is the segment of the diameter.
               Since $u, p_k$ are connected by the diameter path, and there is no alternative path (it's a tree), the distance $d(u, p_k)$ measured by query MUST equal the sum of edge weights on the diameter.
               So yes, $d(u, p_k)$ from query is exactly the distance along diameter.
               So we have a list of numbers $D_{path}[k] = d(u, p_k)$.
               And for each $z$, we have a target $T_z$.
               We find $k$ such that $D_{path}[k] == T_z$.
               Since it's a matching problem (each off-node maps to one vertex), and $D_{path}$ is sorted, this is a lookup.
               Wait. We have multiple $z$'s. Each produces a $T_z$.
               For a specific $z$, the calculated $T_z$ MUST match one of the $D_{path}[k]$.
               If it doesn't match EXACTLY, then my assumption about attaching to a vertex is wrong?
               Wait. What if $T_z$ falls between two values $D[k]$ and $D[k+1]$?
               Then $z$ attaches to the EDGE $(p_k, p_{k+1})$.
               In that case, $z$ is NOT attached to a vertex.
               But we already argued that a valid labeled tree cannot have an edge split.
               Wait. Why?
               The edge $(p_k, p_{k+1})$ has weight $W = D[k+1] - D[k]$.
               If $z$ attaches to the edge, it introduces a degree-1 vertex into the interior?
               If we connect $z$ to $p_k$, we change the distance $d(p_k, p_{k+1})$? No, we add a branch.
               But the ORIGINAL graph has an edge between $p_k$ and $p_{k+1}$ with weight $W$.
               The new edge $z$ brings a node.
               Is it possible for $z$ to share the edge? No.
               Wait, if $z$ attaches to the edge, does it violate the tree structure?
               Let's trace carefully.
               We have a graph with edges $E_{diam}$ (forming a line).
               $z$ is a vertex in the graph.
               $p_k, p_{k+1}$ are consecutive on line.
               Suppose $z$ is connected to the edge $(p_k, p_{k+1})$.
               This implies the graph contains the configuration $p_k --(W)--> p_{k+1}$ and somehow $z$ interacts.
               But in a simple graph, nodes have explicit incident edges.
               If $z$ is connected to $(p_k, p_{k+1})$, it means one of the endpoints $p_k$ or $p_{k+1}$ is connected to $z$?
               No, it means the distance $d(z, p_k) + d(z, p_{k+1}) = W + \text{something}$?
               If $z$ is off the diameter, attached to edge midpoint, then the path from $p_k$ to $p_{k+1}$ goes through $z$?
               Then $d(p_k, p_{k+1}) = d(p_k, z) + d(z, p_{k+1})$.
               So $W = w(p_k, z) + w(z, p_{k+1})$.
               Then we must replace the edge $(p_k, p_{k+1})$ with path $p_k - z - p_{k+1}$.
               This is VALID.
               The edge was $(p_k, p_{k+1})$ in the abstract sense (shortest path), but topologically it could be subdivided.
               However, the INPUT gives us the tree structure.
               If the input tree has an edge of weight $W$, does it mean $p_k$ is directly connected to $p_{k+1}$?
               Yes, "labeled tree" means edges connect vertices.
               If we replaced it with two edges $p_k-z$ and $z-p_{k+1}$, we would have added a vertex $z$ to the path.
               But the input graph has $N$ vertices.
               Wait. If the input has edge $(p_k, p_{k+1})$, and there is a node $z$ that makes $z$ lie on the path...
               That implies $p_k - p_{k+1}$ is NOT an edge. It is a path of length $\ge 2$.
               BUT we defined the diameter as the path of vertices $u, \dots, v$ where CONSECUTIVE vertices are adjacent.
               If $p_k, p_{k+1}$ are consecutive on the diameter path, then by definition they ARE connected by an edge.
               If $z$ is a separate node (degree $\ge 1$, not $p_k$ or $p_{k+1}$), then $z$ cannot lie on the edge $(p_k, p_{k+1})$.
               The only way $z$ connects to the edge is by attaching to a vertex.
               So my conclusion holds: **Every off-diameter node attaches to a vertex on the diameter.**
               So $T_z$ MUST equal some $D_{path}[k]$ exactly.
               Wait, floating point issues? Integers only. Exact match.
               So, algorithm:
               Construct `dists_on_diam` list from diameter nodes (values of $d(u, x)$ for $x \in PathNodes$).
               Sort this list (it will be sorted since we picked $u$ as origin and sorted by $d(u, x)$? No, we need to be sure $u$ is on the diameter).
               Wait, $u$ is on diameter. So $D_{path}[0]=0$.
               Sorting $PathNodes$ by $d(u, x)$ ensures correct order.
               Map: value -> vertex_id.
               For each off-node $z$:
                 $T_z = (d(u, z) - d(z, v) + d(u, v))/2$.
                 If $T_z$ not in map -> Error (should not happen).
                 Attach $z$ to `map[T_z]`.
               
               Wait. Is it guaranteed that $T_z$ is an integer?
               $d(u, z) + d(z, v) - d(u, v) = 2 d(z, p_k)$.
               Since $d(u, z)$ are integers (sum of ints), $2 d(z, p_k)$ is integer.
               Wait, does $d(u, z) + d(z, v) - d(u, v)$ divide by 2?
               $d(u, z) + d(z, v)$ has parity of $d(u, v)$?
               In a tree, distances form a bipartite coloring (colors 0/1).
               $d(x, y) \equiv \text{color}(x) - \text{color}(y) \pmod 2$.
               So $d(u, z) \equiv C_u - C_z$.
               $d(z, v) \equiv C_z - C_v$.
               $d(u, v) \equiv C_u - C_v$.
               Sum $(d(u, z) + d(z, v)) \equiv C_u - C_z + C_z - C_v = C_u - C_v \equiv d(u, v)$.
               So the numerator is always divisible by 2. $T_z$ is an integer.
               
               What about the $d(z, v)$ query?
               Wait, calculating $T_z$ requires $d(z, v)$.
               So we must query $d(z, v)$ for each off-path node.
               Wait, is it possible to avoid it?
               We found $T_z = d(u, z) - H_z$.
               $H_z = d(z, \text{projection})$.
               We don't know projection.
               But $T_z$ is distance from $u$.
               Is there any dependency?
               Maybe not. $3N$ is fine.
               Total Queries: $2(N-1) + (N-1) \approx 3N$.
               Time Limit 2s. Memory 512MB.
               Complexity $O(N \log N)$ due to sorting.
               Implementation details:
               - Fast I/O / flushing.
               - Handling multiple test cases.
               - Output formatting.
               - Be careful with 1-based indexing.
               
               Corner case: $T_z$ might fall on a vertex, but due to precision (none here) or logic error...
               Is it possible $T_z$ corresponds to a vertex that is already a child of $z$? No, $z$ is off-path.
               Is it possible $T_z$ points to $u$? Yes, if $z$ hangs directly off $u$.
               Is it possible $T_z$ points to $v$? Yes.
               
               Wait. One tiny catch.
               If $z$ hangs off the edge midpoint (my argument said impossible), we would have $T_z$ landing on the interval $(D[k], D[k+1])$.
               But if my argument is wrong and it DOES happen, how to handle?
               If it lands on interval, $z$ is connected to edge $(p_k, p_{k+1})$ at distance $D[z] = d(z, \text{mid})$.
               This implies the edge $(p_k, p_{k+1})$ is SPLIT into $(p_k, z)$ and $(z, p_{k+1})$.
               But we need to report edges.
               If we assume it lands on vertex, and it actually landed on edge midpoint, which vertex do we pick?
               If we pick $p_k$, we create edge $(p_k, z)$.
               Distance to $p_{k+1}$? Original edge weight was $W$.
               New path $p_k \to z \to p_{k+1}$.
               Distances: $p_k \to z$ is $w$. $z \to p_{k+1}$ is $W-w$.
               But we reported $(p_k, z)$ with weight $w$. We didn't report $(z, p_{k+1})$?
               We must report ALL edges.
               If $z$ is inserted on an edge, we must split it.
               So we MUST output $(p_k, z)$ and $(z, p_{k+1})$.
               But wait, if the original input had edge $(p_k, p_{k+1})$ with weight $W$, and we output $(p_k, z)$ and $(z, p_{k+1})$, that's a tree of $N$ nodes?
               No. $z$ is a node.
               If we insert $z$, we have $N$ nodes.
               The edge set changed from $\{(p_k, p_{k+1})\}$ to $\{(p_k, z), (z, p_{k+1})\}$.
               Does this preserve the distance matrix?
               Old: $d(p_k, p_{k+1}) = W$.
               New: $d(p_k, p_{k+1}) = w + (W-w) = W$.
               Old: $d(z, p_k) = w$. (Implicitly $z$ wasn't in graph or was isolated? No, $z$ was a node).
               Wait, if $z$ was originally connected to the edge, it means $z$ was an internal node of the path.
               So originally, the path went $p_k - z - p_{k+1}$.
               But we thought $(p_k, p_{k+1})$ was an edge.
               This means $p_k, p_{k+1}$ are NOT consecutive in the path?
               But we extracted the diameter vertices by checking $d(u, z) + d(z, v) = d(u, v)$.
               If $p_k - z - p_{k+1}$ were the path, then $z$ satisfies the condition.
               So $z$ would be included in $PathNodes$.
               So $z$ would be on the diameter.
               Thus, $z$ being OFF-diameter implies it does NOT lie on the path between ANY two vertices satisfying the condition.
               So it cannot lie on any edge of the diameter.
               So it MUST attach to a vertex.
               Proof complete.

               Double check logic:
               We picked $PathNodes = \{z \mid d(u, z) + d(z, v) = d(u, v)\}$.
               If $z$ is on an edge $(p_k, p_{k+1})$, then $z$ lies on the segment.
               Then $d(u, z) = d(u, p_k) + d(p_k, z)$.
               $d(z, v) = d(z, p_{k+1}) + d(p_{k+1}, v)$.
               Sum $= d(u, p_k) + d(p_k, z) + d(z, p_{k+1}) + d(p_{k+1}, v)$.
               Since $p_k, p_{k+1}$ adjacent, $d(p_k, z) + d(z, p_{k+1}) = d(p_k, p_{k+1})$.
               So Sum $= d(u, p_k) + d(p_k, p_{k+1}) + d(p_{k+1}, v) = d(u, v)$.
               So YES, any node lying on an edge (internal to the edge) satisfies the condition.
               So such a node IS IN $PathNodes$.
               Therefore, any node NOT in $PathNodes$ CANNOT lie on any diameter edge.
               So it must attach to a vertex.
               Excellent. The logic holds.

               So the code will:
               - Solve() for each test case.
               - Query $d(1, \cdot)$ for all $i$. Let this be array $D1$.
               - $u = \arg\max D1$.
               - Query $d(u, \cdot)$ for all $i$. Let this be array $Du$.
               - $v = \arg\max Du$.
               - Verify diameter endpoints $u, v$ (we implicitly trust the farthest-farthest heuristic? No, the problem doesn't ask to prove correctness, just solve. The heuristic is robust for competitive programming problems of this type).
                 Wait, is it 100% correct?
                 With a diameter heuristics, sometimes it fails on weird cases?
                 Actually, for finding diameter endpoints, the "twice BFS" is optimal for unweighted graphs.
                 For weighted graphs, it's a good heuristic.
                 However, does it fail on weighted trees?
                 Yes, if $u$ is far but on a light branch, while the real diameter is in a heavy branch.
                 But the problem says edges have random weights in $[1, K]$.
                 This makes the diameter "heavy" and random sampling less skewed.
                 Wait, "Little Cyan Fish wants you to determine...".
                 The problem might be adversarial? No, "Little Cyan Fish has a weighted tree generated...". It's a constructive/random generation.
                 BUT interaction library is non-adaptive.
                 So weights are fixed.
                 The heuristic might fail deterministically on a crafted case (e.g. very flat tree with a few long edges, but random seed picks bad node).
                 Can we safeguard?
                 We can check: $v$ maximizes $d(u, \cdot)$.
                 Is it guaranteed that $v$ is an endpoint?
                 Yes, if $u$ is an endpoint of the diameter.
                 But $u$ is random.
                 If $u$ is on a long path, farthest node $v$ is an endpoint of diameter.
                 Probability that random node $u$ has a farthest node that is NOT diameter endpoint is high?
                 Actually, for any node $u$, the farthest node is an endpoint of SOME diameter? No, of a diameter passing near $u$.
                 Actually, a theorem says the farthest node from ANY node is an endpoint of A DIAMETER.
                 Let's verify.
                 Let $D_{diam} = d(a, b)$. Let $u$ be arbitrary. Let $v$ be farthest from $u$.
                 Claim: $d(u, v) \ge d(a, b)$? No. $d(u, v)$ can be larger? No, diameter is max.
                 So $d(u, v)$ is some value. Is $d(a, v) = d(a, b)$?
                 If $u$ is "inside" $a-b$, then $v=a$ or $v=b$ is farthest? No, not necessarily.
                 Wait. If $u$ is on the diameter, $v$ must be $a$ or $b$.
                 Proof: $u$ is on $a-b$. $a-u-b$.
                 $d(u, x) \le \max(d(u, a), d(u, b))$.
                 So $v$ is $a$ or $b$.
                 If $u$ is not on diameter.
                 Let $c$ be on diameter path $a-b$ that is closest to $u$.
                 Then $d(u, x) = d(u, c) + d(c, x)$ for all $x \in \{a, b\}$.
                 Since $v$ maximizes $d(u, x)$, and $\max_x d(u, x)$ is attained at $x$ maximizing $d(c, x)$,
                 and $c$ is on diameter, $\max(d(c, a), d(c, b)) = d(a, b)$.
                 So $v$ is an endpoint ($a$ or $b$).
                 So yes! **THE FARTEST NODE FROM AN ARBITRARY NODE IS ALWAYS A DIAMETER ENDPOINT.**
                 This is a standard tree property.
                 So $u \to v$ IS an endpoint-to-endpoint path.
                 Wait. If $v$ is an endpoint, then $d(u, v)$ is the distance from $u$ to the furthest endpoint.
                 Does that imply $d(u, v)$ is the diameter length? No.
                 It implies $v$ is an endpoint.
                 So we have ONE endpoint $v$.
                 To get the OTHER endpoint, we find farthest from $v$.
                 Let $y$ be farthest from $v$. Since $v$ is an endpoint of the diameter (say $b$),
                 and we find farthest from $b$, it MUST be the other endpoint $a$.
                 Because farthest from an endpoint is the opposite endpoint.
                 So:
                 1. Pick random $u$. Find $x$ farthest from $u$. $x$ is an endpoint of SOME diameter.
                    Wait, is $x$ an endpoint of THE diameter?
                    Let diameter be $A-B$.
                    If $u$ is arbitrary. Farthest from $u$ is $x$. $x$ is endpoint of diameter passing near $u$.
                    Is it endpoint of $A-B$?
                    Not necessarily.
                    Example: Long line with a small leaf.
                    Line $1-2-3-4-5$. Leaf 6 attached to 3.
                    Weights on line: 1. Leaf 6: 100.
                    Diameter: $6-3-4-5$ length 102? Or $1-2-3-6$ length 100?
                    Wait, weights.
                    Suppose $6-3$ is 100. $3-4$ is 1. $4-5$ is 1.
                    $d(6, 5) = 102$.
                    $d(6, 1) = 100 + 2 = 102$.
                    So endpoints are 1 and 5, and 6.
                    If we pick random $u=3$.
                    Farthest from 3 is 5 (dist 2) or 6 (dist 100)? 6.
                    6 is endpoint of diameter.
                    Suppose $6-3$ is 1. $3-4$ is 100.
                    $d(6, 5) = 102$.
                    $u=3$. Farthest is 6 (dist 1). $d(6, 5)=101$. Wait.
                    $d(3, 6)=1$. $d(3, 5)=101$.
                    Farthest is 5. 5 is endpoint.
                    It seems correct.
                    Theorem: For ANY vertex $u$, the node $v$ farthest from $u$ is an endpoint of a DIAMETER.
                    Source: Many CP resources state this.
                    So, finding $x$ maximizes distance gives us a DIAMETER ENDPOINT.
                    Then finding $y$ farthest from $x$ gives the OTHER ENDPOINT.
                    This works!
                   So the heuristic is rigorous. No randomness issues.

               Final Algorithm Summary:
               1. Pick $u$ randomly (or 1).
               2. Scan all $z$: query $d(u, z)$. Find $u' = \arg\max d(u, z)$.
               3. Scan all $z$: query $d(u', z)$. Find $v' = \arg\max d(u', z)$.
               4. Diameter length $L = d(u', v')$.
               5. Identify Diameter Nodes: $P = \{z \mid d(u', z) + d(z, v') = L\}$.
               6. Sort $P$ by $d(u', z)$ to get ordered diameter.
               7. For each $z \notin P$:
                  Calculate $target = (d(u', z) - d(z, v') + L)/2$.
                  Query $d(z, v')$ (needed).
                  Find $p \in P$ with $d(u', p) == target$.
                  Connect $(p, z)$ with weight $d(u', z) - d(u', p)$.
               Wait, we need $d(u', z)$? We have it from step 2 (if $z \in P$) or step 3 (if $z \notin P$).
               Wait, step 2 scanned $d(u, \cdot)$. Step 3 scanned $d(u', \cdot)$.
               So for all $z$, we have $d(u', z)$.
               So $d(u', z)$ is known for free.
               So we only need $d(z, v')$.
               Total queries:
               Step 2: $N$.
               Step 3: $N$.
               Step 7: For each off-node, query $d(z, v')$.
               Wait, Step 7 queries $d(z, v')$.
               Is there a way to avoid this?
               If $target$ is computed from $d(u', z)$ and $d(u', p)$, wait.
               $target = d(u', p)$.
               $p$ is found by $d(u', p) = \text{expression}$.
               Wait. $d(u', z)$ is known.
               Do we need $d(z, v')$ in the expression?
               $target = d(u', z) - (d(u', z) - (d(z, v') - (L - d(u', p))))$.
               Let's recheck the formula.
               $target = \frac{d(u', z) - d(z, v') + L}{2}$.
               If we don't have $d(z, v')$, we can't compute it.
               So we must query $d(z, v')$ for every off-path node.
               But wait.
               Is it possible to guess $d(z, v')$?
               $d(z, v') = d(z, p) + d(p, v')$.
               $d(z, p) = \text{height} = target$? No, height is $d(z, p) = d(z, u') - d(u', p)$.
               Wait. $target$ was defined as $d(u', p)$.
               Wait, $target$ is $d(u', p)$?
               My formula was $2d(u', p) = d(u', z) - d(z, v') + L$.
               Yes, target is the distance of parent along diameter.
               So we need $d(z, v')$.
               BUT, do we really need to query it?
               We established that if $z$ attaches to $p$, then $d(u', z) = d(u', p) + d(z, p)$.
               So $d(z, p) = d(u', z) - target$.
               Also $d(z, v') = d(z, p) + d(p, v')$.
               So $d(z, p) = d(u', z) - target$.
               Substitute into eq:
               $target = d(u', z) - (d(u', z) - target) + d(p, v') - target$? No.
               $2 target = d(u', z) - (d(z, p) + d(p, v')) + L$.
               $2 target = d(u', z) - (d(u', z) - target) + d(p, v') + L - d(u', p)?$
               No. $d(u', z) = target + d(z, p)$.
               $d(z, v') = d(z, p) + d(p, v')$.
               Eq: $2 target = d(u', z) - d(z, v') + L$.
               $2 target = (target + d(z, p)) - (d(z, p) + d(p, v')) + L$.
               $2 target = target - d(p, v') + L$.
               $target = L - d(p, v') = d(u', p)$.
               This is a tautology! It doesn't help compute target.
               Wait. It proves consistency. It doesn't let us eliminate $d(z, v')$.
               So we need $d(z, v')$.
               Wait, do we need $d(z, v')$ for all nodes?
               We only query it for nodes NOT in diameter.
               Wait, if $z$ is in diameter, $d(z, v') = L - d(z, u')$. We have $d(z, u')$. So we don't query.
               If $z$ is NOT in diameter, we must query.
               Is it possible to avoid?
               Wait, $d(z, v') = d(z, p) + d(p, v')$.
               $d(z, p) = d(u', z) - d(u', p)$.
               So $d(z, v') = d(u', z) - d(u', p) + d(p, v')$.
               We know $d(u', z)$. We can find $p$ by iterating $D_{path}$?
               Wait. We don't know $p$ yet.
               So we are stuck. We need $d(z, v')$ to FIND $p$.
               Is it possible that $p$ is determined by something else?
               Maybe $p$ is the vertex that minimizes $d(z, p)$?
               We don't have $d(z, p)$.
               Wait. We have $d(z, p) = d(u', z) - d(u', p)$.
               This requires assuming $p$ is parent.
               So we are going in circles.
               Wait, $d(z, v')$ is simply the distance to a fixed point $v'$.
               Since we have to query distances to verify edges, it feels natural.
               Maybe the query limit is generous enough? $5N$.
               $3N$ queries used. So budget allows $2N$ extra.
               Wait, I estimated $3N$.
               Scan $u \to x$ (N).
               Scan $x \to y$ (N).
               Scan $z \to y$ (N).
               Is Scan $z \to y$ necessary?
               Can we use $z \to x$?
               Symmetry. $z \to x$ gives similar formula.
               Is it worth querying both? No.
               Is there a way to deduce $z \to v'$ from $z \to x$ and diameter?
               If we assume the tree structure is solved, then yes.
               But we use it to solve the tree.
               So it seems unavoidable. $3N$ is fine.
               Wait, is there a case where we need to query $d(z, \cdot)$ more often?
               Maybe for recursion?
               But we only recurse to handle the diameter.
               Wait.
               Does solving for diameter cover everything?
               Off-diameter nodes are handled by projecting.
               Their projections land on diameter nodes.
               So every node is accounted for.
               So $3N$ is the final count.
               $N=10^5$, $3 \cdot 10^5$ queries. Limit $5 \cdot 10^5$. Safe.

               Wait, is it possible to merge steps?
               We query $d(u', z)$ for all $z$. This step 3 gives us both the path nodes AND the off-nodes distances.
               Wait.
               For a node $z$, we have $d(u', z)$.
               We don't know if it's on path or off.
               To decide, we check $d(z, v')$.
               So we DO need $d(z, v')$.
               Wait, can we predict $d(z, v')$?
               No.
               So, the order should be:
               1. Scan $u' \to \cdot$ (store in map/list). Cost $N$.
                  Find $v' = \arg\max$.
               2. Scan $z \to v'$ (store in map/list). Cost $N$.
                  Check if $d(u', z) + d(z, v') == L$.
                  If yes, add to PathNodes.
               3. Process off-nodes. We have $d(u', z)$ and $d(z, v')$.
                  Compute target. Match to PathNodes.
                  Assign parent.
               4. (Optional?) Check if $z$ attaches to edge? (Already ruled out).
               Wait.
               Wait, $d(u', z)$ and $d(z, v')$ are distances to endpoints.
               Is it possible to find $p$ using only $d(u', z)$?
               No, the off-node location depends on $v'$ (other side).
               Unless the diameter is unique?
               If diameter is unique, $d(u', z)$ determines the path from $u'$ to the branch.
               The branch is at distance $d(u', p)$.
               Is $d(u', p) = d(u', z) - d(z, p)$?
               Yes.
               But we don't know $d(z, p)$.
               Wait.
               $d(z, v') = d(z, p) + d(p, v')$.
               $d(u', z) = d(u', p) + d(z, p)$.
               Subtracting: $d(u', z) - d(z, v') = d(u', p) - d(p, v')$.
               $d(u', z) - d(z, v') = d(u', p) - (L - d(u', p)) = 2d(u', p) - L$.
               So $2d(u', p) = d(u', z) - d(z, v') + L$.
               This confirms the formula.
               So yes, we need both distances to cancel out $d(z, p)$ and isolate $d(u', p)$.
               So $d(z, v')$ is required.
               
               Wait, $d(z, v')$ is $L - d(p, v') + d(z, p)$? No.
               $d(z, v') = d(z, p) + d(p, v')$.
               So $d(z, p) = d(z, v') - d(p, v')$.
               We need to know $d(z, v')$ to calculate $d(z, p)$?
               Wait. If we can find $d(z, p)$, then $d(u', p) = d(u', z) - d(z, p)$.
               But finding $d(z, p)$ is the goal (locating $p$).
               We can't use $d(z, v')$ to find $d(z, p)$ without solving for $p$.
               Circular dependency.
               BUT, the formula $2d(u
````
