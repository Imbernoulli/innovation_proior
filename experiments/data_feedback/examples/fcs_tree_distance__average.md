# FrontierCS：Tree distance（交互式竞赛题） — Average（q35 a100 method model-soup soup10）

> **本文件是完整、未截断的真实评测记录。** 模型类别：`average`。
> 评测：FrontierCS（thinking 口径，竞赛算法编程，官方评测器抽最长 ```cpp 块编译跑测试点给连续部分分）。
> 题目内部编号 `problem_idx=2` / 题库目录 `algorithmic/problems/10` / 第 `sample_idx=0` 个采样（共 5 采样/题）。
>
> **本采样得分（FCS score，0–100 连续部分分）：`85.875`**　|　生成 token 数：`30613`　|　error：`None`

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
The problem asks us to reconstruct a weighted tree with $n$ vertices. We can query distances between any two vertices. We need to find all edges and their weights, or equivalently, the tree structure and edge weights, using as few queries as possible.

We are given $T$ test cases. The sum of $n$ over test cases is at most $10^5$. The time limit is 2 seconds. Memory limit is 512 MB.
The weights are in $[1, K]$, where $K \le 10^4$. This is an interactive problem. The scoring scheme rewards using fewer than $5n$ queries. Since the goal is to minimize queries, we should aim for something close to $O(n)$ or even better if possible, but since we have limited budget per case, usually getting the answer within linear queries is fine. However, the constraint says "as less as you can" for scoring, and explicitly mentions $5n$ as a threshold. So $O(n)$ queries is definitely acceptable. $O(n^2)$ is likely too many ($n=10^5$).

Let's think about what information we get from distance queries.
In a tree, the distance $d(u, v)$ is the length of the simple path between $u$ and $v$.
If we know distances from a root to all other nodes, we can determine the tree structure and edge weights.
Specifically, fix vertex 1 as the root. For any other vertex $v$, if we know $d(1, v)$, does that help?
Not directly the parent, because there might be multiple paths of same length? No, it's a tree, unique path.
But knowing $d(1, v)$ alone doesn't tell us who is the parent of $v$.
However, if we know the distances between all pairs, we could check which one satisfies triangle equality? That's $O(n^2)$.

We want to do it in roughly $2n$ queries or so.
A standard technique to build a tree when we can query distances is to fix a root (say 1) and try to identify parents.
For each node $i > 1$, we need to find its parent $p$ such that $d(i, p) = d(1, i) - d(1, p)$.
Wait, $d(1, i) = d(1, p) + d(p, i)$.
So if we guess $p$, we can verify if $d(1, i) = d(1, p) + d(p, i)$. If this holds, $p$ is on the path from 1 to $i$.
This suggests we don't necessarily need to find the exact parent immediately.

Actually, finding the Lowest Common Ancestor (LCA) is related to distances. But without the tree structure, we don't know LCA easily.

Let's look at the constraints again. Sum of $N$ is $10^5$. Time limit 2s.
Interactive problems often allow $O(N)$ queries. The formula allows up to $5N$ queries without penalty reduction, but we should ideally aim lower or efficient enough.
Actually, if we just use a randomized approach or specific constructive strategy.
Since it's a random labeled tree, maybe we can assume some properties?
But the structure is fixed for the interaction. Wait, "It is guaranteed that the interaction library is non-adaptive."
This means the tree is generated randomly once, then fixed for all queries in this instance.

A very powerful strategy when we have distance oracle and a tree is to find the diameter or just a BFS-like structure.
Consider fixing root at 1. Let $dist[i]$ denote $d(1, i)$.
We can query $d(1, i)$ for all $i$. This takes $n-1$ queries.
With these values, we know the distance from root to every node.
Is that enough?
Suppose we know $dist[u]$ and $dist[v]$. If $dist[w] = dist[u] + dist(v, w)$? No, $w$ is child of $u$ implies $dist[w] = dist[u] + d(u, w)$.
If we knew edge weights were 1, we could sort by distance. But weights are arbitrary.
However, we need to reconstruct the edges $(u, v)$ and weight $w_{uv}$.
Output format: "! u_1 v_1 w_1 ...". We need to list $n-1$ edges with weights.
If we determine the tree structure (who is parent of whom), we can calculate the weight $d(child, parent) = dist[child] - dist[parent]$.
So the task reduces to identifying the parent of each node in a BFS/DFS tree rooted at 1.
Given $d(1, x)$ for all $x$, how to find parent of $x$?
If $y$ is parent of $x$, then $d(1, x) = d(1, y) + d(y, x)$.
So we need to find $y$ such that $d(1, y) < d(1, x)$ (assuming no cycles, tree structure guarantees depth increase if we traverse away from root) AND $d(x, y) = d(1, x) - d(1, y)$.
But checking all pairs is slow.
We need a faster way.
Note that weights are integers $\ge 1$. This is useful.
Also, $K \le 10^4$. Edge weights are small? Actually $K \le 10^4$ is small compared to max possible path lengths, but not negligible.

Can we simply perform BFS?
To simulate BFS, we want to order nodes by distance from root?
No, topological order depends on the graph.
Wait, if we pick a node $u$ that is currently known (distance from 1 is known and it has smaller distance than current candidate), can we determine if it is parent?
The issue is finding parent without querying $d(x, y)$ for every potential parent $y$.
Querying $d(x, y)$ for all $y$ takes $O(deg(x))$. Total could be large? No, sum of degrees is $2(n-1)$.
Wait! In a tree, for any node $x$, its parent is connected to $x$ by exactly one edge.
We can find the parent of $x$ by iterating over all other nodes? No, that's $O(n)$ per node $\implies O(n^2)$ total queries. Too slow.

We need $O(n)$ total queries.
If we already know $d(1, i)$ for all $i$, can we deduce structure without more queries?
Maybe using GCD? Or prime properties?
Or maybe we can exploit the property that weights are small? No.
Wait, the scoring allows $Z - Q$ terms. $Z$ is some baseline. Usually $Z$ around $2n$ or something. If we use $O(n)$ queries we pass.
Key idea: **Bipartite matching?** No.
How about this:
Pick an arbitrary root, say 1. Query $d(1, i)$ for all $i \in [2, n]$. This costs $n-1$ queries.
Now we have distances from root.
Let $S_u$ be the set of nodes $v$ such that $d(1, v) = d(1, u) + \text{something}$.
This seems insufficient.
Let's reconsider the distance property.
$d(u, v) = d(1, u) + d(1, v) - 2 d(1, lca(u, v))$.
This involves LCA, which we don't know.
But we have the ability to query ANY pair.
If we fix root 1.
For a node $u$, let's look for its neighbor $p$ on the path to 1.
$d(1, u) = \min_{v \in Neighbors(u)} (d(1, v) + d(u, v))$.
Since $u$ is connected to its neighbors, and only one neighbor is closer to root (its parent), the parent $p$ must satisfy:
1. $d(1, p) < d(1, u)$ (unless weights allow non-tree shortest path, but it is a tree).
   Wait, $d(1, p) + d(p, u) = d(1, u)$.
   So for the true parent $p$, $d(p, u) = d(1, u) - d(1, p)$.
   Also, for any other neighbor $z$, $d(z, u)$ would be such that $d(1, z) + d(z, u) > d(1, u)$ because the path from 1 to $u$ goes through $p$, so going through $z$ (which is on the other branch) would mean traversing $z-u-z'...p...1$ which is longer.
   Specifically, in a tree rooted at 1, for any node $u$, the parent $p$ minimizes $d(1, v)$ among all $v$ satisfying $d(1, u) = d(1, v) + d(u, v)$.
   Actually, if $d(u, v) = d(1, u) - d(1, v)$, then $v$ is on the path from 1 to $u$. Since it's a tree, there is only one such neighbor.
   So if we find $v$ such that $d(1, u) = d(1, v) + d(u, v)$, then $v$ is the parent of $u$.
   To avoid $O(n^2)$ queries, we cannot check all $v$.

Wait, can we construct the tree using just $d(1, i)$?
Maybe we can iterate through nodes sorted by $d(1, i)$?
If we process nodes in increasing order of $d(1, i)$, say $u_1, u_2, \dots, u_n$ where $u_1=1$.
When considering $u_k$ (not 1), we want to find its parent $p$.
We know $d(1, p) < d(1, u_k)$.
$p$ must be one of the previously visited nodes $u_j$ ($j < k$).
We still need to verify which one is the parent.
However, notice that $d(u_j, u_k)$ query is needed only if we suspect $u_j$ is parent.
We can't do that efficiently.

Let's step back. Is there a simpler way?
We can use the property that we want to output ALL edges.
If we can determine the connectivity, we get weights too.
What if we treat this as finding a spanning tree (the whole tree) by querying?
Standard MST algorithms like Kruskal's use weight comparison. Here we have distance queries.
Prim's algorithm requires knowing connection weights. We don't know edge weights.
We know distance sums.
$d(u, v)$ gives weight of path.

Maybe we can determine the structure by building it layer by layer.
Actually, the simplest way to solve this is to find the diameter endpoints first?
If we find the furthest node from 1, say $X$, then find furthest from $X$, say $Y$. Path $1..X..Y$ is a diameter (maybe not unique).
This doesn't give full tree info.

Wait, looking at the scoring: $Score = 100 * (Z - Q) / (Z - 5n)$.
Usually $Z$ is defined somewhere else, e.g., number of queries allowed to get perfect score. The text says range $[5n, Z]$. If $Q \le 5n$ we get good score?
Wait, "linear formula determined by the range [5n, Z]". It probably means if we use more than $Z$ queries we get 0?
And if we use between $5n$ and $Z$ we get partial points. Ideally $Q$ close to 0?
Wait, $Q \ge 5n$? The formula looks weird if $Z$ is small.
Usually in competitive programming, optimal strategies for trees with distance oracle take $O(n)$ queries, specifically $2n$ or similar.
If the base threshold is $5n$, we are safe with something like $3n$ or $2n$.
Since $n$ is up to $10^5$, $O(n)$ is mandatory.
Wait, I missed something crucial. "It is guaranteed that 1 <= K <= 10^4". Small weights.
But edge weights are not bounded by 1. They are bounded by $K$.
This implies $d(u, v)$ fits in integer.
Is there a way to determine parent-child relationships using fewer queries?
What if we just pick pairs $(u, v)$ that are NOT connected yet, add edge?
But we don't know the edges exist.

Let's revisit the condition: $d(1, x)$ for all $x$.
For a fixed $x$, there is a unique neighbor $p$ on the path to 1.
$d(1, x) = d(1, p) + d(p, x)$.
So $d(p, x) = d(1, x) - d(1, p)$.
This value is strictly positive.
If we knew $d(1, p)$ for all candidates $p$, we could check $d(1, x) - d(1, p) == d(p, x)$.
But we don't want to check all $p$.
Wait, what if we run a "randomized" strategy or something based on bitmasks?
Wait, we can simply ask for $d(i, j)$ for specific pairs.
Actually, consider this:
For any node $u$, we want to find its neighbor $p$ towards the root.
We can binary search? No, nodes are not ordered.
However, since $K$ is small, maybe we can use parity? No.

Let's rethink. Is there a specific algorithm known for this?
Reconstructing a tree with distance queries.
If we can compute $d(root, v)$ for all $v$, can we reconstruct?
Yes, if we can match nodes.
Actually, if we have distances from 1, can we determine the tree uniquely?
Suppose we have nodes 1, 2. $d(1, 2)=w$.
Node 3. $d(1, 3)=a, d(2, 3)=b$.
Then either $a+b=w$ (if 3 is attached via 2) or $a=b+w$ (3 attached via 1? No, 1 is root) or $b=a+w$.
Actually $d(1, 3) + d(3, 2) - d(1, 2) = 0$ implies 3 is on path 1-2? No, that's impossible as 3 is distinct from 1, 2.
In a tree, for any three nodes, one is on the path between the other two.
So either $d(1, 2) = d(1, 3) + d(3, 2)$ OR $d(1, 3) = d(1, 2) + d(2, 3)$ OR $d(2, 3) = d(1, 2) + d(1, 3)$.
This is the "median property" or "tree metric".
So for a fixed root 1, for every pair $(u, v)$, we know distances from 1.
If we query $d(u, v)$, we can determine the relationship.
If $d(1, u) + d(u, v) = d(1, v)$, then $u$ is on the path 1-v. Thus $u$ is an ancestor of $v$.
If we know the set of ancestors for each node, we can form the tree.
Ancestors of $v$: $\{a : d(1, a) + d(a, v) = d(1, v)\}$.
One of them is parent.
If we find ONE ancestor $a$ such that $d(a, v) = d(1, v) - d(1, a)$, we know $a$ is on path.
But $a$ could be grandparent etc.
However, we need $a$ such that $deg(a)$ in the tree restricted to ancestors?
Actually, if we pick $u$ such that $d(1, u)$ is maximal among proper ancestors of $v$ (closest to $v$), then $u$ is parent.
The problem is finding such $u$ efficiently.

Wait, I recall a solution to this problem. It's from Codeforces or similar. "Tree Distance Interactive".
Strategy:
We maintain the set of nodes whose position relative to root is unknown. Initially $\{2, \dots, n\}$. Root is 1.
We want to attach each node $u$ to some node $p \in \{1, \dots, n-1\}$ such that $d(1, u) = d(1, p) + d(p, u)$.
We can do this greedily?
Sort nodes by $d(1, \cdot)$.
Iterate $u$ in increasing order of distance.
Find a parent.
Wait, sorting takes $O(n)$. Finding parent is the hard part.
Can we query $d(u, v)$ for $v$ already processed?
Since $v$ is processed, we know its parent and $d(1, v)$.
Does $d(1, v) < d(1, u)$ imply $v$ is a candidate parent?
Yes, because in the true tree parent must be closer to root.
But which one?
If we pick $v$ randomly?
What if we try to build a Minimum Spanning Tree using Prim's but with a twist?
Actually, since we know $d(1, u)$ for all $u$, the value $d(1, u)$ acts like a height.
The edge weights on the path are integers.
There's a specific trick involving the parity of coordinates or something? No, weights are not bipartite coloring.

Let's look at the constraints again. $N \le 10^5$.
If we just query $d(1, i)$ for all $i$. Cost $N$.
Then for each $i \neq 1$, we want to find parent.
Is there a way to verify parent identity with constant number of extra queries?
Suppose we have a set of active nodes $S$. We add $u$ to $S$.
If we can determine for which $v \in S$ (where $d(1, v) < d(1, u)$) the relation holds.
Actually, checking $d(u, v)$ for all $v$ is too much.
Wait, if we just query $d(u, v)$ for $v$ being the *closest* to 1? No.
What if we use the fact that $d(u, v) \ge 1$?
Consider the case where we have found a subset of nodes forming a valid subtree.
Actually, the tree structure is unique.
Maybe we can just pick $d(1, u)$ for all $u$.
Then, for each $u \neq 1$, there exists a neighbor $p$ with $d(1, p) < d(1, u)$.
Specifically, $d(1, u) - d(1, p) = w(p, u) \le K$.
So $d(1, u) - K \le d(1, p) < d(1, u)$.
This bounds the possible parents to a small range of distances!
Since edge weights are in $[1, K]$, the parent of $u$ must be at distance $D'$ from 1 such that $D' \in (d(1, u) - K, d(1, u))$.
If $K$ is small, we only need to check nodes $v$ such that $d(1, v) > d(1, u) - K$.
Since $K \le 10^4$ and max path length can be large, but for a specific node $u$, we just need to check candidates $v$ where $d(1, v)$ falls in $[d(1, u) - K, d(1, u)]$.
Wait, there might be many such nodes $v$ with different parents.
But we can't iterate all $v$ with appropriate distance.
However, note that we process nodes in increasing order of $d(1, u)$.
At step $u$, we have processed nodes with smaller distances.
Let's refine this.
For $u$, the parent $p$ satisfies $d(1, p) = d(1, u) - w$, where $1 \le w \le K$.
So $p$ is one of the processed nodes (since $d(1, p) < d(1, u)$) such that $d(1, p) \ge d(1, u) - K$.
Is it possible to have too many candidates?
Yes, potentially $O(N)$ candidates if all previous nodes had distance in range.
BUT, we can group them.
What if we only keep track of "representatives"?
Actually, for a fixed $u$, the parent $p$ is unique.
Among all $v$ with correct distance, exactly one will satisfy $d(u, v) = d(1, u) - d(1, v)$.
Others will have $d(u, v) > d(1, u) - d(1, v)$.
Actually, if $v$ is not parent, the path from $u$ to 1 goes through $p$, then $v$ branches off earlier.
So $d(u, v) = d(u, p) + d(p, v)$.
Thus $d(u, v) = (d(1, u) - d(1, p)) + d(1, v)$.
Wait, if $v$ is also in the "processed" set, then $d(1, v) < d(1, p)$ (if $v$ is further down from 1? No $d(1, p) < d(1, u)$).
Let's re-evaluate.
If $p$ is parent of $u$, then $d(u, p) = d(1, u) - d(1, p)$.
If $v$ is another node with $d(1, v) \approx d(1, u)$, say $v$ is sibling or in another branch.
If $v$ is in a different branch meeting at some ancestor $a$, then $d(u, v) = d(u, a) + d(a, v) = (d(1, u) - d(1, a)) + (d(1, v) - d(1, a))$.
So $d(u, v) = d(1, u) + d(1, v) - 2 d(1, a)$.
Notice that $d(u, v) - (d(1, u) - d(1, v)) = 2 d(1, v) - 2 d(1, a)$.
If $v$ was parent, $a=v$, term is 0.
If $v$ is not parent, $d(1, v)$ vs $d(1, a)$.
Basically, we can't check all candidates.

Wait, is there a randomized approach?
Or maybe just picking the closest?
If we have $d(1, v)$, we can sort by $d(1, v)$.
Let the sorted order be $u_1, u_2, \dots, u_n$ (where $u_1=1$).
Consider $u_i$. Its parent $p$ must be some $u_j$ with $j < i$ (since $d(1, p) < d(1, u_i)$).
Moreover $d(1, p) \ge d(1, u_i) - K$.
This restricts $p$ to a subset of indices.
But maybe we can prune?
What if we query $d(u_i, u_j)$ only for $j$ near $i$?
Wait, if we use a Disjoint Set Union (DSU)? No.

Actually, the bound on $K$ ($K \le 10^4$) is quite specific.
If $K$ were large ($O(N)$), this restriction wouldn't hold.
With $K$ small, the "window" of candidate parents based on distance difference is small.
Is it possible that for some $u$, there are few candidates? Not necessarily.
But wait, we only need to FIND the parent.
We can iterate over the sorted list of candidates $v$ such that $d(1, v) \in [d(1, u) - K, d(1, u))$.
The size of this window might be large.
HOWEVER, notice that $d(1, v)$ are integers.
We can check if $d(u, v) == d(1, u) - d(1, v)$.
If yes, then $v$ is on the path from 1 to $u$.
If we find such a $v$, is it the parent?
Yes, because among all nodes on the path to 1, the one with largest $d(1, v)$ is the direct parent (assuming $d(1, v) + d(v, u) = d(1, u)$).
Wait, if $v$ is grandparent, then parent $p$ satisfies $d(1, p) = d(1, v) + d(v, p)$.
So $d(1, u) = d(1, p) + d(p, u) = d(1, v) + d(v, p) + d(p, u) > d(1, v) + d(v, u)$?
No. $d(v, u) = d(v, p) + d(p, u)$.
So $d(1, u) - d(1, v) = d(v, p) + d(p, u) = d(v, u)$.
So ANY ancestor $v$ satisfies $d(1, u) = d(1, v) + d(v, u)$.
The condition for "on path" is satisfied by all ancestors.
But we want the PARENT.
So we need the ancestor $v$ that maximizes $d(1, v)$ (or minimizes $d(v, u)$ among those satisfying the condition).
Since we process in increasing order of $d(1, u)$, we encounter nodes in the window.
Actually, if we check all $v$ in the window and find the one maximizing $d(1, v)$ such that $d(u, v) = d(1, u) - d(1, v)$, that MUST be the parent.
But checking all is expensive.
However, observe that we just need to identify WHICH node is the parent.
Maybe we don't need to scan all?
If we have multiple candidates, they all satisfy $d(u, v) + d(1, v) = d(1, u)$.
Can we narrow down?
If $u$ has parent $p$, and $q$ is another ancestor. $d(u, q) = d(u, p) + d(p, q)$.
This means $d(u, q) > d(u, p)$.
Also $d(u, p) \le K$.
So $d(u, q) - d(u, p) = d(p, q) \ge 1$.
If we compare $d(u, v)$ for various $v$ in the candidate window.
If $v$ is a valid ancestor, $d(u, v) = d(1, u) - d(1, v)$.
If $v$ is NOT an ancestor, $d(u, v) > d(1, u) - d(1, v)$.
So we just need to find if any $v$ satisfies the equality.
If there are multiple, any one is on the path, and since we want parent, we just pick the one with largest distance (closest to $u$).
But we still don't know WHICH one is closest.
Wait, if we just pick ANY $v$ that satisfies the equation, we find an ancestor.
Can we then query $d(u, v)$ for its parent?
Wait, if $v$ is not the parent, but a grandparent, we still found a path to root.
Then we can recurse to find the parent of $v$?
This sounds like we might need multiple queries.
But notice that we can just build the tree bottom-up.
Actually, if we process nodes $2 \dots n$ in some order, say by $d(1, x)$, then for each $x$, we just need to link it to the "best" predecessor.
If we find *any* $v$ such that $d(u, v) = d(1, u) - d(1, v)$, then $u$ belongs to the subtree rooted at $v$ (where $v$ is treated as root for now).
Actually, this allows us to merge sets?
Wait, the structure is a tree.
If we pick $v$ satisfying the condition, we know $u$ is a descendant of $v$ in the "current partial tree".
If we simply link $u$ to $v$, and later discover $v$'s parent is $w$, then $u$ becomes a descendant of $w$.
Wait, the edge weights depend on the direct parent.
If we link $u$ to $v$, we record edge $(u, v)$ with weight $d(1, u) - d(1, v)$.
Later if we realize $v$ connects to $w$ (so $w$ is parent of $v$), then $u$ is still connected to $v$, and path to root is via $v, w$. The weights on edges are preserved.
So, finding *any* ancestor is sufficient to place the node into the correct component relative to root.
Algorithm:
1. Query $d(1, i)$ for all $i$. Store pairs $(i, d(1, i))$. Sort by distance.
2. Initialize DSU structure or just a list of available nodes.
   Since it's a tree, all nodes must eventually belong to the component of 1.
   Let's maintain a data structure of "roots of subtrees" formed so far?
   Actually, simpler: Iterate nodes $u$ in increasing order of $d(1, u)$.
   Skip $u=1$.
   Find an ancestor $v$.
   Condition: $d(1, v) < d(1, u)$ and $d(u, v) + d(1, v) = d(1, u)$.
   If we find such $v$, add edge $(u, v)$ with weight $d(1, u) - d(1, v)$.
   Mark $u$ as visited/included.
   
   How to find $v$ quickly?
   For current $u$, the parent $p$ (and other ancestors) must satisfy $d(1, v) > d(1, u) - K$.
   Since we sorted by $d(1, \cdot)$, we can check $v$ in the range $[d(1, u)-K, d(1, u)-1]$.
   We check candidates $v$ from this window.
   Check if $d(u, v) = d(1, u) - d(1, v)$.
   If we find a match, we accept it.
   But which one?
   If we find $v_1$ and $v_2$ both satisfying the condition, one is parent, other is higher ancestor.
   The "higher" one has smaller $d(1, v)$.
   The "parent" has larger $d(1, v)$.
   We want any ancestor because it defines the edge $(u, v)$.
   Wait, is $(u, v)$ with $v=$ancestor a valid edge in the tree?
   NO. The tree edges are only between immediate neighbors.
   The problem asks to output edges and weights.
   If I output $(u, v)$ with weight $W = d(1, u) - d(1, v)$ where $v$ is grandparent, this is WRONG. The actual edges are $(u, p)$ and $(p, v)$. I would miss $(u, p)$ or invent wrong edge.
   So I MUST find the DIRECT parent.
   To do this, I must find the $v$ satisfying the condition with MAXIMAL $d(1, v)$.
   Since $d(1, v) > d(1, u) - K$, there might be multiple.
   BUT, note that if $v$ is a parent, $d(u, v) = d(1, u) - d(1, v)$.
   If $z$ is a parent of $v$ (so ancestor of $u$), $d(u, z) = d(u, v) + d(v, z) = (d(1, u) - d(1, v)) + (d(1, v) - d(1, z)) = d(1, u) - d(1, z)$.
   Both satisfy the condition.
   And $d(1, v) > d(1, z)$.
   So we prefer $v$ with largest distance.
   
   Can we find the best $v$ efficiently?
   Maybe we don't need to check all.
   Consider the set of candidates $C_u = \{v \mid d(1, v) \in (d(1, u)-K, d(1, u))\}$.
   We check each $v \in C_u$.
   If we find a $v$ satisfying condition, let's call it $cand$.
   Then we know $cand$ is an ancestor.
   Can we stop here and just link $u$ to $cand$?
   Wait, linking to $cand$ creates edge $(u, cand)$ of length $d(cand, u)$.
   If $cand$ is not parent, we fail.
   Wait, but if $cand$ is grandparent, does it matter?
   We need to construct the EXACT tree.
   So we need to output all edges.
   If we output edge $(u, \text{grandparent})$, the system will check if that's the actual edge.
   Since the problem statement says "Little Cyan Fish wants you to determine all the edges", we must return the $n-1$ edges existing in the tree.
   So yes, we must find the direct parent.
   
   However, wait!
   If we link $u$ to $v$ (where $v$ is an ancestor), and later we link $v$ to $w$ (ancestor of $v$), we effectively reconstruct the path $u-v-w-\dots$.
   Does this cover all edges?
   If we connect $u$ to $v$ (ancestor), we output edge $(u, v)$? No.
   If we output $(u, v)$, we claim there is an edge. If $v$ is grandparent, there isn't.
   So we must find the parent.
   
   But notice: If we successfully find $v$ such that $d(u, v) = d(1, u) - d(1, v)$, we know $v$ is on the path.
   Among all such $v$, the one with largest $d(1, v)$ is the parent.
   How to ensure we pick the one with largest $d(1, v)$?
   If we iterate $v$ from largest $d(1, v)$ downwards in the candidate window.
   The FIRST one that satisfies the condition IS THE PARENT.
   Because if there was another ancestor $z$ with larger $d(1, z)$, we would have seen it earlier.
   So we just need to iterate $v$ in the window $[d(1, u)-K, d(1, u)-1]$ in descending order of $d(1, v)$ and pick the first match.
   
   Problem: The window size could be up to $K$. $K=10^4$.
   Checking each takes 1 query.
   Total queries $\sum K \approx N \times K = 10^9$ worst case. Too slow.
   Time limit 2s. Operations $\approx 10^8$.
   We need to reduce the number of queries.
   Note that sum of $N$ over test cases is $10^5$.
   Wait, sum of $N$ is small, but $K$ is per testcase parameter.
   The constraint $K \le 10^4$ applies.
   However, we can't afford $O(N \cdot K)$.
   But average degree is low? No.
   
   Is it possible to optimize the search?
   We need $v$ such that $d(u, v) = d(1, u) - d(1, v)$.
   This is equivalent to $d(u, v) - d(1, v) + d(1, u) = 0$ ? No.
   It is $d(u, v) + d(1, v) - d(1, u) = 0$.
   We know $d(1, u)$ and $d(1, v)$ beforehand.
   We only query $d(u, v)$.
   Wait, $d(1, v)$ is already computed.
   So for $v$ in window, we just check $d(u, v)$.
   We need to check $v$'s until we find the one with largest $d(1, v)$ that works.
   Wait, maybe we don't need to check ALL.
   Most nodes won't be on the path to 1.
   If we pick $v$ randomly from the window, probability of being on path is small.
   BUT, if we have multiple candidates, we must distinguish.
   
   What if we don't check all $v$?
   Wait, is there a way to reduce the number of queries per node?
   Yes!
   Notice that we have processed nodes $1 \dots u-1$.
   If we store the tree built so far (edges $(x, parent(x))$).
   Then $u$ must attach to some node in the current tree.
   Since the tree is already constructed for "closer" nodes (by distance), $u$ attaches to someone with $d(1, p) < d(1, u)$.
   The parent $p$ MUST be connected to the rest of the structure.
   Wait, if we process in order of $d(1, \cdot)$, then when we are at $u$, all its ancestors are ALREADY processed and added to the tree.
   Why? Because for any ancestor $a$ of $u$, $d(1, a) < d(1, u)$ (since edge weights $\ge 1$).
   So $a$ was processed earlier!
   So we don't need to check ALL $v$ in window.
   We only need to check nodes that are part of the constructed tree?
   All nodes $1 \dots u-1$ are part of the tree? No, they are processed, but maybe some didn't connect to the main component?
   No, the final result is a single connected tree.
   So all processed nodes are part of the forest.
   Wait, if we process in distance order, all nodes closer to 1 are connected to 1.
   Why? Because 1 is the root.
   Any node $v$ with $d(1, v)$ minimal in the remaining set must connect to someone with even smaller $d(1, \dots)$.
   Since $d(1, 1)=0$, we start with $\{1\}$.
   Then we take node with smallest $d(1, v) > 0$. Its parent MUST be 1.
   Why? Because parent $p$ has $d(1, p) < d(1, v)$.
   And since $d(1, v)$ is minimal among unconnected nodes, and $d(1, p)$ is even smaller, $p$ must be connected (or be 1).
   By induction, if we process nodes by increasing $d(1, v)$, the parent of $v$ is ALWAYS already connected to 1 (since $d(1, p) < d(1, v)$ and we processed all $w$ with $d(1, w) < d(1, v)$).
   Wait, this is true ONLY IF $d(1, p)$ is small enough.
   Since $d(1, p) < d(1, v)$, and we process in strict increasing order, $p$ is definitely processed.
   Therefore, $p$ is already "active" in our tree construction.
   So for $u$, we only need to check candidates $p$ which are ALREADY IN THE TREE.
   And since $p$ is in the tree, it's connected to 1.
   We just need to find which $p$ in the tree is the parent.
   Wait, but checking all processed nodes is $O(u^2)$. Still slow.
   BUT, remember the constraint $d(u, p) \le K$.
   So $d(1, p) \ge d(1, u) - K$.
   Also, $p$ must be in the tree (connected to 1).
   Actually, since we process in order, if we use a data structure to query distances?
   Wait, checking all nodes in the tree is $O(N)$ per node.
   
   Wait, there's a simpler observation.
   The parent $p$ is the node $v$ in the processed set such that $d(1, p)$ is maximal AND $d(u, v) = d(1, u) - d(1, p)$.
   Wait, is it unique?
   Yes, only the parent satisfies this with the largest $d(1, p)$?
   Actually, any ancestor satisfies the equation.
   But among ancestors, the parent is the one with largest $d(1, p)$.
   So we need to check $v$ with $d(1, v) \approx d(1, u)$.
   
   Can we simply query $d(u, v)$ for $v$ in the window?
   If the number of nodes in the window is small, it's fast.
   If the number of nodes is large, maybe we can optimize.
   However, is it possible to have many nodes with distance in $[d(1, u)-K, d(1, u)]$?
   Yes.
   But most of them are not on the path to 1.
   For any node $v$ NOT on the path to 1, $d(u, v) > d(1, u) - d(1, v)$.
   Wait, $d(1, u) = d(1, a) + d(a, u)$ where $a$ is LCA($u, v$).
   $d(u, v) = d(u, a) + d(a, v)$.
   $d(1, u) - d(1, v) = (d(1, a) + d(a, u)) - (d(1, a) + d(a, v)) = d(a, u) - d(a, v)$.
   So $d(u, v) = d(a, u) + d(a, v)$.
   Condition for parent $p$: $d(u, p) = d(a, u) - d(a, p) = d(1, u) - d(1, v)$? No.
   Parent $p$: $a=p$ (since $p$ is on path). $d(u, p) = d(a, u)$.
   RHS: $d(1, u) - d(1, v) = d(a, u) - d(a, v)$.
   So if $v$ is parent, $d(a, v) = 0 \implies a=v$.
   If $v$ is not parent, but $v$ is a node on path to 1 (ancestor), then $a=v$.
   Then $d(u, v) = d(a, u) = d(1, u) - d(1, v)$.
   So ANCESTORS satisfy equality.
   Other nodes $z$ have $a=LCA(u, z)$.
   Then $d(u, z) = d(u, a) + d(a, z) = (d(1, u) - d(1, a)) + (d(1, z) - d(1, a))$.
   We check if $d(u, z) = d(1, u) - d(1, z)$.
   Substitute: $d(1, u) + d(1, z) - 2 d(1, a) = d(1, u) - d(1, z)$.
   $\implies 2 d(1, z) - 2 d(1, a) = 0 \implies d(1, z) = d(1, a)$.
   This implies $z$ is on the path from $a$ to $u$??
   Wait, if $d(1, z) = d(1, a)$, then $z$ is at same level as LCA.
   In a tree, if $d(1, z) = d(1, a)$ and $a=LCA(u, z)$, then $z$ must be on the path 1 to $a$.
   Since $d(1, z) = d(1, a)$, $z=a$.
   So ONLY ANCESTORS satisfy the equation.
   Wait, really?
   $d(u, v) = d(1, u) - d(1, v)$.
   If $v$ is NOT an ancestor, then $LCA(u, v) \neq v$. Let $l = LCA(u, v)$.
   Then $d(u, v) = d(u, l) + d(l, v) = (d(1, u) - d(1, l)) + (d(1, v) - d(1, l))$.
   Equation requires $d(1, u) - d(1, l) + d(1, v) - d(1, l) = d(1, u) - d(1, v)$.
   $\implies 2 d(1, v) = 2 d(1, l) \implies d(1, v) = d(1, l)$.
   This implies $v$ lies on the path 1 to $l$.
   Since $l$ is on the path 1 to $v$, and $d(1, v) = d(1, l)$, it implies $v=l$?
   Actually, if $v$ is on path 1-l, and $d(1, v) = d(1, l)$, then $v=l$.
   So indeed, only the ancestors satisfy the condition.
   And we know parent is the deepest ancestor.
   
   So the strategy remains: Find ancestor $v$ with largest $d(1, v)$.
   To do this efficiently, we need to avoid checking all nodes.
   But we know $v$ is in the window $[d(1, u)-K, d(1, u)]$.
   Also $v$ is an ancestor, so $v$ must be "reachable" from 1.
   Wait, can we use the fact that ancestors form a chain?
   For a fixed $u$, the ancestors form a sequence $p_1, p_2, \dots, p_k=u$ (reversed).
   $d(1, p_{i+1}) - d(1, p_i) = w_i$.
   This is just distance along the path.
   
   Actually, we can use the randomized approach or just hope for the best?
   No, must be deterministic/correct.
   Wait, is there an $O(N \log N)$ or $O(N)$ method?
   Given $K$ is small, maybe we can maintain a set of active nodes and query them?
   But we can query at most $O(N)$ times total.
   Wait, $N=10^5$. We can afford $O(N)$ queries.
   But here for EACH node $u$ we need to find $v$.
   If we check too many $v$, total exceeds $N$.
   
   Key realization:
   If we have $d(1, u)$ for all $u$.
   We can simply compute the MST? No, edge weights unknown.
   
   Wait, what if we query $d(1, u)$ and then for each $u$, we query $d(u, v)$ for the BEST candidates?
   Candidates are nodes $v$ with $d(1, v) \ge d(1, u) - K$.
   There might be many.
   But notice that if $v$ is not an ancestor, $d(u, v)$ will be STRICTLY GREATER than $d(1, u) - d(1, v)$.
   Specifically $d(u, v) \ge d(1, u) - d(1, v) + (\text{something positive})$.
   Wait, $d(u, v) = d(1, u) - d(1, v) + 2(d(1, v) - d(1, l))$.
   Since $d(1, v) \ge d(1, l)$, the difference is non-negative.
   It is 0 iff $v$ is ancestor.
   So for all non-ancestors, $d(u, v) > d(1, u) - d(1, v)$.
   For ancestors, $d(u, v) = d(1, u) - d(1, v)$.
   
   So for each $u$, we need to find $v$ such that $d(u, v) == d(1, u) - d(1, v)$.
   If we check nodes in descending order of $d(1, v)$, the FIRST one satisfying the equation is the PARENT.
   The question is: how many nodes do we need to check before we find the parent?
   It could be $K$.
   If $K=10^4$, and we do this for $10^5$ nodes, total operations $10^9$.
   However, sum of $N$ over test cases is $10^5$.
   And $K$ is fixed? No, per test case.
   Is it possible that $K$ is always small on average?
   Or maybe we don't need to check $K$ nodes.
   What if we maintain the "last seen valid ancestor"?
   No, structure changes.
   
   Let's check constraints again.
   Maybe we can use the fact that we can only make queries.
   Wait! We don't need to know the structure to answer the problem?
   We need to output the structure.
   
   Is there a trick with $K$?
   If $K$ is small, we can use buckets.
   Group processed nodes by $d(1, v) \pmod M$? No.
   Group by $d(1, v)$.
   Since we process by distance, we only care about nodes with $d(1, v)$ in range.
   Let $Count(d)$ be the list of nodes with distance $d$.
   For node $u$ at distance $D$, we check lists $Count(D-K), \dots, Count(D-1)$.
   If there are many nodes, we can't check all.
   
   Wait, the problem is interactive. The judge generates the tree.
   Is it possible to simply try to query $d(u, v)$ where $v$ is a random node from the "window"?
   If $u$ has many ancestors in the window (unlikely if $K$ is small), then checking one might fail.
   But ancestors are unique. Only one parent.
   Grandparents are also ancestors.
   If we check the one with largest distance, it's parent.
   
   Wait, what if we simply assume that the parent is among the nodes in the previous "layer" of the BFS?
   If we organize nodes by layers (by distance), then for $u$ in layer $L$, its parent is in layer $L-1$?
   Not necessarily, edge weights can be anything in $[1, K]$.
   So parent could be in $L-1, L-2, \dots, L-K$.
   
   Actually, the key insight for this problem on online judges (this looks like a specific contest problem, possibly Codeforces Round #786 Div 1 A or something similar? Or "Interactive Tree Reconstruction") is often simpler.
   Re-read carefully.
   "Little Cyan Fish... generate a random labeled tree".
   "Uniformly randomly select... nn-2".
   "Assign random integer edge weights".
   "K is hidden parameter".
   
   Wait, the example shows $N=3$, output `! 1 2 3 2 3 4`.
   Edges (1,2) w=3, (2,3) w=4.
   Queries used: (1,2)->3, (2,3)->4, (1,3)->7.
   This confirms $d(1, 2) + d(2, 3) = d(1, 3)$.
   
   The problem is essentially: Given $d(u, v)$ queries, find tree edges.
   We can implement a simplified version:
   For each $u$, query $d(1, u)$.
   Then iterate $u=2 \dots n$.
   Try to find parent $p$ by checking nodes $v$ with $d(1, v) \in [d(1, u)-K, d(1, u)]$.
   Since we must finish in time, maybe the "randomized" heuristic or "check only nearby" is intended.
   BUT, we need to be sure.
   Wait, is there a theoretical bound on number of queries?
   The problem statement says "determine... as less as you can".
   Scoring suggests $O(N)$ is fine.
   If the logic holds that we just need to check a few nodes, great.
   But worst case, checking $K$ nodes is bad.
   However, note that we only check nodes $v$ that are ALREADY CONNECTED to 1 (ancestors).
   Wait, if we process in order of $d(1, u)$, then ALL $v$ with $d(1, v) < d(1, u)$ are candidates.
   Do we know which ones are ancestors?
   Only if we know the tree structure.
   But we don't.
   
   Wait, I found a reference to a similar problem (Codeforces 1835C? No).
   The problem is likely "Interactive Tree" or "Tree Reconstruction".
   A common solution involves maintaining a "frontier" of nodes.
   If we pick a random node from the set of nodes with $d(1, v) < d(1, u)$ that hasn't been "matched" as a child yet?
   No.
   
   Let's think about the complexity.
   Total $N \le 10^5$.
   If we just check all $v$ with $d(1, v) < d(1, u)$? $O(N^2)$.
   We must limit to $O(N)$.
   Maybe we don't need to check $d(u, v)$.
   Wait, if we just use $d(1, u)$?
   If we have $d(1, u)$ for all $u$, can we recover edge weights?
   Suppose we sort by $d(1, u)$.
   If $d(1, u) - d(1, v)$ is minimal among all $v < u$?
   Not necessarily.
   
   Let's consider the specific property: $K$ is small.
   This means depth of tree is not restricted, but local depth change is small? No.
   This means $d(1, u) - d(1, p) \in [1, K]$.
   This is the key.
   This implies that for any $u$, its parent $p$ has $d(1, p) \in [d(1, u)-K, d(1, u)-1]$.
   Is it possible to quickly find the right $p$?
   What if we keep track of the "potential parents" for the next nodes?
   For a node $v$, if it satisfies $d(1, v) \in [d(1, u)-K, d(1, u)]$ and $d(u, v)$ matches, it is a candidate.
   Actually, if we just collect all pairs $(u, v)$ with $v$ processed and $d(1, v) \ge d(1, u) - K$, can we optimize?
   
   Wait! The number of queries is what matters.
   If we use the following:
   For each $u \in [2, n]$:
     We know $d(1, u)$.
     We look for $v$ such that $d(1, v) < d(1, u)$ and $d(u, v) == d(1, u) - d(1, v)$.
     We want the one with largest $d(1, v)$.
     But we can't check all.
     Wait, is it guaranteed that such $v$ is unique in the window?
     Yes, parent is unique.
     Are there other nodes $z$ in window that satisfy $d(u, z) == d(1, u) - d(1, z)$?
     Yes, ancestors.
     So if we check $v$'s in descending order of $d(1, v)$, the first match is parent.
     
     Now, how many matches can there be? Up to $K$.
     But we only care about the FIRST match.
     Wait, if there are many ancestors in the window, we have to skip them.
     If we can prove that typically there are few ancestors in the window?
     In a random tree, path length is logarithmic? No, $K$ restricts step sizes, but random tree has varying depths.
     Average depth is $O(\sqrt{N})$? No, expected depth is $O(\sqrt{N})$ for uniform random tree.
     Max depth $O(N)$.
     With weights, distances are spread.
     
     Wait, actually, is there a constraint that prevents $K$ being large for many nodes?
     $K \le 10^4$.
     Maybe the intended solution assumes we can just check the window?
     If $N=10^5$ and $K=10^4$, $NK = 10^9$.
     However, time limit is 2s. Typical ops $10^8$.
     It is risky.
     BUT, sum of $N$ over test cases is $10^5$.
     If $N$ is small, $K$ is large? No constraint.
     Wait, the constraints say "sum of all n in the test data does not exceed 10^5".
     It does NOT bound $T$ times $K$ directly, but $T \le 10^4$.
     So if we have $10^4$ test cases with small $N$ and $K=10^4$?
     E.g. $N=10, T=10^4$.
     Then $\sum N = 10^5$.
     If for each case we check $K=10^4$ nodes, total queries $10^4 \times 10^4 = 10^8$.
     This is acceptable! $10^8$ operations might fit in 2s.
     The check is simple subtraction and lookup.
     Wait, we don't need to query $d(u, v)$ if $v$ is not in the window?
     The check condition is $d(u, v) = d(1, u) - d(1, v)$.
     We need to query $d(u, v)$ to verify.
     So for each $u$, we iterate through all nodes $v$ in the window.
     Number of nodes in window $\le K$.
     So total queries $\approx \sum N \times \min(N, K)$.
     Wait, window size is at most $K$.
     If $K > N$, window is size $N$.
     But sum of $N$ is $10^5$.
     If $K$ is large (e.g. $10^4$) and $N$ is small (e.g. $10$), cost is $N \times K \approx 10 \times 10^4 = 10^5$.
     If $N$ is large (e.g. $10^5$) and $K$ is small (e.g. 1), cost is $N \times 1 = 10^5$.
     The worst case is $N \approx 10^5$ and $K \approx 10^5$ (but $K \le 10^4$).
     So worst case is $N \approx 10^5$ and $K \approx 10^4$.
     Wait, $\sum N \le 10^5$.
     So $N$ cannot be $10^5$ in many cases.
     In the worst case (one test case), $N=10^5$.
     Then queries $10^5 \times 10^4 = 10^9$.
     Too many.
     We need a faster way to find the parent.
     
     Wait, we only need to query $d(u, v)$ for candidates.
     But we can optimize the search.
     We need the *largest* $d(1, v)$ such that $v$ is an ancestor.
     Notice that if $v$ is an ancestor, it is connected to $u$.
     Also, $v$ must be part of the tree built so far.
     Wait, we can maintain the set of active nodes.
     For each node $u$, we want to find $p \in ActiveNodes$ s.t. $p$ is ancestor.
     This seems hard to filter.
     
     Alternative Idea:
     Use Binary Search?
     No.
     
     What if we don't use $K$?
     Wait, what if we use the fact that for ANY node $v$, the parent is $u$ such that $d(1, u) \ge d(1, v) - K$?
     Wait, the problem is asking to reconstruct.
     Maybe we can simply query $d(u, v)$ for a subset of nodes?
     For example, if we pick $v$ randomly from the whole set of processed nodes?
     Probability that a random $v$ is the parent is $1/deg(u)$?
     Or rather, the chance it is ON the path is $1/|PathFromRoot(u)|$.
     Path length can be up to $K$.
     So prob is $1/K$. Too low.
     
     Wait, look at the constraints and type of problem again.
     Maybe there is a simpler invariant.
     If we just pick any two nodes $u, v$, we get distance.
     What if we run Prim's algorithm?
     Start with set $S=\{1\}$.
     At each step, choose $u \notin S$ such that $\min_{v \in S} d(1, u) - d(1, v)$ is minimized?
     Wait, $d(1, u)$ is known.
     We need to know $d(u, v)$.
     But we know $d(1, u) = d(1, v) + d(v, u)$ if $v$ is parent.
     So $\min_{v \in S} (d(1, u) - d(1, v) = d(u, v))$.
     This looks like we are searching for $v$ that minimizes $d(1, u) - d(1, v)$ subject to $d(1, u) - d(1, v) \le K$ and $d(u, v) = d(1, u) - d(1, v)$.
     Wait, we need $d(u, v)$ to verify.
     So we iterate $v \in S$ such that $d(1, v) > d(1, u) - K$.
     We check $d(u, v)$.
     If match, then $v$ is an ancestor.
     Among all matches, pick one with largest $d(1, v)$ (closest parent).
     
     Can we speed up "find any match"?
     If we simply check $v$'s in decreasing order of $d(1, v)$.
     Is it possible to prune?
     Maybe the number of ancestors is small?
     If the tree is deep (line), ancestors are many.
     But if tree is deep, edges are long? No, weights $\le K$.
     If line, depth $\approx N/K$.
     Number of ancestors in window $[D-K, D]$ is 1?
     In a line $1-2-3-\dots$, $d(1, i) = c \cdot i$.
     Window width $K$.
     If $c$ is small, window covers many nodes.
     Wait, $d(1, p)$ decreases by $w_{parent}$.
     If $w=1$, then $d(1, p) = d(1, u) - 1$.
     Then for $u$, parent is $u-1$, grandparent $u-2$.
     Window $[d(1, u)-K, d(1, u)]$ contains $u, u-1, \dots, u-K$.
     All are ancestors!
     So we check $u-1$, find match (dist=1). It is parent. Done.
     If we check $u-2$, it matches (dist=2).
     But since we check descending distance ($u-1$ first), we find parent immediately.
     In a star graph (center 1, leaves $2 \dots n$).
     $d(1, u) = w_u$.
     Window $[w_u - K, w_u]$.
     For $u$, we check candidates $v$ with $d(1, v)$ in range.
     $v$ is not an ancestor unless $v=1$ (root) or $v$ is on path.
     In star graph, only 1 is ancestor.
     So if $w_1=0$ (impossible, min 1? No 1 is vertex). $d(1, 1)=0$.
     If $w_u < K$, 1 is in window.
     Check 1. $d(u, 1) = w_u$. $d(1, u) - d(1, 1) = w_u$. Match!
     So we find 1. Correct.
     
     What if it's a line $1 - 2 - 3 \dots$ but with weights $K$?
     $d(1, u) = (u-1)K$.
     Window $[(u-2)K, (u-1)K]$.
     Contains nodes $u-1, u-2$ (approx).
     Actually, $d(1, u-1) = (u-2)K$. Matches lower bound.
     $d(1, u-1) = d(1, u) - K$.
     Check $u-1$. $d(u, u-1) = K$.
     Formula: $d(1, u) - d(1, u-1) = K$. Match.
     Found parent $u-1$.
     
     So in "regular" cases, parent is found quickly.
     When is it bad?
     If there are MANY ancestors in the window, we have to skip them.
     But we process them in descending order.
     We check $v_1$ (dist $D$). Match? Yes -> Parent.
     Check $v_2$ (dist $D-1$). Match? Maybe.
     Wait, if we find a match, is it GUARANTEED to be the parent?
     Yes! Because parent has the largest distance among all ancestors.
     Proof: Ancestors form a chain. $p_1$ (parent), $p_2$ (grandparent)...
     $d(1, p_1) > d(1, p_2) > \dots$.
     So iterating descending ensures we see parent first.
     So we STOP at the first match.
     The worst case is when the first candidate in the window is NOT an ancestor.
     How many non-ancestors can be in the window?
     Potentially many.
     But maybe the density of ancestors in the distance space is low?
     For a random tree, $d(1, v)$ are somewhat evenly distributed?
     If so, checking is efficient.
     Worst case: Line graph with small weights, many ancestors.
     But we checked that in line graph with small weights, parent is at $d=K$ or $d=1$.
     If weights are small, steps are small.
     If weights are large (e.g. $K$), then parent is at distance $\approx K$.
     Actually, parent distance $d(1, p)$ is in $[d(1, u)-K, d(1, u)-1]$.
     We check candidates $v$ with $d(1, v)$ in that range.
     The parent itself is in that range.
     But other nodes might also be in that range and happen to satisfy the condition.
     If they satisfy the condition, they are ANCESTORS.
     So any match we find is an ancestor.
     And since we iterate descending distance, the FIRST match is the PARENT.
     So we never check non-ancestors if they are in the window.
     Wait, if a node $z$ is NOT an ancestor, it CANNOT satisfy the condition $d(u, z) = d(1, u) - d(1, z)$.
     So checking $z$ returns false.
     So we only "pay" for non-matches.
     If $z$ is an ancestor, we find a match.
     So we only check $v$'s that are not ancestors until we hit the parent?
     Wait, if $z$ is an ancestor, it matches.
     And we want the ONE with LARGEST distance.
     If there are multiple ancestors in the window, say $p$ (parent) and $gp$ (grandparent).
     $d(1, p) > d(1, gp)$.
     So if we iterate descending $d(1, v)$, we check $p$ before $gp$.
     $p$ matches. We stop.
     So we never check $gp$.
     So the cost is determined by: How many non-ancestors are in the window?
     We check $v$ until we find the parent.
     The window size is $K$.
     So in worst case we check $K$ nodes.
     Total $N \times K$.
     Is there a case where we check many non-ancestors?
     We need many nodes $v$ with $d(1, v) \in [d(1, u)-K, d(1, u)]$ such that $d(u, v) \neq d(1, u) - d(1, v)$.
     This happens if $v$ is far away but distance from 1 is close.
     This is exactly the "dense packing" case.
     Can we optimize?
     Maybe we can check a subset?
     Wait, if $v$ is NOT an ancestor, then $d(u, v) + d(1, v) > d(1, u)$.
     This inequality is strict.
     So we are searching for $v$ satisfying $d(u, v) + d(1, v) = d(1, u)$.
     This is a Diophantine-like equation check.
     
     Wait! The problem is interactive.
     We can use the result of queries.
     Is it possible to just submit the solution assuming the worst case is rare?
     Or is $N \cdot K$ acceptable?
     With $N=10^5, K=10^4$, $10^9$ ops is slightly too much for 2s.
     Typically $10^8$.
     Maybe the number of valid candidates in the window is small?
     Or the constant factor is small.
     
     Actually, there is a trick.
     We don't need to check ALL nodes in the window.
     We only need to find ONE node that satisfies the condition.
     Can we do a random sampling?
     Pick a random $v$ in the window. Check if $d(u, v) = d(1, u) - d(1, v)$.
     If yes, check if $v$ is parent?
     How? Query $d(v, u')$ for child of $u$?
     If we find ANY ancestor $a$, then $p$ (parent of $u$) is a descendant of $a$.
     Wait, $d(1, p) < d(1, u)$.
     If we have an ancestor $a$, then $p$ is either $a$ or a child of $a$ on path to $u$.
     Wait, $a$ is ancestor. Path $1 \to \dots \to a \to \dots \to u$.
     $p$ is next node after $a$ on this path.
     But $p$ might not be connected to $a$ directly in our knowledge.
     However, we know $p$ is a neighbor of $u$.
     $d(u, p) = d(1, u) - d(1, p)$.
     So if we find $a$, we haven't narrowed down much, except $d(1, a) \ge d(1, u) - K$? No.
     Wait, if $a$ is ancestor, $d(1, a)$ can be anything.
     But we established $p$ must satisfy $d(1, p) > d(1, a)$? No.
     $d(1, p)$ is between $d(1, u)$ and $d(1, a)$? No.
     $d(1, p)$ is close to $d(1, u)$? Not necessarily.
     But $d(1, p) > d(1, a)$ is false. $d(1, p) < d(1, u)$.
     The window condition $d(1, v) \ge d(1, u) - K$ is derived from $w_{up} \le K$.
     So $d(1, u) - d(1, p) = w \le K \implies d(1, p) \ge d(1, u) - K$.
     So PARENT is definitely in the window.
     Any OTHER ancestor $a$ satisfies $d(1, u) - d(1, a) = d(a, u)$.
     Since $d(p, u) \le K$, $d(a, u) = d(a, p) + d(p, u)$.
     If $d(a, p) \ge 1$ (distinct nodes), then $d(a, u) > K$.
     So $d(1, u) - d(1, a) > K$.
     So $d(1, a) < d(1, u) - K$.
     WAIT! This is huge.
     If $a$ is an ancestor (other than parent), then $d(a, u) = d(a, p) + d(p, u)$.
     Since edge weights are $\ge 1$, and $a$ is grandparent, $d(a, p) \ge 1$.
     So $d(a, u) \ge 1 + 1 = 2$.
     Wait, $d(a, u)$ is path length.
     The condition is $w_{pa} \le K$.
     $d(1, a) = d(1, p) + d(p, a)$.
     $d(1, u) = d(1, p) + d(p, u)$.
     $d(1, u) - d(1, a) = d(p, u) + d(p, a)$.
     This can be larger than $K$!
     My previous deduction "parent in window" relied on $w_{parent} \le K$.
     The parent $p$ has $d(1, p) \ge d(1, u) - K$.
     Other ancestors $a$:
     $d(1, u) - d(1, a) = \text{path } a \to u$.
     If $a$ is parent, path len is $w \le K$.
     If $a$ is grandparent, path len is $w_{ap} + w_{pu}$.
     This could be $> K$.
     So, GRANDPARENTS ARE NOT NECESSARILY in the window $[d(1, u)-K, d(1, u)]$.
     In fact, they are usually further away.
     So in the window $[d(1, u)-K, d(1, u)]$, the ONLY ANCESTOR is the PARENT!
     (Assuming edge weights $\ge 1$).
     Wait, is it possible to have two edges of weight $\le K/2$ such that path from grandparent to leaf is $\le K$?
     Yes. $w(a, p) = 1, w(p, u) = 1$. $K=100$.
     Then $d(1, u) - d(1, a) = 2 \le K$.
     So grandparents COULD be in the window.
     However, they are sparse.
     In the window, we have the parent.
     Any other node $v$ in window is either an ancestor (grandparent etc.) or not an ancestor.
     If it's not an ancestor, it fails check.
     If it is an ancestor, it passes check.
     Since we want the parent, we need to distinguish it from grandparent.
     Since $d(1, p) > d(1, gp)$, we just check descending order.
     The first match IS the parent.
     So we just need to iterate descendants (in distance space) in the window.
     And since we want to find the first match, we can just check them.
     If the parent is at distance $D-1$ and grandparent at $D-2$.
     We check $v$ at $D-1$. If match, stop.
     If not match, check $D-2$.
     Wait, checking $D-1$ fails. This means $D-1$ is NOT an ancestor.
     So we check $D-2$. If match, stop (it is GP, but wait, parent must be between? No, we assumed $p$ is in window).
     If $p$ is in window, we will eventually find it.
     Are there cases where we have many non-ancestors in window?
     Yes.
     Can we bound the number of checks?
     Maybe we can maintain a list of "potential parents"?
     
     Actually, let's look at the constraint $\sum N \le 10^5$.
     If we write efficient code, maybe $10^9$ operations is overkill estimate?
     Wait, we perform a query for EVERY check.
     The query is the bottleneck.
     We cannot afford $10^9$ queries.
     There MUST be a way to skip non-ancestors.
     Or use the fact that $K$ is small.
     
     Wait, I missed something.
     $K$ is a HIDDEN parameter.
     But we know $K \le 10^4$.
     If $K$ is small, the number of nodes in window is bounded by $K$?
     Not by $K$. Nodes are arbitrary.
     But if $K$ is small, then the number of ancestors in the window is SMALL?
     No, in a line graph, it can be large.
     
     Is there any property of "Random Labeled Tree"?
     Maybe the tree is unlikely to be a line with small weights?
     But we must handle worst case.
     
     What if we use a randomized hashing?
     Map $d(1, v)$ to some bucket?
     If we assign random IDs to nodes?
     Wait, what if we simply check the "best" candidate based on distance?
     Let's just implement the naive solution but optimized:
     We store all nodes sorted by $d(1, v)$.
     For each $u$, we check candidates $v$ starting from index corresponding to $d(1, u)-1$ backwards.
     Wait, we can maintain a list of nodes for each distance.
     Since distances are integers.
     If distance values are dense, we have many buckets.
     If sparse, few.
     Wait, $d(1, v)$ can be up to $N \times K \approx 10^9$.
     We can't use array. Map or vector of vectors.
     
     Actually, looking at the limits again.
     The scoring allows up to $5n$ queries.
     Wait, $5n = 5 \times 10^5$.
     Total $N \le 10^5$.
     So $O(N)$ queries is fine.
     But if we do $O(N \cdot K)$, that's $10^9$, which is $O(K \cdot N)$, much worse.
     We need $O(N)$.
     
     There is a specific technique called "Distance Oracle" reconstruction.
     If we query $d(1, u)$ for all $u$, we get distances.
     Then for each $u$, we want to find parent.
     The problem is finding the parent in $O(1)$ time.
     Wait, if we can maintain the tree as we go.
     When we add $u$, we link to $p$.
     This updates the tree.
     
     Is there a randomized greedy approach that works in expectation?
     Or maybe the test cases are weak?
     Usually, "Little Cyan Fish" implies a Codeforces style problem (Cyan Fish is a character in CF?).
     If it's Codeforces 1835C, the solution is to use randomization?
     Actually, for this specific problem (likely from a contest like "ICPC Asia Shenyang"), the solution uses the property that with random weights, the parent is unique and can be identified.
     But we need exact.
     
     Let's assume the number of checks is small on average.
     But we must pass worst case.
     Maybe $N$ is small when $K$ is large?
     Wait, if $N=10^5$, then $\sum N=10^5$, so only 1 test case.
     In that case, if $K=10^4$, we can check $10^4$ nodes for each of $10^5$ nodes? No.
     Total time 2s.
     If we spend $10^9$ queries, we TLE.
     
     Wait! We can optimize the parent search.
     We have $d(1, v)$.
     We want $v$ such that $d(1, v) \ge d(1, u) - K$ AND $d(u, v) = d(1, u) - d(1, v)$.
     Note that $d(u, v) + d(1, v) = d(1, u)$.
     Let $f(u, v) = d(u, v) + d(1, v)$. We want $f(u, v) = d(1, u)$.
     Since $d(u, v)$ is symmetric.
     What if we maintain the set of active nodes $S$.
     For a new node $u$, we want $p \in S$ minimizing $d(1, u) - d(1, p)$ such that $d(u, p) + d(1, p) = d(1, u)$.
     Actually, we know $p$ must be in $S$.
     And $d(1, p) > d(1, u) - K$.
     
     Wait, if we just check $v$'s that are "recently added" or something?
     What if we just pick the $v$ with largest $d(1, v)$?
     If we check only ONE candidate: $v$ with max $d(1, v)$ in the window.
     If it works, done.
     If not, it's not an ancestor.
     Then check next largest?
     If the tree is dense with non-ancestors, this is slow.
     BUT, note that $d(1, v)$ determines the layer.
     In a random tree, the number of nodes with specific distance $X$ is small.
     Actually, maybe we can just check $O(\text{degree})$? No.
     
     Wait, is there an alternative interpretation?
     "Little Cyan Fish... weighted tree... uniform random labeled tree... random integer edge weights".
     This implies the input is statistically typical.
     Usually random graphs have small diameter.
     For random labeled tree (uniform), average depth is $O(\sqrt{N})$.
     With random weights, distances are spread out.
     The probability that many nodes fall into the distance window $[D-K, D]$ is low if weights are continuous.
     But weights are integers in $[1, K]$.
     However, even with integers, the distribution of distances tends to follow Poisson or normal.
     Density is low.
     So checking nodes in descending order of $d(1, v)$ will quickly find the parent (if it exists) or confirm it's not in the window (wait, parent IS in window).
     So if the window contains non-ancestors, we might skip them.
     BUT, due to randomness, the chance of non-ancestors being in the window and "blocking" is low?
     Or rather, the window might contain few nodes at all.
     For $N=10^5$, if distances are random in $[1, N \cdot K]$, the density is low.
     Number of nodes with distance $\approx D$ is small (around $N / \sqrt{N}$?).
     So checking the few candidates is fast.
     This explains why $O(N \cdot K)$ analysis is pessimistic.
     The actual number of candidates in the window is likely $O(1)$ or $O(\log N)$.
     So the naive check should pass.
     
     One detail: $d(1, u)$ can be same for multiple nodes?
     Yes. If $d(1, u) = d(1, v)$, they are at same level.
     Then neither can be parent of other.
     So parent must have strictly smaller distance.
     We can group by distance.
     Process layers from 0 upwards.
     This avoids checking nodes with equal distance.
     
     Implementation Plan:
     1. Read T. Loop T times.
     2. Read N.
     3. Query $d(1, i)$ for $i=2 \dots n$.
        Store in pairs `(d, u)`.
     4. Sort pairs by distance.
     5. Iterate $i$ from 0 to $n-1$.
        Let $u$ be the node.
        If $u == 1$, continue.
        We need parent $p$.
        Candidates are nodes $v$ with $d(1, v) > d(1, u) - K$.
        Since we sorted, we can use a sliding window or pointer.
        Maintain a list of active nodes $S$ with distance $> d(1, u) - K$.
        Actually, since we process in increasing distance, we can maintain a collection of all previously processed nodes.
        But we want to check efficiently.
        Collect all processed nodes into a list/vector? No, $O(N)$.
        Better: Since we process in sorted order, for current $u$, we consider $v$ with $d(1, v)$ just below $d(1, u)$.
        Actually, if $d(1, v) \approx d(1, u)$, they are likely siblings or cousins.
        Parents must be somewhat close in distance (due to $K$).
        So we can maintain a vector of processed nodes.
        Since we sort by distance, we just look at indices.
        But checking all is slow.
        Wait, if we store nodes in a vector `nodes_by_dist`.
        For current $u$, iterate backwards from the node with distance just below $d(1, u)$?
        We can start search from index `max_idx` such that `nodes[max_idx].d >= d(1, u) - K`.
        Go downwards.
        Stop when we reach end or find a match.
        If we find match, parent is $v$. Add edge.
        Else (no match?), then something wrong?
        Wait, parent MUST be an ancestor, so it MUST match.
        If we check enough nodes (up to $K$), we must find it.
        Wait, do we need to check all $K$?
        As argued, candidates are few.
        If $N$ is large and tree is random, the number of nodes with distance $\approx D$ is small.
        So we check few nodes.
        
        Wait, do we need to store all nodes? Yes.
        The number of nodes is $10^5$.
        Vector of vectors? Or just one sorted vector?
        Sorted vector of structs `{id, dist}`.
        Iterate $u$.
        Identify range $[idx\_start, idx\_end]$ such that $d$ in window.
        Since we process in order, $idx\_start$ moves slowly.
        We can maintain `ptr` such that `nodes[ptr].dist >= d(1, u) - K`.
        Start searching from `ptr`.
        Go backwards (decreasing index).
        For each `candidates[j]`:
          Query $d(u, candidates[j].id)$.
          If $d == d(1, u) - candidates[j].dist$:
             This is an ancestor.
             We need the DEEPEST ancestor.
             Since we search backwards (highest dist first), the first match is the parent.
             Break and stop.
          
        Complexity: Sum of loop iterations.
        In worst case, loop runs $K$ times.
        Total $O(N \cdot K)$.
        Given time limit and nature of problem, this should pass.
        
        Corner case: $u=1$ handled.
        Output format: `! u1 v1 w1 ...`
        Need to store edges.
        
        Wait, memory limit 512MB. Storing everything is fine.
        
        One optimization:
        We only need to check nodes $v$ with $d(1, v)$ close to $d(1, u)$.
        Actually, since we iterate $u$ in increasing order of distance, we only need to check $v$ that appear AFTER the "valid ancestor" for current $u$.
        Wait, we need $v$ to be already processed.
        The condition $d(1, v) < d(1, u)$ is automatically satisfied by sorted order.
        The condition $d(1, v) > d(1, u) - K$ is enforced by `ptr`.
        The condition that $v$ is ancestor is checked by query.
        
        Wait, do we need to query for ALL candidates?
        What if we pick a random candidate?
        If we pick random, we risk failure.
        Checking in descending distance order is safer.
        
        Final check on constraints:
        "interaction library is non-adaptive".
        This means tree is fixed.
        But my query strategy depends on $K$?
        Wait, $K$ is hidden.
        We cannot use $K$ in the query count logic?
        Wait, we don't need to know $K$.
        We just know edge weights are $\ge 1$.
        And we query $d(1, u) - d(1, v)$.
        Wait, if edge weights can be large, we might not find parent in window.
        But weights are in $[1, K]$.
        If $K$ is large, say $10^4$, window is large.
        If $K$ is small, window is small.
        But wait, we are GIVEN $K$ is hidden.
        Wait, does the problem give us $K$?
        "K is a hidden parameter."
        But we know "guaranteed that $1 \le K \le 10^4$".
        So we know the upper bound.
        But we don't know the value.
        Should we assume worst case $K=10^4$?
        If we assume $K=10^4$, we check 10000 nodes.
        This is consistent.
        Wait, is it possible to solve without knowing $K$?
        Yes, because we just stop when we find parent.
        If weights are 1, we check at most $N$ nodes?
        No, if weights are 1, parent distance is $d(1, u) - 1$.
        If we check nodes with distance $d-1$, we find it.
        We only need to search down to $d(1, u) - K$.
        Wait, if we don't know $K$, how do we decide when to stop?
        We can stop as soon as we find a match.
        But we might check non-ancestors.
        If we check many non-ancestors, it's slow.
        Is there a way to know when to stop without checking?
        No.
        But we rely on the fact that non-ancestors rarely satisfy the distance equation.
        So we will mostly reject non-ancestors and accept parent.
        The question is "how many non-ancestors to check".
        If we check them in random order, we might hit a parent early?
        No, we need parent.
        But we know parent is the match.
        If we just sample candidates?
        If we query a few random candidates and find a match.
        Then check if it's the parent.
        If we find an ancestor $a$, is it the parent?
        Only if $d(u, a) \le K$.
        We don't know $K$.
        But we know weights are $\le K$.
        Wait, if $a$ is an ancestor and we found it via distance check $d(u, a) = d(1, u) - d(1, a)$.
        This implies $d(u, a) \le K$ is implicitly required?
        No, $d(u, a)$ is the sum of weights on path.
        If $a$ is grandparent, $d(u, a) = w(p, u) + w(p, a)$.
        If we stop at first match, and we picked candidates randomly.
        If we pick grandparent, we find a match.
        But we output $(u, a)$ with weight $d(u, a)$.
        This is incorrect if $a$ is not parent.
        So we MUST find the PARENT.
        The only way to guarantee parent is to check in decreasing order of $d(1, v)$ (deepest ancestor first).
        Then first match is parent.
        
        So we rely on the window size.
        The window is determined by $K$.
        But we don't know $K$.
        However, we know $w \ge 1$.
        If we find a match $v$, then $d(u, v) = w_{edge}$ (if parent).
        Wait, if $v$ is parent, $d(u, v) = w$.
        If $v$ is grandparent, $d(u, v) = w + w_{prev}$.
        We just need to distinguish.
        Since we want to minimize queries, we can't check all.
        But given the problem statement "Hidden parameter K" and "Score based on queries", maybe we are allowed to use a heuristic.
        Or maybe we assume $K$ is reasonably large but we check enough.
        Actually, if we just check ALL nodes in the window, and we don't know $K$, we can just pick a reasonable large $K_{limit}$ like $10^4$ (since it's guaranteed).
        If $K$ is actually larger than $10^4$, this fails.
        But guaranteed $K \le 10^4$.
        So window size $10^4$ is sufficient.
        Wait, if $K$ is small, say 1, then parent is at dist $d-1$.
        Checking up to $10^4$ is safe.
        
        So algorithm:
        Fix $MAX\_STEP = 10000$.
        For each $u$, find $v$ in window $[d(1, u) - MAX\_STEP, d(1, u)]$.
        This ensures we cover all possible parents.
        Even if $K$ is unknown, we know it's at most $10^4$.
        So we search enough.
        
        Sorting by distance takes $O(N \log N)$.
        Processing takes $O(N \cdot K)$.
        With $N=10^5$, $K=10^4$, it's risky.
        But given random weights/structure, checks will terminate early.
        We will not check all $K$ nodes.
        We check only a few non-ancestors.
        Because for a random node, $d(u, v) \approx d(1, u) + d(1, v) - 2 d(1, lca)$.
        Unless $v$ is close to $u$ on path, $d(1, lca)$ is small.
        Then $d(u, v) \gg d(1, u) - d(1, v)$.
        Wait, if $v$ is close in distance but on different branch.
        Example: $d(1, u) = 100, d(1, v) = 99$.
        $u$ is at depth 100. $v$ at depth 99.
        If they are on different branches, LCA is at 1 (depth 0).
        $d(u, v) = 100 + 99 = 199$.
        Check condition: $d(1, u) - d(1, v) = 1$.
        $199 \neq 1$. Reject.
        So we check $v$, reject.
        We keep rejecting until we find parent.
        The number of nodes $v$ with $d(1, v) \approx d(1, u)$ on different branches is small in random tree.
        The probability of collision is low.
        So expected number of checks is small.
        So $O(N)$ queries effectively.
        
        Code Structure:
        Headers.
        IO setup.
        Main loop T.
        Read N.
        Vectors for distances.
        Loop $i=2 \dots n$: query $d(1, i)$.
        Sort nodes by dist.
        Store original index.
        Loop sorted nodes.
        For each node $u$, determine parent.
        Iterate candidates in window (descending dist).
        Check condition.
        If match, add edge, break.
        If no match after scanning enough candidates?
        This shouldn't happen.
        Actually, we can limit the inner loop to not exceed $K_{max}$ iterations?
        If we don't find parent after checking all in window, then error.
        But theoretically parent exists in window.
        However, we might check all nodes in window and none match?
        Wait, if parent is in window, we MUST find it.
        Wait, parent matches. Others don't.
        So we will find it.
        
        Wait, what if we don't know $K$?
        We just iterate $v$ such that $d(1, v) > d(1, u) - 10000$.
        Since we don't know $K$, we can't know where window starts.
        Wait, the window is $[d(1, u) - K, d(1, u))$.
        Since $K \le 10000$, we just need to scan down at least 10000 distance units.
        Or just scan nodes with index such that $d(1, v) > d(1, u) - 10000$.
        Since we have sorted nodes, we can maintain a pointer.
        The pointer will point to the smallest index with $d(1, v) > d(1, u) - 10000$.
        We iterate from there up to the last processed node.
        Since we process in order, we can just look at a range.
        
        Wait, we need to be careful about $d(1, u)$ ordering.
        Sorting handles it.
        Inner loop iterates indices $j$ from `count_nodes_with_dist_ge_threshold` down to `0`?
        Actually, we want $d(1, v) > d(1, u) - K$.
        Let `min_dist = d(1, u) - 10000`.
        We check nodes with $d(1, v) \in (\text{min\_dist}, d(1, u)]$.
        Wait, we need $d(1, v) < d(1, u)$ strictly.
        So $v \neq u$.
        If multiple nodes have same distance $< d(1, u)$, they are siblings/cousins.
        Parent cannot be same distance.
        So strict inequality.
        
        Also, the query output must be flushed.
        
        ```cpp
        #include <iostream>
        #include <vector>
        #include <algorithm>
        #include <numeric>
        
        using namespace std;
        
        struct Node {
            int id;
            int dist_from_root;
        };
        
        // Function to flush output
        auto flush_out = []() {
            cout << endl; // or just newline if handled elsewhere
            fflush(stdout); 
        };
        
        void solve() {
            int n;
            cin >> n;
            if (n == 1) {
                cout << "! ";
                cout.flush();
                return;
            }
            
            // Query distances from 1 to all other nodes
            vector<Node> nodes;
            nodes.reserve(n);
            nodes.push_back({1, 0});
            for (int i = 2; i <= n; ++i) {
                cout << "? 1 " << i << endl;
                int d;
                cin >> d;
                nodes.push_back({i, d});
            }
            
            // Sort nodes by distance from root
            sort(nodes.begin(), nodes.end(), [](const Node& a, const Node& b){
                return a.dist_from_root < b.dist_from_root;
            });
            
            // We need to store edges
            vector<tuple<int, int, int>> edges;
            
            // Pointer to maintain the start of the "window" in the sorted array
            // However, we don't know K exactly, but we know K <= 10000.
            // Actually we can just iterate from the beginning? No.
            // We need v with d(1, v) > d(1, u) - 10000.
            
            // Since nodes are sorted by dist, we can use two pointers or just binary search.
            // But simply iterating back from the node just below u is better.
            
            // Let's group nodes by distance?
            // Actually, just iterate u, and maintain a candidate list.
            // Since we process u in increasing order of dist, 
            // any valid parent v must have appeared before u in the sorted list.
            // And v must have dist > dist(u) - 10000.
            
            // Optimization:
            // We can store all processed nodes in a list.
            // For current u, we scan the list of candidates.
            // But scanning from beginning is slow.
            // We can restrict to range [idx_start, idx_end].
            // idx_start is where dist(idx_start) > dist(u) - 10000.
            // We can track idx_start with a pointer that moves forward.
            
            int ptr_start = 0; // Points to first element in 'nodes'
            // We will iterate u from index 1 to n-1.
            // For each u, we want candidates v with v.index < u.index.
            // Condition: nodes[v].dist_from_root > nodes[u].dist_from_root - 10000.
            
            // Note: We need to iterate v in descending order of distance.
            // Since nodes are sorted ascending, we iterate v from u.index - 1 downwards.
            // But we only care about v where nodes[v].dist > nodes[u].dist - 10000.
            // So we stop when nodes[v].dist <= nodes[u].dist - 10000.
            
            // Wait, we need to check multiple v.
            // Which v?
            // We check v starting from highest distance downwards.
            // Highest distance < nodes[u].dist is just before u.
            // Then u-1, u-2...
            // Until we find a match or condition fails (dist too small).
            
            // However, nodes with same distance as u are processed after u?
            // No, we sort by distance. Nodes with same distance order is undefined/stable.
            // If v has same distance, it cannot be parent. So we can skip.
            // We need v.index < u.index AND nodes[v].dist > nodes[u].dist.
            // And nodes[v].dist > nodes[u].dist - 10000.
            
            // So we can iterate v from u.index - 1 down to 0.
            // If nodes[v].dist <= nodes[u].dist - 10000: break.
            // If nodes[v].dist == nodes[u].dist: continue (skip).
            // Else: query d(u, v.id).
            // If match: add edge, break.
            
            // This logic is sound.
            
            // Wait, we can optimize:
            // If nodes[u].dist == nodes[u-1].dist, they are same level.
            // If many nodes have same distance, iterating u-1 is slow?
            // Actually if nodes[u].dist is same as nodes[u-1], then nodes[u-1] is not parent.
            // We skip.
            // Then nodes[u-2].
            // So we skip all same-level nodes.
            // The inner loop runs only for valid candidates.
            
            for (int i = 1; i < n; ++i) {
                int u_idx = i;
                int u_dist = nodes[u_idx].dist_from_root;
                
                // We want to find parent among nodes[0...i-1]
                // We iterate backwards from i-1
                // Optimization: skip nodes with dist <= u_dist - 10000
                
                // But since we scan backwards, we might hit many nodes with dist > u_dist - 10000.
                // Is it possible to have many nodes with dist ~ u_dist?
                // Yes. But we expect to find parent quickly.
                // Also we need to query.
                // Query cost is dominant.
                // We can try to limit number of queries?
                // If we don't find match after checking a few candidates, we might waste time.
                // But as reasoned, for random tree, we find match quickly.
                // And if we hit non-match, it's rejected.
                // But if many nodes satisfy dist condition but are not ancestors, we check them and reject.
                // In dense graph (star), many nodes might satisfy.
                // But star graph, dists are varied.
                
                // Just implement straightforward loop.
                // Ensure we don't query self.
                
                for (int j = i - 1; j >= 0; --j) {
                    int v_idx = j;
                    if (nodes[v_idx].dist_from_root <= u_dist - 10000) {
                        // If we drop below the minimum possible parent distance,
                        // we can stop.
                        // BUT, we must ensure we don't stop before parent.
                        // Since parent distance >= u_dist - K, and K <= 10000.
                        // If current j has dist <= u_dist - 10000, then it cannot be parent.
                        // AND since v_idx decreases, subsequent nodes will also have smaller dist.
                        // So we can safely break.
                        break;
                    }
                    
                    // Same distance check
                    if (nodes[v_idx].dist_from_root == u_dist) {
                        continue; // Cannot be parent
                    }
                    
                    // Candidate
                    cout << "? " << nodes[u_idx].id << " " << nodes[v_idx].id << endl;
                    int d_uv;
                    cin >> d_uv;
                    
                    // Check condition
                    // parent implies d(u, v) + d(v, 1) = d(u, 1)
                    // d(u, v) + nodes[v_idx].dist_from_root == u_dist
                    
                    if (d_uv + nodes[v_idx].dist_from_root == u_dist) {
                        // It is an ancestor.
                        // Since we iterate descending distance, this is the deepest ancestor, i.e., parent.
                        // Add edge
                        edges.push_back({nodes[v_idx].id, nodes[u_idx].id, abs(d_uv)}); // weight is positive
                        // Wait, d_uv = u_dist - nodes[v_idx].dist. 
                        // Weight is d_uv.
                        // Store as {v, u, d_uv}
                        // Or just {parent_id, child_id, weight}. Order doesn't matter.
                        
                        goto next_node; // Found parent
                    }
                }
                next_node:; // Label
            }
            
            // Output edges
            cout << "!";
            for (auto &e : edges) {
                cout << " " << get<0>(e) << " " << get<1>(e) << " " << get<2>(e);
            }
            cout << endl;
            cout.flush();
        }
        
        int main() {
            ios_base::sync_with_stdio(false);
            cin.tie(NULL);
            int t;
            if (cin >> t) {
                while(t--) {
                    solve();
                }
            }
            return 0;
        }
        ```
        Wait, I put the edge storage inside `solve`.
        Need to reset. `edges` is vector inside solve.
        Wait, `abs(d_uv)`? Distance is positive.
        `d_uv` is input distance.
        Wait, `d_uv` is distance between `u` and `v`.
        If `v` is parent, `d_uv` = `u_dist - v_dist`.
        Yes.
        One correction: The `break` condition.
        If `nodes[v_idx].dist_from_root <= u_dist - 10000`, we break.
        This assumes parent is guaranteed to be in range `[u_dist - 10000, u_dist)`.
        Since `w <= K` and `K <= 10000`, parent distance is at least `u_dist - 10000`.
        This logic holds.
        
        Wait, what if `K` is small, say 1?
        Then parent is at `u_dist - 1`.
        We check range `[u_dist - 10000, u_dist)`.
        We might check many useless nodes above.
        But if `dist` gaps are large, we might check few.
        If `dist` are dense (all 1s), we check `u-1` (match).
        If `dist` are sparse, loop breaks early due to gap.
        The worst case is `dist` values are dense in `[u-10000, u]`.
        In that case, we check many nodes.
        But `dist` values correspond to path lengths.
        If many nodes have distances close to each other, the tree must have short branches.
        Short branches mean small weights.
        Wait, if we have many nodes with same distance `d`, they are on same layer.
        We skip them (`== u_dist`).
        So we only check nodes with distance `> u_dist - 10000` and `< u_dist`.
        So we check nodes at layers `d-1, d-2, ..., d-10000`.
        In a random tree, expected degree is small?
        Yes, average degree is 2.
        So number of nodes at a specific distance layer is small?
        No, layer depth can vary.
        But if weights are uniform, expected number of nodes with distance $D$ is roughly $N / (MaxDist)$.
        MaxDist $\approx N \cdot K$.
        So density is low.
        So checking a few is fast.
        
        One small bug: `goto next_node` works in C++.
        Using label `next_node`.
        
        Wait, the prompt says "Your response should ONLY contain the C++ code".
        I must wrap in `cpp`.
        The problem says "read T".
        I included that.
        
        One detail: `cout << "? " << nodes[u_idx].id << " " << nodes[v_idx].id << endl;`
        Then `cin >> d_uv`.
        If the query fails or buffer issue?
        Standard competitive programming practice.
        Also ensure `flush` on every query?
        The problem says "After you output a query, do not forget to output a newline character and flush the output stream."
        My code does `endl` which flushes in C++ usually?
        Wait, `endl` inserts newline and flushes stream buffer.
        However, for interactive problems, it's safer to explicitly flush or use `cout << flush`.
        But `endl` is fine.
        Wait, `cout << endl` flushes.
        But `ios_base::sync_with_stdio(false)` might interact with `fflush`?
        It's safer to use `cout << "\n"` and `cout.flush()`?
        The problem says `System.out.flush()`.
        Using `endl` is safe.
        
        Wait, `t` reading.
        "First, you need to read an integer T".
        My code does `cin >> t`.
        
        Double check the problem statement "scoring".
        "Score = 100 * (Z - Q) / (Z - 5n)".
        This implies we should use as few queries as possible.
        My solution uses roughly $2n$ (N for root dists + N/2 for parent searches).
        Wait, for each node we might query parent.
        In the loop `for (int j = i - 1; j >= 0; --j)`, we only query if we need to check.
        On average, we query only when we check a candidate.
        If we hit parent on first try, queries = 1 per node.
        So total queries $\approx N$.
        Plus initial $N$ queries.
        So $\approx 2N$.
        $2N$ is well within $5N$.
        
        Wait, I should confirm `edges` storage.
        `edges.push_back({nodes[v_idx].id, nodes[u_idx].id, abs(d_uv)});`
        `abs(d_uv)` is redundant since `d_uv >= 0`.
        `d_uv` is input from judge.
        `nodes[v_idx].dist_from_root` is from 1.
        If match, `d_uv + v_dist == u_dist` => `d_uv == u_dist - v_dist`.
        Since weights are positive integers, `d_uv >= 1`?
        Weights $\in [1, K]$.
        So `d_uv` will be $\ge 1$.
        `abs` is safe.
        
        Also need to include `<tuple>` header.
        And `<cmath>` for `abs`.
        Although `std::abs` is in `<cstdlib>` or `<cmath>`.
        Since `d_uv` is int, `<cstdlib>` is better for `std::abs(int)`.
        But `<cmath>` overloads.
        Using `abs` from `cstdlib` is fine.
        
        Wait, `u != v` constraint.
        `u_idx` and `v_idx` are distinct indices in sorted array.
        Wait, `nodes` vector stores unique nodes.
        Indices `i` and `j` are distinct.
        However, `nodes[i].id` might be same as `nodes[j].id`? No, IDs are unique 1..N.
        
        Final sanity check:
        Input `n=1`.
        My code `if (n == 1)` outputs nothing?
        Wait, loop for `n=1`:
        Reads `T`, enters `solve`.
        Reads `n=1`.
        `nodes` gets `{1, 0}`.
        Loop `i` from 1 to 1 (empty).
        Output `! ` (newline).
        Seems okay.
        Wait, problem says `! u_1 v_1 w_1 ...`.
        If n=1, 0 edges.
        Is empty line allowed? Or `!` followed by nothing?
        The example shows `! ...`.
        For n=1, no edges.
        Likely `!` is enough.
        But `n=1` case is trivial.
        If `n > 1`, edges exist.
        
        One thing: `nodes.reserve(n)` is good.
        `nodes` contains `Node` structs.
        `sort` works.
        
        Is it possible `nodes` size is `n+1` (indices 1 to n)?
        `push_back({1, 0})`.
        Then loop `2` to `n`.
        Size is `n`.
        Indices 0 to `n-1`.
        `nodes[0]` is ID 1.
        `nodes[i]` corresponds to node `ID`.
        IDs are `1` to `n`.
        Correct.
        
        Complexity:
        Sorting: $O(N \log N)$.
        Outer loop: $N$.
        Inner loop: amortized constant?
        Depends on number of nodes with distance in window.
        In worst case $O(N \cdot K)$.
        But practically fast.
        Given constraints and type of problem, this is the intended solution.
        There is no way to beat $O(N \cdot K)$ without stronger assumptions or advanced structures (like segment tree on distances), but maintaining structure dynamically is hard with limited queries.
        With $N=10^5$, $O(N \log N)$ is trivial.
        The query count is the main constraint.
        With $2N$ queries, we are safe.
        
        One final optimization:
        We can check if `d_uv` is too small?
        Since weights $\ge 1$, `d_uv >= 1`.
        If `d_uv == 0` (should not happen for distinct nodes), ignore.
        But `d_uv` comes from judge.
        
        Also, `abs` usage.
        `std::abs` in `<cmath>`.
        
        Let's write the code.
        
```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <tuple>
#include <cstdlib>

using namespace std;

struct Node {
    int id;
    int dist_from_root;
};

void solve() {
    int n;
    if (!(cin >> n)) return;

    if (n == 1) {
        cout << "!";
        cout << endl;
        cout.flush();
        return;
    }

    vector<Node> nodes;
    nodes.reserve(n);
    // Query distances from root (vertex 1)
    // We query d(1, i) for all i from 2 to n
    nodes.push_back({1, 0});
    for (int i = 2; i <= n; ++i) {
        cout << "? 1 " << i << endl;
        int d;
        cin >> d;
        nodes.push_back({i, d});
    }

    // Sort nodes by distance from root to process in increasing order
    sort(nodes.begin(), nodes.end(), [](const Node& a, const Node& b) {
        return a.dist_from_root < b.dist_from_root;
    });

    // Edges to output
    vector<tuple<int, int, int>> edges;
    
    // Process each node to find its parent
    for (int i = 1; i < n; ++i) {
        int u_id = nodes[i].id;
        int u_dist = nodes[i].dist_from_root;

        // We look for a parent v among previously processed nodes (index < i)
        // The parent v must satisfy d(1, v) < d(1, u)
        // Since edge weights >= 1, d(1, v) >= u_dist - K_max
        // We know K <= 10000, so we check candidates in window [u_dist - 10000, u_dist)
        // We iterate candidates with decreasing distance to find the nearest ancestor (parent) first.
        
        // To optimize, we iterate backwards from i-1.
        // We stop if distance is too small.
        
        for (int j = i - 1; j >= 0; --j) {
            int v_dist = nodes[j].dist_from_root;
            
            // Optimization: Stop if distance difference is too large
            // Even though we don't know exact K, we know K <= 10000
            // So if u_dist - v_dist > 10000, v cannot be parent.
            // Since we iterate j downwards (dist decreases), once it drops below threshold, it stays below.
            if (u_dist - v_dist > 10000) {
                break;
            }
            
            // Parent cannot have same distance
            if (v_dist == u_dist) {
                continue;
            }
            
            // Query distance between u and v
            cout << "? " << u_id << " " << nodes[j].id << endl;
            int d_uv;
            cin >> d_uv;
            
            // Check if v is an ancestor of u
            // If v is ancestor, d(u, v) + d(v, 1) = d(u, 1)
            // => d_uv + v_dist == u_dist
            if (d_uv + v_dist == u_dist) {
                // Found an ancestor. Since we check candidates with decreasing distance,
                // this is the deepest ancestor (direct parent).
                // Record edge.
                // The weight of the edge is d_uv.
                edges.push_back(make_tuple(nodes[j].id, nodes[i].id, d_uv));
                
                // Move to next node
                goto next_node;
            }
        }
        
        next_node:;
    }

    // Output the reconstructed tree
    cout << "!";
    for (const auto& edge : edges) {
        cout << " " << get<0>(edge) << " " << get<1>(edge) << " " << get<2>(edge);
    }
    cout << endl;
    cout.flush();
}

int main() {
    // Optimize I/O operations
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    int t;
    if (cin >> t) {
        while (t--) {
            solve();
        }
    }
    return 0;
}
```
Wait, the check `u_dist - v_dist > 10000` relies on `v_dist` being accessed.
Since `nodes[j]` is accessed, `j` must be valid. `i` starts at 1, so `j` starts at 0. Valid.
One minor point: `make_tuple` needs `<tuple>`. Included.
Headers are good.
The code seems robust.
Also handling the `n=1` case separately is good.
Is `abs` needed? `d_uv` is positive. `u_dist` and `v_dist` are from queries, also positive.
Equation `d_uv + v_dist == u_dist` implies `d_uv = u_dist - v_dist`.
So `d_uv` is naturally non-negative.
Actually `d_uv` from judge is sum of weights, so $\ge 1$ for distinct vertices.
Wait, if `u` and `v` are same, `d_uv=0`.
But loop `j < i` ensures `u` != `v` (different elements in `nodes` vector? Wait, `nodes` has unique vertices).
`nodes[j].id` != `nodes[i].id`.
So `d_uv >= 1`.
Wait, what if `u_dist - v_dist` is negative?
Condition `u_dist - v_dist > 10000` ensures `u_dist > v_dist` significantly.
Wait, if `u_dist - v_dist` is negative (i.e. `v_dist > u_dist`), then `u_dist - v_dist > 10000` is false.
But we check `j < i`. Since `nodes` is sorted, `nodes[j].dist <= nodes[i].dist`.
So `v_dist <= u_dist`.
So `u_dist - v_dist >= 0`.
So `d_uv + v_dist == u_dist` check is safe.

Final check on time limit: 2s.
$10^5$ nodes.
Loop overhead is minimal.
Queries $\approx 2 \cdot 10^5$ in ideal case.
Even with loose bounds, it should pass.
The logic assumes $K \le 10000$. This is given in problem description.
The only risk is "hidden parameter K" meaning we might not know it precisely, but the bound is known.
Using 10000 as bound is correct per constraints.

One tiny thing: `nodes` is pushed back `i`th time.
Indices: `0` to `n-1`.
`nodes[i]` is accessed in loop.
Correct.

Double check the `?` command syntax.
`? u v`
Example output: `? 1 2`.
Matches.
Answer: `! u1 v1 w1 ...`
Matches.

One subtle issue:
What if the parent is NOT in the window `u_dist - 10000`?
Since $K \le 10000$ and $d(u, parent) = w \le K$, then $d(1, parent) = d(1, u) - w \ge d(1, u) - 10000$.
So parent IS in the window.
The code checks exactly this range.
If $w > 10000$, it violates constraints.
So logic is solid.

```cpp
/*
Problem Analysis:
We need to reconstruct a weighted tree. We can query distances.
Strategy:
1. Pick a root (node 1). Query d(1, u) for all u. (Cost: N-1 queries)
2. Sort nodes by d(1, u).
3. For each node u (except root), find its parent p.
   Properties:
   - d(1, p) < d(1, u) (since edge weights >= 1).
   - d(1, u) = d(1, p) + d(p, u).
   - d(p, u) <= K (given K <= 10000).
   Therefore, d(1, p) >= d(1, u) - 10000.
   
   We iterate through previously processed nodes v (sorted by distance) and check if v could be the parent.
   Candidates v must have d(1, v) in [d(1, u) - 10000, d(1, u)).
   If multiple candidates satisfy the distance equation d(u, v) + d(1, v) = d(1, u), v is an ancestor.
   Among ancestors, the one with the largest d(1, v) is the direct parent.
   
   Algorithm:
   - Iterate u in increasing order of d(1, u).
   - Scan v with d(1, v) just below d(1, u) downwards.
   - Check if v satisfies ancestor condition using d(u, v) query.
   - First match is the parent.
   - Add edge (v, u) with weight d(u, v).
   
   Complexity: O(N log N) for sorting. O(N * K) worst case for scanning, but practically O(N) due to random tree structure and early termination.
*/
```
I'll put the code block now.

One detail: `cout << endl` in query loop.
`endl` forces flush.
In C++, `cout << endl` performs `buffer_flush`.
So explicit `cout.flush()` after `cout << endl` is redundant but harmless.
In interactive problems, flushing is critical.
My code uses `endl` which flushes.
Wait, `ios_base::sync_with_stdio(false)` does NOT disable flushing on `endl`.
`endl` inserts `\n` and calls `base->flush()`.
So it is safe.
Just ensuring `flush` is called.
Using `"\n"` + `flush` is generally preferred in fast I/O to avoid `endl` overhead.
`endl` adds overhead.
However, correctness of flush is priority here.
`endl` is safer.
Time limit 2s is generous for $10^5$ inputs/outputs.

Final code looks good.
</think>

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <tuple>
#include <cstdlib>

using namespace std;

struct Node {
    int id;
    int dist_from_root;
};

void solve() {
    int n;
    if (!(cin >> n)) return;

    // Handle n=1 case trivially
    if (n == 1) {
        cout << "!";
        cout << endl;
        cout.flush();
        return;
    }

    vector<Node> nodes;
    nodes.reserve(n);
    // Query distances from root (vertex 1) to all other nodes
    nodes.push_back({1, 0});
    for (int i = 2; i <= n; ++i) {
        cout << "? 1 " << i << endl;
        int d;
        cin >> d;
        nodes.push_back({i, d});
    }

    // Sort nodes by distance from root to process in increasing order
    // This simulates a BFS traversal where parents appear before children in sorted order
    sort(nodes.begin(), nodes.end(), [](const Node& a, const Node& b) {
        return a.dist_from_root < b.dist_from_root;
    });

    // Edges to output
    vector<tuple<int, int, int>> edges;
    
    // Process each node to find its parent
    for (int i = 1; i < n; ++i) {
        int u_id = nodes[i].id;
        int u_dist = nodes[i].dist_from_root;

        // We look for a parent v among previously processed nodes (index < i)
        // The parent v must satisfy:
        // 1. d(1, v) < d(1, u)
        // 2. d(1, u) = d(1, v) + d(v, u)  => d(v, u) = u_dist - v_dist
        
        // We know edge weights are <= 10000. Thus d(v, u) <= 10000.
        // Consequently, d(1, v) >= u_dist - 10000.
        // We iterate candidates with decreasing distance to find the deepest ancestor (parent) first.
        
        for (int j = i - 1; j >= 0; --j) {
            int v_dist = nodes[j].dist_from_root;
            
            // Optimization: Stop if distance difference is too large (> 10000)
            // Since K <= 10000, parent distance cannot be less than u_dist - 10000.
            if (u_dist - v_dist > 10000) {
                break;
            }
            
            // Parent cannot have same distance as u (weights >= 1)
            if (v_dist == u_dist) {
                continue;
            }
            
            // Query distance between u and v
            cout << "? " << u_id << " " << nodes[j].id << endl;
            int d_uv;
            cin >> d_uv;
            
            // Check if v is an ancestor of u
            // Condition: d(u, v) + d(v, 1) == d(u, 1)
            if (d_uv + v_dist == u_dist) {
                // Found an ancestor. Since we iterate candidates in descending order of distance,
                // this is the deepest ancestor, i.e., the direct parent of u.
                edges.push_back(make_tuple(nodes[j].id, nodes[i].id, d_uv));
                
                goto next_node; // Found parent, move to next u
            }
        }
        
        next_node:;
    }

    // Output the reconstructed tree
    cout << "!";
    for (const auto& edge : edges) {
        cout << " " << get<0>(edge) << " " << get<1>(edge) << " " << get<2>(edge);
    }
    cout << endl;
    cout.flush();
}

int main() {
    // Optimize I/O operations, but keep synchronization false for speed
    // Interactive problems require flushing, which endl provides
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    int t;
    if (cin >> t) {
        while (t--) {
            solve();
        }
    }
    return 0;
}
```
````
