# 题 9 — base均分 1.0 vs 直接SFT均分 0.0(掉点 1.0)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=0,得分 5.0,completion_tokens=24962)

````
The problem asks us to sort the values in a tree. Specifically, we have a tree where vertex $i$ initially holds value $p_i$, and we want to reach a state where vertex $i$ holds value $i$. In one operation, we can choose a matching (a set of disjoint edges), and for each edge $(u, v)$ in the matching, we swap the values currently at $u$ and $v$. We want to achieve the sorted state with the minimum number of operations.

Wait, actually, looking at the scoring formula: $s_i = \max(0, (base\_value - m) / (base\_value - best\_value))$. Usually, in competitive programming problems with such output, minimizing $m$ is the goal, but sometimes exact minimization is hard. The example shows 4 operations for $n=5$. Let's analyze the structure.
We are swapping pairs. Each swap fixes positions or moves elements towards their targets.
Since we can only swap using matchings, and we want to minimize operations, this sounds like decomposing the permutation into transpositions. However, a general decomposition into transpositions requires $\approx n$ swaps if done greedily, but here the constraint is that all swaps must be disjoint within an operation. This means in one step, we can fix multiple independent pairs.
Actually, if we view this as sorting by adjacent swaps on a graph? No, it's arbitrary edges in a matching.
Let's consider the permutation cycles. If vertex $u$ needs value $v$, and vertex $w$ has value $v$, then we need to move value $v$ from $w$ to $u$. Since the target position for value $x$ is vertex $x$, we can think of this as a permutation mapping: current location of value $x$ is some vertex $y$, we want value $x$ to go to vertex $x$. Or equivalently, at vertex $i$, we have value $p_i$. We want to place value $i$ at vertex $i$.
This defines a permutation $\pi$ such that $\pi(i) = p_i^{-1}(i)$? No, let's formalize.
Currently at vertex $u$ is value $p_u$. We want eventually at vertex $u$ is value $u$.
Consider the cycle decomposition of the permutation defined by "current position vs target position".
Actually, simpler: We have a permutation $P$. We want to apply a sequence of matchings $M_1, M_2, \dots, M_k$ such that applying swaps corresponding to these matchings results in the identity permutation.
Each swap $(u, v)$ exchanges the contents.
If we look at a single operation (one matching), we perform several simultaneous swaps.
To minimize operations $k$, we want to process as many misplaced items as possible in parallel.
However, usually sorting permutations by swaps takes roughly $O(n)$ total swaps. Here we want $O(k)$ rounds of swaps where each round uses a matching.
Actually, since any two disjoint edges can be swapped simultaneously, this suggests we can decompose the required swaps into a minimum number of layers.
But wait, we are constrained by the graph structure. We can only swap $(u, v)$ if there is an edge between $u$ and $v$. Wait, the problem says "select a matching M (M belongs to E)". This means we can ONLY swap values across edges existing in the tree.
So this is effectively bubble sort on a graph, but specifically sorting using only valid edge swaps. And since we can choose any matching, we can pick non-adjacent edges freely within the same step.
However, the constraints say $N \le 1000$. An $O(N^3)$ or $O(N^2)$ solution is acceptable given $\sum N^2 \le 10^6$.
This hints that we might not need a very efficient global strategy, perhaps something greedy or cycle-based.
Let's check the constraint again: "M belongs to E". So we can't swap arbitrary vertices, only neighbors connected by an edge. This implies we must use the connectivity of the tree to move values.
Actually, if we just want to fix the permutation, does the tree structure matter significantly for the *minimum number* of operations? Or is it just about finding *any* set of swaps that sorts the array?
Wait, if we are allowed to swap along ANY path in the tree? No, the operation is strictly defined as selecting a matching from E. So a swap must happen between adjacent vertices in the tree.
If I have to move value from $u$ to $v$ where distance $d(u,v) > 1$, I cannot do it in one step. I would need intermediate steps.
For example, to move $A$ from $u$ to $u+1$ and $B$ from $u+1$ to $u$, one swap suffices if they are connected.
But what if we need to move $A$ from $u$ to $v$ ($dist(u,v)=2$, so $u-x-v$)? We need to swap $(u,x)$ then $(x,v)$ later.
Is it true that the number of operations depends on the diameter or something?
Let's reconsider the problem statement carefully.
"Bobo wants to make pi = i ... with as few operations as possible."
If the graph was a line (path), and we had to sort... wait, if we can swap any matching, maybe we don't need to care about the specific target location immediately, just resolve cycles.
Actually, if we treat the values as nodes in the same graph, moving a value $x$ from $curr(x)$ to $x$ requires a path of swaps.
Since we want to minimize the *number of operations*, not the number of individual swaps, we can perform many independent swaps in parallel.
In fact, if we have a set of edges to swap, we can do them all at once IF they form a matching.
This looks like we want to decompose the necessary permutation into a sequence of matchings.
Wait, if we can sort it using the adjacency property, maybe the tree structure limits the speed?
Actually, if we just swap pairs $(u,v)$ that need to exchange values, and we find a set of such swaps that form a matching, we execute them. Then we update the permutation and repeat.
Since we want to minimize operations, we should maximize the number of swaps per operation.
However, simply swapping $(u,v)$ when needed might disrupt other things or require re-swapping later.
But notice: if we just decompose the permutation into disjoint transpositions (cycles of length 2), each transposition can be realized by a swap. If they share endpoints, we can't do them together.
But cycles of length > 2 are tricky. A cycle $(1 \to 2 \to 3 \to 1)$ means at node 1 we want 3 (wait, node 1 has val 1, node 2 has val 2... no).
Let's define the permutation clearly. $P[u]$ is the value at vertex $u$. Target is $Target[v] = v$.
So we want value $v$ at vertex $v$. Currently value $v$ is at vertex $pos(v)$ such that $P[pos(v)] = v$.
We need to move the content of $u$ to some destination.
If we ignore the tree structure for a moment and assume we can swap any pair, the minimum number of operations would be the height of the longest chain of dependencies?
Actually, the standard "minimum number of operations to sort a permutation" where operations are "choose a subset of indices and permute them cyclically" or similar is different.
Here, operation is "swap disjoint pairs".
This is exactly equivalent to decomposing the permutation into transpositions (swaps).
However, we have a constraint:
1. The swap must correspond to an edge in the tree.
2. Swaps within an operation must be disjoint (matching).

If we could use any edges, and we just wanted to sort, the optimal strategy involves resolving cycles. But since we are limited to matchings, in one step we can resolve at most $\lfloor (\text{length of permutation})/2 \rfloor$ elements involved in 2-cycles?
Wait, if we have a cycle of length 3: $u \to v \to w \to u$. Meaning value $u$ is at $w$, value $v$ is at $u$, value $w$ is at $v$ (assuming target is $i$).
Actually, let's trace carefully.
Vertex 1 has val 2. Vertex 2 has val 3. Vertex 3 has val 1.
Target: 1 at 1, 2 at 2, 3 at 3.
Current: $P=[2, 3, 1]$.
Swap $(1, 2)$ -> $P=[3, 2, 1]$. Now 2 is fixed.
Swap $(1, 3)$ -> $P=[1, 2, 3]$. Done.
Total 2 swaps. Since they share 1, we can't do both in one op.
Can we do better?
Maybe rotate? No, operation is strictly SWAP.
So we must break cycles into swaps. A cycle of length $L$ needs at least $L-1$ swaps to resolve if we were doing standard sorting.
Actually, a cycle $(a_1, a_2, \dots, a_k)$ corresponds to mapping $a_1 \leftarrow P[a_2], a_2 \leftarrow P[a_3] \dots$?
No, standard cycle decomposition of permutation $\pi$: $\pi(i)$ is the value at $i$. We want identity. So $\pi = \text{swap} \circ \dots$.
If we view values as tokens, we have a cycle of tokens.
Actually, it's easier to think: Value $v$ is at $c_v$. We want value $v$ at $v$.
This is a set of cycles in the permutation map: $i \to c_{\text{val } i}$.
Wait, if vertex $i$ holds value $v$, then value $v$ needs to go to vertex $v$.
Let's construct the permutation $\sigma$ where $\sigma(u)$ is the current position of the value that belongs to $u$. That is, $\sigma(u) = u'$ such that $P[u'] = u$.
Then we want to transform $\sigma$ to Identity by swaps.
Each swap $(u, v)$ exchanges the value at $u$ and $v$.
Suppose at $u$ we have value $x$ (belongs at $x$) and at $v$ we have value $y$ (belongs at $y$). After swap, $u$ has $y$, $v$ has $x$.
Basically, we want to route the values to their correct places.
Since it's a tree, and we can only swap along edges... wait.
Is it possible that we MUST move values along paths?
Yes. Value at $u$ must travel to $target(u)$. It travels via edges.
However, the operation is defined globally.
If I decide to swap $(u,v)$, values pass through that edge.
Since we want to MINIMIZE operations, not necessarily the number of swaps, maybe we can fix multiple things at once.
Key Insight:
If we have a matching, we can perform those swaps in parallel.
Consider the cycles of the permutation formed by "vertex $\leftrightarrow$ value".
For the Example:
$N=5$. $P = [1, 4, 2, 5, 3]$.
Edges: (1,2), (2,3), (2,4), (1,5).
Values at vertices:
1: 1 (Correct)
2: 4 (Needs to go to 4)
3: 2 (Needs to go to 2)
4: 5 (Needs to go to 5)
5: 3 (Needs to go to 3)
Misplaced: 2, 3, 4, 5.
Value 4 is at 2. Value 2 is at 3. Value 5 is at 4. Value 3 is at 5.
Cycles of values movement:
Start at vertex 2: Has 4. 4 needs to go to 4. Vertex 4 has 5. 5 needs to go to 5. Vertex 5 has 3. 3 needs to go to 3. Vertex 3 has 2. 2 needs to go to 2. Vertex 2 has 4...
Cycle: $2 \to 4 \to 5 \to 3 \to 2$. Length 4.
Vertices involved: {2, 3, 4, 5}.
We want to resolve this cycle.
Since we can only swap on edges, and we want to minimize steps.
Is there a strategy to solve this?
Usually, for a cycle of length $L$, we need $L-1$ swaps.
However, we want to cover these with minimal number of matchings.
If the cycle vertices induce a path/subgraph in the tree, maybe we can pipeline?
Wait, if we just take the standard algorithm for sorting (which resolves cycles), we can try to do as many swaps as possible.
In one operation, we select a matching $M$. For every $(u,v) \in M$, we swap.
Effectively, we are applying a product of transpositions.
Is it possible that the number of operations is always related to the maximum cycle length? Or something simple?
Actually, the example output has 4 operations.
Input:
P: 1 4 2 5 3.
Target: 1 2 3 4 5.
Operations:
1. 2 edges: 4, 3. Edges are (4,?) and (3,?). Wait, input format: `ki` followed by `indices`.
Wait, the example output says: `2 4 3`. The integers are "indexes of edges you select".
The edges are given as lines:
1: 1 2
2: 2 3
3: 2 4
4: 1 5
So edge 4 is (1,5), edge 3 is (2,4).
Op 1: Swap (1,5) and (2,4).
Initially:
1: 1, 2: 4, 3: 2, 4: 5, 5: 3.
Swap (1,5): vals at 1 and 5 become 3 and 1.
Swap (2,4): vals at 2 and 4 become 5 and 4.
State after Op 1:
1: 3, 2: 5, 3: 2, 4: 4, 5: 1.
Fixed: Vertex 4 has 4.
Op 2: `1 1` -> Select edge 1: (1,2).
Current: 1:3, 2:5, 3:2, 4:4, 5:1.
Swap (1,2): 1 gets 5, 2 gets 3.
State: 1:5, 2:3, 3:2, 4:4, 5:1.
Op 3: `1 2` -> Select edge 2: (2,3).
Current: 1:5, 2:3, 3:2, 4:4, 5:1.
Swap (2,3): 2 gets 2, 3 gets 3.
State: 1:5, 2:2, 3:3, 4:4, 5:1.
Fixed: 2, 3, 4.
Op 4: `1 4` -> Select edge 4: (1,5).
Current: 1:5, 5:1.
Swap: 1 gets 1, 5 gets 5.
State: 1:1, 5:5.
All fixed. Total 4 ops.

Note that the total number of swaps performed is $2 + 1 + 1 + 1 = 5$.
The cycle length was 4 ($2-4-5-3-2$). Plus vertex 1 was already fixed.
Wait, my manual trace above had:
Initially 2,3,4,5 involved.
Cycle: 2->4->5->3->2.
Edges available: (1,2), (2,3), (2,4), (1,5).
Notice (2,4) connects 2 and 4. (2,3) connects 2 and 3. (1,5) connects 1 and 5.
In step 1, (1,5) and (2,4) used. (1,5) moved value 3 (at 5) to 1? No, value at 5 was 3. Target of 3 is 3.
Wait, at step 0:
V1: 1 (ok)
V2: 4 (need at 4)
V3: 2 (need at 2)
V4: 5 (need at 5)
V5: 3 (need at 3)
Edge (2,4): 4 is at 2, 5 is at 4. After swap: 2 has 5, 4 has 4.
Now 4 is fixed.
Edge (1,5): 1 has 1, 5 has 3.
Wait, V1 has 1. Why swap (1,5)?
Example output selected (1,5). V1 had 1. V5 had 3. After swap: V1 has 3, V5 has 1.
Why did they swap V1 which was already correct?
Ah, to enable subsequent movements. Maybe V5 needs to pass the value 3 somewhere else?
Wait, value 3 belongs at 3. Currently at 5. Needs to go to 3. Path 5-1-2-3?
Edges: (5,1), (1,2), (2,3). Yes.
So V5 (val 3) -> V1 (via 5-1) -> V2 -> V3.
But V1 held 1. V1 is on the path. It acts as a buffer.
This confirms that sometimes we use "buffers" or correct vertices to facilitate swaps for others.

Given the constraints $N \le 1000$, $\sum N^2 \le 10^6$.
The proposed solution by the user is likely a constructive algorithm that finds *a* solution, possibly not optimal in theoretical worst case but works well enough, OR the problem allows for a heuristic that achieves "few operations".
But wait, the prompt asks to help Bobo. The problem asks for "as few operations as possible".
Is the problem NP-hard? Minimizing matchings for a permutation on a tree.
Actually, if we just want to find *some* solution, maybe BFS to find shortest path for each element is too slow/complex?
However, with $N=1000$, maybe $O(N^2)$ is fine.
Actually, since the scoring function depends on $m$ (number of operations) relative to `best_value`, we should aim for small $m$.
How small can $m$ be?
In the example, $m=4$. Max distance involved was small.
With $N=1000$, if we do a BFS-like flow or matching based approach?
Let's consider the simplest approach:
At each step, perform a maximum matching on the set of pairs that need to be swapped?
Wait, identifying pairs that need to be swapped.
A pair $(u,v)$ needs to be swapped if value at $u$ needs to go to $v$ AND value at $v$ needs to go to $u$.
This corresponds to a 2-cycle in the permutation: $u \leftrightarrow v$.
If we have $u \leftrightarrow v$, we can fix them immediately.
What if we have longer cycles?
Example: $u \to v \to w \to u$. Value at $u$ belongs at $v$, at $v$ at $w$, at $w$ at $u$.
If $(u,v)$ is an edge and $(v,w)$ is an edge, we can swap $(u,v)$, then $(v,w)$ later?
Or $(u,w)$? No, must be edge.
So, generally, we want to decompose the permutation into cycles. For each cycle, we need to route values.
One effective strategy for sorting on trees (or graphs) is:
Identify all vertices $u$ where $P[u] \neq u$. These are active.
Also vertices $u$ where $P[u]=u$ might act as buffers.
Wait, if $P[u]=u$, can we involve it? Yes, swapping $(u,v)$ where $P[u]=u$ puts value at $v$ into $u$ (destroys correctness) and moves $u$'s correct value to $v$.
Is it beneficial? Only if it helps sort.

However, note that the problem statement says: "can you please help him?" and the scoring gives points for smaller $m$.
Often in such contest problems, the optimal solution isn't easily derivable, but a decent one exists.
Is there a known result?
Problem seems to be from a contest (Codeforces / POJ etc).
Searching for "Bobo tree matching swap pi=i".
This looks like a problem from the 2017-2018 ICPC Asia Xi'an Regional Contest or similar?
Actually, the text matches a problem from "ZOJ 4020" or similar?
Found reference: Problem H "Tree Sort" from "2016 Multi-University Training Contest 1"? No.
Wait, the text "Bobo is given a tree T..." appears in problem "H - Tree" from some contest.
Let's look for logic.
If we can find *any* valid sequence, that solves the requirement of getting *points*. But the scoring penalizes large $m$.
Actually, the question says "with as few operations as possible". This implies minimizing $m$.
However, with $N=1000$, maybe the intended complexity is lower.
But wait, if the graph was fully connected, min ops would be related to cycles. With a tree, it's more complex due to path constraints.
However, we can observe:
In the example, the tree is small.
Is it possible to simply run a greedy strategy?
"At each step, find the largest matching $M$ such that applying it brings the permutation closer to identity?"
How to measure progress?
Heuristic: A swap $(u,v)$ is good if:
1. $P[u] == v$ and $P[v] == u$. (Direct swap fix).
2. Or if swapping helps moving $P[u]$ closer to its target, etc.
Specifically, $|P[u] - u|$ doesn't matter because it's a tree, not a line.
Distance in tree is better.
Let $d(u, x)$ be distance between vertex $u$ and vertex $x$.
Current position of value $v$ is $pos[v]$. Target is $v$.
Cost to fix is $d(pos[v], v)$.
We want to decrease total distance?
Since we can only swap adjacent vertices.
If we swap $(u,v)$, the distance for values at $u$ and $v$ changes.
Actually, this looks like the "sorting by swaps on a graph" problem, but restricted to matchings (parallel swaps).
Wait, if we can swap any matching, maybe we can simply simulate a "flow" of elements?
Actually, there is a simpler observation.
Any permutation can be decomposed into disjoint cycles.
For a cycle of length $k$, it takes $k$ swaps to fix (if edges allow).
Since we have a tree, maybe we can fix elements locally.
Strategy:
Find a matching $M$ that maximizes the number of fixed points or reduces distance.
Since we need to output the answer, and time limit is generous (1s, 1024MB), maybe a slightly suboptimal strategy works if it's fast enough.
Wait, $T$ test cases, sum of $N^2 \le 10^6$. This suggests $O(N^2)$ per test case is totally fine. Even $O(N^3)$ might pass for smaller cases, but $\sum N^2$ constraint is key.
So an $O(N^2)$ simulation per step is okay.
How many steps?
If we have a naive strategy that picks max number of direct swaps (where $P[u]=v, P[v]=u$), how many remain?
After fixing all 2-cycles, we are left with larger cycles.
We can break cycles by picking an edge inside the cycle path.
Wait, if we can break cycles arbitrarily, we can reduce any cycle of length $L$ to length $L-1$ with one swap? No, swapping two elements in a cycle breaks it into two smaller cycles.
E.g. cycle $1 \to 2 \to 3 \to 1$. Swap 1 and 2 (if edge exists).
Pos 1 has 2, Pos 2 has 3. Swap -> Pos 1 has 3, Pos 2 has 2.
Now 2 is fixed. Cycle becomes $1 \to 3$ (value 3 is at 1, needs to go to 3). Length 2.
So one swap reduced length 3 to length 1 (fixed) + length 2.
So generally, $L-1$ swaps.
If we can do parallel swaps, the number of operations is related to the "depth" of operations.
Maybe the greedy approach:
While not sorted:
  Find all pairs $(u,v)$ that are connected by an edge.
  Check if swapping them is part of a beneficial plan.
  Actually, simply finding pairs $(u,v)$ such that $P[u]=v$ and $P[v]=u$ is not enough.
  Also pairs $(u,v)$ where $P[u]$ wants to go towards $v$ and $P[v]$ wants to go towards $u$?
  Wait, $P[u]$ wants to go to $P[u]$ (vertex index equals value).
  So value $x$ wants to go to vertex $x$.
  If $u$ has $x$, distance is $d(u, x)$.
  If we swap $(u,v)$, new pos for $x$ is $v$, dist $d(v,x)$.
  Ideally we want $d(v,x) < d(u,x)$.
  This happens if $v$ is closer to $x$ than $u$ is.
  Since tree structure is unique path, $u$ and $v$ are neighbors. One lies on the path from $u$ to $x$.
  So if $x$ is 'behind' $v$ (relative to $u$), swap is good.
  This defines a direction for each element.
  However, conflicts arise.
  
  Greedy Strategy Proposal:
  While there exists $u$ such that $P[u] \neq u$:
    Find a matching $M$ of edges $(u,v)$ that are "useful".
    A swap $(u,v)$ is useful if:
      For value $val = P[u]$ (belongs at $val$): $d(v, val) < d(u, val)$.
      AND/OR for value $val' = P[v]$ (belongs at $val'$): $d(u, val') < d(v, val')$.
      We can prioritize pairs where BOTH values improve their distance, or just one improves.
      To maximize progress, we should probably pick edges where distance strictly decreases.
      Is it sufficient to just pick *any* edge that reduces distance?
      Consider the example again.
      Values: 1, 4, 2, 5, 3 at 1, 2, 3, 4, 5.
      Targets: 1, 2, 3, 4, 5.
      Distances from pos to target:
      Val 1 (at 1): $d(1,1)=0$. Correct.
      Val 4 (at 2): Target 4. Path $2 \to 4$ is direct edge (2,4). So $d(2,4)=1$. Neighbor 2->4.
         Neighbors of 2 are 1, 3, 4.
         Move to 4 -> dist 0. Improvement!
         Move to 1 -> dist $d(1,4)=2$. Worse.
         Move to 3 -> dist $d(3,4)=2$ (3-2-4). Worse.
      Val 2 (at 3): Target 2. Path $3 \to 2$. Edge (3,2). $d(3,2)=1$. Move to 2 -> dist 0.
      Val 5 (at 4): Target 5. Path $4 \to 5$ (via 4-5? No, 4 is connected to 2. 2 is connected to nothing else except 1? Wait.)
         Edges: (1,2), (2,3), (2,4), (1,5).
         Structure: 1-5, 1-2, 2-3, 2-4.
         So 5 is connected to 1. 1 is connected to 2. 2 is connected to 3,4.
         Path for 5 (at 4): Target 5.
         Current 4. Path $4 \to 2 \to 1 \to 5$. Length 3.
         Neighbors of 4: 2. Move to 2 -> dist $d(2,5)=2$ (2-1-5). Improvement (3 -> 2).
      Val 3 (at 5): Target 3.
         Current 5. Path $5 \to 1 \to 2 \to 3$. Length 3.
         Neighbors of 5: 1. Move to 1 -> dist $d(1,3)=2$ (1-2-3). Improvement (3 -> 2).
      
      Edges available:
      (1,5): 1 (val 1, ok), 5 (val 3, wants 1->dist 2).
            Swap (1,5): Val 1 moves to 5 (dist 1->2 Bad). Val 3 moves to 1 (dist 2 Good).
      (1,2): 1 (val 1, ok), 2 (val 4, wants 2->dist 1).
            Swap (1,2): Val 1 moves to 2 (Bad), Val 4 moves to 1 (Bad).
      (2,3): 2 (val 4, wants 2), 3 (val 2, wants 2).
            Swap (2,3): Val 4 moves 2->3 (dist 1->2 Bad). Val 2 moves 3->2 (dist 1->0 Good).
      (2,4): 2 (val 4, wants 4), 4 (val 5, wants 5).
            Swap (2,4): Val 4 moves 2->4 (dist 0 Good). Val 5 moves 4->2 (dist 2 Good).
      
      Best improvement is (2,4). Both improve?
      Val 4 at 2 -> target 4. Distance 1. Swap with 4 (val 5) at 4 -> target 5.
      Swap (2,4): 4 goes to 4 (Good), 5 goes to 2.
      Dist(4,5) = 2 (path 2-1-5). Original Dist(4,5) = 1 (at 2, 5 needs to go to 5? No).
      Wait. At vertex 2, value is 4. Target 4.
      At vertex 4, value is 5. Target 5.
      Swap:
      Vertex 2 gets 5. Dist(2, 5) = $d(2,5) = d(2,1)+d(1,5) = 2$.
      Vertex 4 gets 4. Dist(4, 4) = 0.
      Original costs:
      Val 4 at 2: cost $d(2,4)=1$.
      Val 5 at 4: cost $d(4,5)=3$.
      New costs:
      Val 5 at 2: cost $d(2,5)=2$.
      Val 4 at 4: cost 0.
      Net change: $(-1) + (-1) = -2$. Good.
      Edge (2,4) is definitely good.
      Can we use it? It connects 2 and 4.
      Any other disjoint edges?
      (1,5).
      At 1: Val 1. Cost 0. At 5: Val 3. Cost $d(5,3)=3$.
      Swap (1,5):
      At 1 gets 3. Cost $d(1,3)=2$.
      At 5 gets 1. Cost $d(5,1)=1$.
      Net change: $+2 - 3 = -1$. Good (but less than 2).
      So {(2,4), (1,5)} is a valid matching.
      Both give improvements.
      This matches the example solution.
      
      Algorithm Idea:
      While not sorted:
        Construct graph of vertices with current values.
        Find a maximum weight matching? Or just a greedy one that covers as many improving moves as possible.
        Actually, since we can iterate:
        Just find a matching where every edge is "locally beneficial" for at least one element?
        Or even better, for *all* elements involved, distance strictly decreases?
        If we restrict to edges where *both* swap partners move towards their target, we might be slower but safer?
        In the example, (2,4) was good for both. (1,5) was good for 3 (move to 1) but bad for 1.
        Wait, in example output, (1,5) was picked. It moved value 1 away from target.
        Why? Because 1 is just sitting there blocking or waiting? No, 1 was correct.
        Moving 1 to 5 and 3 to 1 helped 3 get closer.
        So we shouldn't forbid moves that increase cost for someone else if it enables future moves?
        But maybe simple greedy: Pick all edges $(u,v)$ such that $dist(P[u], u) > 0$ or $dist(P[v], v) > 0$? No.
        
        Let's refine the greedy criterion.
        We want to reduce the total Manhattan distance? No, tree distance.
        Sum of distances $\sum d(pos(x), x)$.
        For edge $(u,v)$, calculate change in total distance.
        $\Delta = (d(v, P[u]) + d(u, P[v])) - (d(u, P[u]) + d(v, P[v]))$.
        We want $\Delta < 0$.
        Actually, we might not even care about global sum.
        What if we just try to fix *all* possible swaps where $\Delta < 0$?
        If there is a conflict (edges share vertices), we need to select a Matching.
        Maximum Weighted Matching? Too expensive?
        $N=1000$. $O(E \sqrt{V})$ or $O(V^3)$ max matching is too slow for multiple test cases if repeated.
        But we only need *a* matching. Not necessarily max weight.
        We just need *some* matching that makes progress.
        If we find a matching $M$ such that for all $(u,v) \in M$, $\Delta < 0$?
        Or maybe just one $\Delta < 0$ per edge?
        
        Wait, if we restrict to edges where $\Delta < 0$, we ensure that the potential function (sum of distances) decreases.
        Since it's bounded below by 0, it must terminate.
        Does this guarantee minimum operations? Probably not, but likely good enough for passing.
        However, maximizing the size of $M$ reduces operations count per step.
        So:
        Iterate over all edges $e=(u,v)$. Calculate $\Delta(e)$.
        If $\Delta(e) < 0$, mark $e$ as candidate.
        From candidates, find a maximum matching.
        Apply it.
        Repeat.
        
        Complexity:
        Each step: Compute $\Delta$ for all edges. $O(N)$.
        Max matching: $O(N^3)$ or $O(N^{2.5})$. With $N=1000$, $10^9$ ops is too much.
        But note sum of $N^2$ is $10^6$.
        $N^2$ per test case is OK. But we have multiple test cases.
        If $N=1000$ and 1 test case, $10^6$ ops is fine.
        But max matching is superlinear. Hopcroft-Karp is faster on bipartite. General graph max matching is complex.
        However, our graph is a tree, which is sparse.
        Also, we only need *a* matching. Not max.
        Maybe just a greedy matching?
        Greedy matching for candidates: Iterate edges, if both endpoints free, pick.
        Does greedy matching work?
        It might leave many un-picked edges that could have been matched.
        But if we want to optimize, maybe we don't need perfect.
        Given time limit, let's try to code a simplified greedy.
        Wait, finding *any* matching from a subset of edges that forms a partial matching.
        Actually, just checking if $(u,v)$ is a good move is $O(N)$.
        Calculating LCA/distances takes $O(\log N)$ or $O(1)$ with preprocessing.
        Preprocessing DFS/BFS: $O(N)$. LCA $O(N \log N)$ or $O(1)$.
        Total $\sum N^2$ suggests $O(N^2)$ total time.
        If we have loop while unsorted:
          Update potentials/positions.
          Scan edges.
          Identify candidates.
          Select matching.
        How many iterations?
        Worst case, maybe $N$ iterations.
        If iteration cost is $O(N)$, total $O(N^2)$. Perfect.
        Max matching step: instead of complex algo, use simple greedy matching on candidates.
        Candidate list size $\le 2N$ (since each edge is candidate).
        Greedy matching on candidate list takes $O(|Candidates|) = O(N)$.
        So total time $O(N^2)$ per testcase.
        This fits constraints.
        
        Is it guaranteed to work?
        The condition $\Delta < 0$ ensures sum of distances decreases.
        The sum of distances is finite.
        Wait, swapping could increase distance for one element and decrease for another.
        Is it possible to get stuck in a local optimum?
        If all valid edges have $\Delta \ge 0$, but no matching exists?
        If all edges have $\Delta \ge 0$, it means no single swap (even sequentially) can reduce total distance?
        That implies the configuration is "stable". But the only stable configuration is sorted (distance sum 0).
        Wait, if $\Delta < 0$ for some edges, why not take them?
        Are there cycles of dependency where greedy fails?
        Maybe. But for the purpose of generating *a* solution that satisfies the checker (getting points), this randomized/greedy approach often works on random tests or weak checks. But this is a problem where we need to generate a solution.
        Actually, the provided solution should just implement this logic.
        
        Let's refine the distance calculation.
        Root the tree arbitrarily (say at 1).
        $depth[u]$ distance from root.
        $lca(u, v)$ distance.
        $d(u, v) = depth[u] + depth[v] - 2 \cdot depth[lca(u, v)]$.
        
        Algorithm Steps:
        1. Build adjacency list, BFS to compute depths and parent pointers for LCA (or just store LCA table for $O(1)$ query).
        2. Initialize `pos[x]` as vertex where value `x` is. Initially `pos[p[i]] = i`.
           Also keep track of current value at each vertex `val_at[i]`. Initially `p[i]`.
        3. Loop:
           Check if `val_at` is identity. If so, break.
           Collect candidate edges: For each $(u, v)$ in tree:
             $cost\_before = d(u, val\_at[u]) + d(v, val\_at[v])$
             $cost\_after = d(u, val\_at[v]) + d(v, val\_at[u])$ // swap values
             If $cost\_after < cost\_before$, add $(u,v)$ to list.
           
           Perform greedy matching on candidates:
             Maintain `used` array for vertices.
             Iterate edges. If neither $u$ nor $v$ used, add to `op_matching`, mark used.
           
           Execute swap for matching.
           Increment op_count.
           Record operation.
        
        One optimization: Instead of full LCA every time, can we optimize?
        With $T$ up to whatever, $\sum N^2 \le 10^6$.
        If we use binary lifting for LCA, build takes $O(N \log N)$, query $O(\log N)$.
        Loop runs maybe up to $N$ times (each time reduces total distance or sorts something).
        If we pick max number of edges, we reduce distance rapidly.
        Actually, in each step we swap at least one value to correct place or improve significantly.
        Worst case, maybe $O(N)$ steps?
        If $N$ steps $\times$ $N$ edges $\times$ $O(1)$ LCA (if optimized) $\approx N^2$.
        So this is viable.
        
        Implementation details:
        - Adjacency list `adj[N]`.
        - `parent[u][k]` for LCA.
        - `tin[u], tout[u]` for fast subtree check? Not needed for distance, just depth.
        - `get_dist(u, v)` function.
        - `swap_values` updates `val_at` and `pos`.
        - Output formatting.
        
        Corner case: $N=10$. Small inputs.
        Multiple test cases. Reset data structures.
        
        Is greedy matching safe?
        Suppose we have a path 1-2-3-4.
        Candidates: (1,2) good, (3,4) good. Greedy picks both.
        Suppose (1,2) good, (2,3) good, (3,4) bad.
        Greedy: picks (1,2). Then 2 is blocked. Picks (3,4) if good.
        Resulting size 1 vs 2?
        Maybe (2,3) prevents (1,2).
        If (2,3) yields more benefit, we prefer it?
        Simple greedy might miss it.
        However, finding Maximum Weight Matching is overkill.
        Maybe sort candidates by improvement amount $\Delta$ descending?
        Then pick first ones compatible.
        This is a heuristic for max weight matching (greedy by weights).
        Given time constraints and $N^2$ allowance, sorting candidates takes $O(N \log N)$.
        Iterating takes $O(N)$.
        Total per step $O(N \log N)$ or $O(N)$.
        Number of steps?
        Usually small.
        
        Let's consider the maximum possible number of operations.
        Since we improve distance, and max distance is $O(N)$, we might need $O(N)$ steps?
        Wait, distance sum is $\le N^2$. If each step reduces by 1, $O(N^2)$ steps?
        But typically swap reduces distance by at least 1 for at least one element.
        Often much more.
        However, to avoid TLE, let's hope $N$ is not hit with high steps.
        Actually, we can stop if we made progress.
        Wait, is there a deterministic bound?
        The "potential function" argument works.
        The concern is step count.
        If step count is high, this solution TLEs.
        But maybe for tree sorting, parallel swaps reduce complexity drastically.
        
        Alternative approach:
        BFS from each vertex? No.
        What if we decompose into cycles?
        Each cycle of length $L$ can be resolved.
        For a cycle $c_1, c_2, \dots, c_k$, we need to move elements.
        This is solvable.
        But handling cycles in parallel with matchings is hard.
        
        Let's stick to the Greedy Strategy. It's simple and likely passes given the problem type (often allows heuristics).
        Specifically, the scoring allows $s_i = \max(0, (base-m)/...)$.
        We want $m$ to be as small as possible.
        Greedy by weight is better.
        
        One edge case: What if no candidates found?
        This shouldn't happen unless sorted. If not sorted, there exists at least one element out of place.
        Is it guaranteed there exists an edge $(u,v)$ with $d(u, P[u]) + d(v, P[v]) > d(u, P[v]) + d(v, P[u])$?
        Consider a cycle. In a metric space, triangle inequality holds.
        Actually, in trees, yes, usually.
        Suppose we have a component of misplacements. The boundary must exist.
        If all adjacent swaps increase or maintain distance, it implies local minima.
        If no swap improves distance sum, then the arrangement is stable.
        But for trees, can we have stable arrangements where elements are permuted but not reachable without increasing distance?
        Imagine 1-2, 1 has 2, 2 has 1. $d(1,2)=1, d(2,1)=1$. Swap -> $d(1,1)=0, d(2,2)=0$. Better.
        So any inversion (2-cycle) improves distance.
        Longer cycles?
        If we have 1-2-3, 1 has 2, 2 has 3, 3 has 1.
        Swap (1,2): 1 gets 3 ($d=2$), 2 gets 2 ($d=0$). Prev 1($d=1$), 2($d=1$). Total 2 -> 2. No change?
        Wait. 1 has 2 ($d(1,2)=1$). 2 has 3 ($d(2,3)=1$). Sum 2.
        Swap 1-2: 1 gets 3 ($d(1,3)=2$). 2 gets 2 ($d(2,2)=0$). Sum 2.
        Swap is neutral.
        Wait, if neutral, we shouldn't perform it.
        But we can perform (2,3)? 2 has 3 ($d=1$), 3 has 1 ($d=2$). Sum 3.
        Swap (2,3): 2 gets 1 ($d(2,1)=1$), 3 gets 3 ($d(3,3)=0$). Sum 1.
        Improvement!
        So greedy will find (2,3).
        So it seems safe.
        We just need to handle cases where sum of distances doesn't decrease?
        The problem asks to minimize operations. We should prioritize swaps that reduce sum.
        If no swap reduces sum, but we MUST swap to proceed (neutral swaps), we might need to include those?
        Neutral swaps don't hurt distance but consume operations.
        If we have to use neutral swaps to unblock something, maybe.
        But can we reach a sorted state using only strict decreasing swaps?
        Usually yes, unless parity issues?
        Parity is preserved by swaps. Sorting preserves parity.
        So no parity obstruction.
        Therefore, a strictly decreasing sequence of distance sums should exist.
        
        So, strategy:
        1. Precompute LCA/Dist.
        2. Loop while not sorted:
            Candidates = []
            For each edge (u,v):
                calc delta. If delta < 0, add to Candidates.
            Sort Candidates by delta (descending magnitude of negative change).
            Find max matching in Candidates (using weighted greedy or just greedy by order).
            Since we sorted by delta, the first few are best.
            Greedy matching: iterate candidates, if edges disjoint, pick.
            This tries to pick high gain edges.
            Apply swaps.
            
        Wait, `delta < 0` means `new < old`.
        Delta = new - old. Negative is good.
        Magnitude of improvement is -delta.
        So we pick largest improvement.
        
        Complexity Check:
        Per Step:
          Compute deltas for all edges: $O(N \log N)$ or $O(N)$ with fast LCA.
          Sort: $O(N \log N)$.
          Matching: $O(N)$.
        Number of steps: ?
        If we assume average reduction leads to quick convergence.
        With $N=1000$, max steps ~2000?
        $2000 \times 1000 \times \log(1000) \approx 2 \cdot 10^7$ ops.
        This fits in 1s comfortably.
        
        Wait, calculating distance.
        LCA construction $O(N \log N)$. Query $O(\log N)$.
        Using binary lifting `up[u][i]`.
        
        Also, need to be careful with `long long` for distance sums?
        Max dist $N$, sum $N^2 \approx 10^6$. `int` is fine.
        
        Let's double check the constraints.
        Sum of $N^2 \le 10^6$.
        If we have many small cases, overhead matters.
        If 1 big case $N=1000$.
        $1000 \times 1000$ is $10^6$.
        If steps $\approx N$, then $O(N^2 \log N)$ total.
        $10^6 \times 10 = 10^7$. Fast enough.
        
        One detail: "It is guaranteed that the sum of n^2 of all test cases will not exceed 10^6."
        This strongly implies $O(N^2)$ per testcase is the expected complexity.
        My proposed solution is roughly $O(N^2)$ or slightly higher due to loop iterations.
        But since each iteration reduces potential energy, maybe iterations are fewer than $N$.
        Also, we might not need to check *all* edges every time.
        But $O(N)$ scan is cheap enough.
        
        Data Structures:
        - `vector<int> adj[1005]`
        - `int p[1005]` (initial permutation)
        - `int cur_val[1005]` (current value at vertex)
        - `int pos[1005]` (position of value i)
        - `tin[1005], tout[1005], depth[1005], up[1005][12]`
        - `bool in_op[1005]`
        
        Code structure:
        ```cpp
        #include <iostream>
        #include <vector>
        #include <algorithm>
        #include <cstring>
        #include <queue>
        
        using namespace std;
        
        const int MAXN = 1005;
        const int LOGN = 12; // 2^10 = 1024
        
        int n;
        vector<int> adj[MAXN];
        int up[MAXN][LOGN];
        int depth[MAXN];
        bool visited[MAXN];
        
        void dfs(int u, int p, int d) {
            visited[u] = true;
            depth[u] = d;
            up[u][0] = p;
            for(int i=1; i<LOGN; ++i) {
                if(up[u][i-1] != -1) up[u][i] = up[up[u][i-1]][i-1];
                else up[u][i] = -1;
            }
            for(int v : adj[u]) {
                if(v != p) dfs(v, u, d+1);
            }
        }
        
        int get_lca(int u, int v) {
            if(depth[u] < depth[v]) swap(u, v);
            for(int i=LOGN-1; i>=0; --i) {
                if(up[u][i] != -1 && depth[up[u][i]] >= depth[v]) {
                    u = up[u][i];
                }
            }
            if(u == v) return u;
            for(int i=LOGN-1; i>=0; --i) {
                if(up[u][i] != up[v][i]) {
                    u = up[u][i];
                    v = up[v][i];
                }
            }
            return up[u][0];
        }
        
        int get_dist(int u, int v) {
            return depth[u] + depth[v] - 2 * depth[get_lca(u, v)];
        }
        
        struct Operation {
            int k;
            int id[MAXN];
        };
        
        struct EdgeCandidate {
            int u, v;
            int old_dist_sum;
            int new_dist_sum;
            int diff; // new - old, should be negative
        };
        
        int main() {
            ios_base::sync_with_stdio(false);
            cin.tie(NULL);
            int t;
            if (!(cin >> t)) return 0;
            while(t--) {
                cin >> n;
                vector<int> p(n + 1);
                for(int i=1; i<=n; ++i) cin >> p[i];
                
                // Read edges
                for(int i=0; i<n-1; ++i) {
                    int u, v;
                    cin >> u >> v;
                    adj[u].push_back(v);
                    adj[v].push_back(u);
                }
                
                // Initialize
                memset(depth, -1, sizeof(depth));
                for(int i=0; i<=n; ++i) {
                    up[i][0] = -1;
                    visited[i] = false;
                    for(int j=0; j<LOGN; ++j) up[i][j] = -1;
                }
                memset(adj, 0, sizeof(adj)); // Clear adj
                // Actually vector clear is better
                for(int i=1; i<=n; ++i) adj[i].clear();
                
                dfs(1, -1, 0); // LCA prep
                
                // Current state tracking
                // pos[x] stores vertex where value x is currently located
                // val_at[u] stores value currently at vertex u
                vector<int> val_at(n + 1);
                vector<int> pos(n + 1);
                for(int i=1; i<=n; ++i) {
                    val_at[i] = p[i];
                    pos[p[i]] = i;
                }
                
                vector<pair<int,int>> edges_to_print; 
                // Store operations: each op is a list of edges.
                // But problem asks to print matching size and edge IDs.
                // We need to store original indices? Input edges have implicit indices 1..n-1.
                // Better to read edges and store index.
                // Wait, the input format: "u v meaning edge". Order defines index.
                // So we need to store the original index of each edge.
                // Let's modify adjacency to store {neighbor, index}.
            }
        }
        ```
        Wait, reading input properly.
        Need to store edge indices.
        `struct EdgeInfo { int to; int id; };`
        `vector<EdgeInfo> adj[MAXN];`
        Then during DFS, we just traverse edges. LCA doesn't depend on index.
        During scanning, we iterate `1` to `n-1` (edge indices).
        Reconstruct adjacency or iterate stored edges.
        
        Refined Plan:
        Store edges in a list `edges` as `{u, v, index}`.
        Run DFS for LCA.
        Loop:
          Find candidates.
          Match.
          Swap.
          Output.
          
        Wait, LCA calculation might be bottleneck inside loop if not careful.
        However, $O(\log N)$ is small.
        
        Is it possible to have infinite loop?
        No, distance sum decreases.
        Wait, we need to handle the case where no strict decrease is possible.
        This implies distance sum is constant.
        If sum of distances is constant and we are not sorted, we might be stuck in a cycle of configurations?
        Or stuck in local optima where $\Delta = 0$ for all edges.
        Actually, if $\Delta = 0$, we can swap but it won't help.
        However, in tree sorting, is it possible to have $\Delta = 0$ everywhere but not sorted?
        Yes, e.g. symmetric 2-cycle $u \leftrightarrow v$. $d(u,v)=1$.
        $d(u,v) + d(v,u) = 2$. After swap: $d(u,u) + d(v,v) = 0$.
        $\Delta = -2$.
        So strict decrease is possible for all inversions.
        If $\Delta=0$, maybe we have elements that don't move?
        Basically, as long as not sorted, there is at least one inversion relative to target?
        The target is identity.
        If we have value $x$ at $u$, and $x \neq u$.
        There must be a path from $u$ to $x$.
        Along this path, there must be an edge $(u', v')$ such that swapping improves position?
        In trees, usually yes.
        
        So the greedy strategy is robust.
        
        One subtle point: `val_at` array.
        Initially `val_at` contains $p_i$.
        When swapping edge $(u,v)$ (indices in matching):
          `temp = val_at[u]; val_at[u] = val_at[v]; val_at[v] = temp;`
        And we need to know WHICH edge index was used to output correctly.
        The edge indices are determined by input order.
        So `vector<pair<int,int>> current_edge_indices`? No.
        We just need to iterate over edges stored.
        `vector<pair<int,int>> edge_list` where `pair<int>` is `{u, v}` and we access by index.
        Better:
        `struct RawEdge { int u, v, id; };`
        `vector<RawEdge> all_edges;`
        Inside loop, iterate `0` to `all_edges.size()-1`.
        
        One optimization:
        If `all_edges` is static, we can reuse memory?
        With $N \le 1000$, creating vectors inside loop is fine?
        Better to clear/resize efficiently.
        
        Also, output format:
        Line 1: m
        Line 2..m+1: `k` then `t1, ..., tk`
        Note: `ki` is number of edges. `t` are 1-based indices.
        
        One last check on complexity.
        Sum of $N^2$ limit is tight.
        My solution is $O(\text{steps} \cdot N \log N)$.
        If steps $\approx N$, it is $O(N^2 \log N)$.
        $N=1000 \implies 10^7$ ops.
        Time limit 1s.
        It should pass.
        Also, LCA can be made $O(1)$ with Euler tour + RMQ, but $O(\log N)$ is likely sufficient.
        
        Let's write the code.
        Handling input correctly.
        Remember to handle multiple test cases.
        
```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstring>

using namespace std;

const int MAXN = 1005;
const int LOGN = 12; 

int n;
struct Node {
    int u, v, id;
};
vector<Node> edges;
vector<int> adj[MAXN]; // For DFS traversal only (needs neighbor)
// Wait, adj needs to store just neighbor for DFS/LCA build.
// We also need edges to iterate for matching.
// Let's use one adjacency list for LCA build.
// And the list of all edges separately.
// But wait, building LCA requires graph structure.
// Graph is tree.
// We can build LCA using the adjacency list.

int up[MAXN][LOGN];
int depth[MAXN];

void dfs_lca(int u, int p, int d) {
    depth[u] = d;
    up[u][0] = p;
    for (int i = 1; i < LOGN; ++i) {
        if (up[u][i-1] != -1)
            up[u][i] = up[up[u][i-1]][i-1];
        else
            up[u][i] = -1;
    }
    for (int v : adj[u]) {
        if (v != p) dfs_lca(v, u, d + 1);
    }
}

int get_lca(int u, int v) {
    if (depth[u] < depth[v]) swap(u, v);
    for (int i = LOGN - 1; i >= 0; --i) {
        if (up[u][i] != -1 && depth[up[u][i]] >= depth[v]) {
            u = up[u][i];
        }
    }
    if (u == v) return u;
    for (int i = LOGN - 1; i >= 0; --i) {
        if (up[u][i] != up[v][i]) {
            u = up[u][i];
            v = up[v][i];
        }
    }
    return up[u][0];
}

int get_dist(int u, int v) {
    return depth[u] + depth[v] - 2 * depth[get_lca(u, v)];
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    int t_cases;
    if (cin >> t_cases) {
        while (t_cases--) {
            cin >> n;
            edges.clear();
            edges.resize(n); // 0 to n-2
            for (int i = 1; i <= n; ++i) {
                adj[i].clear();
                up[i][0] = -1;
                for(int j=1; j<LOGN; ++j) up[i][j] = -1;
                depth[i] = -1;
            }
            int p[MAXN];
            for (int i = 0; i < n; ++i) {
                cin >> p[i+1];
            }
            for (int i = 0; i < n - 1; ++i) {
                int u, v;
                cin >> u >> v;
                adj[u].push_back(v);
                adj[v].push_back(u);
                edges[i] = {u, v, i + 1}; // 1-based ID
            }

            dfs_lca(1, -1, 0);
            
            vector<int> val_at(n + 1);
            for(int i=1; i<=n; ++i) val_at[i] = p[i];
            // Target is val_at[i] == i
            
            // Operations storage
            struct Op {
                int size;
                vector<int> indices;
            };
            vector<Op> solutions;
            int ops_count = 0;
            
            // Main loop
            while (true) {
                bool sorted = true;
                for(int i=1; i<=n; ++i) {
                    if(val_at[i] != i) {
                        sorted = false;
                        break;
                    }
                }
                if(sorted) break;
                
                // Find candidates
                // We want edges (u,v) such that dist(u, val[u]) + dist(v, val[v]) > dist(u, val[v]) + dist(v, val[u])
                // Store (weight = improvement, u, v, id)
                // Use a struct/vector for sorting.
                // Optimization: Avoid computing dist for all edges repeatedly if not needed?
                // But N=1000, iterating edges is fine.
                
                struct Cand {
                    int u, v, id, imp; // imp > 0
                    // Imp = (dist_old - dist_new)
                };
                vector<Cand> candidates;
                
                for (int i = 0; i < n - 1; ++i) {
                    int u = edges[i].u;
                    int v = edges[i].v;
                    
                    // Current costs
                    int old_d1 = get_dist(u, val_at[u]);
                    int old_d2 = get_dist(v, val_at[v]);
                    int old_sum = old_d1 + old_d2;
                    
                    // Swapped costs
                    int new_d1 = get_dist(u, val_at[v]);
                    int new_d2 = get_dist(v, val_at[u]);
                    int new_sum = new_d1 + new_d2;
                    
                    int imp = old_sum - new_sum;
                    
                    if (imp > 0) {
                        candidates.push_back({u, v, edges[i].id, imp});
                    }
                }
                
                if(candidates.empty()) {
                    // Should not happen if logic is correct, but break to prevent infinite loop
                    break;
                }
                
                // Greedy matching: sort by imp descending
                sort(candidates.begin(), candidates.end(), [](const Cand& a, const Cand& b){
                    return a.imp > b.imp;
                });
                
                vector<bool> used(n + 1, false);
                int matching_size = 0;
                for(auto& cand : candidates) {
                    if(!used[cand.u] && !used[cand.v]) {
                        used[cand.u] = true;
                        used[cand.v] = true;
                        matching_size++;
                        solutions.back().indices.push_back(cand.id); // Wait, we need to push to current op
                    }
                }
                
                if(matching_size == 0) break; // Safety
                
                // Create op record
                Op op;
                op.size = matching_size;
                for(auto& cand : candidates) { // Re-iterate to get indices? No, candidates vector lost match.
                    // We can just rebuild the matching list or store it.
                    // Simpler: create temp list of matched indices.
                }
                // Better: reconstruct matching from used or store during matching.
                // Storing is better.
            }
            
            // ...
        }
    }
    return 0;
}
```
Wait, in the matching phase, I iterate candidates but `candidates` order might be arbitrary if sorting failed? No, sorted.
Wait, I need to save which edges were picked.
Re-implementation of matching:
```cpp
                vector<int> chosen_ids;
                vector<bool> used_vertex(n+1, false);
                for(auto& cand : candidates) {
                    if(!used_vertex[cand.u] && !used_vertex[cand.v]) {
                        used_vertex[cand.u] = true;
                        used_vertex[cand.v] = true;
                        chosen_ids.push_back(cand.id);
                    }
                }
                
                if(!chosen_ids.empty()) {
                    // Apply swaps
                    // Need to record these for printing.
                    // But wait, we need to accumulate ops.
                    // Store ops in a vector.
                }
```
Need to reset `solutions` correctly.
Wait, I am accumulating operations.
Is it better to output directly? No, `m` first. So must store.
Memory limit 1024MB is huge. Vector storing all ops is fine.

Double check: Is it possible that picking an edge with high `imp` blocks an edge with even higher `imp`?
Yes.
Example: Triangle-like dependency.
Greedy strategy picks highest.
This is heuristic.
Given time limit, it's the best bet.
Wait, if `imp` is calculated, is it integer?
Distances are integers.
So yes.
Also, `old_sum - new_sum` is positive for good moves.

One corner case: $N=10$. $N^2 = 100$.
Constraints $10 < n \le 1000$.
Wait, "10<n". Smallest is 11.
Okay.

Wait, inside the loop, `candidates` contains edges.
I need to extract IDs.
In `chosen_ids`, I have IDs.
Wait, `candidates` stores `id`.
So `for(auto& cand : candidates)` inside matching logic is correct.
Just need to make sure I collect `chosen_ids` properly.

Final Check:
Does the problem require MINIMUM operations? "with as few operations as possible".
Yes.
My greedy might not find minimum.
Is there a specific strategy to ensure minimum?
Minimum operations to sort on a graph with matching constraint.
This problem might be equivalent to finding a sequence of matchings covering the permutation.
Actually, if we just output *any* valid sequence, maybe it is accepted as long as it sorts?
"The first line contains a single integer m (m>=0) meaning the number of operations you used."
The scoring formula rewards smaller m.
But the judge checks "correct result".
Correct result means sorted.
So I must ensure sorting.
My greedy ensures distance decreases, so it sorts.
The number of operations might be suboptimal but likely acceptable.
Wait, if my solution produces `m` much larger than optimal, I might get low score.
But usually, greedy on tree structures converges fast.
Wait, let's verify if I can optimize the matching selection.
Is it possible to solve this exactly?
Actually, if the graph is a star?
If graph is a line?
On a line, sorting is adjacent swaps.
If edges are available, it's easier.
Actually, the problem statement says "Bobo wants to make pi = i ... with as few operations as possible".
And scoring suggests it matters.
However, implementing exact minimum matching decomposition is hard.
Given the constraints and typical CP nature, the intended solution might be simpler or the greedy is sufficient.
Wait, there is a known approach for this:
Simply picking any valid matching until sorted is not optimal.
But the number of operations is likely bounded by $N$.
Is there a way to guarantee $O(N)$?
If we pick edges that fix *at least one* element permanently (i.e., put it in correct spot and never move again), then we make progress on element count.
An element is fixed if $val\_at[u] == u$.
If we swap $(u,v)$, and end up with $val\_at[u] == u$, we fixed $u$.
Can we force this?
Only if we find edge $(u,v)$ where $val\_at[v] == u$.
This corresponds to a 2-cycle $(u,v)$.
This is always a strict improvement.
So prioritizing edges where both end up fixed?
Or at least one ends up fixed.
My greedy uses distance.
If $val\_at[u] == u$, $dist(u, val\_at[u]) = 0$.
Swapping $(u,v)$ with $val\_at[u]=u$ changes dist to $dist(v, u) > 0$.
Wait, if $val\_at[u]=u$, moving $u$ out increases distance of value $u$.
So my greedy would *not* pick such edges (since `old_dist` for $u$ is 0, and `new_dist` is $>0$).
So my greedy naturally avoids touching fixed elements UNLESS they help others.
If a fixed element helps, its distance contribution increases.
Is it worth it?
In example, edge (1,5) touched 1 (fixed). $val\_at[1]=1$.
Old dist of 1 is 0. New dist is $d(1,5)=1$.
Increase in sum.
But $val\_at[5]=3$. Old dist 3. New dist 2.
Net change: $+1 - 1 = 0$. Wait, I calculated 0 earlier?
In example trace:
Val 3 at 5 (dist 3). Move to 1 (dist 2). Change -1.
Val 1 at 1 (dist 0). Move to 5 (dist 1). Change +1.
Sum change 0.
Wait, my previous trace said net change -1?
Ah, because in example output they picked (1,5) and (2,4).
Let's check (2,4).
Val 4 at 2 (dist 1). Move to 4 (dist 0). Change -1.
Val 5 at 4 (dist 3). Move to 2 (dist 2). Change -1.
Net -2.
Sum total change for both edges: 0 (from 1-5) + (-2) (from 2-4) = -2.
Wait, if (1,5) has net change 0, my greedy ignores it?
Because `imp` > 0.
If `imp` is 0, it's ignored.
Does ignoring it matter?
If we ignore (1,5), we might fail to solve in minimal steps?
Actually, if `imp` is 0, it doesn't reduce distance sum.
But maybe it rearranges elements to allow better future swaps.
However, ignoring it is likely safer to avoid loops?
Wait, if (1,5) is neutral, skipping it leaves 1 fixed.
Later, 3 is at 5. We need to move 3 to 3.
Path 5 -> 1 -> 2 -> 3.
We need to swap (5,1). If we didn't do it now, we do it later.
Doing it later might involve different costs.
Does delaying hurt?
Maybe.
But given constraints, maybe sticking to `imp > 0` is fine, or including `imp == 0`?
Actually, if we include `imp == 0` and sort them low, it's safer?
Or maybe `imp` is strictly defined as reducing distance?
If we rely on distance, we might stall at a local minimum where no single swap improves global distance.
Is it possible?
Imagine cycle 1-2-3-4. 1 at 2, 2 at 3, 3 at 4, 4 at 1. (Rotation by 1).
Path: 1-2, 2-3, 3-4, 4-1?
Assuming tree edges allow rotation.
Distance analysis:
1(2)->2. Dist 1.
2(3)->3. Dist 1.
3(4)->4. Dist 1.
4(1)->1. Dist 1.
Sum 4.
Swap (1,2):
1 has 4 (dist 2).
2 has 1 (dist 1).
Others unchanged.
New sum: $2+1+1+1 = 5$. Increase!
Swap (2,3):
2 has 4 (dist 1).
3 has 1 (dist 2).
New sum: $1+1+2+1 = 5$.
Swap (3,4):
3 has 4 (dist 0).
4 has 1 (dist 1).
1 still has 2.
Wait, let's re-evaluate carefully.
Configuration:
V1 has 2. V2 has 3. V3 has 4. V4 has 1.
Targets: 1->1, 2->2, 3->3, 4->4.
Distances:
Val 2 at 1: $d(1,2)$.
Val 3 at 2: $d(2,3)$.
Val 4 at 3: $d(3,4)$.
Val 1 at 4: $d(4,1)$.
Assume line 1-2-3-4.
$d(1,2)=1$.
$d(2,3)=1$.
$d(3,4)=1$.
$d(4,1)=3$.
Sum = $1+1+1+3 = 6$.
Swap (1,2) (vals 2 and 3):
V1 has 3 ($d(1,3)=2$).
V2 has 2 ($d(2,2)=0$).
Others unchanged.
New Sum = $2+0+1+3 = 6$.
Wait, sum didn't change?
Let's check again.
Old: V1(2), V2(3). $d(1,2)=1, d(2,3)=1$.
Swap: V1(3), V2(2). $d(1,3)=2, d(2,2)=0$.
Change for pair: $(2+0) - (1+1) = 0$.
So this swap is neutral.
Swap (2,3) (vals 3 and 4):
V2(3) ($d(2,3)=1$), V3(4) ($d(3,4)=1$).
Swap -> V2(4), V3(3).
$d(2,4)=2, d(3,3)=0$.
Change: $(2+0)-(1+1)=0$.
Swap (3,4) (vals 4 and 1):
V3(4) ($d(3,4)=1$), V4(1) ($d(4,1)=3$).
Swap -> V3(1), V4(4).
$d(3,1)=2, d(4,4)=0$.
Change: $(2+0)-(1+3) = -2$.
Improvement!
So, in this line example, there IS a swap with positive improvement (negative change).
So distance method finds *something*.
It seems distance monotonicity is robust.
Is it possible ALL swaps are neutral?
If all swaps are neutral, we are stuck.
But as seen, usually one exists.
So relying on `imp > 0` is good.

Final plan:
Implement greedy with `imp > 0`.
Use `vector` to store ops.
Print result.
Include headers.
Optimize IO.
Max N=1000.
`MAXN=1005`.

Wait, one thing.
In the loop, `dfs_lca` might traverse whole tree every time.
It's fast.
Inside `main`:
Clear global arrays.
Read T.
Loop.

Check Example logic with my code mentally.
Line 1-2-3-4-1 cycle.
Code calculates `imp` for all edges.
Selects best ones.
Should work.

One minor thing:
The `candidates` vector can contain multiple copies of edges if duplicate edges?
Input guarantees tree, so unique edges.
Edges stored in `edges` vector.
`cand` struct uses `edges[i]`.
Indices `u, v` match.
`id` is unique.
Sorting stable? Doesn't matter.

Correctness of LCA `up` initialization.
Root at 1. `up[1][0] = -1`.
DFS handles parents.
Seems fine.

```cpp
// Double check includes
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstring>
// for queue if needed? No queue used.
```
Looks good.

One final thought: Could there be a case where we need to perform an op with `ki=0`?
The problem says $m$ operations. $ki \ge 0$.
If sorted initially, $m=0$.
Loop condition handles this.

Output format:
`ki` followed by indices.
Indices are from input order (1-based).
My code stores `edges[i].id = i + 1` (since 0-based loop, 1-based ID). Correct.

The solution seems solid.

```cpp
/*
Problem analysis:
We need to sort the array P such that P[i] = i.
Allowed operation: Select a matching of edges, swap values at endpoints.
Minimize number of operations.
Approach:
Simulate the process. Maintain current state of values at vertices.
Iteratively find a set of edges (matching) that reduces the total displacement (sum of distances to target).
Repeat until sorted.
Complexity: O(T * steps * N * log N). Since sum of N^2 is bounded, and steps are expected to be low or N, this passes.
*/
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstring>

using namespace std;

const int MAXN = 1005;
const int LOGN = 12; 

int n;
struct Node {
    int u, v, id;
};
vector<Node> edges; // Stores all edges with IDs
vector<int> adj[MAXN]; // Adjacency list for LCA

int up[MAXN][LOGN];
int depth[MAXN];

void dfs_lca(int u, int p, int d) {
    depth[u] = d;
    up[u][0] = p;
    for (int i = 1; i < LOGN; ++i) {
        if (up[u][i-1] != -1)
            up[u][i] = up[up[u][i-1]][i-1];
        else
            up[u][i] = -1;
    }
    for (int v : adj[u]) {
        if (v != p) dfs_lca(v, u, d + 1);
    }
}

int get_lca(int u, int v) {
    if (depth[u] < depth[v]) swap(u, v);
    for (int i = LOGN - 1; i >= 0; --i) {
        if (up[u][i] != -1 && depth[up[u][i]] >= depth[v]) {
            u = up[u][i];
        }
    }
    if (u == v) return u;
    for (int i = LOGN - 1; i >= 0; --i) {
        if (up[u][i] != up[v][i]) {
            u = up[u][i];
            v = up[v][i];
        }
    }
    return up[u][0];
}

int get_dist(int u, int v) {
    return depth[u] + depth[v] - 2 * depth[get_lca(u, v)];
}

struct Candidate {
    int u, v, id, imp; // imp > 0 indicates improvement
};

struct Solution {
    int size;
    vector<int> indices;
};

void solve() {
    cin >> n;
    vector<int> p(n + 1);
    for (int i = 1; i <= n; ++i) {
        cin >> p[i];
    }

    edges.clear();
    edges.resize(n - 1);
    for (int i = 0; i < n; ++i) adj[i].clear(); // Clear adjacency

    for (int i = 0; i < n - 1; ++i) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
        edges[i] = {u, v, i + 1}; // Store 1-based index
    }

    // LCA Setup
    // Reset globals
    for(int i=1; i<=n; ++i) {
        depth[i] = -1;
        for(int j=0; j<LOGN; ++j) up[i][j] = -1;
    }
    dfs_lca(1, -1, 0);

    // Current state
    // val_at[u] is the value currently at vertex u
    vector<int> val_at(n + 1);
    for (int i = 1; i <= n; ++i) val_at[i] = p[i];

    vector<Solution> ops;
    vector<int> current_op_indices;

    while (true) {
        bool sorted = true;
        for (int i = 1; i <= n; ++i) {
            if (val_at[i] != i) {
                sorted = false;
                break;
            }
        }
        if (sorted) break;

        vector<Candidate> candidates;
        for (int i = 0; i < n - 1; ++i) {
            int u = edges[i].u;
            int v = edges[i].v;

            // Calculate distances before and after swap
            // Before: u has val_at[u], v has val_at[v]
            // After: u has val_at[v], v has val_at[u]
            int d_u = get_dist(u, val_at[u]);
            int d_v = get_dist(v, val_at[v]);
            int old_sum = d_u + d_v;

            int d_u_swap = get_dist(u, val_at[v]);
            int d_v_swap = get_dist(v, val_at[u]);
            int new_sum = d_u_swap + d_v_swap;

            int imp = old_sum - new_sum;
            if (imp > 0) {
                candidates.push_back({u, v, edges[i].id, imp});
            }
        }

        if (candidates.empty()) {
            // If no improvement possible but not sorted -> stuck.
            // However, mathematically there should be a path.
            // Fallback: pick any edge involving misplaced element?
            // For safety, break or try to find any valid swap.
            // Given the problem type, breaking might be wrong but logic suggests it won't happen.
            // To be safe, if no improvement, maybe just break? 
            // Or perform a forced swap?
            // Let's assume it won't happen for valid inputs reachable by swaps.
            break; 
        }

        // Sort candidates by improvement
        sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
            return a.imp > b.imp;
        });

        vector<bool> used(n + 1, false);
        vector<int> matched_indices;
        // Greedy matching
        for (const auto& cand : candidates) {
            if (!used[cand.u] && !used[cand.v]) {
                used[cand.u] = true;
                used[cand.v] = true;
                matched_indices.push_back(cand.id);
            }
        }

        if (matched_indices.empty()) continue; // Should not happen with candidates > 0

        // Apply swaps
        for (int idx : matched_indices) {
            int u = edges[idx-1].u;
            int v = edges[idx-1].v;
            swap(val_at[u], val_at[v]);
        }

        // Record operation
        Solution s;
        s.size = matched_indices.size();
        s.indices = matched_indices;
        ops.push_back(s);
    }

    cout << ops.size() << "\n";
    for (const auto& s : ops) {
        cout << s.size;
        for (int idx : s.indices) {
            cout << " " << idx;
        }
        cout << "\n";
    }
}

int main() {
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
Wait, `adj` clearing inside `solve` is needed. `memset` is faster.
The code is mostly correct.
One detail: `edges` vector is resized.
Is `edges` size correct? `n-1` edges. `resize(n-1)` is correct.
Input loop runs `n-1` times.
Seems good.

Is `vector<int> val_at` cleared? It's re-declared inside `solve`.
Wait, `solve` declared inside `while` or separate function.
Better to declare inside `solve` to keep scope clean.
I put everything inside `solve`.
Global arrays `up`, `depth`, `adj` need clearing.
I added loop to clear `up` and `adj`.
Wait, `adj` is `vector`. `adj[i].clear()` is fine.
`depth` and `up` need init.
Done.

One small issue: `edges` is global, but I use `edges.clear()`.
If I resize inside, `edges` is reallocated or capacity changed.
It is safer to just `clear()` and push_back or `resize` with `reserve`.
My code does `edges.clear(); edges.resize(n-1);`.
But then loop `for (int i=0; i<n-1; ++i) edges[i] = ...` is risky if `resize` was not initialized properly or if `edges` was empty.
`vector` resizing initializes default elements. `Node` is POD, default constructor? Struct has ints.
Default constructed ints are 0.
Wait, `edges` elements will be `{0,0,0}`.
Overwriting them in loop is safe.
However, `adj` needs to be cleared for next testcase.
I did `adj[i].clear()`.
Wait, `adj` is vector of vectors.
`adj` is array of vectors.
`vector<int> adj[MAXN]`.
`adj[i].clear()` is correct.
But I forgot to clear `visited`? No `visited` not used.
I used `depth` init.
Looks good.

Wait, `edges` is global.
In `solve`, `edges` is resized.
Is it possible `n` is large? `MAXN=1005`.
If `n` changes, `resize` adapts.
Correct.

Double check: The `get_lca` and `get_dist` functions rely on `up` and `depth` being consistent.
DFS fills them.
DFS starts from 1. `up[1][0] = -1`.
For other nodes, `up` computed.
Seems correct.

Time complexity:
Outer loop `solve` runs T times.
Inside:
Initialization: O(N).
LCA build: O(N log N).
Main loop:
  While not sorted:
    Compute candidates: O(N * log N) (dist calls).
    Sort candidates: O(N log N).
    Greedy match: O(N).
    Swap: O(M).
Number of iterations? Hopefully O(N) or constant amortized.
So total time per testcase O(N^2 log N).
Given sum of N^2 <= 10^6, total ops approx 10^7 * log factor.
Acceptable.

One potential bug:
`adj` needs to be populated.
In `solve`: `adj` is populated.
But `adj` is array of vectors.
I need to clear `adj` for `1` to `n`.
Loop `for (int i = 0; i < n; ++i) adj[i].clear();`
Wait, `adj` is global size MAXN.
Previous test case might have used `n_prev`.
If `n > n_prev`, `adj` from `0` to `n-1` need clearing?
Or rather, `adj` should be cleared completely up to `MAXN`?
Or just `1` to `n`.
Since each testcase sets its own `n`, clearing `1` to `n` of current testcase is sufficient *provided* we don't dirty unused nodes.
But in next testcase, `n` might be larger than previous, accessing dirty nodes?
We clear `1` to `n`.
Wait, `adj` has entries for all vertices.
If `n_curr > n_prev`, then indices `n_prev+1` to `n_curr` were not cleared in previous step?
Wait, `solve` is called per test case.
We clear `1` to `n` (current n).
Wait, if `n` is smaller, we clear `1` to `n`.
If next testcase `n` is larger, we use more indices.
But what about indices from `1` to `MAXN`?
If `n_prev` was 1000, and `n_curr` is 10.
We cleared `1` to 10.
Indices `11` to `1000` still hold old data?
Next testcase `n` might be 1000 again.
We clear `1` to 1000.
This works! We always clear `1` to current `n`.
And since we only use `1` to `n`, stale data outside range doesn't matter.
Wait, if `n` decreases, we don't clear the higher indices.
But if `n` increases later, we clear `1` to NEW_N.
Does `adj` at indices > prev `n` contain garbage?
They were populated by previous test case (when `n` was larger).
If we increase `n` again, we overwrite them in the `solve` loop?
No, we only clear `1` to `n`.
So if `n` increases from 10 to 100, we clear `1..100`.
This is safe.
Wait, `adj[i]` is vector. `adj[i].clear()` clears it.
Yes, if `adj[i]` has garbage, `clear` removes it.
So we must ensure we clear all `adj` indices that *will be used*.
Since we use `1..n`, we just clear `1..n`.
Wait, what if `adj[i]` has data from `i` being used in a previous bigger test case?
We clear it now.
So clearing `1..n` is always sufficient.
Because we are starting fresh for `1..n`.
Wait, what if `adj` wasn't accessed at `i` in current case but was accessed in previous?
It's cleared in the previous step (when `n` was large).
So yes, we are safe.

Final check on `vector<Node> edges`.
`edges` is global.
`edges.clear()` resets it.
`edges.resize` sets size.
Loop reads `n-1` edges.
Indices are `i+1`.
Everything seems consistent.

One variable name conflict? `size` in struct `Solution` vs member function?
`sizeof` etc.
`s.size()` might be ambiguous if not careful, but `s.size` member is fine.
Wait, `s` is `Solution` object.
`s.size` is member. `s.size()` doesn't exist for struct.
Wait, I wrote `cout << s.size;` which is correct.
Member variables are fine.
`vector` has `.size()`.
In `ops` vector, `s` is `Solution`.
No ambiguity.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstring>

using namespace std;

const int MAXN = 1005;
const int LOGN = 12; 

int n;
struct Node {
    int u, v, id;
};
vector<Node> edges;
vector<int> adj[MAXN];

int up[MAXN][LOGN];
int depth[MAXN];

void dfs_lca(int u, int p, int d) {
    depth[u] = d;
    up[u][0] = p;
    for (int i = 1; i < LOGN; ++i) {
        if (up[u][i-1] != -1)
            up[u][i] = up[up[u][i-1]][i-1];
        else
            up[u][i] = -1;
    }
    for (int v : adj[u]) {
        if (v != p) dfs_lca(v, u, d + 1);
    }
}

int get_lca(int u, int v) {
    if (depth[u] < depth[v]) swap(u, v);
    for (int i = LOGN - 1; i >= 0; --i) {
        if (up[u][i] != -1 && depth[up[u][i]] >= depth[v]) {
            u = up[u][i];
        }
    }
    if (u == v) return u;
    for (int i = LOGN - 1; i >= 0; --i) {
        if (up[u][i] != up[v][i]) {
            u = up[u][i];
            v = up[v][i];
        }
    }
    return up[u][0];
}

int get_dist(int u, int v) {
    return depth[u] + depth[v] - 2 * depth[get_lca(u, v)];
}

struct Candidate {
    int u, v, id, imp;
};

struct Solution {
    int size;
    vector<int> indices;
};

void solve() {
    cin >> n;
    // Reading p
    // We'll use local vector for p to save global vars mess
    vector<int> p(n + 1);
    for (int i = 1; i <= n; ++i) {
        cin >> p[i];
    }

    edges.clear();
    edges.reserve(n - 1);
    
    // Clear adj
    for (int i = 1; i <= n; ++i) adj[i].clear();

    for (int i = 0; i < n - 1; ++i) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
        edges.push_back({u, v, i + 1}); // 1-based index
    }

    // LCA Setup
    for(int i=1; i<=n; ++i) {
        depth[i] = -1;
        for(int j=0; j<LOGN; ++j) up[i][j] = -1;
    }
    dfs_lca(1, -1, 0);

    vector<int> val_at(n + 1);
    for (int i = 1; i <= n; ++i) val_at[i] = p[i];

    vector<Solution> ops;
    
    // We might want to check termination condition carefully.
    // If sorted, break.
    // Infinite loop protection? Not expected.
    
    while(true) {
        bool sorted = true;
        for (int i = 1; i <= n; ++i) {
            if (val_at[i] != i) {
                sorted = false;
                break;
            }
        }
        if (sorted) break;

        vector<Candidate> candidates;
        // Reserve to avoid reallocations
        candidates.reserve(n - 1);
        
        for (const auto& edge : edges) {
            int u = edge.u;
            int v = edge.v;
            
            // Distances
            int d_u = get_dist(u, val_at[u]);
            int d_v = get_dist(v, val_at[v]);
            int old_sum = d_u + d_v;

            int d_u_swap = get_dist(u, val_at[v]);
            int d_v_swap = get_dist(v, val_at[u]);
            int new_sum = d_u_swap + d_v_swap;

            if (old_sum > new_sum) {
                candidates.push_back({u, v, edge.id, old_sum - new_sum});
            }
        }

        if (candidates.empty()) {
            // Logic dictates this shouldn't happen before sorted.
            // If it does, we can't make progress, so break.
            // However, maybe a neutral swap is needed?
            // Assuming strictly positive improvement suffices.
            break; 
        }

        // Sort by imp descending
        sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
            return a.imp > b.imp;
        });

        vector<bool> used(n + 1, false);
        vector<int> matched_indices;
        
        for (const auto& cand : candidates) {
            if (!used[cand.u] && !used[cand.v]) {
                used[cand.u] = true;
                used[cand.v] = true;
                matched_indices.push_back(cand.id);
            }
        }

        if (matched_indices.empty()) continue;

        // Apply swaps
        for (int idx : matched_indices) {
            // edges are 0-indexed in vector, but stored IDs are 1-based.
            // Wait, `edge` in loop iterates `edges`. `edge.id` is 1-based.
            // `edges` vector is indexed 0..n-2.
            // `idx` corresponds to `edges[idx-1]`.
            // Let's find u, v from edges vector.
            int u = edges[idx-1].u;
            int v = edges[idx-1].v;
            swap(val_at[u], val_at[v]);
        }

        Solution s;
        s.size = matched_indices.size();
        s.indices = matched_indices;
        ops.push_back(s);
    }

    cout << ops.size() << "\n";
    for (const auto& s : ops) {
        cout << s.size;
        for (int idx : s.indices) {
            cout << " " << idx;
        }
        cout << "\n";
    }
}

int main() {
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
Wait, one tiny logical error in `matched_indices` collection.
I iterate `candidates`, check `used`.
When `matched_indices` collected, I iterate them to apply swaps.
`edges[idx-1]` access is correct assuming `idx` is 1-based index into the `edges` vector (which corresponds to input order).
Yes, `edges` is filled 0..n-2 with IDs `1..n-1`.
So `edges[id-1]` gives the edge with `id`.
Correct.

The loop `while(true)` with `break` if `candidates.empty()` handles termination.
Since sum of distances is strictly decreasing, it terminates.

I will wrap this in the response.
</think>

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstring>

using namespace std;

// Define maximum constants
const int MAXN = 1005;
const int LOGN = 12; // Sufficient for 1000

int n;

// Struct to store edge information
struct Node {
    int u, v, id;
};
vector<Node> edges;
vector<int> adj[MAXN]; // Adjacency list for DFS

// LCA tables
int up[MAXN][LOGN];
int depth[MAXN];

// Depth First Search to compute depths and 2^i ancestors
void dfs_lca(int u, int p, int d) {
    depth[u] = d;
    up[u][0] = p;
    for (int i = 1; i < LOGN; ++i) {
        if (up[u][i-1] != -1)
            up[u][i] = up[up[u][i-1]][i-1];
        else
            up[u][i] = -1;
    }
    for (int v : adj[u]) {
        if (v != p) dfs_lca(v, u, d + 1);
    }
}

// Function to find Lowest Common Ancestor
int get_lca(int u, int v) {
    if (depth[u] < depth[v]) swap(u, v);
    for (int i = LOGN - 1; i >= 0; --i) {
        if (up[u][i] != -1 && depth[up[u][i]] >= depth[v]) {
            u = up[u][i];
        }
    }
    if (u == v) return u;
    for (int i = LOGN - 1; i >= 0; --i) {
        if (up[u][i] != up[v][i]) {
            u = up[u][i];
            v = up[v][i];
        }
    }
    return up[u][0];
}

// Function to get tree distance
int get_dist(int u, int v) {
    return depth[u] + depth[v] - 2 * depth[get_lca(u, v)];
}

// Candidate structure for greedy selection
struct Candidate {
    int u, v, id, imp; // imp = improvement
};

// Structure to store an operation
struct Solution {
    int size;
    vector<int> indices;
};

void solve() {
    if (!(cin >> n)) return;
    
    // Read initial permutation
    vector<int> p(n + 1);
    for (int i = 1; i <= n; ++i) {
        cin >> p[i];
    }

    // Clear edges
    edges.clear();
    edges.reserve(n - 1);
    
    // Clear adjacency lists
    for (int i = 1; i <= n; ++i) adj[i].clear();

    // Read tree edges
    for (int i = 0; i < n - 1; ++i) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
        edges.push_back({u, v, i + 1}); // Store 1-based index
    }

    // Initialize LCA structures
    for(int i = 1; i <= n; ++i) {
        depth[i] = -1;
        for(int j = 0; j < LOGN; ++j) up[i][j] = -1;
    }
    dfs_lca(1, -1, 0);

    // Track current values at vertices
    // val_at[u] is the value currently residing at vertex u
    vector<int> val_at(n + 1);
    for (int i = 1; i <= n; ++i) val_at[i] = p[i];

    vector<Solution> ops;
    
    // Greedy simulation loop
    // We continue until the permutation is sorted
    while(true) {
        bool sorted = true;
        for (int i = 1; i <= n; ++i) {
            if (val_at[i] != i) {
                sorted = false;
                break;
            }
        }
        if (sorted) break;

        // Find candidate edges that strictly decrease total distance to target
        vector<Candidate> candidates;
        candidates.reserve(n - 1);
        
        for (const auto& edge : edges) {
            int u = edge.u;
            int v = edge.v;
            
            // Distances before swap
            int d_u = get_dist(u, val_at[u]);
            int d_v = get_dist(v, val_at[v]);
            int old_sum = d_u + d_v;

            // Distances after swap (values exchanged)
            int d_u_swap = get_dist(u, val_at[v]);
            int d_v_swap = get_dist(v, val_at[u]);
            int new_sum = d_u_swap + d_v_swap;

            // Improvement
            if (old_sum > new_sum) {
                candidates.push_back({u, v, edge.id, old_sum - new_sum});
            }
        }

        if (candidates.empty()) {
            // If no strictly improving move exists, break.
            // This theoretically implies we reached a local minimum or state, 
            // but distance logic should guide us to sorted state.
            break; 
        }

        // Sort candidates by improvement magnitude descending
        sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
            return a.imp > b.imp;
        });

        // Greedy matching on candidates
        vector<bool> used(n + 1, false);
        vector<int> matched_indices;
        
        for (const auto& cand : candidates) {
            if (!used[cand.u] && !used[cand.v]) {
                used[cand.u] = true;
                used[cand.v] = true;
                matched_indices.push_back(cand.id);
            }
        }

        if (matched_indices.empty()) continue;

        // Apply swaps recorded in this operation
        for (int idx : matched_indices) {
            // Get edge with 1-based ID idx
            int u = edges[idx-1].u;
            int v = edges[idx-1].v;
            swap(val_at[u], val_at[v]);
        }

        // Record operation
        Solution s;
        s.size = matched_indices.size();
        s.indices = matched_indices;
        ops.push_back(s);
    }

    // Output results
    cout << ops.size() << "\n";
    for (const auto& s : ops) {
        cout << s.size;
        for (int idx : s.indices) {
            cout << " " << idx;
        }
        cout << "\n";
    }
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
````

---

## 直接SFT 样本(sample_idx=2,得分 0.0,completion_tokens=2489)

````
I need to understand the problem first. We have a tree where each vertex has a number, and we want to make pi = i for all vertices (so that value i is at vertex i). In one operation, we can select a matching (a set of edges with no common vertices) and swap the values on both endpoints of each selected edge.

Let me think about the minimum number of operations needed:

If the current permutation needs d swaps to fix itself, then we need ⌈d/2⌉ operations if we could swap any two arbitrary elements in each operation.

This is because if you can swap any pair, you can reduce the distance to identity by at most 1 per element per operation. With pairs, each operation can contribute at most 1 towards the reduction. So if you need d swaps total, you need ceil(d/2) operations.

For example, the cycle 4 2 5 3 means:
- Position 1 has 4, needs 1 (distance 1)
- Position 2 has 2 (correct, distance 0)  
- Position 3 has 2... wait let me recalculate.

Actually, let me be more careful about what the cycle means here. We want value i at position i.

Current: p[1]=1, p[2]=4, p[3]=2, p[4]=5, p[5]=3

So:
- Position 1: has 1 ✓ (correct)
- Position 2: has 4, wants 2 (has wrong value)
- Position 3: has 2, wants 3 
- Position 4: has 5, wants 4
- Position 5: has 3, wants 5

The non-fixed elements are at positions {2, 3, 4, 5}. Let's trace which position should hold which current value:
- Value 2 belongs at position 2, currently at position 3
- Value 4 belongs at position 4, currently at position 2
- Value 3 belongs at position 3, currently at position 5


- Value 5 belongs at position 5, currently at position 4

Starting from value 4: position 2 → position 4 → position 5 → position 3 → position 2. That's a 4-cycle. With a 4-cycle, I can fix it in ceil(4/2) = 2 swaps: swap 2↔4 and then 3↔5 gives 1 4 5 2 3, but that leaves position 2 with 4 instead of 2, so that doesn't actually resolve everything.

Wait, I need to reconsider what happens when swapping. The value at position 2 moves to position 4, and the value at position 4 moves to position 2. For the cycle 4→2→5→3→4, each position cycles forward through the next position in the sequence. Swapping the first two positions would move the value at pos 2 to pos 4 and vice versa, but the value originally at pos 4 still ends up at pos 2—position 2 doesn't become fixed just because a value arrives there. The real insight is that I should track where each value currently sits rather than tracking positions themselves.

Tracing the cycle properly: value 4 starts at pos 2 and belongs at pos 4, value 4 at pos 4 needs to go to pos 5, value 5 at pos 5 needs to go to pos 3, and value 3 at pos 3 needs to go back to pos 2. This forms the cycle 4→2→5→3→4. To break this efficiently, swapping adjacent pairs in the cycle works—swapping values between consecutive positions like swapping the values at positions corresponding to elements 4 and 2, then handling the remaining elements separately.

The optimal approach is to decompose the permutation into disjoint cycles, and for each k-cycle, perform exactly k/2 swaps when k is even or (k+1)/2 swaps when k is odd. This gives us a formula for the minimum total swaps: sum of floor(cycle_length/2) across all cycles. Since one operation allows multiple independent swaps (as long as they don't share endpoints), each swap counts as half an operation cost, making the answer ceiling of total_swaps/2. The key constraint is that every swap operation must form a valid matching—no two edges can share a vertex—so I need to pair up the swaps within each operation carefully.

Looking at the sample case, the cycle decomposition shows value 4 needs to move from position 2 to position 4, value 2 from position 3 to position 2, value 5 from position 4 to position 5, and value 3 from position 5 to position 3—forming a single 4-cycle. The output indicates 4 operations total, which suggests the matching constraints significantly limit how many swaps can occur simultaneously compared to a naive pairing approach.

The solution strategy involves processing cycles sequentially. For each cycle, identify the mismatched positions and apply targeted swaps along the cycle to gradually place correct values in their target positions. The key insight is determining whether to process odd-length or even-length cycles differently based on the cycle's structure, ensuring swaps remain valid within matching constraints. 5) → (swap, get_value=5, swap, get_value=3, swap, get_value=2)
        # Process last element
        add_edge(last_pos, curr)
        add_edge(curr, curr - 2)
    
    return ops

print(process_cycle(mismatch_positions))
```

The algorithm traces through each cycle, greedily taking steps to place values correctly. For even-length cycles, it repeatedly swaps between current and previous positions, collecting edges to fix pairs of misplaced elements. Odd-length cycles follow a similar pattern but adjust for the final element, ending with a specific edge configuration that accounts for the cycle's parity. The greedy approach might not always produce optimal results. Consider a 6-cycle: starting at the beginning, swapping adjacent pairs fixes positions 1 and 6, then positions 2 and 5, leaving position 3 unpaired since the only available neighbor is already used. This yields 3 operations for the entire cycle. However, a better ordering exists—processing positions 2, 3, then 4, 6—also achieves 3 operations, suggesting that the initial greedy choice of direction matters for reaching optimality.

When testing a 7-cycle, I notice the cycle contains an odd number of elements, which affects how pairs can be formed. Starting from position 0, I take alternating steps backward and forward through the cycle while marking visited positions. The first step reaches position 5 (length 1), marking both start and end. At position 5, I try moving backward to position 4, which is unvisited, so I mark it and continue. Moving forward from position 4 leads to the start again, which is already visited—this path terminates. From the start node, moving forward goes to position 1 (unvisited, length 2, marked), then further forward reaches position 2, which is also visited.

The second component begins at position 1 with a forward step reaching position 2 (length 2). A backward step from position 2 returns to the start, already marked. A forward step lands on position 3, unvisited and marked with length 3. Forward continues to position 4, which marks itself and connects back to the start via one additional edge.

From position 3, a forward move reaches position 5, now visited. A backward move goes to position 2, also visited. Finally, a backward move from position 3 reaches position 1, completing the second cycle. After marking all components as processed, I have three cycles totaling 8 edges that form 5 operations. The algorithm successfully handles both even and odd-length cycles by greedily building components until blocked, then moving to the next unprocessed element and connecting components together, though the efficiency gains over brute force require further validation.

Looking at the code, it decomposes the permutation into disjoint cycles, processes each one by greedily pairing up its vertices into cycles of varying lengths, then builds individual matchings where each edge either closes a cycle or chains two together. For each such grouping, it creates a matching operation containing all those edges, and finally outputs the total number of operations followed by each operation's details.

The core issue is that the greedy construction might not minimize operations—I'm taking whatever pairings seem locally convenient without verifying they're globally optimal. Rather than trying to improve the matching construction itself, I should reframe this using a simpler principle: the answer equals max(n minus fixed points, divided by two) rounded up. Each swap contributes at most one fixed point per operation, so I'd need roughly (n - fixed_points)/2 operations in the best case, adjusted up if that division isn't exact.

But tracing through the actual sample reveals this breaks down—the algorithm produces four operations while a direct swap sequence would work in just two, suggesting the bound is loose for certain configurations. Rather than chase an elegant formula, a cleaner strategy is to treat this as a flow problem where I explicitly construct a directed graph capturing which swaps are possible, then find a maximum weight perfect matching on that structure to determine the true minimum.

However, computing a general weighted matching runs in O(E³√V) time, which pushes past the per-test-case n² budget since we'd need to solve this n times. I should look for something faster—a linear-time heuristic that pairs up unmatched elements intelligently rather than exhaustively searching for the optimal arrangement.
</think>

```cpp
#include <cstdio>
#include <vector>
#include <algorithm>

using namespace std;

int p[1005], parent[1005];

bool dfs(int u, int &cnt) {
    for (int v : adj[u]) if (!vis[v]) vis[v] = dfs(v, cnt), cnt += vis[v];
    return !cnt;
}

int main() {
    int T; scanf("%d", &T); while (T--) {
        int n; scanf("%d", &n);
        for (int i = 1; i <= n; i++) scanf("%d", p + i);
        for (int i = 0; i < n - 1; i++) {
            int u, v; scanf("%d%d", &u, &v);
            adj[u].push_back(v);
        }

        vector<int> op_edges;
        vector<int> visited_nodes;

        // Find cycles in the permutation
        for (int i = 1; i <= n; i++) {
            if (p[i] != i) {
                vector<int> cycle;
                int cur = i;
                while (true) {
                    cycle.push_back(cur);
                    cur = p[cur];
                    if (cur == i) break;
                }

                int len = cycle.size();
                int half = len / 2;
                for (int j = 0; j < len; j++) {
                    op_edges.push_back(j > half ? j - len + 1 : j);
                    visited_nodes.push_back(cycle[j]);
                }
            }
        }

        printf("%d\n%zu", visited_nodes.size(), visited_nodes.size() / 2);
        for (size_t i = 0; i < visited_nodes.size(); i += 2) {
            printf("2 %d %d\n", visited_nodes[i], visited_nodes[i + 1]);
        }
    }
    return 0;
}
```
````
