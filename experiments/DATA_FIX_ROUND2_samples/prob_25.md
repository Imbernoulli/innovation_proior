# 题 25 — base均分 0.0 vs 直接SFT均分 14.0(掉点 -14.0)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=0,得分 0.0,completion_tokens=32768)

````
The user wants a solution for an interactive competitive programming problem.
The goal is to determine if a graph $G=(V, E)$ is connected using a limited number of queries.
Graph parameters: $N \le 200$ vertices.
Query limit: $3500$ queries per testcase. $T \le 5$. Total queries allowed $\approx 17500$, but we should be efficient within the per-testcase limit.
Query type: `? s` where $s$ is a binary string representing a subset $S \subseteq V$. The response is $|N(S) \setminus S|$. This is the number of neighbors of $S$ that are outside $S$.
This value is also equivalent to the number of edges between $S$ and $V \setminus S$. Let this count be $cut(S, \bar{S})$.
If the graph is connected, then for any proper non-empty subset $S \subsetneq V$, there must be at least one edge leaving $S$, so $cut(S, \bar{S}) > 0$.
If the graph is disconnected, there exists a proper non-empty subset $S$ (a union of one or more connected components) such that no edges leave $S$, i.e., $cut(S, \bar{S}) = 0$.

So the problem reduces to finding if there exists a cut with weight 0.
Wait, actually, determining connectivity is usually done by BFS/DFS if we knew the adjacency list. Here we don't know the edges explicitly. We only know the size of the boundary.
Actually, if we find a subset $S$ such that query returns 0 and $0 < |S| < N$, then the graph is definitely disconnected because the set $S$ is isolated from the rest.
However, just checking random cuts isn't enough because we might pick $S$ inside a component.
Let's rephrase: If the graph is connected, then for all $S \subsetneq V, S \neq \emptyset$, the answer is $>0$.
But we can't check all subsets.
We need to find if we can reach all nodes.
Let's try to build a connected component starting from vertex 1.
Pick $S = \{1\}$. Query gives $d(1)$ (degree of 1).
This tells us how many neighbors 1 has.
But it doesn't tell us *who* they are.
Actually, the query result is $|N(S) \setminus S|$. It counts the size of the neighbor set excluding self-loops (which don't exist in simple graphs).
If we fix $S$, we learn nothing about which specific vertices are in $N(S) \setminus S$. Just the count.
This looks like we can't identify individual edges directly.
Is it possible to deduce the structure?
Maybe we can grow a known connected component.
Suppose we maintain a set $C$ of vertices known to be in the same connected component as node 1. Initially $C=\{1\}$.
If we query $S=C$, we get the number of edges crossing the cut $(C, V \setminus C)$.
If this count is 0 and $C \neq V$, then $C$ is an isolated component (or part of one), so $V$ cannot be connected unless $C=V$. Since we assume we want to check full connectivity, if count=0 and $C \neq V$, we output 0.
If count > 0, it means there is at least one edge connecting $C$ to $V \setminus C$. However, we still don't know *which* vertex connects.
We need to find a specific vertex $v \in V \setminus C$ such that $(u, v) \in E$ for some $u \in C$.
How can we find such a $v$?
Let $C = \{c_1, c_2, \dots, c_k\}$.
Query $S=\{c_i\}$ gives degree of $c_i$. But we don't know the sum.
Wait, let's look at the constraints. $N \le 200$. Queries $\le 3500$.
We have plenty of queries to do something close to $O(N^2)$ or even slightly more.
Can we find all edges? No, max edges $N(N-1)/2 \approx 20000$. That's too many queries if we iterate edges.
But we don't need all edges. We just need to see if the graph is connected.
We want to construct the whole component containing vertex 1.
Let $CurrentSet$ be the set of vertices we have successfully added to the component. Start with $\{1\}$.
While $|CurrentSet| < N$:
  Let $U = CurrentSet$.
  Query $S=U$. Result $k$.
  If $k == 0$ and $|U| < N$: Disconnected! Return 0.
  If $k == 0$ and $|U| == N$: Connected (trivially). Return 1.
  If $k > 0$: There exists some $v \in V \setminus U$ adjacent to some $u \in U$. We need to find one such $v$ to add to $U$.
  
  How to find such $v$ efficiently?
  We know $\sum_{u \in U} d(u)$ is related to internal edges + external edges. Specifically, query($U$) = $\sum_{u \in U} d(u) - 2|E(U)|$.
  This doesn't give individual neighbors.
  
  Consider the property: $v$ is in $V \setminus U$ is "outside".
  If we query $S=\{u\}$ for each $u \in U$, we get $d(u)$. This tells us how many neighbors $u$ has in total.
  But again, we don't know who they are.
  
  Wait, the query is: select subset $S$. Response is $|N(S) \setminus S|$.
  This effectively tells us the cut size.
  If we could somehow force the cut to isolate candidates, we might learn something.
  Suppose we pick $u \in U$ and $v \in V \setminus U$.
  Can we test if edge $(u, v)$ exists?
  Not directly.
  
  However, notice the query definition: $|N(S) \setminus S|$.
  If we fix $S$, we get a number.
  What if we vary $S$ by moving one element at a time?
  Consider $S = U \cup \{v\}$.
  Let $R_S = query(S)$.
  Then $R_{U \cup \{v\}} = |N(U \cup \{v\}) \setminus (U \cup \{v\})|$.
  $N(U \cup \{v\}) = N(U) \cup N(\{v\})$.
  So $R_{U \cup \{v\}} = |(N(U) \cup N(\{v\})) \setminus (U \cup \{v\})|$.
  Note $N(U) \setminus U$ are vertices outside $U$ connected to $U$. Let's call this set $Ext(U)$.
  So $query(U) = |Ext(U)|$.
  Now consider $Ext(U \cup \{v\}) = Ext(U) \cup (Ext(v) \cap \overline{U \cup \{v\}})$.
  Basically, $Ext(U \cup \{v\})$ contains all old external neighbors plus any new external neighbors coming from $v$, minus $v$ itself if $v$ has self loops (none here) or neighbors in $U$ which were already counted? No.
  Let's be precise.
  $query(S)$ counts $y \notin S$ such that $\exists x \in S, (x, y) \in E$.
  For $S=U$, it counts neighbors of $U$ not in $U$. Let these be $y_1, \dots, y_k$.
  If we add $v$ to $S$, i.e., $S' = U \cup \{v\}$.
  New neighbors are those $z \notin S'$ connected to some node in $S'$.
  Nodes in $Ext(U)$ are already $\notin U$. Are they in $S'$? Only if $z=v$.
  If $v \in Ext(U)$, it means $v$ was a neighbor of some $u \in U$.
  If we remove $v$ from being in the complement (i.e. add it to $S$), the contribution of $v$ to the cut count changes.
  Specifically, if $v \in N(U)$, then adding $v$ to $S$ removes $v$ from the count (since now $v \in S$, so $v \notin S'$ is false). But does it introduce new neighbors? Yes, $N(v) \setminus (U \cup \{v\})$.
  So $query(U \cup \{v\}) = query(U) - [v \in N(U)] + |N(v) \setminus (U \cup \{v\})|$.
  Wait, $|N(v) \setminus (U \cup \{v\})|$ is exactly the number of neighbors of $v$ outside $U \cup \{v\}$.
  Actually, $N(v) \setminus (U \cup \{v\}) = (N(v) \setminus U) \setminus \{v\}$.
  
  This seems complicated. Maybe simpler logic:
  We have a set $U$ of vertices in our component.
  We know that the boundary of $U$ in the graph is $B = N(U) \setminus U$.
  $|B| = query(U)$.
  If $|B| > 0$, we know the component extends into $B$.
  We need to find *one* vertex $v \in B$ that is connected to some $u \in U$.
  Actually, every vertex in $B$ is connected to some vertex in $U$ by definition.
  So if $query(U) > 0$, then there is at least one vertex in $V \setminus U$ connected to $U$.
  However, knowing $|B| > 0$ doesn't tell us *which* vertex is in $B$.
  To identify vertices in $B$, we need to distinguish them from vertices not in $B$.
  
  Vertex $y$ is in $B$ iff it is connected to some $u \in U$.
  This is equivalent to saying $(y \in V \setminus U) \land (\exists u \in U : u \sim y)$.
  We want to find such a $y$.
  Is there a way to detect connectivity to $U$?
  What if we query singletons? $S=\{y\}$. $query(\{y\}) = deg(y)$. This doesn't help with relation to $U$.
  
  What if we query $S = U \cup \{y\}$ for various $y \in V \setminus U$?
  As derived before: $query(U \cup \{y\}) = query(U) - I(y \in N(U)) + |N(y) \setminus U \cup \{y\}|$? No.
  Let $k = query(U) = |N(U) \cap (V \setminus U)|$.
  Let's take a candidate $y \in V \setminus U$.
  Consider $S_y = U \cup \{y\}$.
  Let $res_U = query(U)$.
  Let $res_{S_y} = query(S_y)$.
  Difference:
  $N(S_y) = N(U) \cup N(y)$.
  $S_y = U \cup \{y\}$.
  $N(S_y) \setminus S_y = (N(U) \cup N(y)) \setminus (U \cup \{y\})$.
  $= (N(U) \setminus (U \cup \{y\})) \cup (N(y) \setminus (U \cup \{y\}))$.
  Note $N(U) \setminus (U \cup \{y\}) = (N(U) \setminus U) \setminus \{y\}$.
  Also $N(U) \setminus U$ is exactly the set of vertices connected to $U$ (and not in $U$).
  So if $y \in N(U) \setminus U$, then removing $y$ reduces the count by 1.
  If $y \notin N(U) \setminus U$, then $y$ was not contributing.
  Additionally, we need to consider neighbors of $y$ that are not in $U \cup \{y\}$.
  These are $N(y) \cap (V \setminus (U \cup \{y\}))$.
  Let this be $deg_{out}(y)$. Note that neighbors in $U$ are ignored (since they are in $S_y$). Neighbors equal to $y$ impossible.
  So $res_{S_y} = res_U - I(y \in N(U)) + |N(y) \cap (V \setminus (U \cup \{y\}))|$.
  Wait, this logic assumes $N(y) \setminus (U \cup \{y\})$ does not contain elements of $N(U) \setminus U$.
  Actually, it's a set operation.
  Let $A = N(U) \setminus U$. This is the set of boundary nodes. $|A| = res_U$.
  Let $B = N(y) \setminus (U \cup \{y\})$. These are neighbors of $y$ outside $U \cup \{y\}$.
  Then $N(S_y) \setminus S_y = (A \setminus \{y\}) \cup B$.
  Since $y \in A$ (assuming we test a neighbor) or $y \notin A$ (if not neighbor).
  Wait, $y$ is in $V \setminus U$. $A \subset V \setminus U$.
  If $y \in A$, then $y$ is removed from $A$ when moving from $S=U$ to $S'=U \cup \{y\}$.
  So term $(A \setminus \{y\})$ contributes $|A| - 1$ if $y \in A$, and $|A|$ if $y \notin A$.
  The term $B$ contributes $|B|$.
  Does $B$ overlap with $A \setminus \{y\}$?
  An element $z \in A$ means $z \sim u$ for some $u \in U$.
  An element $z \in B$ means $z \sim y$ and $z \notin U$.
  So intersection is $\{z \in V \setminus U \mid z \sim U \text{ AND } z \sim y \}$.
  So $res_{S_y} = |A \setminus \{y\}| + |B| - |(A \setminus \{y\}) \cap B|$.
  $res_{S_y} = res_U - I(y \in A) + |B| - |(A \setminus \{y\}) \cap B|$.
  This looks messy because of the intersection term.
  Intersection term: vertices connected to both $U$ and $y$.
  
  Let's simplify.
  If we know $y$ is connected to $U$, does $res_{S_y}$ increase, decrease or stay same compared to $res_U$?
  Usually, adding $y$ brings its own external neighbors.
  Most likely $res_{S_y} \ge res_U - 1$.
  This doesn't seem to isolate membership in $A$ easily.
  
  Wait, let's look at the constraint again.
  $N \le 200$. Queries $\le 3500$.
  This is $O(N^2 / 2) = 20000$ operations roughly? No $20000$ is $100 \times 200$.
  Actually $3500$ is about $N \times N / 60$?
  Wait, $200 \times 200 = 40000$.
  Allowed queries is 3500 per test case.
  So $O(N^2)$ is feasible in terms of constant factor.
  
  We need a way to detect edges.
  Or simply to find if all nodes are reachable.
  
  Another approach:
  Iterate over pairs $(u, v)$ and check if edge $(u, v)$ exists?
  To check edge $(u, v)$:
  Query $S=\{u\}$. Returns $deg(u)$.
  Query $S=\{v\}$. Returns $deg(v)$.
  Doesn't help.
  Query $S=\{u, v\}$.
  $res = |N(u) \cup N(v) \setminus \{u, v\}|$.
  If we knew $u \not\sim v$, then $N(u)$ and $N(v)$ might intersect?
  This looks like standard graph reconstruction from degrees/cuts is hard.
  
  BUT we don't need to reconstruct the graph. We only need connectivity.
  Connectivity check can be done via Max Flow Min Cut or just BFS.
  Here we have a "Black Box" oracle for cut sizes.
  Is there a strategy?
  Strategy Idea:
  Maintain a set $S$ of connected vertices. Initially $S=\{1\}$.
  While $|S| < N$:
    Query $S$. Get $k$.
    If $k=0$ and $|S|<N$, return Disconnected.
    Else ($k>0$): We need to extend $S$.
    We need to find $v \in V \setminus S$ that is adjacent to some $u \in S$.
    Let's iterate through all potential candidates $v \in V \setminus S$.
    For a fixed $v$, how to check if $v \in N(S)$?
    Recall: $query(S) = k$.
    If we query $S \cup \{v\}$, we get $k'$.
    $N(S \cup \{v\}) \setminus (S \cup \{v\}) = (N(S) \cup N(v)) \setminus (S \cup \{v\})$.
    We know $N(S) \setminus S$ is the set of nodes connected to $S$ outside $S$. Let this be $Ext(S)$.
    Then $N(S) \cup N(v) = (Ext(S) \cup \{neighbors of S inside S\}) \cup N(v)$.
    Actually simpler: $N(S)$ includes neighbors inside $S$ and outside.
    Let's denote $N_{out}(S) = N(S) \setminus S$. Size is $k$.
    Let $N_{out}(S \cup \{v\})$ be the set of nodes connected to $S \cup \{v\}$ outside it.
    Nodes connected to $S \cup \{v\}$ outside it are:
    1. Nodes connected to $S$ but not in $S \cup \{v\}$. (These are in $N_{out}(S)$ except maybe $v$ if $v \in N_{out}(S)$).
       So count is $k - I(v \in N_{out}(S))$.
    2. Nodes connected to $v$ but not in $S \cup \{v\}$. These are $N(v) \setminus (S \cup \{v\})$.
       Let this count be $deg_{ext}(v)$.
    3. Overlaps?
       Nodes in group 1 are connected to $S$. Group 2 connected to $v$.
       They can share nodes if some node $z$ is connected to both.
       So $|N(S \cup \{v\}) \setminus (S \cup \{v\})| = |N_{out}(S) \setminus \{v\}| + |N(v) \setminus (S \cup \{v\})| - |((N_{out}(S) \setminus \{v\}) \cap (N(v) \setminus (S \cup \{v\})))|$.
       This intersection term represents neighbors of $v$ that are already connected to $S$ (and outside $S$).
    
    This dependency on intersections makes direct testing hard.
    However, notice we can compute $query(S \cup \{v\})$ for ALL $v$.
    Let's analyze what happens if we iterate all $v \in V \setminus S$.
    Sum of $query(S \cup \{v\})$ over all $v \in V \setminus S$?
    Maybe not.
    
    Let's step back.
    We have $O(3500)$ queries. $N=200$.
    We can afford $O(N)$ queries per expansion step if $O(N)$ steps occur (worst case).
    So we have enough budget to check each unconnected node individually against $S$.
    
    How to check if $v$ is connected to $S$?
    Condition: $v \in N(S) \setminus S$. Equivalent to: $\exists u \in S$ such that $u \sim v$.
    We want to check if $\exists u \in S$ s.t. edge $(u,v)$ exists.
    We don't know edges.
    But we know $query(\{u\}) = deg(u)$.
    This gives the number of neighbors of $u$.
    It doesn't identify them.
    
    However, look at this:
    $query(S)$ returns the size of $N(S) \setminus S$.
    If we query $S=\{u\}$ for all $u \in S$, we get degrees.
    If we query $S=\{u, v\}$, we get something else.
    
    Key Observation from similar problems:
    If we consider a partitioning of vertices.
    We want to find if we can reach everyone from 1.
    If we find any subset of vertices $U$ such that $query(U)=0$ and $U \subsetneq V$, then the graph is disconnected (and specifically $U$ is separated from $V \setminus U$).
    If $query(U)>0$ for all proper non-empty $U$, then it is likely connected (but we don't check all).
    But actually, if the graph is connected, there are NO subsets with cut size 0.
    So if we find ANY subset with cut size 0 and size $< N$, we can conclude Disconnected.
    Conversely, if we explore the component of 1 and eventually cover all vertices, we prove connectedness.
    
    How to expand the component?
    We have a set $U$ (known connected). $query(U)$ gives the size of the cut.
    If $query(U) == 0$, then no edges leave $U$. Since we started with $\{1\}$, $U$ is a connected component. If $U \neq V$, disconnected.
    If $query(U) > 0$, there are edges going out. We need to pick one endpoint $v$ of an outgoing edge.
    Since all neighbors of $U$ (in $V \setminus U$) are valid candidates, we just need to find *one*.
    
    Can we find one efficiently?
    Suppose we pick an arbitrary $u \in U$.
    We want to find $v \in V \setminus U$ such that $u \sim v$.
    We know $deg(u)$.
    But we don't know who the neighbors are.
    Wait, maybe we can use the fact that we can query subsets.
    
    Consider the following algorithm idea:
    Try to "isolate" potential bridges.
    Or better:
    Since we can choose any subset $S$, let's fix $u \in U$.
    Let's test every $v \in V \setminus U$.
    Query $S = U \cup \{v\}$.
    Result $res_v$.
    Recall formula: $res_v = query(U \cup \{v\})$.
    We know $k = query(U)$.
    If $v \in N(U)$ (meaning $v$ is connected to some $u \in U$):
      $query(U \cup \{v\}) = query(U) - 1 + |N(v) \setminus (U \cup \{v\})| - \dots$
      This depends on overlaps.
      
    Is there a simpler invariant?
    What if we query $S = V \setminus \{u\}$?
    Then $query(S) = deg(u)$?
    Let's check.
    $S = V \setminus \{u\}$.
    $N(S) = N(V \setminus \{u\})$.
    $N(S) \setminus S$.
    Since $V \setminus \{u\}$ contains almost all nodes, $N(S)$ is either $S$ or $V$.
    If $u$ has a neighbor in $S$, then $N(S)$ contains $u$ (and possibly others).
    Wait, $N(S) \setminus S$ counts nodes NOT in $S$ that are connected to $S$.
    The only node not in $S$ is $u$.
    So $query(S)$ is 1 if $u$ is connected to at least one node in $S$, and 0 otherwise.
    This tells us $deg(u) > 0$ (for $n > 1$).
    This is trivial (unless $u$ is isolated).
    Does it tell us anything about who? No.
    
    What if we want to find the exact set of neighbors of $U$?
    Actually, we just need to find one node in $N(U) \setminus U$.
    Let's assume we process nodes $v \in V \setminus U$ one by one.
    Is it possible to determine if $v$ is connected to $U$?
    This is the "connectivity decision problem" on a black box.
    
    Let's try to verify if $v \in N(U)$.
    Query $S = U$. Count $k$.
    Query $S = U \setminus \{u\}$. Count $k_u$.
    If we remove $u$ from $U$, how does $k$ change?
    $k = |N(U) \setminus U|$.
    $N(U) = N(U \setminus \{u\}) \cup N(u)$.
    So $N(U) \setminus U = (N(U \setminus \{u\}) \setminus U) \cup (N(u) \setminus U)$.
    Wait, $U \setminus \{u\}$ is a subset of $U$.
    $N(U \setminus \{u\}) \setminus U$ is the set of neighbors of $U \setminus \{u\}$ that are outside $U$.
    Since $U \setminus \{u\} \subset U$, any neighbor of $U \setminus \{u\}$ that is outside $U$ is also a neighbor of $U$ outside $U$.
    And $N(u) \setminus U$ are neighbors of $u$ outside $U$.
    These two sets are disjoint?
    Wait. $X = N(U \setminus \{u\}) \setminus U$.
    $Y = N(u) \setminus U$.
    Is $X \cap Y = \emptyset$?
    If $z \in X \cap Y$, then $z \in N(U \setminus \{u\})$ AND $z \in N(u)$.
    And $z \notin U$.
    So $z$ is connected to something in $U \setminus \{u\}$ AND to $u$.
    In general, $query(U) = k$.
    $query(U \setminus \{u\}) = k'$
    $query(U \setminus \{u\})$ counts neighbors of $U \setminus \{u\}$ outside $U$.
    $query(U)$ counts neighbors of $U$ outside $U$.
    Neighbors of $U$ are $N(U \setminus \{u\}) \cup N(u)$.
    Outside $U$: $(N(U \setminus \{u\}) \setminus U) \cup (N(u) \setminus U)$.
    So $k = | (N(U \setminus \{u\}) \setminus U) \cup (N(u) \setminus U) |$.
    $k' = |N(U \setminus \{u\}) \setminus U|$.
    Also $N(u) \setminus U$ are neighbors of $u$ outside $U$.
    This doesn't isolate $v$ from $U$ clearly.
    
    Wait! Let's reconsider the query definition carefully.
    $query(S) = |N(S) \setminus S|$.
    If we define $f(S) = query(S)$.
    Consider $S = \{u\}$. $f(\{u\}) = deg(u)$.
    Consider $S = V \setminus \{v\}$. $f(V \setminus \{v\}) = 1$ if $v$ is not isolated, 0 otherwise.
    This is useful to check isolation.
    
    What if we query $S = \{v, u_1, u_2, \dots \}$?
    
    Alternative Approach:
    We have plenty of queries. $N \le 200$.
    We can try to determine the degrees of all nodes.
    Query $S=\{i\}$ for all $i$. Store $d[i]$.
    Check if $\sum d[i]$ implies connectivity? No.
    But if there is an isolated node $v$, $d[v]=0$.
    If $d[i]=0$ for any $i \neq 1$, then $i$ is isolated.
    Since the graph is undirected, if $i$ is connected to $j$, $d[j]$ must be $>0$.
    But a node can have $d[i]>0$ but belong to a small component not containing 1.
    Example: Two components of size 10. All nodes in both have degree > 0.
    So degrees aren't enough.
    
    However, we can combine degrees with the subset query.
    We want to find if the graph has more than one component.
    Equivalently, find a non-trivial cut with weight 0.
    A cut corresponds to partition $(V_1, V_2)$.
    If we can guess the partition...
    
    Let's go back to building the component of 1.
    $Current = \{1\}$.
    Loop while $|Current| < N$:
      Query $S=Current$.
      Get $k = query(Current)$.
      If $k=0$: return Disconnected (Assuming $Current \neq V$).
      If $k > 0$: There exists a node $v \in V \setminus Current$ adjacent to some node in $Current$.
      We need to find such $v$.
      Since we have a lot of queries, maybe we can iterate over $v \in V \setminus Current$ and check if it's a neighbor.
      But how to check if $v$ is a neighbor of $Current$ efficiently?
      We can check if $v$ is a neighbor of $Current$ by seeing if adding $v$ to $Current$ changes the boundary count significantly?
      
      Let's revisit:
      $k = query(Current)$.
      We want to find $v$ such that $v \in N(Current) \setminus Current$.
      This means $v$ is connected to at least one $u \in Current$.
      Suppose we pick a $u \in Current$ with maximal degree (among nodes in $Current$ that are in $Current$?).
      Or simply iterate all $u \in Current$.
      For a fixed $u \in Current$, we want to find its neighbors outside $Current$.
      Let $S_u = Current \cup \{v\}$? No.
      
      Wait, if we check every $v \in V \setminus Current$:
      Check if $v$ is a neighbor of $u$?
      If we knew edges of $u$, we are good.
      Can we find neighbors of $u$?
      $u$ is in $Current$. $d[u]$ is known from previous singleton queries.
      Neighbors of $u$ are a subset of $V$.
      How to identify them?
      This is equivalent to finding neighbors of a vertex given degrees.
      Actually, $d[u]$ is small? No, could be up to $N$.
      But we have queries to subsets.
      
      Let's assume we found ONE neighbor $v$ of $u$ that is not in $Current$.
      Then we add $v$ to $Current$.
      Repeat.
      The challenge is finding *that* neighbor $v$.
      
      Here is a trick:
      Query $S = V \setminus \{x\}$. Result is $1$ iff $x$ is not isolated.
      Wait, $N(V \setminus \{x\}) \setminus (V \setminus \{x\})$.
      If $x$ is connected to someone in $V \setminus \{x\}$, then that someone is in the neighbor set.
      So $x$ is a neighbor of the set $S$.
      Wait, $N(S)$ contains all neighbors of $S$.
      If $x \in N(S)$, then there is an edge between $x$ and $S$.
      Since $S = V \setminus \{x\}$, this is true if $deg(x) > 0$.
      And $N(S) \setminus S$ will contain $x$ (and possibly other nodes? No, $x$ is the only node not in $S$).
      So $query(V \setminus \{x\}) = 1$ if $deg(x) > 0$, else 0.
      This only checks isolation.
      
      What if we use subsets to separate $x$ from $y$?
      $query(S) = |N(S) \setminus S|$.
      Consider $S = Current \cup \{x\}$.
      If $x$ is already in $Current$, redundant.
      If $x \notin Current$.
      $query(Current \cup \{x\}) = |N(Current \cup \{x\}) \setminus (Current \cup \{x\})|$.
      Let $N(Cut)$ be the set of neighbors of $Current$ outside $Current$.
      $N(Cut) = N(Current) \setminus Current$. Size is $k$.
      When we add $x$ to $Current$, $x$ is no longer "outside".
      So any neighbor of $x$ that was in $N(Cut)$ is now inside $NewCut$.
      Wait, the set of neighbors of the NEW set is $N(NewS) = N(Cut) \cup N(x) \cup \dots$ (edges within $Current$ don't contribute).
      Actually $N(Current) = \{v \mid \exists u \in Current, u \sim v\}$.
      $N(Current \cup \{x\}) = N(Current) \cup N(x)$.
      The difference $N(Current \cup \{x\}) \setminus (Current \cup \{x\})$ vs $N(Current) \setminus Current$.
      The elements in $N(Current) \setminus Current$ are those connected to $Current$.
      If $x$ is one of these elements, it gets removed from the set (since it becomes part of the set).
      If $x$ is NOT one of these elements, the count from $N(Current)$ remains unchanged.
      However, $x$ brings in new elements: $N(x) \setminus (Current \cup \{x\})$.
      So, $query(Current \cup \{x\}) = query(Current) - [x \in N(Current)] + |N(x) \setminus (Current \cup \{x\})| - |(N(Current) \setminus \{x\}) \cap (N(x) \setminus (Current \cup \{x\}))|$.
      This intersection term is problematic.
      
      However, what if we select $Current$ such that $x$ has NO neighbors in $Current$ yet?
      Initially $Current=\{1\}$. For any $x \notin \{1\}$, $x$ has neighbors in $\{1\}$ iff $(1,x) \in E$.
      If $(1,x) \in E$, then $x \in N(\{1\}) \setminus \{1\}$. So $query(\{1\}) \ge 1$.
      Adding $x$: $Current=\{1, x\}$.
      $query(\{1, x\}) = |N(\{1, x\}) \setminus \{1, x\}|$.
      $N(\{1, x\}) = N(\{1\}) \cup N(\{x\})$.
      So $query(\{1, x\}) = | (N(\{1\}) \cup N(\{x\})) \setminus \{1, x\} |$.
      We know $N(\{1\}) \setminus \{1\}$ contains $x$ (if edge exists).
      Let's analyze the value change.
      Let $d_1 = deg(1)$. Let $d_x = deg(x)$.
      If $1 \sim x$:
        $x$ is in $N(\{1\})$. Removing $x$ from the set reduces count by 1.
        We gain neighbors of $x$. $N(x) \setminus \{1, x\}$.
        These are neighbors of $x$ distinct from 1.
        Are any of them already in $N(\{1\}) \setminus \{x\}$?
        If yes, we subtract overlap.
      This seems to require tracking sets.
      
      But wait. We don't need to know exact overlaps if we just want to know IF there are neighbors.
      Actually, we just need to find one $x$ that is connected to $Current$.
      Is it possible to check connectivity of $x$ to $Current$ without full details?
      
      Consider iterating $x$ over all unvisited nodes.
      Check if $x$ is connected to $Current$.
      We can calculate $query(Current)$ first. Call it $K$.
      Then for a specific $x$, we calculate $query(Current \cup \{x\})$. Call it $K'$.
      Relationship:
      $K' = K - I(x \in N(Current)) + \Delta_x$.
      Where $\Delta_x = |N(x) \setminus (Current \cup \{x\})| - | (N(Current) \setminus \{x\}) \cap N(x) \setminus (Current \cup \{x\}) |$.
      This $\Delta_x$ is always $\ge 0$ minus overlaps.
      Actually $\Delta_x$ could be negative?
      Suppose $x$ is very well connected. $N(x)$ includes many neighbors of $Current$.
      $x$ itself is a neighbor of $Current$.
      If we move $x$ from outside to inside, we lose the contribution of $x$ to the count $K$ (so $-1$).
      We gain the neighbors of $x$ outside.
      If neighbors of $x$ overlap with $N(Current)$, they don't contribute extra count.
      So $K'$ can be less than $K$?
      Example: $Current=\{u\}$. $N(u)=\{v\}$. $deg(u)=1$. $K=1$.
      Pick $x=v$. $N(x)=\{u\}$. $deg(x)=1$.
      $S=\{u, v\}$. $N(S)=\{u, v\}$. $S=Current \cup \{x\} = \{u, v\}$.
      $N(S) \setminus S = \emptyset$.
      $K' = 0$.
      $K = 1$.
      So $K'$ decreased.
      This suggests if $K' < K$, then $x$ is definitely connected to $Current$.
      What if $K' > K$?
      Suppose $N(u) = \{y\}$. $x$ is not connected to $u$.
      $Current=\{u\}$. $K=1$ (node $y$).
      Add $x$ (isolated). $N(x)=\emptyset$.
      $K' = 1$.
      Suppose $N(u) = \{y\}, N(z) = \{w\}$. $Current=\{u\}$.
      Add $x=z$. $K'=2$.
      
      So:
      If $x$ is connected to $Current$, $I(x \in N(Current))=1$.
      The change in count is $\Delta = |N(x) \setminus (Current \cup \{x\})| - |(N(Current) \setminus \{x\}) \cap N(x)| - 1$.
      Wait, if $x \in N(Current)$, it was counted in $K$. Now it is not (removed). So we subtract 1.
      Then we add new neighbors of $x$. Some might be already counted.
      So $K' = K - 1 + |N(x) \setminus (Current \cup \{x\}) \setminus (N(Current) \setminus \{x\})|$.
      This new term is $\ge 0$.
      So $K'$ could be $K - 1$ or larger.
      If $x$ is NOT connected to $Current$:
      $I(x \in N(Current))=0$.
      $K' = K + |N(x) \setminus (Current \cup \{x\})| - |(N(Current)) \cap N(x)|$.
      Again $K'$ could be smaller if intersection is large?
      Suppose $N(u) = \{y\}$. $x$ is connected to $y$ and nothing else.
      $N(x) = \{y, z\}$. $y \in Current$. (Impossible, $y \notin Current$).
      Wait, $N(Current) \setminus \{x\}$ are nodes outside $Current$.
      Let $N_{ext}(Current)$ be the set of nodes outside $Current$ connected to $Current$. $|N_{ext}(Current)| = K$.
      $N(x) \setminus (Current \cup \{x\})$ are neighbors of $x$ outside.
      Intersection: neighbors of $x$ that are also connected to $Current$.
      If $x$ connects to $Current$ (directly), $x \in N_{ext}(Current)$.
      If $x$ does NOT connect to $Current$, then $x$ is in $V \setminus (Current \cup N_{ext}(Current))$.
      If $x$ connects to some $z \in N_{ext}(Current)$, then $x$ is connected to a neighbor of $Current$ (distance 2).
      In that case $x$ is not in $N_{ext}(Current)$.
      So $x \notin N_{ext}(Current)$.
      If $x \notin N_{ext}(Current)$, then $x$ was not contributing to $K$.
      $K' = K + \text{new neighbors} - \text{overlaps}$.
      It's possible $K'$ stays same or increases or decreases?
      Actually, if $x$ is far away (dist 2), $K'$ might decrease if $N(x)$ overlaps heavily with $N_{ext}(Current)$? No, $N_{ext}(Current)$ is defined by edges to $Current$.
      If $x$ is connected to some $z \in N_{ext}(Current)$, that $z$ is connected to some $u \in Current$.
      $N(x)$ contains $z$. $N(u)$ contains $z$.
      This seems complicated.
      
      Let's look for a deterministic condition.
      We can use the following observation:
      $query(S) = |N(S) \setminus S|$.
      If we pick $S = Current$, $K = query(Current)$.
      We want to find $v \in N(Current) \setminus Current$.
      If we check all $v \in V \setminus Current$.
      For a specific $v$, consider $S_v = Current \setminus \{u\} \cup \{v\}$? No.
      
      Wait, what if we simply check connectivity of 1 to everyone else using queries $S=\{1, u, v, \dots\}$?
      Actually, $N \le 200$.
      We have 3500 queries.
      This is roughly $N$ iterations.
      In each iteration we need to add ONE node to the connected component.
      If we spend $N$ queries per iteration, total is $N^2 \approx 40000$. Too slow.
      We need amortized constant or near $O(N)$ per node.
      Wait, we have $3500$ queries. $N=200$.
      Average allowed per node is $17.5$.
      So we can afford checking a few nodes.
      
      Let's try this strategy:
      1. Identify isolated nodes. (Degree 0).
         Query $\{i\}$ for all $i$. If $query(\{i\}) == 0$ and $n > 1$, then $i$ is isolated.
         If we find any isolated node $\ne 1$, then Disconnected (unless $1$ is the only node).
         If $1$ has degree 0 and $n > 1$, Disconnected.
         Also if there is an isolated node $i \ne 1$, Disconnected.
         Note: Checking all $\{i\}$ takes 200 queries. We have space.
         
      2. Find the connected component of 1.
         Let $S = \{1\}$.
         We need to find all $v$ such that $path(1, v)$ exists.
         This is essentially running BFS/DFS.
         The issue is identifying neighbors.
         However, we can query $S = \{u\}$ for any $u$. This gives $deg(u)$.
         It doesn't give neighbors.
         But we can iterate candidates.
         
         Wait! We can determine edges!
         Given $N \le 200$, we can find all edges?
         Edges $\le 19900$. Queries $\approx 3500$.
         We cannot find all edges.
         But we don't need all edges.
         We just need to find if the graph is connected.
         
         Is there a randomized approach?
         Pick random $S$?
         If $query(S) = 0$ and $S \subsetneq V$, disconnected.
         If $query(S) > 0$, it might still be disconnected (e.g. S is one component).
         But if we keep expanding a component...
         
         Back to the core problem:
         Can we find if $v$ is connected to $Current$?
         Let's try:
         Compute $K = query(Current)$.
         Pick $v \in V \setminus Current$.
         Compute $K_v = query(Current \setminus \{u\} \cup \{v\})$? No.
         
         How about this:
         $K_v = query(Current \cup \{v\})$.
         If $query(Current \cup \{v\}) < query(Current)$, then $v$ MUST be in $N(Current) \setminus Current$ ?
         Let's verify.
         $K = |N(Current) \setminus Current|$.
         $K' = |N(Current \cup \{v\}) \setminus (Current \cup \{v\})|$.
         $N(Current \cup \{v\}) = N(Current) \cup N(v)$.
         $(N(Current) \cup N(v)) \setminus (Current \cup \{v\})$.
         This is $(N(Current) \setminus (Current \cup \{v\})) \cup (N(v) \setminus (Current \cup \{v\}))$.
         $N(Current) \setminus (Current \cup \{v\}) = (N(Current) \setminus Current) \setminus \{v\}$.
         So $K' = |(N(Current) \setminus Current) \setminus \{v\}| + |N(v) \setminus (Current \cup \{v\})| - |(N(Current) \setminus Current) \setminus \{v\} \cap N(v) \setminus (Current \cup \{v\})|$.
         Term 1 is $K$ if $v \notin N(Current) \setminus Current$. If $v \in N(Current) \setminus Current$, it is $K-1$.
         So $Term 1 \le K$.
         If $v \in N(Current) \setminus Current$, term is $K-1$.
         If $v \notin N(Current) \setminus Current$, term is $K$.
         
         Term 2 is $\ge 0$.
         Intersection term is $\ge 0$.
         So $K' = Term 1 + Term 2 - Term 3$.
         We want to know if $v$ is connected to $Current$. i.e. $v \in N(Current) \setminus Current$.
         Case 1: $v \in N(Current) \setminus Current$.
           Then $Term 1 = K - 1$.
           So $K' = K - 1 + (\dots) \ge K - 1$.
           Could be $\ge K$.
           Example: $v$ has no neighbors, $N(v)=\{Current \text{ nodes}\}$.
           Then $Term 2 = 0$. Intersection = 0 (no neighbors outside).
           Wait, $N(v)$ contains nodes in $Current$. $N(v)$ does not contain nodes in $V \setminus (Current \cup \{v\})$.
           If $N(v) \subseteq Current \cup \{v\}$, then $Term 2 = 0$. Intersection = 0.
           Then $K' = K - 1$.
           If $v$ has external neighbors. $K' \ge K$.
         Case 2: $v \notin N(Current) \setminus Current$.
           Then $Term 1 = K$.
           $K' = K + Term 2 - Term 3$.
           If $v$ is completely isolated (disconnected from $Current$ and no external neighbors), $K' = K$.
           
         This doesn't strictly differentiate.
         However, note that if $K' < K$, then $v$ MUST be connected to $Current$.
         Proof: If $v \notin N(Current)$, $Term 1 = K$.
         Then $K' = K + (\dots)$.
         Since Term 2 counts external neighbors of $v$, Term 3 counts overlaps.
         It's possible $Term 2 < Term 3$?
         $Term 2 = |N(v) \setminus (Current \cup \{v\})|$.
         $Term 3 = | (N(Current) \setminus Current) \cap N(v) \setminus (Current \cup \{v\}) |$.
         Note that $N(Current) \setminus Current$ is a set of nodes outside $Current$.
         $N(v) \setminus (Current \cup \{v\})$ is a set of nodes outside $Current \cup \{v\}$.
         Both are subsets of $V \setminus (Current \cup \{v\})$.
         It is possible that every neighbor of $v$ outside is already a neighbor of $Current$.
         In that case $Term 2 = Term 3$.
         Then $K' = K$.
         But can $K' < K$ if $v \notin N(Current)$?
         This would require $Term 2 < Term 3$. Impossible since $Term 3$ is intersection of Term 1 and Term 2 (roughly speaking).
         Wait, $Term 1$ in Case 2 is just the count of existing neighbors.
         $K' = | (Existing) \cup (New) |$. Union size is $\ge |Existing|$.
         So $K' \ge K$.
         So YES, if $K' < K$, then $v \in N(Current)$.
         Specifically $K' \le K$ implies $v \in N(Current)$?
         If $v \in N(Current)$, we saw $K' \ge K - 1$.
         If $v \notin N(Current)$, we saw $K' \ge K$.
         So if $K' = K - 1$, $v$ is connected.
         If $K' \ge K$, $v$ might or might not be connected.
         
         So we have a necessary condition: $query(Current \cup \{v\}) < query(Current)$ implies $v$ is a neighbor of $Current$.
         This helps identify a subset of candidates, but maybe not all.
         However, what if we reverse the test?
         What if we query $Current \setminus \{u\}$ instead?
         Or simply, we can iterate over all $u \in Current$.
         For a specific $u \in Current$, we want to find its neighbors.
         We can use the fact that $u$ has degree $d_u$.
         Neighbors of $u$ are a subset of size $d_u$.
         If we query $S=\{u\}$, we get $d_u$.
         If we query $S=\{u, v\}$, etc.
         
         Wait, this problem is solvable by iterating all nodes and trying to "link" them.
         Actually, with $N \le 200$, we can check for every pair $(u, v)$ if they are connected? No, too expensive.
         
         Let's rethink. Is there a simple condition?
         $query(V \setminus \{u\})$ checks isolation.
         What if we query $S = \{u\}$ for all $u$.
         Then we have degrees.
         If $\sum_{u \in V} d[u] = 2|E|$.
         If the graph is a tree, $|E| = N-1$. Sum = $2N-2$.
         But we don't know number of edges.
         
         Let's look at the "Note" example.
         T=1, N=4. Graph 1-2-3-4-2. (Edges: 1-2, 2-3, 3-4, 2-4).
         Query 1100 ({1,2}). Answer 2. (Neighbors of {1,2} outside: {3,4}. 1-2 connected to 3,4. 2 connected to 3,4).
         Query 0010 ({3}). Answer 2. (Neighbors of {3}: {2,4}).
         
         The goal is to find if we can reach all vertices.
         Let's build the component of vertex 1 greedily.
         $Current = \{1\}$.
         Loop:
           If $query(Current) == 0$ and $|Current| < N$: Disconnected. Break.
           We need to find some $v \in V \setminus Current$ such that $v \in N(Current)$.
           To do this efficiently:
           We can iterate all $v \in V \setminus Current$.
           Check if $query(Current \cup \{v\}) == query(Current) - 1$.
           Wait, we established earlier that if $K' = K - 1$, then $v$ IS a neighbor.
           So we just search for $v$ such that $query(Current \cup \{v\}) == query(Current) - 1$.
           Is this sufficient?
           Suppose $v$ is a neighbor. Then $K' \ge K - 1$.
           It could be $K, K+1, \dots$ if $v$ brings new neighbors.
           But if $v$ brings new neighbors, $K'$ increases.
           So if we rely on $K' == K-1$, we might miss neighbors that bring new neighbors.
           But wait, we can simply check all $v$ and find ONE that reduces the cut size?
           Or rather, if we find any $v$ that decreases the cut size (or keeps it same but adds nothing?), no.
           
           Actually, we can just check if $v$ is in $N(Current)$ by querying $S=\{u, v\}$ for some $u \in Current$.
           No.
           
           Let's use the property that $query(Current) \le N - |Current|$.
           Also $query(Current) \ge 1$ if connected.
           
           Let's refine the expansion strategy.
           We have a current set $U$.
           We want to add a neighbor $v$.
           We iterate over all $v \in V \setminus U$.
           For each $v$, we perform query $S=U \cup \{v\}$.
           Let $val(v) = query(U \cup \{v\})$.
           We know $val(base) = query(U)$.
           If $val(v) < val(base)$, then $v$ is connected to $U$ (as proven above, $val(base) - val(v) \ge 0$ if not connected, contradiction).
           Actually, if $v \notin N(U)$, $val(v) \ge val(base)$.
           So if $val(v) < val(base)$, then $v \in N(U)$.
           So we can filter candidates: Candidates $\{v \mid val(v) < val(base)\}$.
           If this set is empty but $query(U) > 0$?
           Wait, if $query(U) > 0$, there exists at least one neighbor $w$.
           So the set cannot be empty.
           So we will find at least one $v$.
           Then we pick the first $v$ from Candidates and add it to $U$.
           
           Is it possible that $val(v) \ge val(base)$ even if $v \in N(U)$?
           Yes, if $v$ brings many new neighbors.
           But in that case, we might NOT find $v$ using the strict inequality condition.
           However, does it matter?
           If we miss a neighbor $v$ because $val(v) \ge val(base)$, does it mean $U$ cannot be expanded?
           If there are OTHER neighbors $w$ with $val(w) < val(base)$, we will find $w$.
           If ALL neighbors satisfy $val(w) \ge val(base)$, we won't find any.
           Then we get stuck with $U$. But the graph is connected, so this shouldn't happen unless our logic is flawed or we missed something.
           Actually, is it guaranteed that there exists a neighbor with $val < val(base)$?
           Consider $Current = \{1\}$. $N(1) = \{2, 3, \dots, k\}$.
           $query(\{1\}) = k-1$ (all neighbors except themselves).
           Pick $v \in N(1)$. $query(\{1, v\})$.
           $N(\{1, v\}) \setminus \{1, v\}$.
           Old neighbors $\{2, \dots, k\} \setminus \{v\}$.
           Plus neighbors of $v$ outside $\{1, v\}$.
           If $v$ is connected to $1$ and only to $1$ (leaf).
           Then $N(v) \setminus \{1, v\} = \emptyset$.
           Then $query(\{1, v\}) = k-1$.
           Wait. $k$ was count of neighbors.
           If $N(1)=\{v_1, \dots, v_m\}$. $query(\{1\}) = m$.
           Take $v = v_1$.
           $N(\{1, v_1\}) = \{v_1, \dots, v_m\}$. (If no other edges).
           $N(\{1, v_1\}) \setminus \{1, v_1\} = \{v_2, \dots, v_m\}$. Size $m-1$.
           So $query(\{1, v_1\}) = m-1$.
           So it decreases by 1.
           So leaf nodes will trigger the decrease.
           What if $v_1$ is connected to many nodes in $V \setminus \{1, v_1\}$?
           Say $N(v_1) = \{1, w\}$.
           $query(\{1\}) = 1$ (only $v_1$).
           $query(\{1, v_1\}) = 1$ (contains $w$).
           Wait. $N(\{1\}) = \{v_1\}$.
           $query(\{1\}) = 1$.
           $N(\{1, v_1\}) = \{v_1\} \cup \{w\} = \{v_1, w\}$.
           Remove $\{1, v_1\}$. Remaining $\{w\}$.
           $query(\{1, v_1\}) = 1$.
           So $val(v_1) = 1 = val(base)$.
           So $v_1$ did NOT decrease the count.
           So my condition "$val(v) < val(base) \implies v \in N(U)$" holds.
           But the converse "if $v \in N(U)$ then $val(v) < val(base)$" is FALSE.
           It is possible that a neighbor $v$ causes $query$ to stay same or increase.
           
           However, in the case where $query$ stays same, $v_1$ had a neighbor $w$ outside.
           If we picked $v_1$, we add it to $U$. Then next step we might pick $w$.
           Wait, if we add $v_1$ to $U$, $N(1 \cup v_1)$ grows.
           But wait, if we can't distinguish $v_1$ (neighbor) from $w$ (non-neighbor) using the decrease check?
           Actually, if $v_1$ was a neighbor but didn't decrease, does it mean we skip it?
           If we iterate all $v$, and for all $v$ (including real neighbors), $val(v) \ge val(base)$.
           This means for every $v \in V \setminus U$, adding $v$ does not reduce the cut size.
           Is it possible that $N(U) \neq \emptyset$ (so $query(U)>0$) but no $v$ satisfies $val(v) < val(base)$?
           Yes, if every $v \in N(U)$ has degree $>0$ outside $U \cup \{v\}$ such that they compensate for the loss of $v$.
           AND every non-neighbor $z$ has neighbors in $N(U)$ such that it compensates.
           
           So this heuristic might fail to find neighbors.
           However, we can try another check.
           Since $N \le 200$, maybe we can verify connectivity of 1 to $v$ via specific queries.
           
           Wait! There is a query: $? s$.
           Can we query $S = \{u, v\}$?
           No, we want to know if $u \sim v$.
           If we query $S=\{u\}$ and $S=\{v\}$ and $S=\{u, v\}$?
           $a = deg(u)$, $b = deg(v)$, $c = |N(u \cup v) \setminus \{u, v\}|$.
           $N(u \cup v) = N(u) \cup N(v)$.
           $c = | (N(u) \cup N(v)) \setminus \{u, v\} |$.
           This doesn't give edge existence.
           
           Let's go back to basics.
           We need to verify if the graph is connected.
           If we can show that the size of the connected component of 1 is $N$, we are done.
           Or if we find a cut of size 0.
           
           Algorithm Proposal:
           1. Check if there is an isolated vertex.
              Query $\{i\}$ for all $i$.
              If $d(i)=0$ for any $i \neq 1$ (if $N>1$), Disconnected.
              If $d(1)=0$ and $N>1$, Disconnected.
           2. Find component of 1.
              Let $U = \{1\}$.
              Repeat until $U=V$ or $query(U)=0$:
                Calculate $K = query(U)$.
                If $K==0$ and $|U| < N$: Disconnected.
                
                We need to find a neighbor $v \in V \setminus U$.
                If $K > 0$, there is at least one neighbor.
                We need to find one efficiently.
                Since $N$ is small, maybe we can afford $O(N)$ checks per node?
                $N=200$. Max queries 3500.
                We can perform approx $3500/N = 17$ checks per node? No, total queries is 3500.
                So roughly 17 checks TOTAL per step?
                Wait, if we iterate $V \setminus U$, that's $O(N)$ checks.
                So we can only do this if $N$ is small or queries are cheap.
                With $N=200$, doing 200 checks per node is 40000 total. Too high.
                We need faster identification.
                
                Let's optimize finding neighbors.
                We know $K = query(U) > 0$.
                We want to find $v$ s.t. $v \in N(U)$.
                Consider the sum of degrees? No.
                
                Let's try to verify if a specific node $v$ is connected to $U$.
                Query $S = U \cup \{v\}$.
                Query $S = U \setminus \{u\} \cup \{v\}$?
                No.
                
                Let's try the following:
                Assume graph is connected.
                For each $v \in V \setminus U$, $query(U \cup \{v\})$ is computed.
                Let $K = query(U)$.
                For each $v$, let $val_v = query(U \cup \{v\})$.
                We established if $v \in N(U)$, $val_v \ge K-1$.
                If $v \notin N(U)$, $val_v \ge K$.
                So the minimal value of $val_v$ among all $v \in V \setminus U$ will correspond to a neighbor of $U$ (specifically, $K-1$).
                If the global minimum is $K-1$, then there is a neighbor with this property.
                If the global minimum is $\ge K$, it might be that no neighbor exists (impossible if $K>0$) OR all neighbors satisfy $val_v \ge K$.
                But wait, if $K>0$, there exists $w \in N(U)$.
                So there is at least one $v$ with $val_v \ge K-1$.
                So the minimum will be $\le K$.
                Actually, we can just pick $v$ with the smallest $val_v$.
                This $v$ is likely a neighbor.
                If $min\_val == K-1$, definitely a neighbor.
                If $min\_val == K$, it could be a neighbor or not.
                If it is a non-neighbor, then for ALL neighbors, $val_v \ge K$.
                But intuitively, leaves minimize $val_v$.
                Can we force leaves to appear?
                Maybe shuffle?
                
                Wait, if $val_v = K$ for all $v$, it means for every $v$, adding $v$ to $U$ keeps the cut size same.
                This implies $v$ is "attached" in such a way that its new edges replace $v$'s removal.
                
                But wait. $N \le 200$. $3500$ queries.
                We can afford to check EVERY $v \in V \setminus U$ once per level of BFS?
                If we do this for all levels, it's $O(N^2)$.
                Total queries $\approx 200^2 = 40000$. Too high.
                Wait, we only need to visit each node ONCE.
                So we perform $N$ steps.
                In each step, we need to scan $V \setminus U$ (size decreasing).
                Sum of costs $\approx \sum_{i=1}^{N} i \approx N^2/2$.
                Still 20000 queries.
                We have budget 3500.
                We need to reduce the constant factor.
                Maybe we don't need to query all $v$.
                
                How about picking a random sample? No, worst case needed.
                
                Let's use the property that if $v$ is connected to $U$, $v \in N(U)$.
                If we query $S=V \setminus \{v\}$, we get 1 iff $deg(v) > 0$.
                Wait, if $v \in N(U)$, it implies $v$ is not isolated (unless $U$ is just $\{v\}$ which is impossible).
                So checking isolation is a quick filter.
                If $query(\{v\}) == 0$, ignore.
                But this leaves connected vs non-connected non-isolated nodes.
                
                Let's consider the structure again.
                If we find any $S \subsetneq V$ such that $query(S) == 0$, we are done (Disconnected).
                So if during the expansion $query(U)==0$ (with $U \neq V$), we stop.
                If we manage to grow $U$ until $U=V$, we are Connected.
                The only difficulty is growing $U$.
                
                Maybe we can grow $U$ slower?
                Instead of BFS, can we just pick an arbitrary partition?
                No, we don't know partitions.
                
                Wait, we can just try to isolate a component.
                If the graph is connected, min-cut is at least 1.
                If disconnected, there is a cut of size 0.
                We want to find if there is a cut of size 0.
                Or rather, we can find one by trying to partition the vertices.
                But we can't check all partitions.
                
                Actually, we can run the following:
                Try to isolate vertex $v$ from the rest.
                Is it possible to check if $v$ belongs to a different component than $1$?
                If so, $1$ and $v$ are disconnected.
                How?
                If $1$ and $v$ are in different components, then $query(S=Component(1)) == 0$?
                Or $query(S=\{v\}) \dots$
                
                Let's use the constraint: 3500 queries for $N=200$.
                This allows roughly $3500/200 = 17$ queries per node on average.
                Or $17$ queries per level?
                If we can identify neighbors of 1 quickly.
                
                What if we query $S=\{1\}$ to get $d(1)$.
                If $d(1) = 0$, done.
                If $d(1) > 0$, we know 1 has $d(1)$ neighbors.
                We don't know who.
                But we know there are $d(1)$ neighbors.
                Can we find them?
                We can use $S=\{1, v\}$ for all $v$.
                Wait, we established $val(v)$ for $v \in V \setminus \{1\}$.
                $val(v) = query(\{1, v\})$.
                $base = d(1)$.
                If $v$ is a neighbor of 1, $val(v) = base - 1 + |N(v) \setminus \{1, v\}| - overlaps$.
                If $v$ is not a neighbor, $val(v) = base + \dots \ge base$.
                So if $val(v) < base$, then $v$ is neighbor.
                We can identify neighbors by checking $val(v) < base$.
                If we find such $v$, we add it to $U$.
                What if all neighbors satisfy $val(v) \ge base$?
                This happens if $val(v) \ge base$ for all $v \in N(1)$.
                This requires $|N(v) \setminus \{1, v\}|$ to be large enough or overlaps to compensate.
                If $val(v) \ge base$ for all neighbors, we fail to identify.
                
                HOWEVER, note that if $v$ is a neighbor, it is IN $N(1)$.
                If we check $val(v) < base$, we might not catch it.
                But can we catch IT?
                If $v \in N(1)$, then $query(\{1, v\})$ removes $v$ from boundary.
                So contribution drops by 1.
                Unless added neighbors compensate perfectly (net 0 change).
                If net change is positive or zero, we miss.
                
                Is it possible to guarantee catching a neighbor?
                What if we query $S = \{1\}$, get $K$.
                Then for EACH $v \in V \setminus \{1\}$, we compute $query(\{1, v\})$.
                Wait, if we find ANY $v$ with $val(v) < K$, we add it.
                If none found, but $K>0$, then all neighbors satisfy $val(v) \ge K$.
                This implies for every $u \in N(1)$, $u$ has enough external connections or overlaps to keep cut size $\ge K$.
                This means $u$ has degree at least $K$? No.
                
                Is there another query?
                Query $S = V \setminus \{v\}$. Result 1 if $deg(v)>0$.
                Query $S = V \setminus \{u, v\}$. Result ...
                
                Let's try to check if $1$ and $v$ are connected directly.
                Actually, we can use the property:
                $query(\{1\}) = d(1)$.
                $query(\{1, v\})$.
                If $1 \sim v$, then $N(\{1, v\}) = N(\{1\}) \cup N(v)$.
                If $1 \not\sim v$, $N(\{1, v\}) = N(\{1\}) \cup N(v)$. (Same union).
                But in case $1 \sim v$, $v \in N(\{1\}) \setminus \{1\}$.
                So $v$ is removed from the boundary when moving from $\{1\}$ to $\{1, v\}$.
                In case $1 \not\sim v$, $v \notin N(\{1\}) \setminus \{1\}$.
                So $v$ is NOT removed from boundary (it wasn't there).
                Thus, $query(\{1, v\}) = query(\{1\}) - 1$ IF $1 \sim v$ AND $N(v) \cap N(\{1\}) \setminus \{v\}$ is empty and no other neighbors added.
                Generally $query(\{1, v\}) \ge query(\{1\}) - 1$ is true regardless of edge?
                Wait.
                $K = |N(1) \setminus \{1\}|$.
                $K' = |(N(1) \cup N(v)) \setminus \{1, v\}|$.
                If $1 \sim v$, then $v \in N(1) \setminus \{1\}$.
                So $v$ is removed from $N(1) \setminus \{1\}$.
                So count from original part is $K-1$.
                Then we add neighbors of $v$.
                So $K' \ge K-1$.
                If $1 \not\sim v$, then $v \notin N(1) \setminus \{1\}$.
                So $v$ is NOT removed.
                So count from original part is $K$.
                Then we add neighbors of $v$.
                So $K' \ge K$.
                
                Conclusion:
                If $query(\{1, v\}) = query(\{1\}) - 1$, THEN $1 \sim v$.
                Because if $1 \not\sim v$, count is at least $K$.
                
                So checking $K' < K$ works to identify neighbors!
                And if $K' == K$, it's inconclusive (could be neighbor with extra neighbors).
                BUT, crucially, if $1 \sim v$, then $v$ is a neighbor.
                So if we iterate all $v$, and for all $v$, $K' \ge K$, then no $v$ satisfied the strict inequality.
                Does this imply no neighbor?
                No, as discussed, neighbors might increase the count.
                BUT WAIT.
                If $1 \sim v$, then $v$ is removed from boundary. This is a HARD reduction by 1.
                Adding neighbors increases the count.
                Overlap decreases further.
                So $K' = K - 1 + \text{something}$.
                If $\text{something} > 0$, $K' \ge K$.
                However, notice the "hard reduction".
                If $1 \not\sim v$, there is NO reduction from $v$.
                So $K' = K + \text{something}$.
                So:
                If $1 \sim v$, $K' \ge K - 1$.
                If $1 \not\sim v$, $K' \ge K$.
                
                So if we observe $K' < K$, we know $1 \sim v$.
                If we observe $K' \ge K$, we DON'T know.
                But we can't simply discard $v$.
                However, notice that $N \le 200$.
                If $1 \sim v$, $v$ contributes to $K$.
                If we can't find $v$ via $K'<K$, maybe we can find it differently.
                
                Wait! $K = d(1)$.
                If we iterate $v \in V \setminus \{1\}$.
                If $1 \sim v$, then $v$ is counted in $K$.
                If $K$ is small, say 1.
                Then there is only 1 neighbor, say $u$.
                Then for all $v \ne u$, $1 \not\sim v$. So $K' \ge K$.
                For $u$, $1 \sim u$. $K'$ could be anything $\ge 0$.
                If $K=1$, then $u$ is the unique neighbor.
                Then for $v=u$, $K' = query(\{1, u\})$.
                If $u$ has other neighbors, $K' \ge 0$.
                If $u$ is leaf, $K'=0$.
                If $K=1$ and $u$ is not leaf, $K' \ge 1$.
                In this case ($K=1, u$ not leaf), $K'_u \ge 1 = K$.
                And for all $v \ne u$, $K'_v \ge K = 1$.
                So ALL $v$ yield $K' \ge K$.
                We wouldn't distinguish $u$ from others.
                So the condition fails.
                
                BUT, if $K=1$, then $d(1)=1$.
                The graph must contain an edge (1, u).
                If we assume graph is connected (hypothetically), then all nodes must be reachable.
                But we need to VERIFY.
                If $K=1$, we know 1 has 1 neighbor.
                Let's call the neighbor set $N(1)$. $|N(1)|=1$.
                If we don't identify it, we fail.
                
                However, note that if $K=d(1)$ is small, we have fewer queries needed to find?
                No.
                Wait, if $K=1$, we can iterate all $v$.
                Check $query(\{1, v\})$.
                Wait, if $K=1$, we have only one neighbor $u$.
                $query(\{1\}) = 1$.
                $query(\{1, u\})$ involves $u$.
                $query(\{1, u\})$ will be determined by $u$'s neighbors.
                Is there any property distinguishing $u$?
                $query(\{1, u\}) = |N(\{1, u\}) \setminus \{1, u\}|$.
                $N(\{1, u\}) = N(1) \cup N(u) = \{u\} \cup N(u)$.
                So $N(\{1, u\}) \setminus \{1, u\} = N(u) \setminus \{1, u\} = \{ \text{neighbors of } u \text{ other than } 1, u \}$.
                So $query(\{1, u\}) = deg(u) - 1$ (since $1 \in N(u)$).
                For $v \ne u$, $1 \not\sim v$.
                $query(\{1, v\}) = |N(1) \cup N(v) \setminus \{1, v\}|$.
                $N(1) = \{u\}$.
                So $N(\{1, v\}) \setminus \{1, v\} = |\{u, N(v) \setminus \{1, v\}\} \setminus \{v\}|$.
                If $v$ connected to $u$, then $u$ is neighbor.
                If $v$ not connected to $u$, then $u$ is in set.
                So count is roughly $1 + deg(v)$.
                Wait, $deg(v)$ could be anything.
                So $query(\{1, u\}) = d(u) - 1$.
                $query(\{1, v\}) = d(v) + (\text{overlap?})$.
                If $u, v$ not connected, overlap is empty.
                So $query(\{1, v\}) = 1 + d(v)$ (since $u$ is in boundary).
                If $u, v$ connected, overlap $\{u\}$. $u$ removed from count?
                $N(1) \cup N(v) = \{u\} \cup N(v)$.
                Set outside $\{1, v\}$.
                Contains $u$ (if $u \ne 1, v$).
                If $v \sim u$, then $u \in N(v)$.
                Wait, $u$ is the ONLY neighbor of 1.
                So $u$ is in $N(1)$.
                So $query(\{1, v\})$ is at least 1 (because $u$ is always there, unless $u=v$).
                If $v=u$, $query = d(u)-1$.
                If $v \ne u$. $query \ge 1$.
                But if $d(u)$ is small, $d(u)-1$ could be 0.
                If $d(u)=1$ (leaf), $query(\{1, u\}) = 0$.
                If $d(v)=0$ (isolated), $query(\{1, v\}) = 1$.
                
                So if $d(1)=1$, we can identify $u$ if $u$ has lower query value than others?
                If $d(u)=1$, $K'_u = 0$. Others $K'_v \ge 1$.
                So $K'_u$ is strictly minimal.
                If $d(u) > 1$, $K'_u = d(u)-1$.
                For other $v$, $K'_v \ge 1 + (\dots)$.
                It is possible that $d(u)-1 \ge K'_v$?
                If $d(u)-1$ is large.
                e.g. Star graph centered at 1. $1$ connected to $2, \dots, n$.
                $d(1) = n-1$.
                $K = n-1$.
                Pick $v \in \{2, \dots, n\}$. $1 \sim v$.
                $K'_v = query(\{1, v\})$.
                $N(\{1, v\}) \setminus \{1, v\} = \{ \text{others} \}$. Count $n-2$.
                So for leaves, $K'_v = n-2 = K-1$.
                For center (here no center), leaves are symmetric.
                Pick non-neighbor? None exist.
                So for all neighbors $K' = K-1$.
                For all non-neighbors (none exist).
                So if we find $K'=K-1$, we know it's a neighbor.
                
                So it seems checking $K' < K$ is a strong indicator.
                If $K' \ge K$ for all $v$, it implies either no neighbors (impossible if $K>0$) or neighbors have high degree.
                Wait, if $d(1)=n-1$ (star), $K=n-1$.
                Neighbors $v$ give $K' = n-2 < n-1$.
                So we find them.
                
                Is there a case where $d(1)$ is large but we can't find neighbors?
                If $d(1)$ is large, $K$ is large.
                $K' = K - 1 + \Delta$.
                $\Delta \ge - (\text{overlaps})$.
                Wait, overlaps reduce count.
                $term 3$ is intersection of neighbors of $u$ and neighbors of $v$.
                Max overlap is bounded.
                Is it possible that $K' \ge K$ for all neighbors?
                We need $\Delta \ge 1$.
                $\Delta = |N(v) \setminus (Current \cup \{v\})| - |intersection|$.
                This is basically $d_{out}(v) - |N(v) \cap N(Current)|$.
                If $d_{out}(v) \ge |N(v) \cap N(Current)| + 1$, then $K' > K$.
                This is possible.
                However, notice that $N(Current) \setminus Current$ are neighbors of $Current$ OUTSIDE.
                For $v \in N(Current) \setminus Current$.
                $v$ connects to some $u \in Current$.
                $N(v)$ contains $u$ (inside).
                Other neighbors might be in $N(Current) \setminus \{v\}$ or outside.
                If $N(v)$ is dense inside $N(Current)$, then overlaps are high.
                
                But wait! We have $N \le 200$.
                We can implement the "Iterate all $v$" check.
                Even if the condition $K' < K$ is weak, we can try to use the minimal $K'$.
                Actually, let's assume we can find *at least one* neighbor.
                If $K' < K$, we pick it.
                What if all neighbors have $K' \ge K$?
                Then we fail to find neighbor?
                If we fail, we get stuck.
                BUT, if all neighbors have $K' \ge K$, does it imply anything about graph structure?
                It means neighbors are "rich".
                
                Wait, is there any case where this fails completely?
                Yes. Consider $K=1$. $u$ is neighbor. $d(u)=100$.
                $K'_u = 99$.
                Non-neighbor $v$ ($d(v)=0$). $K'_v = 1$.
                Then $K'_u > K'_v$.
                We would pick $v$ (incorrect).
                So simply minimizing $K'$ is BAD.
                Minimizing $K'$ favors nodes with few neighbors.
                We want nodes WITH neighbors of 1.
                Those nodes $u$ satisfy $1 \in N(u)$.
                So $u$ has a connection to $Current$.
                If $u$ is a non-neighbor, it has NO connection to $Current$.
                
                So we need to prefer nodes with connection.
                $query(\{u\}) = d(u)$.
                If $u$ connected to 1, it's possible $d(u)$ is high or low.
                
                Let's revisit the query $query(U \cup \{v\})$.
                We have $K = query(U)$.
                If $1 \sim v$, $K' \ge K-1$.
                If $1 \not\sim v$, $K' \ge K$.
                This means the minimal possible value for a neighbor is $K-1$.
                The minimal possible value for a non-neighbor is $K$.
                So the range of values for neighbors is $[K-1, \infty)$.
                Range of values for non-neighbors is $[K, \infty)$.
                So if we observe a value $K-1$, it MUST be a neighbor.
                If we observe $\ge K$, it COULD be neighbor or not.
                So if we find a value equal to $K-1$, we have a neighbor.
                If we never see $K-1$, it means either $K=0$ (done/disconnected) or all neighbors have value $\ge K$.
                This happens if for all neighbors $v$, $N(v)$ outside $Current$ is large enough to offset the loss of $v$.
                In particular, if $N(Current)$ is "small" (low cut), then neighbors might be rare.
                Wait, if $K=1$, and neighbor $u$ has $d(u)=1$, then $K'_u = 0$.
                If $d(u) \ge 2$, $K'_u \ge 1$.
                If $K \ge 1$, we can't necessarily distinguish.
                
                BUT! Notice $query(\{1, v\})$ gives info about $1-v$ edge EXISTENCE.
                If $query(\{1, v\}) == d(1)$, then $1 \not\sim v$?
                Let's check.
                $K' = query(\{1, v\})$. $K = d(1)$.
                If $1 \sim v$, $K' \ge K-1$.
                If $1 \not\sim v$, $K' \ge K$.
                So if $K' = K$, it could be either.
                BUT if $K'=K$, we can't distinguish.
                
                However, there is a trick.
                Check $query(V \setminus \{v\})$.
                This is 1 if $v$ not isolated, 0 if isolated.
                Check $query(V \setminus \{1, v\})$.
                This equals $|N(V \setminus \{1, v\}) \setminus (V \setminus \{1, v\})|$.
                The complement of $V \setminus \{1, v\}$ is $\{1, v\}$.
                So this is 1 if there is an edge between $\{1, v\}$ and the rest.
                If $1$ or $v$ is connected to something, this is 1.
                If both are isolated from everything else, it's 0.
                This doesn't tell us about $1-v$.
                
                Let's try a different strategy.
                We have $N \le 200$.
                We have $Q \le 3500$.
                We can try to determine the component of 1 by BFS-like process.
                We need to find neighbors of 1.
                If we can't distinguish perfectly, maybe we can do it probabilistically or by filtering.
                Or simply, we can use the fact that $N$ is small.
                Maybe we can verify ALL nodes.
                But 3500 queries.
                If we run BFS.
                At each step, we have a set $S$.
                We need to find a neighbor $v \notin S$.
                We can iterate all $v$.
                If we use the condition $K' < K$, we filter.
                If we don't find any $v$ with $K' < K$, then for all neighbors $K' \ge K$.
                In that case, we can't be sure.
                
                However, notice the constraint: score inversely linear to queries.
                This implies we should use minimal queries.
                The example queries used 3 queries for $N=4$.
                This is $O(N)$.
                
                Maybe we can check connectivity in $O(N)$ queries?
                Wait, if we check $S=\{1, \dots, N\}$, $K=0$.
                If we check $S=\{i\}$, $d(i)$.
                If we check $S=\{1, v\}$ for all $v$.
                
                Actually, let's consider the maximum matching or min-cut.
                Wait, there is a simpler idea.
                For every $v \in V \setminus \{1\}$, is it connected to 1?
                We can check this pairwise.
                But we don't know edges.
                
                Wait! If we iterate all $v \in V \setminus \{1\}$.
                Query $S = \{1, v\}$. Let $r_v = query(\{1, v\})$.
                Query $S = \{1\}$. Let $r_1 = d(1)$.
                We know $r_v \ge r_1 - 1$.
                Also $r_v \ge r_1$ if not neighbor.
                So neighbors satisfy $r_v \le r_1$? No, $\ge r_1 - 1$.
                So neighbors are in $[r_1 - 1, \infty)$.
                Non-neighbors in $[r_1, \infty)$.
                Wait, if $r_v = r_1 - 1$, it is a neighbor.
                If $r_v \ge r_1$, it might be a neighbor.
                
                BUT! If $r_v = r_1$, then we don't know.
                Wait, $r_1$ is $d(1)$.
                If $r_v = d(1)$, does it mean not neighbor?
                Example: $1 \sim v$, $v \sim w, z, \dots$ ($d(v)$ large).
                $r_1 = 1$ (assume only $v$). $r_v = d(v) - 1$ (since $1$ removed).
                So $r_v = d(v) - 1$.
                If $d(v) = d(1) + 1$, then $r_v = d(1)$.
                So indistinguishable.
                
                Okay, what if we assume we need to identify neighbors.
                We have $d(1)$ neighbors.
                If $d(1)$ is small, say 1, then the interval $[r_1-1, r_1]$ is small?
                No, if $d(1)$ is large, interval is large.
                But if $d(1)$ is large, we have MANY neighbors.
                So probability that we miss all neighbors is low? No, worst case required.
                
                What if we just collect ALL $v$ with $r_v = d(1)$?
                Wait. If $1 \sim v$, $r_v$ could be anything $\ge d(1)-1$.
                If $1 \not\sim v$, $r_v \ge d(1)$.
                So if $r_v = d(1)-1$, it is a neighbor.
                If $r_v > d(1)$, it could be anything.
                If $r_v = d(1)$, it could be anything.
                
                Wait, I missed something fundamental?
                $query(S) = |N(S) \setminus S|$.
                If $S=\{1\}$, count is neighbors of 1.
                If $S=\{1, v\}$.
                $N(S) \setminus S = (N(1) \cup N(v)) \setminus \{1, v\}$.
                If $1 \not\sim v$, then $v \notin N(1)$.
                Then $N(1) \setminus \{1\}$ has $d(1)$ elements.
                $N(v) \setminus \{1, v\}$ has $d(v)$ elements.
                Union size is roughly sum.
                If $1 \sim v$, then $v \in N(1)$.
                So $N(1) \setminus \{1\}$ loses 1 element ($v$).
                So base count is $d(1)-1$.
                So neighbors contribute a shift of $-1$ relative to $d(1)$.
                Non-neighbors contribute 0 shift (relative to $d(1)$).
                Then we add neighbors of $v$.
                So $r_v = d(1) - 1 + \text{something}$ for neighbor.
                $r_v = d(1) + \text{something}$ for non-neighbor.
                
                Is it possible to make $\text{something} = 0$?
                For neighbor: Need $N(v) \setminus \{1, v\}$ to overlap fully with $N(1) \setminus \{1\} \setminus \{v\}$?
                Yes, if $v$ is connected to all neighbors of $1$ except itself?
                This seems specific.
                But generally, we can expect neighbors to have lower scores or different distribution.
                
                Let's just trust that $r_v < r_1$ is a strong signal.
                Wait, if we just run BFS:
                $U=\{1\}$.
                Query $S=\{1\}$. $K = r_1$.
                Iterate $v \in V \setminus U$.
                Compute $r_v = query(\{1, v\})$.
                If $r_v < r_1$, add $v$ to $U$.
                (This finds at least one neighbor if $r_v$ minimizes to $K-1$).
                If no $r_v < r_1$:
                  This means for all $v$, $r_v \ge r_1$.
                  Since $K = r_1 > 0$ (else disconnected), there exists at least one neighbor $u$.
                  So for all $v$, $r_v \ge r_1$.
                  For the true neighbor $u$, $r_u \ge r_1$.
                  This implies $u$ has extra neighbors to compensate.
                  Wait, if no neighbor triggers $r_v < r_1$, we can't expand $U$ using this logic alone.
                  
                  Is there a case where no neighbor triggers it?
                  Yes, constructed earlier.
                  But notice, if we add ANY node $v$ (even if wrong) to $U$, we might progress?
                  No, if we add wrong node (non-neighbor), we break connectivity assumption of BFS.
                  
                  However, notice we can query $S = V \setminus \{1, v\}$?
                  No.
                  
                  Let's look at the constraints again.
                  3500 queries.
                  Maybe we can just check all pairs? No.
                  Maybe we can use randomization?
                  Random permutation of vertices, query prefix sums? No.
                  
                  Wait! $N \le 200$.
                  Is it possible to query $S=\{u\}$ for all $u$, get degrees.
                  Then we can check if $\sum_{v \in V \setminus U} d(v) = \dots$?
                  
                  Let's think about the simplest valid algorithm:
                  Since 3500 is quite generous for 200 vertices (max possible pairs ~20000, max cuts $2^{200}$), we just need an efficient expansion.
                  Maybe checking $r_v < r_1$ is sufficient because the counter-example requires special structure (dense overlaps).
                  If graph is random, neighbors will be identified.
                  But worst case matters.
                  
                  Actually, what if we iterate $v$ and check if $r_v == r_1$ AND $r_v$ is minimal?
                  
                  Let's try to write a solution that implements the expansion using $r_v < r_1$ condition, and fallback if fails.
                  Wait, if it fails, we are stuck.
                  BUT, maybe if we iterate ALL $v \in V \setminus U$ and check $r_v$, we can just pick one that is MINIMAL.
                  Minimal $r_v$ corresponds to "least additional burden".
                  If $v$ is a neighbor, we pay $-1$ initial cost, but pay $d_{out}(v)$. Net $-1 + d_{out}(v)$.
                  If $v$ is not a neighbor, we pay 0 initial cost. Pay $d_{out}(v)$.
                  Wait, we subtract overlap.
                  It's complex.
                  
                  Let's step back and think of the problem as "Check if cut size is 0".
                  If we pick $S = \{1, \dots, 100\}$. Query it.
                  If result 0 -> Disconnected.
                  If result > 0 -> There are edges from 1..100 to 101..200.
                  But this doesn't help finding edges.
                  
                  Wait, what if we pick $S$ such that it isolates the connected component?
                  We can't do that.
                  
                  Actually, there is a very specific query that solves this.
                  If we query $S = \{1, \dots, k\}$ for increasing $k$?
                  
                  Let's assume the condition $query(U \cup \{v\}) == query(U) - 1$ is reliable enough?
                  Wait, in competitive programming, sometimes you need to handle edge cases.
                  What if we iterate $v \in V \setminus U$ and check $query(\{u\} \cup \{v\})$ for all $u \in U$?
                  If $u \sim v$, then $query(\{u, v\}) \le query(\{u\}) - 1 + \dots$?
                  No, for fixed $u$, we want to find neighbors.
                  If we just pick $u \in U$ with MAX degree.
                  Then it is most likely to have neighbors.
                  Let's check all $v \in V \setminus U$ against a fixed $u \in U$ (say $u$ with max degree).
                  Query $S=\{u, v\}$.
                  $r = query(\{u, v\})$. $d_u = query(\{u\})$.
                  If $u \sim v$, $r \le d_u$?
                  Let's trace: $r = |(N(u) \cup N(v)) \setminus \{u, v\}|$.
                  $d_u = |N(u) \setminus \{u\}|$.
                  If $u \sim v$, $v \in N(u)$.
                  So $N(u) \setminus \{u\}$ contains $v$.
                  So removing $v$ from set gives $d_u - 1$.
                  Then add $N(v) \setminus \{u, v\}$.
                  Subtract overlaps.
                  So $r \ge d_u - 1$.
                  If $u \not\sim v$, $v \notin N(u)$.
                  $N(u)$ untouched. $r \ge d_u$.
                  So $r < d_u \implies u \sim v$.
                  So if $u \sim v$, $r$ is potentially smaller than $d_u$.
                  Wait, earlier I said $r$ could be $\ge d_u$.
                  Yes, if $N(v)$ is big.
                  BUT if $u \sim v$, $r \le d_u - 1 + \dots$
                  Wait, $d_u$ is fixed.
                  If $u \sim v$, $r$ CAN be $d_u - 1$.
                  If $u \not\sim v$, $r$ CAN be $d_u$.
                  So if we find $v$ such that $r < d_u$, then $u \sim v$.
                  If $r = d_u$, it's ambiguous.
                  So we check $r < d_u$.
                  If we find such $v$, we add it to $U$.
                  Is it guaranteed that if $u$ has neighbors, at least one $v$ satisfies $r < d_u$?
                  This requires for all neighbors $v$, $r \ge d_u$.
                  This means for all neighbors $v$, $r \ge d_u$.
                  This implies $N(v)$ overlaps sufficiently.
                  
                  What if we check ALL $v \in V \setminus U$ for ALL $u \in U$?
                  Cost $|U| \cdot |V \setminus U|$.
                  For $|U|=1$, cost $N$.
                  For $|U|=N/2$, cost $N^2/4$.
                  Total $\sum_{i=1}^{N} i \cdot (N-i) \approx N^3/6$.
                  For $N=200$, $200^3 / 6 \approx 800000 / 6 \approx 133,000$. Way too many queries.
                  We need $O(N^2)$ total.
                  So we can't check all pairs.
                  We can check each $u$ with remaining $v$ only once?
                  
                  Wait, if we just need to find ONE neighbor to expand.
                  Maybe we only need to check one specific $u$?
                  If we pick $u \in U$ randomly?
                  Or best $u$?
                  The one with highest degree $d_u$.
                  If $u$ has no neighbors left (isolated in bipartite sense), then $r \ge d_u$ for all $v$.
                  But $u$ was chosen from $U$, which is connected. So $u$ must have a path to other nodes? No, edges inside $U$.
                  If $u$ has edges to $U \setminus \{u\}$, but no edges to $V \setminus U$.
                  Then $u$ is not helpful.
                  But $U$ is connected to $V \setminus U$ (since $K > 0$).
                  So there is at least one node in $U$ connected to $V \setminus U$.
                  Let's say we pick the node in $U$ with MINIMUM remaining edges?
                  Or just pick a node $u$ and try to find its neighbor.
                  If $u$ has no external neighbors, we discard it?
                  But we don't know degrees.
                  But we can track "unprocessed" neighbors of $u$.
                  
                  Correct Strategy:
                  Maintain a queue of "potential active vertices" in $U$?
                  Start with $U = \{1\}$.
                  While $|U| < N$:
                     Calculate $K = query(U)$.
                     If $K == 0$, return 0.
                     We need to find a neighbor $v$.
                     Since we don't know who, we try to probe.
                     Pick an arbitrary $u \in U$.
                     Check all $v \in V \setminus U$.
                     Query $S=\{u, v\}$. Let $res = query(\{u, v\})$.
                     Let $du = query(\{u\})$.
                     If $res < du$, then $u \sim v$. Found one!
                     Else ($res \ge du$), $u$ might be not connected to $v$.
                     We iterate $v$.
                     If we find a neighbor, great.
                     If we finish loop over $v$ and find NO neighbor?
                     Then $u$ is not connected to any $v \in V \setminus U$.
                     Remove $u$ from $U$. (It's "dead").
                     If $U$ becomes empty? Impossible since $K > 0$.
                     If $U$ reduces but not empty, repeat loop.
                     
                     Complexity?
                     In each step of outer loop (finding one new node), we check one $u$.
                     But $u$ might be checked multiple times?
                     Actually, if we exhaust all $v$ for a specific $u$, we mark $u$ as exhausted.
                     We pick a fresh $u$ from $U$ (those not exhausted).
                     Wait, if $U$ has size $k$, and we exhaust $k$ nodes?
                     Total work: each $u \in U$ is tested against $V \setminus U$ at most once?
                     No, $V \setminus U$ shrinks.
                     So total queries $\sum_{u \in V} (N - |current U|)$.
                     Roughly $\sum_{u} N = N^2$.
                     $200^2 = 40000$. Still too high.
                     We have 3500.
                     
                     Wait, $U$ grows.
                     In first step $|U|=1$, test 1 against 199 nodes. 199 queries.
                     In second step $|U|=2$, pick $u$ (maybe 2nd node). Test against 198 nodes.
                     We continue this.
                     If we only test ONE node per expansion, we use $N$ nodes *avg $N/2$ tests* = $N^2/2$ queries.
                     Still 20000.
                     
                     We need to drastically reduce the number of tests per expansion.
                     We have $K = query(U)$.
                     We need to find $v \in N(U)$.
                     We know there are at least $K$ edges crossing.
                     Wait, $K$ is number of nodes connected to $U$.
                     We need to find ONE such node.
                     Can we check subsets?
                     
                     Actually, if $K$ is small, it's easier to find.
                     If $K$ is large, it's harder.
                     But wait! If $K$ is large, then almost any random node might be in $N(U)$.
                     
                     Wait, there is a solution involving just $O(N)$ queries.
                     If $query(U) == 0$, disconnect.
                     Otherwise, we can just query $S=V \setminus \{v\}$ for all $v$? No.
                     
                     Wait, look at the constraints and typical solutions for this problem (Farquaad network).
                     This is Codeforces 474D? No.
                     It's an interactive problem.
                     
                     Maybe we can use the fact that $T \le 5$ and Time Limit 2s.
                     But queries are limited.
                     
                     Let's check if we can determine if a node is in the same component as 1.
                     If we query $S=\{1\}$, we get $d(1)$.
                     If we query $S=\{v\}$, we get $d(v)$.
                     If $query(\{1, v\}) < query(\{1\})$, then $v$ is neighbor.
                     If we just check ALL $v$ against $S=\{1\}$, we use 200 queries.
                     This identifies neighbors of 1.
                     If we pick a neighbor $v$, update $U=\{1, v\}$.
                     Then check ALL $w \in V \setminus U$ against $S=\{u, v\}$?
                     Cost is sum of sizes of $V \setminus U$.
                     $\sum (N - i) \approx N^2/2 = 20000$.
                     Still too many.
                     
                     Is there a way to speed up finding ONE neighbor?
                     We have $K$ neighbors.
                     If $K > 0$, we can pick ANY neighbor.
                     Maybe we can query random nodes? No.
                     Maybe we can query $S=\{u, v, w, \dots\}$?
                     If we take $S = U \cup \{random\}$.
                     
                     Wait! 3500 queries is 17 per node.
                     Maybe we can use the property that if we query $S=\{1, \dots, 100\}$ and $S=\{101, \dots, 200\}$, we know cut.
                     But we don't know partition.
                     
                     What if we use a randomized algorithm?
                     Shuffle array $P$.
                     Construct set $S$ incrementally?
                     
                     Let's look at the limits again.
                     $N \le 200$. $Q \le 3500$.
                     We have ample time.
                     Maybe the condition $r_v < r_1$ is sufficient if we interpret correctly?
                     If $r_v = r_1$, it's ambiguous.
                     But what if we try to verify ambiguity?
                     If $r_v = r_1$, maybe $u \not\sim v$ or $u \sim v$ with compensation.
                     Compensation requires $d_{out}(v)$ to be large.
                     If we iterate $v$, and find $v$ with $r_v < r_1$, we pick it.
                     If all $v$ have $r_v \ge r_1$, then for all neighbors $r_v \ge r_1$.
                     This implies neighbors have large external degree.
                     Wait, if neighbors have large external degree, then the graph is "high connectivity"?
                     
                     Let's assume we can identify neighbors.
                     Wait, if we just implement the check $r_v < r_1$ and hope? No, must be correct.
                     
                     Wait! $K = query(U)$.
                     If we select $S = U \cup \{v\}$, $r_v = K - 1 + \dots$
                     We want to minimize $r_v$.
                     If $v$ is NOT a neighbor, $r_v \ge K$.
                     If $v$ IS a neighbor, $r_v \ge K - 1$.
                     So the minimum possible value for a non-neighbor is $K$.
                     The minimum possible value for a neighbor is $K-1$.
                     Thus, if we find $v$ such that $r_v = K-1$, we are done.
                     If the minimum over all $v$ is $\ge K$, then no neighbor yielded $K-1$.
                     This implies for all neighbors, $r_v \ge K$.
                     This implies for all neighbors $v$, $N(v)$ is "heavy" enough.
                     But wait. If we sort all $v$ by $r_v$.
                     The ones with smallest $r_v$ are candidates.
                     If $\min_r = K-1$, it is a neighbor.
                     If $\min_r = K$, we have a candidate set (nodes with $K$).
                     Can we distinguish among them?
                     Maybe some are neighbors and some are not.
                     But wait. If all neighbors have $r_v \ge K$, and non-neighbors also have $r_v \ge K$.
                     Then they are indistinguishable by this metric?
                     If we assume the graph is random, unlikely.
                     But worst case?
                     
                     Actually, the question asks to determine if connected.
                     If we find a cut of size 0, we return 0.
                     If we explore everything and reach $N$ nodes, we return 1.
                     If we get stuck (cannot find neighbor but $K>0$), what does it mean?
                     It means our method failed.
                     
                     But maybe we can simply iterate ALL $v$ and check $r_v < K$?
                     Wait, is it possible that $r_v < K$ for a NON-neighbor?
                     If $v \notin N(U)$.
                     $r_v = query(U \cup \{v\})$.
                     $N(U \cup \{v\}) = N(U) \cup N(v)$.
                     $S = U \cup \{v\}$.
                     $r_v = | (N(U) \cup N(v)) \setminus (U \cup \{v\}) |$.
                     $N(U) \setminus (U \cup \{v\})$.
                     Since $v \notin N(U)$, $v \notin N(U)$.
                     So $N(U) \setminus (U \cup \{v\}) = N(U) \setminus U$.
                     So count is $|N(U) \setminus U| + |N(v) \setminus (U \cup \{v\})| - \text{intersection}$.
                     $= K + d_{out}(v) - \text{intersection}$.
                     Can this be $< K$?
                     We need $d_{out}(v) < \text{intersection}$.
                     Intersection consists of neighbors of $v$ that are also neighbors of $U$ (outside $U$).
                     If $v$ connects to many neighbors of $U$, then intersection is large.
                     If $v$ connects ONLY to neighbors of $U$ (and nothing else outside), then intersection = $d_{out}(v)$.
                     Then $r_v = K + d_{out}(v) - d_{out}(v) = K$.
                     So $r_v \ge K$ strictly for non-neighbors?
                     Wait, if intersection = $d_{out}(v)$, then $r_v = K$.
                     So $r_v$ cannot be $< K$.
                     It can be equal to $K$.
                     So YES!
                     If $v \notin N(U)$, then $r_v \ge K$.
                     If $v \in N(U)$, then $r_v \ge K - 1$.
                     
                     Therefore, if $r_v < K$, it implies $v \in N(U)$.
                     If $r_v \ge K$, it implies $v \notin N(U)$ OR $v \in N(U)$ with compensation.
                     Wait. If $v \in N(U)$, $r_v$ starts at $K-1$.
                     If it ends at $K$, it's "compensated".
                     If it ends at $K-1$, it's "not compensated".
                     So finding $v$ with $r_v = K-1$ guarantees neighbor.
                     Finding $v$ with $r_v < K-1$ is IMPOSSIBLE?
                     Let's check.
                     $r_v = K - 1 + \Delta$.
                     $\Delta = |N(v) \setminus (U \cup \{v\})| - |Intersection|$.
                     Intersection is subset of $N(v) \setminus (U \cup \{v\})$.
                     So $|Intersection| \le |N(v) \setminus (U \cup \{v\})|$.
                     So $\Delta \ge 0$.
                     So $r_v \ge K-1$ ALWAYS.
                     
                     So we have strict bounds:
                     If $v \in N(U)$: $r_v \in [K-1, \infty)$.
                     If $v \notin N(U)$: $r_v \in [K, \infty)$.
                     
                     So if $r_v < K$, it MUST be a neighbor.
                     If $r_v \ge K$, it MIGHT be a neighbor.
                     
                     So to find a neighbor, we need to find a $v$ with $r_v = K-1$.
                     If we don't find any such $v$ (i.e. min $r_v \ge K$), does it mean no neighbors?
                     No. It means neighbors have $r_v \ge K$.
                     Is it possible that ALL neighbors have $r_v \ge K$?
                     Yes, if they all compensate.
                     Example: $U=\{1\}$. Neighbor $v$. $v$ connects to $N(1)$.
                     Then $r_v$ could be higher.
                     
                     However, wait. If we have MANY neighbors, it's unlikely all are compensated?
                     But worst case.
                     
                     But here is a trick:
                     If $r_v \ge K$ for all $v$, we can't distinguish.
                     But notice $K = query(U)$.
                     If $r_v = K$, then for all $v$, $r_v = K$.
                     This implies $d_{out}(v)$ is exactly balanced by overlaps for all $v$.
                     Wait, for non-neighbors, $r_v = K$ means intersection = $d_{out}(v)$.
                     This means ALL neighbors of $v$ (outside) are neighbors of $U$.
                     For neighbors, $r_v = K$ means intersection = $d_{out}(v) + 1$? No, equation is $r_v = K - 1 + d_{out} - int$.
                     If $r_v = K$, then $int = d_{out} - 1$.
                     So intersection is $d_{out} - 1$.
                     This means $v$ is connected to almost all neighbors of $U$.
                     
                     So, if $r_v = K$ for all $v$, it means the graph is very structured?
                     But we just need to output correct.
                     Wait!
                     If we encounter the situation where min $r_v \ge K$.
                     This means no "easy" neighbor found.
                     In that case, we can perhaps swap roles?
                     Or just accept that we can't find it?
                     
                     Wait, is there ANY node with $r_v = K-1$?
                     Consider $v$ with $d_{out}(v) = 0$. (Leaf attached to U).
                     Then $int = 0$.
                     Then $r_v = K - 1 + 0 - 0 = K - 1$.
                     If $U$ has a leaf neighbor, we find it.
                     If $U$ has no leaf neighbors, but has neighbors.
                     If all neighbors have $d_{out}(v) \ge 1$.
                     Then we might miss.
                     
                     Wait, what if we query $S=\{u, v\}$ for $u \in U$?
                     Let's refine the algorithm to be robust.
                     We have $U$. $K=query(U)$.
                     If $K=0$, done.
                     If $K > 0$:
                       We know there exists $v \in N(U)$.
                       We want to find it.
                       If we iterate ALL $v \in V \setminus U$ and check if $r_v < K$.
                       If we find any, we pick the first and add to $U$.
                       If not found (all $r_v \ge K$), then it implies for all neighbors $r_v \ge K$.
                       BUT wait. If $K=1$, and neighbor $v$ is leaf, $r_v=0 < 1$. We find it.
                       If $K=1$ and neighbor $v$ is not leaf ($d_{out} \ge 1$).
                       Then $r_v \ge 1$.
                       If we have non-neighbor $z$ ($d_{out}=0$). $r_z = 1$.
                       Then all $v$ give $r_v \ge 1$.
                       We fail.
                       
                       BUT, in this case ($K=1$, neighbor $v$, $d(v) \ge 2$), the graph structure is:
                       $U=\{1\}$ connected to $v$. $v$ connected to something else $w$.
                       So $1-v-w$.
                       $query(\{1\}) = 1$.
                       $query(\{1, v\}) = 1$.
                       $query(\{1, w\}) = 1$ (since $w$ connected to $v$ (not 1)).
                       Wait, $query(\{1, w\}) = |N(\{1, w\}) \setminus \{1, w\}|$.
                       $N(\{1, w\}) = \{v, \dots\}$.
                       If $1 \not\sim w$. $r_w = 1$.
                       If $1 \sim w$, $r_w \ge 0$.
                       
                       Wait, if $1 \sim w$, then $w$ is a neighbor.
                       If $K=1$, then $v=w$.
                       So neighbor is $v$.
                       If we can't identify it, we fail.
                       
                       BUT notice if $K=1$, then $1$ has only 1 neighbor $v$.
                       So if we assume the graph is connected (we want to check), then $v$ connects to rest.
                       If $v$ doesn't connect to $w$, then $w$ is unreachable?
                       Wait, if graph is connected, $v$ must connect to $w$.
                       So $1-v-w$.
                       Then $v$ is neighbor. $w$ is not.
                       $r_v \ge 1$. $r_w = 1$.
                       They are indistinguishable.
                       
                       However, in this case, we can just try to expand using $U=\{1, v\}$?
                       No, we don't know $v$.
                       
                       Is there a query that distinguishes $v$ and $w$?
                       $query(\{v\}) = d(v)$.
                       $query(\{w\}) = d(w)$.
                       $r_v = d(v) - 1$ (since $1 \sim v$).
                       $r_w = d(w)$. (since $1 \not\sim w$, assuming no other neighbors).
                       So $d(v)$ vs $d(w)$.
                       But we don't know $d(v)$ exactly?
                       Wait, $r_v$ is queried value.
                       $r_v = 1$ in example.
                       $r_w = 1$.
                       Wait, if $w$ has degree 2 ($v$ and $z$).
                       $query(\{1, w\})$. $N(\{1, w\}) = \{v, z\}$.
                       $S=\{1, w\}$. Complement $\{v, z\}$.
                       So $r_w = 2$.
                       So $r_v = 1, r_w = 2$.
                       So they ARE distinguishable if we check $S=\{1, w\}$.
                       Why did I think $r_w=1$?
                       $r_w = query(\{1, w\})$.
                       If $1 \not\sim w$.
                       $N(\{1, w\}) = \{v\} \cup N(w)$.
                       $N(w)$ contains $v$? If $w \sim v$.
                       $N(w) \setminus \{1, w\}$ contains $v$ (if $v \ne 1$) and $z$.
                       So count is $|\{v, z\}| = 2$.
                       Wait, intersection calculation.
                       $K' = |(N(1) \cup N(w)) \setminus \{1, w\}|$.
                       $N(1)=\{v\}$. $N(w)=\{v, z\}$.
                       Union $\{v, z\}$.
                       Removed $\{1, w\}$.
                       Result $\{v, z\}$. Size 2.
                       My manual calc was correct ($r_w=2$).
                       My confusion was assuming intersection removes $v$.
                       $v$ is in $N(w)$. $v$ is NOT in $S$ ($v \notin \{1, w\}$).
                       So $v$ is kept.
                       So $r_w$ includes $v$.
                       $r_v$ removed $v$ (since $v \in S$).
                       So $r_v < r_w$ in this case.
                       Generally, if $u$ is connected to $v$, $r_u$ removes $v$. $r_{other}$ keeps $v$.
                       So neighbors tend to have SMALLER $r$.
                       Thus minimizing $r_v$ is a good heuristic.
                       
                       Wait, if $r_v$ is small, it's likely neighbor.
                       Is it possible for non-neighbor to have small $r$?
                       Only if it connects to nothing and $v$ was already in $N(U)$.
                       Wait, $r_v = K$ for non-neighbors?
                       In my example $K=1$.
                       $r_w = 2$.
                       $r_v = 1$.
                       So $r_v = K-1$ (found neighbor).
                       Wait, previously I said $r_v \ge K$.
                       Why the difference?
                       Because $r_v$ calculation depends on overlaps.
                       $r_v = K + d_{out}(v) - int$.
                       If $v \notin N(U)$, then $v$ not in $N(U)$.
                       If $v$ has NO neighbors in $N(U)$? Then int=0.
                       Then $r_v = K + d_{out} \ge K$.
                       If $v$ has neighbors in $N(U)$, int increases.
                       In my example, $w$ has neighbor $v \in N(U)$?
                       $N(U)=\{v\}$. $v \in N(U)$.
                       $w$ connects to $v$.
                       So $int \ge 1$.
                       $d_{out}(w) = 1$ (only $v$).
                       So $r_w = 1 + 1 - 1 = 1$.
                       Wait, $r_w = 1$.
                       $r_v$ (neighbor of 1).
                       $v \in N(U)$.
                       $r_v = 1 - 1 + d_{out}(v) - int$.
                       $d_{out}(v) = 1$ (to $w$).
                       $int = 1$ (to $w$).
                       $r_v = 0 + 1 - 1 = 0$.
                       So $r_v = 0$. $r_w = 1$.
                       So $r_v < r_w$.
                       
                       It seems neighbors have lower score!
                       Why?
                       Because $v \in N(U)$ is removed from count.
                       Non-neighbors keep it.
                       So neighbors start at $K-1$. Non-neighbors at $K$.
                       Then they add external neighbors.
                       But for neighbor $v$, $v$ connects to $U$ (by definition).
                       If $v$ connects to lots of nodes in $N(U)$, it reduces count further?
                       Wait, intersection term removes common neighbors from count.
                       So $r_v$ decreases if $v$ shares neighbors with $U$ (excluding $v$).
                       
                       So $r_v$ tends to be smaller for neighbors.
                       Specifically $r_v \le K$ is likely?
                       In my example $r_w = K$.
                       $r_v = K - 1$.
                       Is it possible $r_v > K$?
                       Suppose $v$ connects to $U$, and $N(v)$ is huge and disjoint from $N(U) \setminus \{v\}$.
                       Then $r_v = K - 1 + large$.
                       So $r_v$ can be larger.
                       But $r_w$ (non-neighbor) also adds $large$.
                       So comparing absolute values is risky.
                       
                       BUT comparing relative to $K$ is safer?
                       We know $r_v \ge K - 1$.
                       If we find $r_v = K - 1$, it is a neighbor.
                       If we find $r_v < K$, it is a neighbor.
                       If all $r_v \ge K$, then no neighbor yielded $K-1$.
                       This means ALL neighbors yielded $\ge K$.
                       This implies neighbors are "busy".
                       
                       Let's just use the rule:
                       Pick $v$ minimizing $r_v$.
                       If $\min < K$, we assume it's a neighbor.
                       If $\min \ge K$, we assume we missed it?
                       Wait, if $\min \ge K$, we might have missed it.
                       But actually, we proved earlier that $r_v$ for non-neighbors is $\ge K$.
                       So non-neighbors are $\ge K$.
                       Neighbors are $\ge K-1$.
                       So if $\min \ge K$, it's possible all neighbors are $\ge K$.
                       But if we pick ANY $v$ with $r_v < K$ (i.e. $\le K-1$), it's DEFINITELY a neighbor.
                       Since $r_v \ge K-1$ always, $r_v < K \implies r_v = K-1$.
                       So if $\min = K-1$, we found a neighbor.
                       What if $\min > K-1$?
                       Then all candidates have score $\ge K$.
                       Since non-neighbors are $\ge K$, and some neighbors might be $\ge K$, we can't distinguish.
                       BUT!
                       Notice that $K = query(U)$.
                       $K$ is the number of neighbors.
                       If $K$ is large, say $N$. Then many neighbors.
                       Probability that a random $v$ is neighbor is high.
                       If we just pick a random $v$ from $V \setminus U$ and check $r_v < K$.
                       If it passes, add it.
                       If not, try another?
                       If we try all, we might exhaust queries.
                       
                       However, with $N \le 200$, we can run BFS.
                       If we can't find a neighbor, we fail.
                       BUT, notice the problem asks to output 0 if disconnected.
                       If we assume connected and fail, we output 0? No, that's wrong.
                       
                       Is there a simpler check?
                       $query(V \setminus \{v\})$.
                       This is 1 if $deg(v)>0$, 0 else.
                       Wait, if $v$ is isolated, it is a separate component.
                       If graph is connected, no node (except trivial $N=1$) is isolated.
                       If we find isolated node $\ne 1$, disconnected.
                       What if component of 1 is not full?
                       Then there are other components.
                       Any node in other component is "far" from 1.
                       Can we check distance?
                       
                       Wait! The solution might be:
                       For each $v \in V \setminus \{1\}$:
                         Check if $v$ is connected to 1.
                         If not, return 0?
                         But we don't know.
                         
                       Actually, if we just query all pairs $(u, v)$ with $query(\{u, v\})$, it's $N^2$ queries.
                       $200^2 = 40000$.
                       We can only do 3500.
                       So we cannot check all pairs.
                       
                       But wait! $K \le 3500$ is small?
                       No.
                       
                       Let's check the Note again.
                       Interaction example uses 3 queries for N=4.
                       If we implement the BFS expansion using "Find neighbor by minimizing $r_v$".
                       We will likely succeed because neighbors reduce the cut size.
                       In competitive programming, heuristics often pass if data isn't maliciously constructed against them.
                       But for this problem, it's likely deterministic.
                       
                       Let's refine the "Minimizing $r_v$" logic.
                       Since neighbors start at $K-1$ and non-neighbors at $K$.
                       The set of values for neighbors is $[K-1, \dots)$.
                       The set of values for non-neighbors is $[K, \dots)$.
                       So the GLOBAL MINIMUM of $r_v$ over all $v$ MUST be $\ge K-1$.
                       If the minimum is $K-1$, then there exists a neighbor achieving this minimum.
                       If the minimum is $\ge K$, then ALL neighbors achieve $\ge K$.
                       Is it possible to distinguish?
                       Wait. If all neighbors achieve $\ge K$, it means $K' \ge K$.
                       For a neighbor $v$, $K' \ge K \iff \text{overlaps} < d_{out}(v)$.
                       Wait, $K' = K - 1 + d_{out} - int$.
                       $K' \ge K \iff d_{out} - int \ge 1 \iff int \le d_{out} - 1$.
                       This means $v$ has at least one neighbor outside $N(U)$.
                       This is always true if $d_{out} \ge 1$.
                       So if $d_{out}(v) \ge 1$ (which is implied if $K>0$? No, neighbor might be leaf with respect to $U$? No, $d_{out}$ refers to outside $U$).
                       Wait, $v \in N(U) \implies v \in V \setminus U$.
                       $N(v)$ may have nodes inside $U$ and nodes outside $U$.
                       $d_{out}$ is neighbors outside $U$.
                       If $v$ is leaf in the graph, $d_{out} = 0$.
                       Then $int = 0$. $K' = K-1$.
                       So leaves minimize to $K-1$.
                       If $v$ has neighbors outside $U$.
                       Does it ensure $int$ is high enough?
                       
                       Actually, here is a robust logic:
                       If $K=0$, Disconnected.
                       If $K > 0$:
                         Check all $v \in V \setminus U$.
                         Pick $v$ that minimizes $r_v$.
                         If min $r_v == K-1$, then $v$ is a neighbor.
                         If min $r_v > K-1$, then we might have issues.
                         BUT wait.
                         Is it possible that for ALL neighbors $v$, $r_v > K-1$?
                         This requires $d_{out}(v) - int(v) \ge 1$ for all neighbors.
                         i.e., every neighbor has at least one neighbor outside $N(U)$ (counted in $d_{out}$) that is NOT connected to $U$?
                         Or simply $d_{out}(v)$ is large.
                         If we find a neighbor with $r_v = K-1$, we are safe.
                         If we don't find any $v$ with $r_v < K$, we are stuck.
                         However, note that if the graph is connected, there MUST be a path.
                         Eventually we reach leaves?
                         
                         What if we prioritize nodes with $r_v < K$?
                         If we find ANY $v$ with $r_v < K$, we pick it.
                         If we don't find ANY $v$ with $r_v < K$ after scanning all, then we can't expand?
                         But wait. If $K > 0$, there IS a neighbor.
                         Why can't we find it?
                         Because we can't distinguish.
                         BUT, what if we just assume the one with minimal $r_v$ is the neighbor?
                         If $r_v \ge K$ for all $v$, then the minimal value is $\ge K$.
                         Is it possible that the true neighbor has value $> K$ while some non-neighbor has value $K$?
                         Yes.
                         But if we pick a non-neighbor, we make a mistake.
                         However, if we pick a node $v$ with minimal $r_v$, and if the graph is "nice", it's likely a neighbor.
                         
                         Wait, there is one more thing.
                         We can check if $query(S) == 0$ for proper subset $S$.
                         If we find such $S$, return 0.
                         The algorithm is:
                         Maintain connected component $S$ of vertex 1.
                         Initially $S=\{1\}$.
                         Loop:
                           $k = query(S)$.
                           If $k == 0$ and $|S| < N$: return 0.
                           If $k > 0$:
                             Iterate $v \in V \setminus S$.
                             Calculate $val(v) = query(S \cup \{v\})$.
                             Keep track of $best\_v$ which minimizes $val(v)$.
                             Wait, if we minimize, we prefer neighbors.
                             If we find $val(best\_v) < k$, we are certain $best\_v$ is a neighbor.
                             If $val(best\_v) == k$, it might be non-neighbor or neighbor.
                             If $val(best\_v) > k$, then $best\_v$ is definitely non-neighbor?
                             Wait.
                             Non-neighbor: $r_v \ge k$.
                             Neighbor: $r_v \ge k-1$.
                             So if $r_v < k$, neighbor.
                             If $r_v \ge k$, could be either.
                             
                             So if we pick $v$ with $r_v < k$, we are safe.
                             If all $v$ have $r_v \ge k$.
                             Then we have no certainty.
                             However, if the graph is connected, can it happen that for ALL neighbors $r_v \ge k$?
                             This requires $d_{out}(v) \ge 1 + int(v)$.
                             This means every neighbor has many external connections.
                             But non-neighbors have $r_v = k + \dots$?
                             Wait, if $r_v = k$ for all $v$, we have a tie.
                             
                             Actually, let's use the constraint that $r_v \ge k-1$.
                             If $r_v = k-1$, it is a neighbor.
                             If
````

---

## 直接SFT 样本(sample_idx=1,得分 0.0,completion_tokens=1527)

````
This is an interactive problem about determining if a graph is connected using subset queries. A query "?s" returns |N(S) \ S|, the number of vertices outside S that are connected to at least one vertex in S (i.e., edges going out from S).

Key insight: The sum of all query results when querying each single vertex gives us the total number of edges in the graph (summing over all v, |N({v})\{v}| counts each edge twice, once from each endpoint). But more importantly for connectivity, I need to find whether the graph has exactly n-1 edges AND be acyclic (a tree), or use another approach.

Better approach - use the query strategically:
- Query subsets {i} and {j}: Let q(i) = |N({i})|, q(j) = |N({j})|.
- Query subset {i,j}: q({i,j}) = |N({i,j})|.
- Note that N({i,j}) ⊆ N({i}) ∪ N({j}), and |N({i,j})| = |N({i}) ∪ N({j})| = |N({i})| + |N({j})| - |N({i}) ∩ N({j})|.

So |N({i}) ∩ N({j})| = q({i}) + q({j}) - q({i,j}). This tells me how many common neighbors i and j share. If this is 0, then i and j have no common neighbor.

Strategy: For each pair of vertices (i,j), check if they share a common neighbor using these 3 queries. If every pair shares a common neighbor, does that mean connected? No... but it helps.

Even better approach - build the adjacency matrix:
For every ordered pair (i,j):


- Query {i}: get q_i = |N({i})|
- Query {j}: get q_j = |N({j})| 
- Query {i,j}: get q_{ij} = |N({i,j})|
- Then |N({i}) ∩ N({j})| = q_i + q_j - q_{ij}

If i and j are NOT directly connected, then their direct neighbors must all overlap through common paths for connectivity to hold. But checking all pairs individually is too slow with only ~3500 queries available for n up to 200.

The key insight is recognizing that |N({i}) \ N({j})| = q_i - (q_i + q_j - q_{ij}) = q_{ij} - q_j, which directly counts neighbors of i that aren't neighbors of j. Similarly, neighbors of j not neighbors of i equal q_{ji} - q_i. So I can compute these sets by subtracting common neighbor counts from individual degree queries—giving me full control over which neighbors belong to which vertex without needing separate membership queries.

Now I should verify whether the graph is actually connected rather than just computing these values. For any cut partitioning vertices into two non-empty sets U and V, the number of crossing edges equals the sum of degrees into the other set minus twice the internal edges. So crossing_edges(U,V) = deg_internal(V,U) - 2*deg_internal(V) where deg_internal(V,U) counts all edges from V to U.

Since a disconnected graph must have at least one separating cut, I should focus on finding a minimal separator instead of enumerating all cuts. Using BFS, I can maintain the minimum degree of each visited vertex relative to its parent—specifically min_degree[u] = min(min_degree[parent], |{w : w is a child of u}|). This tracks the fewest edges needed to isolate a subtree rooted at u.

For a graph to stay connected after removing vertex x, every other node needs either a direct edge to x or a path to a different neighbor of x. So if x has fewer than 2 neighbors, it's a critical vertex unless isolated. If some nodes never reach a second distinct neighbor of x, x is also essential for connectivity.

I'll track each vertex's "second-neighbor count"—how many other neighbors exist beyond its primary connections—and verify this remains consistent throughout. For isolated vertices, this value stays 0, while truly critical vertices have count less than 2 since they connect only through themselves or lack alternative paths.
</think>

```cpp
#include <bits/stdc++.h>
using namespace std;

void solve() {
    string line;
    if (!getline(cin, line)) return;
    getline(line, line); // consume newline
    
    int n = stoi(line);
    
    if (n == 0 || n == 1) {
        cout << "! 1" << endl;
        fflush(stdout);
        return;
    }
    
    vector<vector<int>> adj(n, vector<int>(n, 0));
    for (int i = 0; i < n; i++) adj[i][i] = 1;
    
    auto ask = [&](string s) -> int {
        cout << "? " << s << endl;
        fflush(stdout);
        string l; getline(cin, l);
        return stoi(l);
    };
    
    int d[205];
    for (int i = 0; i < n; i++) {
        string ss = "";
        for (int j = 0; j < n; j++) ss += '0';
        ss[i] = '1';
        d[i] = ask(ss);
    }
    
    int ans = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i != j && d[j] > 0) {
                int qq = d[j] - d[i];
                if (qq > 1) {
                    for (int k = 0; k < n; k++) {
                        if (k != i && k != j && adj[i][k] == 0 && adj[j][k] == 0) {
                            ans = 1; break;
                        }
                    }
                } else if (ans == 0) {
                    for (int k = 0; k < n; k++) {
                        if (k != i && k != j && adj[i][k] == 0 && adj[j][k] == 0) {
                            ans = 1; break;
                        }
                    }
                }
                if (ans) break;
            }
        }
        if (ans) break;
    }
    
    cout << "! " << ans << endl;
    fflush(stdout);
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int t = 0;
    if (cin >> t) {
        while (t--) {
            cin.ignore();
            solve();
        }
    }
    return 0;
}
```
````
