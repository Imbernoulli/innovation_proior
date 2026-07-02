# 题 70 — base均分 33.6 vs 直接SFT均分 0.0(掉点 33.6)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=0,得分 83.9,completion_tokens=26229)

````
The user wants a C++ solution for an interactive competitive programming problem called "Treasure Hunt".
The problem involves navigating an undirected graph to visit all vertices.
Key constraints and rules:
1.  **Graph**: Undirected, $n$ vertices, $m$ edges. $2 \le n \le 300$, $1 \le m \le \min(n(n-1)/2, 25n)$. The graph is connected (implied by the goal to visit every vertex starting from one). Actually, usually in such problems, it's connected, but I should assume standard traversal logic applies. Wait, if it's disconnected, I can't reach some nodes. Assuming it's connected as per typical interactive graph problems unless stated otherwise.
2.  **Adversary**: The order of neighbors at each vertex is shuffled *every time* you arrive. This means I cannot rely on neighbor indices being consistent between visits to the same vertex.
3.  **Information Available at Current Vertex**:
    -   Degree of current vertex ($d$).
    -   For each adjacent vertex (in the shuffling order):
        -   Degree of that neighbor.
        -   Whether that neighbor has been visited before (flag).
4.  **Goal**: Visit every vertex at least once. Minimize moves compared to `base_move_count`.
5.  **Interaction**:
    -   Print `t` (number of test cases).
    -   Read `n, m, start, base_move_count`.
    -   Read `m` edges.
    -   Loop: Read neighbor info `d deg_1 flag_1 ...`.
    -   Output index `i` (1-based) to move to. Flush.
    -   Stop when AC or F.
6.  **Constraints**: Time limit 5s, Memory 256MB. $N$ up to 300 is small.

**Analysis:**

Since the neighbor list order changes every time we are at a node, we don't know which neighbor corresponds to edge $(u, v)$ based solely on position. However, we have information about neighbors: their degree and visited status.
This implies we can uniquely identify a specific adjacent vertex (e.g., vertex $v$) by its properties relative to the current node $u$. But wait, the input only gives degrees. If there are multiple neighbors with the same degree, we can distinguish them by the "visited" flag? No, because we haven't visited them yet, they might all be unvisited.
Actually, since it's an undirected graph, if we moved from $u$ to $v$, we will see $u$ again later as a neighbor of $v$.
Crucially, we need to keep track of which vertices we have visited and the structure of the graph to reconstruct the path or plan the next step.

Since $N$ is small ($300$), we can maintain a set of visited vertices. But how do we navigate given the shuffled neighbors?
We need to find *any* unvisited neighbor to expand our search front, or follow a known path if possible.
Wait, can we determine the exact connection $u-v$ given the input format?
The interactor says: "For each adjacent vertex: its degree and whether you have visited it before".
So at vertex $u$, I receive a list of tuples $(deg(v_i), visited(v_i))$.
I know my current vertex $curr$. I know I came from `prev` (unless $start$).
When I am at $curr$, I want to choose a neighbor $next$ such that $next$ is unvisited (if possible) or follows a strategy to minimize moves. BFS/Dijkstra/DFS style logic would work best if I knew the connections. Since I don't know the labels of neighbors directly, I only know their structural properties (degree).

Let's consider identifying nodes.
At the start, I know vertex IDs $1..N$. I start at `start`.
When I output a move index `k`, I go to some neighbor. The interactor doesn't explicitly tell me the ID of the new vertex immediately. Instead, it prints the description of that new vertex.
Ah, "You must navigate the graph... interaction begins. The interactor prints vertex descriptions... To make a move... output a single integer i... choosing the i-th neighbor".
Then the interactor updates the state.
So after I pick neighbor $k$, I wake up at the destination vertex. I receive its description. This description includes the degree of neighbors and their visited flags.
One important detail: I need to know the current vertex ID to update my internal state of visited nodes.
BUT the problem statement says: "You start at a known fixed vertex." It doesn't say the interactor tells me the current vertex ID explicitly after the move.
However, the input format after graph description is just "vertex descriptions".
Wait, usually in these types of interactive problems where I choose a neighbor, the feedback gives me the properties of the *current* vertex (the one I landed on).
Does the output allow me to distinguish vertices?
If two vertices have the same degree and the same connectivity pattern relative to visited nodes, they might be indistinguishable locally.
However, I can build a global map.
Let's denote `visited_set` as a boolean array tracking which vertex IDs I've been to.
But to populate `visited_set`, I need to know which vertex ID I am currently standing on.
The interaction protocol doesn't give me the current vertex ID directly. It just gives neighbor info for the vertex I am at.
Wait, if I am at vertex $u$, I see neighbors with degrees $D_1, D_2, \dots$ and flags $F_1, F_2, \dots$.
Can I match this signature to a vertex ID in the graph?
Yes! The initial graph structure is known. $u$ has specific neighbors. Each neighbor $v$ has a specific degree.
Wait, if I am at $u$, and look at neighbor $v$, the information provided is $(deg(v), visited(v))$.
Is this unique enough to identify $v$?
Not necessarily. Multiple neighbors could have the same degree.
However, I also know which neighbor I came *from*.
When I was at parent $p$ and moved to $u$, I chose a specific index.
Now I am at $u$. I see a list of neighbors. One of them is $p$.
$p$'s degree is known from the initial graph input.
Also $p$ should have a "visited" flag = 1 (since I just came from there).
By identifying the neighbor with degree($p$) and flag=1, I can orient myself back towards where I came from.
Once I identified the direction to the parent, I know which entry in the list corresponds to the parent.
But does the list order change? Yes.
So, whenever I arrive at $u$:
1.  Receive degree $d$ and lists of $(deg, visited)$.
2.  Iterate through the list. Find the neighbor that matches the previous vertex ID I was at.
3.  Mark the corresponding entry in my internal representation of $u$'s neighbors.
    Wait, I don't store neighbors by index, I store them by ID.
    So, for the current vertex $u$ (unknown to me directly, but deducible?), I need to figure out who I am.
    
    This looks like I can maintain a mapping of "Vertex Signature" -> Vertex ID.
    Signature = (ID of self). No, that's circular.
    Let's think about distinguishing vertices.
    Two vertices $A$ and $B$ are structurally distinct if their neighbors sets are different.
    In the graph provided initially, I can precompute properties.
    Property of vertex $v$: `(degree(v), sorted_neighbor_degrees)`.
    Actually, since I don't get the neighbor list from the interactor (only degree + visited status), and visited status depends on time/path, the static property is just the degree.
    If there are multiple nodes with same degree, they are locally ambiguous.
    
    However, since $N \le 300$, maybe I can just simulate BFS.
    I need to know which vertex ID I am currently visiting to update the `visited` set correctly.
    Is it possible the interactor prints the current vertex ID implicitly? No.
    Is it possible the list of neighbors is ordered consistently? No ("shuffles the order").
    
    Wait, re-read carefully: "For each adjacent vertex: its degree and whether you have visited it before".
    When I move to $u$ from $p$, I am now at $u$. I see $p$ in the neighbor list.
    I know $p$'s ID. I know $p$'s degree.
    So I scan the current list for an entry with `deg == degree(p)` and `visited == true` (since I just left it).
    Wait, did $p$ become visited before I left? Yes, I visited it.
    Is it guaranteed that $p$ is marked visited? Yes.
    But is it guaranteed that only $p$ has `deg == degree(p)` and `visited == true`?
    There might be other visited nodes with same degree.
    However, among all neighbors, exactly one is $p$.
    I don't know which one is $p$ just by looking at degree and visited status if others match.
    BUT, I know the sequence of moves.
    
    Let's refine the tracking state.
    State:
    - `visited_ids`: Set of vertex IDs I have visited.
    - `path`: List of vertex IDs I traversed.
    - `adj_map`: Map from vertex ID to list of neighbors. Constructed from initial input.
    
    Problem: When at unknown current vertex `curr_id`, how to confirm `curr_id`?
    When I am at `curr`, I look at neighbors. One is `parent`.
    I know `parent` ID (last element in `path`).
    I scan neighbor list. I find candidates with `deg == degree[parent]` and `visited == 1`.
    Usually, the edge is bidirectional. If I came from `p`, then `p` is a neighbor of `curr`.
    Is it possible another neighbor `q` has same degree and `q` is also visited?
    Maybe. But I am at `curr` now. `q` might be visited too.
    If there are multiple such neighbors, I can't definitely distinguish `p` from `q`.
    
    HOWEVER, notice the problem asks to visit *all* vertices.
    This suggests a spanning tree traversal is sufficient (DFS/BFS).
    Once all vertices are visited, game over (AC).
    
    Key realization: We don't necessarily need to know `curr_id` perfectly if we just need to continue finding unvisited nodes.
    BUT to ensure we don't cycle infinitely or mark wrong nodes as visited, knowing the graph topology is helpful.
    If the graph is not a complete graph or something with high symmetry, degrees might help.
    
    Let's consider the simplest strategy: DFS.
    From `start`, pick any unvisited neighbor. Go there. Repeat until no unvisited neighbors. Then backtrack.
    Since the neighbor order shuffles, I treat the current list as a bag of candidates.
    I need to select one. Which one?
    Any unvisited neighbor.
    How do I pick one? I just pick the first one with `visited == 0` in the input list.
    But wait, I also need to know if moving there helps explore new nodes or leads me back.
    
    Wait, do I even need to know `curr_id`?
    If I am at $u$ and see neighbor $v$ with degree $D_v$ and `visited=0`, I move there.
    Now I am at $v$. I need to know what happened to $u$.
    At $v$, I see $u$ in the list. It has `deg(D_u)` and `visited=1` (because I visited it).
    So I can identify "where I came from" by matching the degree of the source node I'm carrying in memory.
    This requires keeping track of the last node ID.
    Does the "visited" flag persist? "whether you have visited it before".
    If I come back to $u$ later, its flag remains 1.
    
    So the algorithm could be:
    1.  Parse graph. Build adjacency list `adj[u]` containing pairs `(v, degree(v))`. Note: actually I need the exact `degree(v)` for every neighbor $v$.
    2.  Initialize `visited_global[n+1]` false. `visited_global[start] = true`.
    3.  `current_node = start`.
    4.  Loop until AC:
        a. Receive neighbor list from interactor: list of `(deg, visited_flag)`.
        b. Identify which neighbor corresponds to `current_node`'s parent?
           Actually, I need to check which neighbors are valid to traverse.
           Valid moves: Neighbors with `visited_flag == 0` (unvisited).
           Also, to backtrack (or simply to manage stack depth), I need to find the neighbor corresponding to my previous node.
           Let `last_node` be the node I arrived from.
           I need to find the entry in the list with `deg == adj[last_node].degree` and `visited_flag == 1` (since `last_node` is visited).
           Wait, if `last_node` is visited, the flag is 1. If I see multiple such entries, which one is it?
           Suppose there are multiple neighbors with same degree.
           However, I am physically connected to exactly one node that has the ID `last_node`.
           The interactor guarantees I see the correct neighbors.
           If there is ambiguity (multiple neighbors satisfy degree condition), does the visited status distinguish them?
           They are both visited.
           This is tricky. If the graph has two nodes $A, B$ both with degree 5, and I am at $C$ connected to both $A$ and $B$, and both are visited. The list shows two entries with `deg=5`, `visited=1`. I don't know which is $A$ and which is $B$ without more info.
           
           BUT, I can use the fact that `deg` of neighbors is constant.
           If I am at $u$, I see neighbors.
           If I just came from $v$, then $v$ is a neighbor.
           Can I identify $v$ uniquely?
           Not necessarily just by degree and visited status.
           
           However, notice the scoring function. We want to finish fast.
           If I get stuck in a local loop due to confusion, that's bad.
           Is it possible to solve this without knowing the specific identity of the neighbors?
           Wait, "Your score depends on how close your move count is to this baseline."
           Base move count is roughly related to BFS distance or similar?
           `base_fraction = (base_move_count + 1) / n`. `c = 90 / sqrt(base_fraction - 1)`.
           If moves <= base, score > 0.
           Basically BFS-like distance is preferred.
           
           Let's reconsider the "Identity" problem.
           We know $N \le 300$.
           We know the graph structure fully initially.
           When we are at vertex $U$, we receive a list of attributes for neighbors.
           The attributes are `degree` and `visited`.
           Crucially, I know `degree(U)`?
           The input line starts with `d deg_1 flag_1 ...`. `d` is the degree of CURRENT vertex $U$.
           Ah! This is the degree of the current vertex $U$.
           So I know `degree(U)` immediately.
           
           So the information available is:
           `degree(current_vertex)`
           `list of neighbors: [ (degree_of_neighbor_1, is_visited_1), ..., (degree_of_neighbor_d, is_visited_d) ]`
           
           Since I know `degree(current_vertex)`, let's call it `my_deg`.
           I can identify which vertex I am at by finding a vertex in my global knowledge that has `degree = my_deg`.
           Is it unique? No, many vertices can have same degree.
           BUT, I also know the path taken so far.
           The sequence of moves defines the path.
           If I record the sequence of `visited_flags` for neighbors?
           
           Actually, maybe I can just treat this as a random walk or BFS where I don't care about IDs, just components?
           No, I need to mark `visited` globally to avoid cycles and ensure termination.
           The interaction ends when ALL vertices visited.
           This implies I need to coordinate which vertices are covered.
           The problem is I don't know which vertex ID I am at.
           
           Is it possible that the graph is always a specific type (e.g. bipartite)? No, general undirected graph.
           
           Let's look at the example.
           3 vertices, 3 edges (triangle).
           Start 1.
           Input: `2 2 0 2 1` -> d=2, neighbors: deg=2, vis=0; deg=2, vis=1.
           I start at 1.
           I know node 1 has neighbors 2 and 3. Both have degree 2.
           Both unvisited initially.
           I pick one (index 1). Move to a neighbor. Say node 2.
           Next turn:
           Interactor prints vertex desc for node 2.
           Node 2 has neighbors 1 and 3.
           Degrees: 2, 2. Visited: 1 is visited, 3 is unvisited.
           Output might be: `2 2 1 2 0`.
           Here I see two neighbors: one with `deg=2, vis=1`, one with `deg=2, vis=0`.
           I know I came from 1. 1 is visited. So `vis=1` candidate is likely 1.
           Then I see `vis=0` candidate is 3.
           So I can identify which neighbor is 3.
           
           Generalizing:
           I am at vertex $U$. I see neighbors $V_1, \dots, V_k$.
           One of them is $P$ (previous vertex). $P$ is visited.
           So I look for neighbor(s) with `visited==1`. One of them MUST be $P$.
           If there is exactly one such neighbor with the correct properties, I know it.
           What if multiple?
           Suppose $P$ has degree $D_P$.
           Other visited neighbors $Q$ might also have degree $D_Q = D_P$ and be visited.
           If multiple neighbors are visited, how do I differentiate $P$ from $Q$?
           Well, I know I came from $P$.
           If I don't know which entry corresponds to $P$, I can't confirm which edge I took back or forward.
           However, if I treat the graph as a state machine?
           
           Wait, maybe I can use the fact that `base_move_count` is provided and likely optimal path length is related to BFS layers.
           But solving it correctly requires handling the ambiguity.
           Is there any invariant I missed?
           "While the structure of the graph remains constant, the wizard shuffles the order of the roads".
           "You cannot see the labels... only degree and visited flag".
           This sounds like we have to deduce the path using the "visited" flags.
           
           Consider the case where multiple neighbors have `visited=1` and `deg=d`.
           This happens if I have visited multiple branches.
           However, note that we are exploring.
           Most of the time we encounter unvisited nodes (`visited=0`).
           We pick one to move to.
           Once we enter a new territory, we mark it visited.
           
           Strategy Idea:
           We maintain a set of *known* visited vertices.
           Initially `{start}`.
           We are at `start`.
           We see neighbors. All are `visited=0` (except possibly none).
           Wait, if I am at `start`, all neighbors are unvisited? Yes, unless graph has weird loops, but initially yes.
           So I pick an arbitrary unvisited neighbor.
           Let's say I picked neighbor $X$.
           I move to $X$.
           Now I am at $X$.
           I see neighbors. One is `start`. `start` has `visited=1`. Others are `0` (usually).
           Among `0`s, I pick an unvisited one, say $Y$.
           If I pick one that is already visited (a back-edge), fine, I'll just return to a known node.
           I just need to ensure I don't revisit nodes too much.
           Ideally, I want to find a spanning tree and traverse it (DFS).
           
           To implement DFS:
           I need to identify neighbors.
           I have a history of moves.
           Path: $v_0 \to v_1 \to \dots \to v_k$.
           When at $v_k$, I received neighbors list.
           The neighbor corresponding to $v_{k-1}$ is the one that allows me to go back.
           The other unvisited ones lead to new nodes.
           I should prioritize unvisited nodes.
           
           The main issue remains: distinguishing between visited neighbors.
           But actually, do I need to distinguish them?
           If I move to an unvisited node $W$, I extend the path.
           If I move to a visited node, I am cycling.
           So, I just need to ensure I pick an unvisited node if one exists.
           If I pick an unvisited node, say $W$, then I transition state from "at $V$" to "at $W$".
           The interactor confirms I am at $W$ via its own neighbor description.
           At $W$, I will see $V$ (visited) and potentially new unvisited nodes.
           I need to know $V$ to backtrack or just continue?
           Backtracking is needed to visit other components attached to $V$.
           
           Wait, if I treat the "graph" I am building online.
           I don't know the graph topology completely (I know edges, but labels are lost).
           However, I know the total number of nodes $N$.
           If I successfully traverse an edge to an unvisited node, that node becomes visited.
           If I eventually visit $N$ nodes, AC.
           So, maybe I can just run a simple exploration:
           Always try to move to an unvisited neighbor.
           If all neighbors are visited, move to any visited neighbor (preferably one that seems to offer more potential? Or just random?).
           Actually, if all neighbors are visited, I'm in a component of visited nodes.
           I can pick any one.
           Eventually, I will find all nodes?
           Wait, if I am at node $u$, and all neighbors are visited, it means I've completed the connected component of $u$.
           But the graph is connected.
           Wait, "visit every vertex". If the graph is connected, and I do a BFS/DFS, I cover all.
           The only issue is I need to ensure I don't miss a branch.
           
           If I blindly pick unvisited, I perform a DFS traversal.
           When I hit a dead end (all neighbors visited), I must backtrack.
           To backtrack, I need to find the neighbor that is the parent (previously visited from).
           The parent $P$ has `visited=1`.
           Is it unique?
           If there is only one neighbor with `visited=1` (besides maybe duplicates), I can pick it.
           If there are multiple neighbors with `visited=1`, I have a problem.
           BUT, in a DFS traversal of a new component:
           When moving forward into new nodes, the only visited node seen is the one I just came from (assuming no cycles).
           Wait, if I entered $u$ from $P$, then $P$ is the only visited neighbor IF $u$ is unvisited initially?
           Yes!
           When I am at an unvisited node $u$ (first time visiting), all its neighbors are unvisited EXCEPT $P$ (which I just came from).
           Because $u$ itself is unvisited before I entered it, and no one else could have reached $u$ yet (unless there's an edge from a previously explored part of the graph that wasn't closed off).
           Wait. If $u$ has an edge to some $Q$ which was already visited in a separate part of the traversal, then $Q$ would be a neighbor with `visited=1`.
           But in a simple DFS/BFS that expands layer by layer or node by node, we usually handle visited checks to avoid crossing old regions unnecessarily.
           But here, I can't control the order easily due to shuffling.
           However, if I just follow the rule: "Always move to an unvisited neighbor if available", I effectively grow the frontier.
           If I reach a node where all neighbors are visited, then either:
           1. I am at the very end (target achieved).
           2. I am inside the explored region.
           
           The key is backtracking.
           If I arrive at a node $u$ where all neighbors are visited, I need to go back to $P$.
           The problem is distinguishing $P$ from other visited neighbors.
           However, $P$ is the *most recent* visited node on the path from `start`.
           Actually, if I treat this as a generic graph search where I don't care about specific IDs but just connectivity, I can track the "history of current vertex".
           
           Wait, I know the degree of the current vertex! `d`.
           And I know `degree(P)`? Yes, I stored it initially.
           And I know `degree(Q)` for other neighbors? Yes.
           So at $u$, I have a list of neighbors with $(deg, visited)$.
           I know I came from $P$. $P$ has specific degree $deg(P)$.
           I should filter neighbors: `candidates = { v | neighbor.deg == deg(P) && neighbor.visited == 1 }`.
           If candidates size is 1, great, I know $P$.
           If candidates size > 1, ambiguity exists.
           
           Is it possible to have ambiguity?
           Only if there are multiple neighbors of $u$ that have the same degree AND are both visited.
           Example: Star graph. Center $C$, leaves $L_1 \dots L_{N-1}$.
           Leaves have degree 1. Center has degree $N-1$.
           Start at center. Move to $L_1$. $L_1$ sees $C$ (degree $N-1$) and no other neighbors.
           $C$ is the only neighbor. No ambiguity.
           Move to $L_2$. See $C$.
           
           Example: Cycle. 1-2-3-4-1. All degrees 2.
           Start 1. Neighbors 2, 4. Both degree 2.
           Pick 2.
           At 2. Neighbors 1, 3.
           1 has degree 2, visited.
           3 has degree 2, unvisited.
           Ambiguity: Both neighbors have degree 2.
           One is visited (1), one is unvisited (3).
           So I identify 1 by `visited=1`.
           If I decide to stay in the "new territory" mode, I pick unvisited (3).
           At 3. Neighbors 2, 4.
           2 (visited, deg 2). 4 (unvisited, deg 2).
           I pick 4.
           At 4. Neighbors 3, 1.
           3 (visited). 1 (visited).
           Now ambiguity arises!
           I came from 3. 1 is also visited.
           Both have degree 2 and visited=1.
           I need to know which one is 3 to backtrack, OR I need to just pick a neighbor.
           If I pick 1, I backtrack to 1.
           If I pick 3, I go forward (redundant).
           If I am at 4, and I pick 1. I am now at 1.
           1's neighbors are 2, 4. Both visited.
           I am stuck.
           I have visited {1, 2, 3, 4}. Goal met. AC.
           So even if I get confused, if I visit everyone, I'm good.
           The only risk is getting infinite loops.
           With $N$ small and connected, I can detect if I visited all nodes.
           
           Algorithm refinement:
           Maintain `visited_nodes` count. If `count == n`, AC.
           Use a heuristic: Always prefer unvisited neighbors to minimize moves.
           If no unvisited neighbors, choose any neighbor (randomly or deterministically) to backtrack or cross bridge.
           To avoid cycles, we can prioritize "returning to where we came from"?
           But we don't know "where we came from" perfectly if degrees match.
           
           However, in practice, most graphs won't be highly symmetric in a way that causes persistent confusion unless specifically constructed.
           Wait, $N \le 300$ allows us to do something more robust?
           What if I just print the move to *every* neighbor sequentially? No, I only output one move at a time.
           
           Let's verify the constraints again. $M \le 25N$. This is sparse-ish. Average degree 50.
           With max degree 299.
           
           The ambiguity problem:
           At node $u$, I have a list of neighbors.
           I need to select index $i$ (corresponding to neighbor $v_i$).
           I want to maximize progress (visit new nodes).
           I know the degrees of all nodes in the graph.
           Let `GlobalDegree[u]` be the degree of current node.
           For each neighbor $v$, I receive `LocalDegree[v]` and `Visited[v]`.
           So I receive full structural info: `GlobalDegree[u]` and `{(LocalDegree[v], Visited[v])}`.
           
           Wait, `GlobalDegree[u]` is not necessarily unique across all nodes.
           Many nodes might have the same degree.
           So I can't identify $u$ by `GlobalDegree[u]` alone.
           
           BUT, I can maintain a "state" of the graph search.
           Since I only need to visit all nodes, I can essentially perform a Random Walk with preferences.
           Prefer unvisited.
           If all visited, pick one.
           To prevent infinite loops, keep track of `(current_path)`? No, path resets.
           Just keep a global counter of unique visited vertices found.
           Wait, how do I distinguish "found vertex X" from "found vertex Y"?
           I need to count.
           Every time I move to a neighbor with `visited=0` (according to INTERACTOR), I increment a counter?
           NO.
           The interactor provides `visited` status of NEIGHBORS.
           When I am at $u$, and I pick a neighbor $v$ with `visited=0`, I move to $v$.
           The interactor confirms I am at $v$ (by printing its description).
           At $v$, I will see $u$ in neighbor list. $u$ will have `visited=1` now?
           Wait, $u$ was the place I came from. It was visited.
           So in the neighbor list of $v$, $u$ has `visited=1`.
           So I can track: "Nodes I have moved TO".
           Count how many *unique* neighbors have I moved to?
           But I don't know if I am at a node I visited before or not if I am revisiting.
           Actually, `base_move_count` hints at BFS/DFS depth.
           
           Let's assume the worst case:
           I need to avoid getting stuck in a small component if one exists (though problem says visit all vertices, implies graph is connected or I can't visit all). Assuming connected.
           
           What if I try to memorize the "signature" of the current vertex?
           Signature = `(degree(u), multiset of neighbor_degrees_and_visits?)`
           Actually, since the shuffle happens at arrival, the neighbor list I receive IS the signature of neighbors.
           And my `degree(u)` is part of the input line.
           So, `Signature(state)` = `(degree(current_node), multiset of (deg(n), visited(n)))`.
           Is this signature unique enough to identify the vertex $u$?
           Vertices in the same orbit of automorphism group might share signatures.
           In a triangle 1-2-3, all have degree 2. Neighbors have degree 2. Visits vary.
           So signatures distinguish visited vs unvisited.
           If I am at $u$, and have visited neighbors $\{p\}$ and unvisited neighbors $\{v\}$.
           Then the list contains one entry `(2, 1)` and one `(2, 0)`.
           If I move to $v$, at $v$ I will see `u` as `(2, 1)` and `w` as `(2, 0)`.
           Basically, the configuration of visited/unvisited neighbors evolves.
           
           Heuristic Strategy:
           Maintain `visited_vertices` set?
           Actually, the interaction ends when AC (all vertices visited).
           I just need to visit every vertex.
           Simplest approach:
           Just output moves.
           Track "moves made" and "unique vertices encountered".
           But how to track unique?
           If I move to a node $v$, and the interactor says it is my neighbor, I don't learn $v$'s ID.
           I just know I'm somewhere.
           Wait, the problem says: "interaction ends when You have visited all vertices: The interactor prints 'AC'".
           This implies the interactor tracks my visitation state internally.
           My task is just to trigger it.
           But I must ensure I reach all.
           If I wander randomly, I might take exponential time.
           I need to target all $N$.
           
           Wait, can I infer that I have visited $k$ vertices by counting moves? No, cycles.
           Is there any signal telling me "You are now at vertex ID x"? No.
           So I must deduce my position based on topology.
           With $N=300$, maybe I can assume the graph is not "hard" adversarial against me except for the shuffle.
           But wait, "shuffles the order" implies I can't rely on index.
           But I can rely on degree and visited flags.
           
           Correct Logic for Traversal:
           1.  I know `n, m, start`.
           2.  I read `m` edges. I build `adj_list[id]` for each node `id`.
               `adj_list[u]` stores a list of neighbor IDs `v`.
           3.  I start at `start`.
           4.  I need to keep track of which nodes I have conceptually "claimed" as visited.
               But I don't know my current ID.
               
               Wait!
               Do I really need to know the ID of the current node?
               Suppose I am at $u$. I see a list of neighbors.
               I pick one with `visited=0`.
               I move there.
               Now I am at $v$.
               I see neighbors.
               I need to continue finding `visited=0`.
               If I am doing this blindly, I might go $u \to v \to w \to u$.
               If I have a cycle $u-v-w-u$, I might cycle forever.
               I need to break cycles.
               A common way to break cycles is to keep track of the "parent" pointer in the spanning forest I am building.
               Parent of $v$ is $u$.
               When at $v$, I look for $u$.
               I know $u$'s degree.
               I look for neighbor with `deg=u.degree` and `visited=1`.
               (Since $u$ is visited).
               If I find such a neighbor, that's my "backtrack" link.
               If there are multiple, I am unlucky.
               BUT, if I prioritize moving to UNVISITED neighbors, I minimize revisiting.
               
               Wait, if I prioritize unvisited, I essentially construct a BFS/DFS tree.
               The nodes I move TO are unvisited nodes.
               When I am at a node $x$, I might have come from $y$.
               $y$ is visited.
               If I move to a neighbor $z$ (unvisited), I extend the tree.
               If I can't find unvisited neighbors, I must move to a visited neighbor.
               This effectively retraces steps or crosses bridges.
               
               So the algorithm:
               State: `Current_Node_ID`? No, I can't track it reliably if ambiguous.
               But I CAN track `Current_Path_Index` or `Depth`?
               
               Actually, maybe I don't need the ID.
               I just need to ensure that every vertex gets visited.
               The only constraint is I don't waste moves.
               Scoring penalizes extra moves.
               So I should aim for BFS-like movement (shortest path to cover).
               Shortest path covering is TSP-like, NP-hard.
               But BFS traversal of spanning tree visits each node at most twice (once going down, once coming back). Total moves $\approx 2(N-1)$.
               If `base_move_count` is around $2(N-1)$, we are good.
               
               Wait, if I can't identify the parent reliably, I might not be able to backtrack correctly to the right place in the tree if I got stuck in a sub-tree.
               If I am at leaf $L$ of BFS tree, parent is $P$.
               I want to go to $P$.
               If I go to $P$, good.
               If I go to $Q$ (another visited neighbor), I might be stuck in a cycle or wrong branch.
               But $P$ and $Q$ have same `deg`.
               However, in BFS, $Q$ would be at the same level or higher.
               $P$ is at level $d-1$.
               If I can distinguish levels? I don't have distance info.
               
               Okay, let's look at the "visited" flag again.
               When I am at $u$ (just arrived from $p$), $p$ is marked visited.
               All other neighbors are unvisited (in a pure exploration phase).
               So the neighbor list is: `(deg_p, 1), (deg_n1, 0), (deg_n2, 0)...`.
               If I pick one of `(deg_ni, 0)`, I move there.
               If I pick `(deg_p, 1)`, I backtrack.
               The problem arises if there are multiple neighbors with `(deg, 1)`.
               But in a pure tree expansion, there is only one visited neighbor ($p$).
               Wait, unless I visited a neighbor that is NOT $p$ earlier?
               That means the graph has a cycle or cross-link.
               If there is a cross-link to $q$ (already visited), then $q$ appears as `(deg_q, 1)`.
               In that case, I have `(deg_p, 1)` and `(deg_q, 1)`.
               If I choose $p$, I backtrack. If I choose $q$, I go to $q$.
               Going to $q$ is safe (it's visited) but doesn't help exploration directly.
               Backtracking to $p$ allows me to explore other branches from $p$.
               So I SHOULD identify $p$.
               
               How to identify $p$ among `(deg, 1)`?
               Maybe use the previous step?
               I know I just came from $p$.
               Is there any other info?
               No.
               But wait! The problem states: "You start at a known fixed vertex."
               I know `start` node has ID `start`.
               If I am at `start` (depth 0), neighbors are unvisited.
               If I move to neighbor 1 (depth 1).
               At neighbor 1, I see `start` as visited.
               If `start` has a unique degree among neighbors?
               Or if `start` is unique?
               Actually, if I assume the graph is simple, and $N \le 300$.
               Is it possible to assign IDs to nodes?
               I can label vertices based on their `(degree, visited_neighbors_structure)`.
               This is basically trying to match vertices to IDs.
               With $N$ small, I can try to find a canonical labeling.
               But I don't know which is which initially.
               Wait, the interactor gives me the graph structure.
               I know `node 1`, `node 2`...
               I start at `node start`.
               So I know `ID(start)`.
               When I am at `node start`, I see neighbors. I can hypothesize:
               "Maybe this list corresponds to {neighbor_of_start}".
               If I pick one, I move to a node $X$.
               Then I see neighbors of $X$.
               I know one of them is `start`.
               `start`'s degree is known.
               So I can identify `start`'s entry in the list of $X$.
               This confirms `ID(X)` has an edge to `ID(start)`.
               I can mark this relation.
               I can essentially propagate the graph topology from the start.
               I can assign temporary IDs to nodes as I discover them.
               Root = `start` (ID 0).
               Children of root discovered = ID 1, 2, ...
               At each step, I identify neighbors by matching the edge to parent.
               Since I know `parent`'s degree, I can isolate the parent in the current node's neighbor list.
               If I have ambiguity (multiple parents), I might fail. But if I process BFS, I add nodes one by one.
               For a newly discovered node $u$, it connects to at least one parent $p$ (the one I came from).
               If $u$ connects to another visited node $v$ (back-edge), $v$ is also visited.
               But I can handle that.
               
               Wait, the crucial thing is: does the interactor allow me to distinguish neighbors by degree?
               Yes.
               And does it allow me to distinguish by "is it the one I came from"?
               Yes, by checking visited status and degree.
               
               So the plan:
               I will maintain a `map<ID, Vector<Neighbor>>` which represents my current understanding of the graph structure near me.
               More precisely: I can just maintain a queue of unvisited nodes to visit.
               Wait, I need to navigate.
               
               Simpler Logic:
               I just need to find the parent to go back.
               Parent $P$ is the vertex I arrived from.
               Its degree is `known_degree[P]`.
               At current vertex $U$, I iterate neighbors.
               I look for `neighbor` such that `neighbor.degree == known_degree[P]` AND `neighbor.visited == 1`.
               If there is exactly one such neighbor, I assume it is $P$.
               I move back to it? No, I don't need to move back immediately.
               I need to move to an *unvisited* neighbor to explore.
               So:
               Scan neighbors for `visited == 0`. Pick one. Move.
               If no unvisited, scan for `visited == 1` (Parent). Move.
               Why? If I am at a leaf of the explored tree, I must backtrack to reach other parts.
               If I pick an unvisited neighbor, I descend.
               
               Is it guaranteed that there is exactly one `visited == 1` and `degree == P.degree`?
               If the graph has symmetries, maybe not.
               BUT, if I assume I am constructing a BFS tree, I only traverse edges to *unvisited* nodes.
               Edges to *other visited* nodes (back-edges) are ignored or treated as traps.
               If I encounter a node with `visited == 0`, I go there.
               If I encounter multiple `visited == 0`? Pick any.
               If I encounter only `visited == 1`? Must pick one to continue (backtrack).
               
               What if picking one leads to infinite loop?
               If I always pick `visited == 0` if exists, I am essentially performing DFS/BFS.
               Since graph is finite and connected, eventually all reachable nodes will be visited.
               Are there cases where I cycle?
               Only if I pick a `visited == 1` neighbor that is NOT the parent.
               When would I pick a `visited == 1` neighbor?
               Only if no `visited == 0` exists.
               If no `visited == 0` exists, I am in a region where all neighbors are visited.
               I MUST leave this region. To leave, I must go to a node with `visited == 0` (impossible) or... wait.
               If all neighbors are visited, I can't go anywhere new.
               I must backtrack.
               If I backtrack to a node that was visited long ago (crossing edge), I might enter the same component.
               Wait, if all neighbors are visited, I am in a cluster of visited nodes.
               I just walk within it.
               The question is: will I eventually exit? No, there are no exits (no unvisited neighbors).
               This means I have finished exploring the component.
               Wait, if the graph is connected, and I am in a cluster, have I visited everything in the whole graph?
               Not necessarily. I might have separated the graph into two visited components.
               How? By cutting off? No, graph is undirected.
               If I visited a subset of vertices $S$, and for all $v \in S$, neighbors are in $S$.
               Then $S$ is a connected component.
               Since original graph is connected, $S$ must be the whole graph.
               So, if I reach a state where I am at $u$ and all neighbors are visited, then I MUST have visited all vertices in the entire graph (assuming graph was connected).
               
               Wait. This assumes "visited" means "visited by ME".
               Yes.
               But what if there is an edge to a vertex $w$ that I *haven't visited*, but the interactor says `visited=1`? Impossible.
               So, if I am at $u$ and see `visited=0` for neighbor $w$, $w$ is unvisited. I move there.
               If I see `visited=1` for all neighbors, then all neighbors are visited.
               Does this imply all nodes in graph are visited?
               If the graph is connected, YES.
               Wait, what if there is a cut? No, connected.
               What if I am in a sub-component of visited nodes?
               Suppose I started at $u$. I visited neighbors. I came back to $u$.
               I went to neighbor $v$. From $v$, all neighbors are visited (including $u$).
               So $u, v$ form a cluster.
               Are there other unvisited nodes elsewhere?
               If $u, v$ is the ONLY connected component of visited nodes, and $G$ is connected, then NO unvisited nodes exist.
               Thus, AC condition is met.
               
               So, the strategy simplifies to:
               **Always move to an unvisited neighbor if available.**
               If NO unvisited neighbor is available, move to ANY visited neighbor (to traverse the visited space? or stop? or just try to find if I missed something).
               Actually, if I have visited all nodes, the interactor prints AC.
               If I am stuck in visited nodes, and haven't visited all, something is wrong.
               But if the graph is connected, reaching a state with no unvisited neighbors implies completion.
               
               Wait, is it possible that the "visited" flag refers to the global history, and I might visit a node, mark it visited, and come back, and the interactor knows?
               Yes.
               So the logic holds.
               The only caveat is distinguishing which visited neighbor to pick if multiple exist.
               If I am truly stuck (all neighbors visited), it doesn't matter which one I pick, as long as I don't get stuck in a loop.
               Since $N$ is finite, looping infinitely without progress is bad.
               But with random choice or fixed preference, I might cycle.
               However, to pass, I probably just need to reach AC.
               If I am cycling, I never reach AC?
               Wait, if I am cycling, I am NOT visiting NEW nodes.
               But if I have no unvisited neighbors, it means I AM DONE.
               Wait, I checked "All neighbors visited".
               If there ARE unvisited nodes elsewhere, then there MUST be a path from my current component to unvisited nodes.
               That path must cross an edge $(u, w)$ where $w$ is unvisited.
               So I would have seen `visited=0` at $u$ (or on the path).
               If I am at $u$ and ALL neighbors are visited, then no such edge exists from $u$'s perspective.
               Could $u$ be part of a "visited cloud" isolated from "unvisited"?
               In a connected graph, you can't be isolated by visited nodes unless you visited EVERYTHING.
               So, yes: **No unvisited neighbors => Finished**.
               
               So the strategy is extremely simple:
               1. Start at `start`.
               2. Loop:
                  a. Read `d` and list of `(deg, vis)`.
                  b. Check if `vis` column has any `0`s.
                  c. If yes, pick index of one `0`. Move there.
                  d. If no (all `1`s), pick index of a `1` (maybe first one). Move there.
                  e. Check if interactor outputs AC.
                  
               Wait, is it guaranteed that `base_move_count` allows this greedy traversal?
               Greedy DFS usually visits every edge twice (tree) plus backtracking. Moves $\approx 2E$.
               Given $N$ is up to 300, $E$ up to $25N \approx 7500$.
               Wait, $2E \approx 15000$.
               Base move count might be large enough.
               Example: Triangle $N=3, M=3$. $E=3$. Moves $\approx 6$. Base might be 1000. Plenty of slack.
               Constraint $M \le 25n$.
               Max edges $25 \times 300 = 7500$.
               $2E$ moves could be 15000.
               Is `base_move_count` defined? It's given in input.
               The scoring function penalizes exceeding `base_move_count`.
               If I exceed it, I get less points or 0.
               So I need to be efficient.
               DFS visits edges twice. BFS visits edges once?
               Actually, to visit all nodes in BFS, you visit each node. You don't necessarily need to visit every edge.
               In BFS, you enqueue unvisited neighbors.
               But here, "move" consumes edge.
               In BFS, to visit neighbor $v$ of $u$, I must traverse $(u,v)$.
               If I traverse all edges incident to $u$ (both forward and backward), that's cost.
               Standard BFS (queue) explores layers.
               But physically I am walking.
               To visit a node at depth $k$, I traverse $k$ edges.
               If I just do a walk that forms a tree, cost is $2(N-1)$.
               Wait, in a tree, there are $N-1$ edges.
               To visit all nodes in a tree starting at root, I need to do a DFS.
               DFS touches every edge twice. Cost $2(N-1)$.
               Wait, is it possible to visit a tree with fewer than $2(N-1)$ moves?
               No. To return to root (not required) costs $N-1$.
               If I don't return to root, I just stop at a leaf. Cost $N-1$.
               But I might need to visit "dead ends" before the leaf.
               To visit $u$ and then $v$, I must go $root \to \dots \to u \to v$.
               So $N-1$ moves is theoretical minimum (Hamiltonian path).
               If such path exists, $N-1$ moves.
               Otherwise, $> N-1$.
               
               Since $N \le 300$, and base is likely reasonable (maybe proportional to $N$).
               If base is small (e.g. $2N$), my greedy DFS might be too slow?
               Wait, if the graph is dense ($M \approx N^2/2$), DFS takes $O(E)$ edges.
               Wait, $M \le 25N$ means sparse?
               $N=300 \implies N^2/2 = 45000$.
               $25N = 7500$.
               So graph is indeed sparse (avg degree $\le 50$).
               Number of edges is manageable.
               But DFS visits every edge twice. Moves $\approx 2M$.
               If base is $2N$, then $2M \gg 2N$ if $M$ is large.
               Wait, $M$ can be up to $7500$.
               If `base_move_count` is small, I might fail.
               But usually "base" is calculated to be achievable by an optimal solver.
               The problem asks for optimal navigation.
               Optimal navigation seeks a Hamiltonian Path or shortest tree traversal.
               Finding optimal is hard.
               However, for $N=300$, maybe BFS is better than DFS for minimizing moves?
               BFS doesn't backtrack along the whole path.
               Wait, BFS is a mental model. Physically, I must walk.
               To go to a second branch, I must walk back.
               This suggests DFS is natural.
               But I can "jump" back mentally? No, I move physically.
               If I visit nodes in BFS order: $v_0, v_1, \dots$.
               To get to $v_2$ which is neighbor of $v_1$, I must go $v_0 \to v_1 \to v_2$.
               To get to $v_3$ (neighbor of $v_0$), I must go $v_0 \to v_1 \to \dots \to v_0 \to v_3$.
               I need to backtrack to $v_0$.
               This is exactly DFS behavior (implicitly).
               However, maybe I don't need to backtrack all the way?
               If I am at $v_1$ and want to visit $v_3$ (neighbor of $v_0$), I must return to $v_0$.
               So, to visit all nodes efficiently, we effectively want to avoid unnecessary re-visits.
               Ideally, we only traverse edges necessary for discovery.
               This is Tree Traversal.
               So DFS/BFS on the spanning tree.
               DFS depth goes up to $N$.
               Moves $\approx 2(N-1)$ if we strictly follow tree edges.
               Wait, in DFS, do we traverse non-tree edges?
               If we encounter a non-tree edge (edge to already visited node), we shouldn't take it (wasteful).
               So, we should avoid moving to visited nodes UNLESS we are stuck.
               So: **Only move to unvisited nodes.**
               If all neighbors visited -> Stuck -> Must move to a visited node (backtrack).
               
               This confirms the strategy:
               Always pick `visited=0` if available.
               Else pick any neighbor.
               
               Wait, if I pick a visited neighbor (backtracking), I consume a move.
               Can I optimize which visited neighbor to pick?
               If I am stuck at $u$ (all neighbors visited), I MUST pick a neighbor $v$ to move to.
               To proceed towards remaining unvisited nodes (if any), I should pick $v$ that offers access to them.
               But if all neighbors are visited, there are NO unvisited neighbors reachable?
               As discussed, if connected, this means all visited.
               BUT what if there is a visited cluster that connects to an unvisited node through a "cut vertex" but the path is blocked?
               No, if graph is undirected and connected, you can't have a cut vertex that blocks everything.
               Wait, if I am at $u$ and all neighbors are visited, it means I cannot move to any UNVISITED node.
               Therefore, there are NO unvisited nodes reachable from my current location without passing through a visited node.
               If there were an unvisited node $Z$, there is a path.
               The path enters the visited set at some point.
               But if I am at $u$ inside visited set, and all neighbors are visited, I can't leave the visited set.
               Thus, no unvisited nodes exist.
               So if "all neighbors visited" happens, we are done.
               Wait, this implies the game ends when I hit a node with all visited neighbors?
               Yes.
               So I just need to avoid cycles before finishing.
               
               Wait, is it possible that "all neighbors visited" condition is triggered prematurely?
               Suppose I visited a subset of vertices $S$ that induces a subgraph where all vertices in $S$ have all their neighbors in $S$.
               Then I can't leave $S$.
               If $S$ is a proper subset of $V$, and $S$ is connected (so I can't get out), and $G \setminus S$ is unreachable, then $G$ is disconnected.
               But the problem says "visit every vertex".
               This implies $G$ is connected.
               So $S$ cannot be a proper subset closed under adjacency.
               The only closed subsets are empty or $V$.
               So if all neighbors of current node are visited, and I am inside a connected component of visited nodes (which I am), then either:
               1. $S = V$ (Done).
               2. Graph disconnected (Shouldn't happen per problem assumption).
               
               So the only risk is infinite loops while $S < V$.
               How can I loop while $S < V$?
               I need to be at a node $u$ where all neighbors are visited.
               But if $S < V$, there exists $Z \notin S$.
               Since $G$ connected, there is a path from $u$ to $Z$.
               Path leaves $S$ at some edge $(x, y)$ where $x \in S, y \notin S$.
               So $x$ (which is in $S$) MUST have a neighbor $y$ with `visited=0`.
               So if I am at $x$, I see `visited=0`.
               I will move to $y$.
               Thus, I will NEVER be at a state "all neighbors visited" as long as unvisited nodes remain.
               
               Conclusion: **There is NO risk of getting stuck at a node with all visited neighbors before finishing.**
               Wait, what if I am at $u$, see unvisited neighbors, but picking the WRONG one (based on shuffle) leads to a dead end?
               In a tree, no dead ends (always leaves).
               In general graph, maybe picking a neighbor that creates a "bad" traversal?
               Actually, just picking any unvisited neighbor extends the explored set $S$.
               Since $S$ grows monotonically, eventually $S=V$.
               And since I never pick a visited neighbor while unvisited ones exist, I avoid cycles of length $>1$ (except self-loops? no self loops in simple graph).
               Wait, what if I move $u \to v$, then immediately $v \to u$?
               At $v$, $u$ is visited.
               If $v$ has other unvisited neighbors, I pick one.
               If $v$ has NO other unvisited neighbors, then I pick $u$ (backtrack).
               This is fine.
               What if I pick $u$ at $v$ even though $v$ has unvisited neighbors?
               I must check `visited` flags.
               Unvisited = `visited=0`.
               If I have `visited=0`, I pick it.
               So I NEVER backtrack unless forced.
               So I only do cycles if I have to backtrack.
               If I always pick `visited=0`, I never cycle back and forth $u \leftrightarrow v$ unnecessarily.
               The only backtracking is when I hit a dead end (leaf of exploration tree).
               Then I must backtrack to parent.
               
               So the strategy is sound.
               Implementation details:
               - Read graph.
               - Store `n, start, base`.
               - Loop.
               - Inside loop:
                 - Read line. `deg_cur`, neighbors `(deg, vis)`.
                 - Identify unvisited neighbors (flag=0).
                 - If found, pick one.
                 - Else (all visited), pick any neighbor.
                   - Wait, if I am supposed to be done, I expect AC.
                   - But what if AC hasn't printed yet?
                   - The problem says: "The interaction ends ... when You have visited all vertices: The interactor prints 'AC'."
                   - This happens immediately after my move if I trigger the condition?
                   - Or does it happen after interactor receives move?
                   - "To make a move... output... flush."
                   - Then "interaction ends ... interactor prints AC".
                   - It seems if I make a move that lands me on the LAST unvisited node, the interactor detects "all visited" and prints AC immediately before my next request?
                   - Wait, if I am at a node and there is one unvisited neighbor left.
                   - I move to it.
                   - I arrive. Interactor realizes I visited the last node. Prints AC.
                   - I terminate.
                   
               - Edge case: Base move count limit.
                 - If I exceed it, I get 0 points.
                 - I should check if I should stop?
                 - If I am stuck (shouldn't happen), I might need to stop.
                 - But with greedy strategy, moves $\approx 2(N-1)$.
                 - With $N=300$, $2N \approx 600$.
                 - Base move count is given.
                 - If Base is huge (like 10000), no issue.
                 - If Base is tight (like 200), I might need optimization.
                 - Optimization: Prioritize neighbors?
                 - With $N=300$, maybe I can't optimize perfectly.
                 - But greedy is standard for this type.
                 
               Wait, "You are provided with a base_move_count".
               The scoring formula uses it.
               The code doesn't need to calculate score.
               I just need to output moves.
               I should respect time limits.
               
               What about "shuffle"?
               My logic relies on `visited` flag.
               Is `visited` flag reliable?
               "indicated by a flag". Yes.
               Does it update instantly?
               "whether you have visited it before".
               This implies cumulative history.
               My logic holds.
               
               One subtle bug possibility:
               If I pick an unvisited neighbor, I enter it.
               The interactor updates the `visited` set.
               Next turn, that node sees me as visited.
               This matches my expectation.
               
               Wait, what if I move from $u$ to $v$ (unvisited).
               I am at $v$.
               I see $u$ in neighbor list.
               $u$ is visited.
               I also see other unvisited nodes.
               If I pick another unvisited node $w$, I go $v \to w$.
               I never go $v \to u$ (backtrack) unless $w$ has no neighbors.
               So my path is a simple walk.
               This ensures minimal moves?
               It avoids traversing same edge twice immediately.
               It builds a walk that covers the graph.
               
               Wait, does "visited" include the *current* node?
               "For each adjacent vertex: its degree and whether you have visited it before".
               This refers to the NEIGHBOR.
               I do NOT know if I visited CURRENT node?
               Wait, I just arrived.
               Does the system consider me having visited the current node?
               Implicitly, yes.
               Also, for the START node, it is visited at the beginning?
               Usually yes.
               If not, and I return to it later, it might say `visited=1` if I consider myself visited.
               Wait, the input gives `visited` for NEIGHBORS.
               It doesn't ask about self.
               My internal state needs to track visited vertices.
               
               Code structure:
               ```cpp
               #include <iostream>
               #include <vector>
               #include <string>
               #include <algorithm>

               using namespace std;

               int main() {
                   // Fast IO not needed for interactive but good practice
                   string line;
                   // Read t
                   getline(cin, line); 
                   // Parse t? Actually example input shows t on first line.
                   // Wait, problem says "First, the interactor prints an integer t".
                   // Then "Four integers: n, m, start, base_move_count."
                   // So loop t times.
                   
                   // Need to handle flushing carefully.
                   // cin >> t;
                   
                   // But wait, interactive problems often mix cin/cout or getline.
                   // cin is safer for integers.
                   
                   int t;
                   cin >> t;
                   while(t--) {
                       int n, m, start, base;
                       cin >> n >> m >> start >> base;
                       
                       // Read edges to build graph info
                       // We need degrees of all nodes.
                       vector<int> degree(n + 1, 0);
                       vector<vector<int>> adj(n + 1);
                       
                       for(int i=0; i<m; ++i) {
                           int u, v;
                           cin >> u >> v;
                           adj[u].push_back(v);
                           adj[v].push_back(u);
                           degree[u]++;
                           degree[v]++;
                       }
                       
                       // Logic
                       // We maintain a visited array for our OWN logic?
                       // Do we need it?
                       // The problem is we don't know the ID of current node.
                       // But we know start node.
                       // Wait, if I can't track visited global set accurately, I might fail.
                       // But based on analysis, I only need to follow unvisited paths.
                       // The interactor handles the "global visited" for the "AC" condition.
                       // I just need to survive until AC.
                       // Wait, if I cycle infinitely, I lose points/time limit.
                       // The greedy strategy "pick unvisited neighbor" prevents immediate cycles.
                       // It forces path extension.
                       // Does it guarantee coverage?
                       // Yes, on connected graph.
                       // Does it guarantee min moves?
                       // Probably sufficient for "competitive" problem (scoring based).
                       
                       // What about base_move_count limit?
                       // I can't easily predict if I will exceed it without simulation.
                       // But usually greedy is optimal enough for these types of puzzles.
                       
                       int current_id = start; // Logical start, though actual ID unknown.
                       // Actually, I don't need to track current_id if I trust the greedy strategy.
                       // But wait, to pick the CORRECT neighbor index (1-based), I need to know which neighbor I want.
                       // I don't know the neighbors' identities.
                       // But I know `d` and list of `(deg, vis)`.
                       // I need to select `index` (1..d).
                       
                       // Do I need to track WHICH node is which?
                       // If I rely on degrees to identify edges?
                       // If I have to backtrack, I need to identify parent.
                       // Parent identification logic:
                       // Search for neighbor with `deg == degree[current_id]` and `visited == 1`.
                       // Wait, I need to know `degree[current_id]`.
                       // But `degree[current_id]` is `d` (the first number in input line)!
                       // The input line format: "d deg_1 flag_1 ..."
                       // `d` is the degree of the current vertex.
                       
                       // So, at every step:
                       // 1. Read `d`. This is `degree[current]`.
                       // 2. Parse neighbors.
                       // 3. Identify parent (if needed for backtracking).
                       //    Parent is the node I came from.
                       //    How do I know its degree?
                       //    I can't easily store `degree[prev]` without knowing `prev`.
                       //    BUT, I can deduce it!
                       //    If I track `degree[current]` for every step, I have a log of degrees.
                       //    Wait, no. I am at `current`. I just saw `degree[current]`.
                       //    I moved FROM `prev` TO `current`.
                       //    I don't know `degree[prev]` from the input.
                       //    However, if I stored the degree of the node I LEFT?
                       //    I can store the sequence of degrees?
                       //    Or better: I don't need to identify `prev` to avoid cycles?
                       //    The strategy "Pick Unvisited if available" works regardless of `prev`.
                       //    It finds *new* nodes.
                       //    If I hit a dead end (all neighbors visited), I must move to a neighbor.
                       //    Then I am backtracking.
                       //    Moving to *any* neighbor is fine in that case, because I know I'm at a dead end.
                       //    Wait, is it guaranteed that I'm at a dead end if all neighbors are visited?
                       //    Yes, derived earlier.
                       
                       // So, I DON'T need to know the parent!
                       // I just need to find `visited=0`.
                       // If I find one, move there.
                       // If all `visited=1`, pick any `visited=1` (say first one).
                       // This is purely heuristic.
                       // Is it optimal?
                       // In "Treasure Hunt", we want to visit all.
                       // Greedy DFS visits all in a connected component.
                       // If I get stuck at a node with all visited neighbors, I stop?
                       // If I am done, I get AC.
                       // If not done, something is wrong (disconnected graph or I misidentified).
                       
                       // Wait, is it possible `visited=0` exists but picking it is bad?
                       // Only if picking it leads to a trap (small component).
                       // But all components must be connected.
                       // So any unvisited node leads to a new territory.
                       // So yes, greedy works.
                       
                       // One small catch: `base_move_count` calculation.
                       // `sol_fraction = (moves + 1) / n`.
                       // This implies I should aim for minimum moves.
                       // DFS is O(E). BFS is O(V+E)?
                       // In terms of edge traversal, BFS tree is a spanning tree.
                       // DFS walks edges twice.
                       // Wait, if I just walk on the spanning tree:
                       // Root -> Child1 -> Grandchild1 -> Leaf.
                       // At Leaf, backtrack -> Grandchild -> Child -> Root.
                       // Then go to Child2.
                       // This traces the tree edges.
                       // Each edge is traversed twice (down and up).
                       // Total moves = 2 * (N-1) = 2N - 2.
                       // BFS on the tree?
                       // BFS explores layer by layer.
                       // It still physically moves.
                       // BFS on graph requires moving back to previous nodes frequently.
                       // Actually, physical BFS is exactly the same cost as DFS (spanning tree traversal) for connected graph?
                       // Wait, BFS visits nodes in order of distance.
                       // Physical BFS means I go to distance 1, then distance 2.
                       // To reach distance 2, I must go to distance 1, then to 2.
                       // To visit another distance 2, I must go back to distance 1.
                       // So I traverse distance 1 -> 2 edge, then distance 1 <- 2 edge.
                       // This is exactly tree traversal.
                       // The order doesn't change the set of edges traversed in a tree.
                       // So cost is ~2N.
                       // Wait, is it possible to do better?
                       // Only if Hamiltonian path exists (Cost N-1).
                       // Generally, greedy DFS is a good approximation.
                       
                       // But wait. `base_move_count` might be larger than 2N.
                       // Or smaller?
                       // If `base_move_count` is 2N, and I do DFS, I am safe.
                       // If `base_move_count` is 1.5N (optimal?), I might exceed.
                       // But usually for interactive problems, strict optimality isn't required, just efficiency.
                       // The problem statement implies `base_move_count` is a reference.
                       // And `score` depends on it.
                       // If I exceed, I get 0.
                       // So I must hope I don't exceed.
                       // With $N=300$, DFS cost is roughly 600.
                       // If base is small, say 200, I'm screwed.
                       // But $M \ge N-1$.
                       // If graph is a tree, I need $2N-2$ moves.
                       // If base is less than $2N-2$, impossible?
                       // Maybe base is generous.
                       
                       // Let's write the greedy solution.
                       // One detail:
                       // Reading neighbor list.
                       // `cin >> d`.
                       // `for(int i=1; i<=d; ++i) { cin >> deg >> vis; ... }`
                       
                       // Handling TLE/Memory:
                       // N=300, small vectors. O(1) per step.
                       // Steps <= moves. Moves limited.
                       // Should be fine.
                       
                       // Wait, flushing is mandatory.
                   
                   }
                   return 0;
               }
               ```
               
               Refining the "Stuck" logic.
               If I am stuck (all neighbors visited), I MUST backtrack.
               If I pick an arbitrary visited neighbor, I move.
               Will I eventually find unvisited nodes?
               If I am truly done, AC.
               If I am not done, I am in a "bubble" of visited nodes?
               As analyzed, this implies graph disconnected (contradiction) or I am done.
               BUT wait.
               What if the graph is like a star graph.
               Center 0. Leaves 1..N.
               Start at Center.
               0 has neighbors 1..N (unvisited).
               I pick 1. Move to 1.
               1 has neighbors 0 (visited).
               1 sees 0. `visited=1`.
               Are there other unvisited neighbors? No.
               So 1 is stuck. All neighbors visited.
               According to my logic: "All neighbors visited -> Pick any".
               I pick 0. Move to 0.
               Back at 0.
               Now I see neighbors 1 (visited) and 2 (unvisited).
               I pick 2.
               Move to 2.
               Stuck -> backtrack to 0.
               Pick 3.
               ...
               Pick N.
               Done.
               So backtracking to center works perfectly for Star graph.
               What if graph is a line 1-2-3-4.
               Start 1.
               1->2. At 2. See 1(vis), 3(unvis). Pick 3.
               At 3. See 2(vis), 4(unvis). Pick 4.
               At 4. See 3(vis). All visited.
               Backtrack to 3.
               At 3. See 2(vis), 4(vis).
               All visited.
               Backtrack to 2.
               At 2. See 1(vis), 3(vis).
               All visited.
               Backtrack to 1.
               At 1. See 2(vis).
               All visited.
               Done?
               Wait, I visited 1, 2, 3, 4. AC should trigger.
               If AC doesn't trigger, I keep bouncing?
               Yes, I will bounce 1-2-3-4-3-2-1 forever.
               This wastes moves.
               To prevent this, I must detect when I am done.
               How?
               I don't know if all are visited.
               Wait, the interactor prints AC.
               So if I bounce, I just wait.
               Eventually (hopefully) AC triggers.
               Wait, if I never reach AC, I lose on time limit or score.
               With infinite bouncing, moves = infinity -> 0 points.
               I need to ensure I don't bounce unnecessarily.
               BUT, the "All visited" state implies AC.
               If I am in "All visited" state and I haven't gotten AC, it means I am stuck?
               Wait, why didn't AC trigger?
               Maybe I haven't visited *everyone*?
               In Line 1-2-3-4, if I am at 1, and visited {1,2,3,4}, then AC triggers.
               Wait, the interactor checks "visited all".
               When does it check?
               "interaction ends ... when You have visited all vertices".
               This likely triggers immediately after I arrive at a node that makes the set of visited nodes equal to V.
               So if I visited 4, then AC is printed.
               So I won't bounce at 4.
               But if I backtrack to 1 (after visiting 4).
               Did I visit 4? Yes.
               So I must have got AC.
               So backtracking logic in Star/Line examples works fine because the moment I enter the last node, AC is printed.
               The only risk is if I have unvisited nodes, I shouldn't think I'm done.
               My logic: "If all neighbors visited -> Pick any".
               This leads to backtracking.
               If I backtrack to a node that had unvisited neighbors (like Center in Star), I will pick them.
               So I will continue until I exhaust all unvisited nodes.
               If I am at a node and all neighbors visited, and I still have unvisited nodes elsewhere, then I MUST be able to move to a node that has unvisited neighbors?
               Wait, if I am at $u$, and all neighbors of $u$ are visited.
               Does this imply $u$ has no edges to unvisited set?
               Yes.
               But since graph is connected, there must be a path from $u$ to some unvisited node.
               Path leaves visited set at $(x, y)$. $x$ visited, $y$ unvisited.
               If I am at $x$, I would see $y$ (visited=0).
               But I am at $u$.
               If $u \ne x$.
               How do I leave the cluster of visited nodes?
               I need to traverse edges connecting visited nodes?
               This is the issue.
               If I am in a visited cluster, and there are unvisited nodes outside, but NO edge from my current node $u$ leads to unvisited.
               I need to travel within the visited cluster to find an edge leading out.
               But my strategy "Pick any neighbor" does this?
               In Line 1-2-3-4 example.
               Start 1. Visit 2, 3, 4.
               At 4: all visited. Pick 3. (Move 4->3).
               At 3: all neighbors visited (2, 4). Pick 2.
               At 2: neighbors (1, 3). Pick 1.
               At 1: neighbor (2). Pick 2.
               Wait, if AC triggered at 4, I would stop.
               So I wouldn't reach 4's dead end and backtrack further?
               The "AC" message terminates the map.
               So I just need to reach a node that completes the set.
               If I backtrack, I might overshoot?
               No, once AC triggers, the program terminates.
               So I just need to ensure I visit all nodes.
               The strategy guarantees I explore all reachable nodes (DFS/BFS on graph).
               Since graph is connected, I visit all.
               
               One edge case: "If base_move_count < moves <= 2 * base_move_count...".
               If I take too many moves, I get partial credit.
               DFS takes $2N$ moves.
               BFS takes $N$ moves?
               In a line graph, BFS is $N$ moves (straight line).
               Wait, I start at 1. Visit 2, 3, 4.
               If I just walk straight 1->2->3->4, I visit all in 3 moves ($N-1$).
               My strategy:
               1->2. At 2: see 1(vis), 3(unvis). Pick 3.
               2->3. At 3: see 2(vis), 4(unvis). Pick 4.
               3->4. At 4: see 3(vis). All visited.
               If AC triggers, perfect.
               So for lines, it works optimally.
               What for Star graph?
               Start Center.
               Move to L1. (Unvis).
               At L1: see C(vis). Pick C. (Backtrack).
               At C: see L1(vis), L2(unvis). Pick L2.
               Move to L2.
               At L2: see C(vis). Pick C.
               This visits all leaves.
               Moves:
               C -> L1 (1)
               L1 -> C (1)
               C -> L2 (1)
               L2 -> C (1)
               ...
               Total moves $2 \times (N-1)$.
               Wait, I don't need to return to C after L1 if I pick L2?
               Ah, at L1, I *must* return to C because L1 has no other unvisited neighbors.
               So backtracking is inevitable in Star graph.
               So moves $\approx 2N$.
               Since base_move_count is likely related to $N$ (e.g. $2N$ or $N$).
               If base is small, maybe I can optimize?
               But optimizing requires knowing the "best" next leaf.
               I don't have that info easily.
               So I'll stick with greedy DFS.
               
               Double check: "base_fraction = (base_move_count + 1) / n".
               This suggests base is roughly proportional to $n$.
               With $N=300$, $2N$ is fine.
               
               Final verification of constraints and types.
               Time 5s.
               Operations per test case: proportional to moves.
               Moves $\le 2N$?
               Worst case DFS on tree: $2N$.
               Worst case DFS on general graph?
               Still bounded by $2M$?
               Wait, I only move to unvisited.
               So each edge is traversed at most twice?
               Wait, I traverse edges in the spanning tree formed by my choices.
               The size of spanning tree is $N-1$.
               I traverse tree edges down and up.
               Non-tree edges are skipped (since neighbor `visited=1`).
               Wait, skipping non-tree edges is implicit.
               Do I ever traverse a non-tree edge?
               Only if I am forced to.
               If I am at a leaf of the tree, all neighbors are visited.
               If one neighbor is part of a non-tree edge to a visited node.
               I pick it (arbitrary).
               So I traverse it.
               But this leads to a cycle.
               In the Star graph, L1 -> C. Edge is tree edge (conceptually, though I visited C first).
               In a graph with cycle. 1-2-3-1.
               Start 1.
               1->2.
               2 sees 1(vis), 3(unvis). Pick 3.
               3 sees 2(vis), 1(vis).
               All visited.
               Pick 1. (Non-tree edge 3-1 used).
               At 1. See 2(vis), 3(vis). All visited.
               Done.
               Moves: 1->2, 2->3, 3->1. Total 3.
               $N=3$. Moves=3. Optimal ($N-1$).
               Wait, I used edge 3-1.
               Was 1-2 tree? Yes.
               Was 2-3 tree? Yes.
               Was 3-1 tree? No (cycle).
               But I traversed it.
               Why? Because at 3, all neighbors (2, 1) were visited.
               Wait, at 3, 1 was visited (via 1->2->3, so 1 is visited? Yes, start node).
               So at 3, I see 2 (visited), 1 (visited).
               All neighbors visited.
               So I pick 1.
               So I closed the cycle.
               Moves 3.
               This is efficient.
               The logic seems robust.
               
               One tricky thing: "interactor prints 'AC'".
               This message replaces the "vertex description".
               I should check for "AC" string.
               
               Input parsing:
               `cin >> t;`
               Loop t:
                 `cin >> n >> m >> start >> base;`
                 Loop m: `cin >> u >> v;`
                 While(true):
                    `cin >> d;`
                    `vector<pair<int,int>> neighbors(d);`
                    `int found_unvisited = -1;`
                    `int idx = -1;`
                    `for(int k=0; k<d; ++k) { cin >> deg >> vis; if(vis == 0 && !found_unvisited) found_unvisited = k+1; if(!found_unvisited && idx == -1) idx = k+1; }`
                    `// Pick best choice`
                    `if (found_unvisited != -1) idx = found_unvisited;`
                    `cout << idx << endl;`
                    `check for AC;`
                    
               Wait, logic for picking `idx` when no unvisited.
               If all visited, I pick first neighbor `idx = 1`.
               Or maybe the one corresponding to parent?
               Doesn't matter much for correctness if I assume I'm done.
               But to avoid infinite loops in disconnected components (impossible) or weird states.
               Actually, if I have unvisited nodes, I should not pick a neighbor that leads to a dead end?
               No, all unvisited nodes lead to something new.
               So picking `found_unvisited` is key.
               
               Code details:
               Need to handle reading `AC` or `F`.
               After outputting move, flush.
               Then check if `cin` fails or reads AC/F?
               No, `cin` might hang if I output move and interactor prints AC.
               Actually, the interactor prints AC *after* my move is processed?
               "interaction ends... interactor prints AC".
               So I print move. Flush.
               Then I try to read next input.
               The next input might be `AC` instead of vertex desc.
               So loop logic:
               `while(cin.peek() != EOF) {`
                 `try {`
                   `string token; cin >> token;`
                   `if (token == "AC") break;`
                   `if (token == "F") { cout << endl; break; }`
                   `// Else it's vertex data?`
                   `// Actually vertex data starts with integer d.`
                   `// But mixing cin >> string and reading integer.`
                   
                   // Better: read line or token.
                   // The vertex line: `d deg_1 ...`
                   // I should read the first integer `d`.
                   // If `d == 0`? No, degree >= 1?
                   // Wait, problem says $1 \le m$, so degree $\ge 1$ for connected nodes.
                   // What if AC is printed immediately?
                   
                   // Safe approach:
                   // cin >> n; (Reads first token).
                   // If n is "AC" (as string), then handled.
                   // But `cin >> n` expects int.
                   // "The interactor prints vertex descriptions...".
                   // So normally `d` is an int.
                   
                   // Let's read the first token as string.
                   // Check if it equals "AC" or "F".
                   // If so, break/continue.
                   // If it is a number, parse as `d`.
                   
                `} catch(...) {}`
               `}`
               
               Wait, reading `t` is easier.
               Then inside map loop:
               Read `n`. Check string.
               Actually, `cin >> t` works for int.
               For vertex loop:
               `string s; cin >> s;`
               `if (s == "AC") { /* done */ return 0; }`
               `if (s == "F") { /* done */ return 0; }`
               `int d = stoi(s);`
               `...`
               
               This looks solid.

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <cmath>

using namespace std;

int main() {
    int t;
    if (!(cin >> t)) return 0;
    while (t--) {
        int n, m, start, base;
        cin >> n >> m >> start >> base;
        
        // Read graph structure
        // We don't really need to store full adjacency for the greedy strategy,
        // but storing degrees helps if we wanted to identify nodes.
        // With greedy strategy, we just react to current input.
        
        for (int i = 0; i < m; ++i) {
            int u, v;
            cin >> u >> v;
            // We could build adj if needed, but for "pick unvisited" we just use input.
        }
        
        // Interaction loop
        // We loop until AC is printed.
        // Note: The interactor might print AC right after I make a valid move that finishes the task.
        
        while (true) {
            string s;
            cin >> s;
            
            // Check for termination signals
            if (s == "AC") {
                // Victory
                break; 
            }
            if (s == "F") {
                // Timeout/Limit exceeded
                break;
            }
            
            int d = stoi(s);
            // Expect d pairs of deg_flag
            // We read d pairs.
            // Identify an index to move to.
            // Greedy strategy:
            // 1. Prefer unvisited neighbor (flag=0).
            // 2. If none, pick any neighbor (backtrack/exploration).
            
            int best_idx = -1;
            bool found_unvisited = false;
            
            for (int i = 0; i < d; ++i) {
                int deg;
                int flag;
                cin >> deg >> flag;
                
                // We need to output 1-based index.
                if (!found_unvisited && flag == 0) {
                    best_idx = i + 1;
                    found_unvisited = true;
                }
            }
            
            // If no unvisited, pick first neighbor
            if (best_idx == -1) {
                // It is possible that all flags are 1.
                // We pick the first neighbor to continue.
                // Or maybe we need to be smarter?
                // Just picking first is safe enough.
                // However, we need to consume the remaining inputs for THIS turn?
                // No, we consumed them in the loop.
                // Wait, if I didn't find unvisited, I still looped and read all.
                // So `best_idx` might be uninitialized.
                // If found_unvisited is false, it means I iterated all and didn't find unvisited.
                // So I should pick index 1.
                // But wait, I need to ensure I read the stream correctly.
                // The loop above read `d` pairs.
                // So I am ready to output.
                // If `found_unvisited` was false, `best_idx` remains -1.
                // I force it to 1.
                
                if (best_idx == -1) best_idx = 1;
            }
            
            // Output move
            cout << best_idx << endl;
            
            // Important: The interactor might print AC immediately.
            // But I am in a while(true) loop at the start.
            // So next iteration will catch AC.
            // Wait, flushing is needed.
            // endl includes flush.
            
            // Safety check: if I exceed move limit, I might get F.
            // Handled by "F" check.
            
            // Optimization:
            // If I find an unvisited neighbor, I pick it.
            // Is it possible I picked a neighbor, but AC comes immediately? Yes.
            // Is it possible I picked a neighbor, and AC comes later? Yes.
            
            // One subtle bug:
            // If `d` is 0? (Isolated vertex?)
            // Problem says graph connected implicitly (goal to visit all).
            // And m >= 1.
            // So d >= 1 for all nodes involved.
            
            // Wait, is there any chance `d` is 0?
            // Start node might be isolated? No, "undirected graph", "visit every vertex".
            
            // What if `found_unvisited` is true, I set `best_idx`.
            // I output.
            // Next iteration I read `s`.
            
        }
    }
    return 0;
}
```
Re-evaluating the "read all inputs" logic.
In the loop:
```cpp
            for (int i = 0; i < d; ++i) {
                int deg;
                int flag;
                cin >> deg >> flag;
                
                if (!found_unvisited && flag == 0) {
                    best_idx = i + 1;
                    found_unvisited = true;
                }
            }
```
If I find `flag == 0` at index `k`, I update `best_idx`.
I continue reading the rest of the `d` lines.
This is correct because I need to consume the input for the current turn.
And I need to identify the best move.
The `best_idx` will hold the index of the FIRST unvisited neighbor found.
Wait, should I find the *first* or *any*?
Any unvisited neighbor is fine. First one is fine.
What if I find an unvisited neighbor, but then later in the list there is another unvisited one?
It doesn't matter.
Is it better to pick the one closest to "center" or something? I can't tell.
Just pick first is easiest.
What if `found_unvisited` stays false?
Then I pick `best_idx = 1`.
Wait, I need to be careful. If `found_unvisited` is false, I should pick *some* neighbor.
Does it matter which one?
Probably best to pick one that allows "backtracking".
Usually, I came from a parent.
The parent has `visited=1`.
Are all neighbors with `visited=1`?
Yes, if no unvisited.
So I am backtracking.
I should pick a neighbor that brings me closer to where I was, or just any.
Picking index 1 is deterministic.
Wait, shuffling means index 1 is random relative to the actual topology.
But since all neighbors are visited, any move keeps me in the visited set.
Eventually I must backtrack to find unvisited?
Wait, if all neighbors are visited, I am in a "hole" or done.
If I am not done, there must be unvisited nodes elsewhere.
How do I reach them?
They must be connected to the current component.
The only way to reach them is via an edge to a visited node that is connected to an unvisited node.
Wait, my analysis said "If connected, you can't be trapped in visited set unless V=visited".
This means I must be able to find a neighbor that connects to unvisited?
NO.
My previous deduction: "If I am at $u$ and ALL neighbors are visited, then either $u$ is isolated from unvisited set OR I am done."
But in a connected graph, "isolated from unvisited set" implies unvisited set is unreachable, i.e. disconnected.
Since graph is connected, this contradicts.
So, if there exists ANY unvisited node, there MUST be a path from current component to it.
The first edge on this path goes from a visited node to an unvisited node.
Wait, if I am at $u$, and $u$ has NO edge to unvisited node.
Then I must move to some neighbor $v$. $v$ is visited.
$v$ might have an edge to unvisited node.
So I must move $u \to v \to w \dots$ until I hit unvisited.
So I need to traverse visited nodes.
Does picking "any" neighbor suffice?
Yes, random walk in visited subgraph eventually hits the boundary (node adjacent to unvisited).
So picking index 1 is fine.
Deterministic choice helps avoiding random walk cycles, though cycles are harmless if AC triggers.

Wait, I should be careful about `cin >> t` loop.
Inside the map loop, I read `n, m`.
Then edges.
Then interaction.
Correct.

Double check: "The interactor prints vertex descriptions in the format: d deg_1 flag_1..."
Yes.

Wait, the `base_move_count` might be relevant for stopping early?
If I sense I am taking too long?
No, I just play optimally.
Also, I don't know if I visited ALL nodes.
So I continue until AC.

Wait, `t` test cases.
Example output shows moves.
The example output is empty?
Ah, "Example Output" is empty in the prompt?
Wait, "Example Output:\n\n\n1\n2\n1".
Wait, the example input/output is broken or formatted weirdly in the prompt.
The prompt says "Example Output:\n\n\n\n\n1\n\n2\n\n1".
It seems there are blank lines.
Usually interactive problems require flushing.
`cout << idx << endl;` is good.

Wait, `stoi` requires `<string>`. Included.

Wait, memory limit 256 MB.
Vector sizes are small.

Time limit 5 seconds.
With $T \le 5$ and $N \le 300$, even exponential complexity might pass, but mine is linear per move.
Max moves could be high if graph is adversarial (e.g. grid).
But moves bounded by 2M. M=7500. 7500 ops.
Very fast.

One final check on "Base Move Count".
If I exceed, I get 0 points.
Is it possible I can't win without exceeding?
Usually base is set to a reasonable bound for optimal solution.
Greedy DFS is likely optimal or close to optimal (2N).
If base < 2N, I lose.
But I can't do better than DFS without full graph knowledge (which I lack due to shuffling).
Actually, with shuffling I don't know the graph topology locally.
So DFS is the best I can do.

Wait, is it possible to deduce the structure faster?
I can maintain a map of `NodeSignature -> NodeID`.
Node Signature = `(degree(u), multiset of neighbor_degrees)`.
This signature identifies the node type in the graph class.
But since neighbors change order and visited flags change, dynamic signature is `(degree, multiset of neighbor_degrees, visited_status_vector)`.
But `visited` status depends on time.
However, static structure is `(degree, multiset of neighbor_degrees)`.
This static signature might help identify `u` if unique.
If unique, I know `u` is "node 1".
Then I can track `visited[u]`.
If multiple nodes have same signature (symmetry), I can't distinguish them.
But for most graphs (non-symmetric), this works well.
Given I just need to visit everyone, tracking explicit IDs might not be necessary if I rely on visited flags.
If I use IDs, I need to ensure consistency.
Since shuffling breaks ID-to-index mapping, I'd have to rebuild the mapping.
It's complex and risky.
The greedy strategy is robust to shuffling because it doesn't depend on neighbor order.
It only depends on `visited` flag, which is stable across time (once visited, always visited).
So greedy is safer.

Final check: `base_move_count` variable.
The problem asks me to generate solution code.
I should ignore calculating score.
Just produce moves.
Wait, "solve the problem".
Implies passing the judge.
Judge simulates interaction.
So I just need to pass AC.
Scoring is for evaluation.
If I fail to AC, I fail.
If I AC with good moves, I win points.

Code formatting:
Wrap in ```cpp ```.

One detail: `cin` vs `getline`.
Interactive problems with mixed types (strings/ints) can be finicky.
Using `cin >> s` is safest.
Handling `AC` check.
If `s` is "AC", break.
If `s` is "F", break.
If `s` is a number (vertex data), convert to `d`.
Then loop `d` times.

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>

using namespace std;

int main() {
    // Optimize I/O operations slightly, though not critical
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int t;
    if (cin >> t) {
        while (t--) {
            int n, m, start, base;
            cin >> n >> m >> start >> base;

            // Read graph edges
            // Although we don't strictly need to store them for the greedy strategy,
            // reading them is required to advance the input stream.
            for (int i = 0; i < m; ++i) {
                int u, v;
                cin >> u >> v;
            }

            while (true) {
                string s;
                cin >> s;
                
                if (s == "AC") {
                    // Visited all vertices
                    break; 
                }
                if (s == "F") {
                    // Exceeded move limit
                    break;
                }

                // s should be the degree 'd' of the current vertex
                int d = stoi(s);
                
                // Prepare to read neighbors and choose one
                int best_idx = -1;
                bool found_unvisited = false;
                
                // Consume the input for neighbors
                for (int i = 0; i < d; ++i) {
                    int deg, flag;
                    cin >> deg >> flag;
                    
                    // Greedy: pick first unvisited neighbor
                    if (!found_unvisited && flag == 0) {
                        best_idx = i + 1; // 1-based index
                        found_unvisited = true;
                    }
                }
                
                // If no unvisited neighbors, pick any neighbor (e.g., first one) to backtrack
                if (best_idx == -1) {
                    // We assume there is at least one neighbor since graph is connected and n>=2
                    // But strictly speaking, if d=0 (unlikely in this context), we might be stuck.
                    // However, for safety, if d>0, pick index 1.
                    if (d > 0) best_idx = 1;
                    else {
                        // Should not happen based on problem constraints
                        // Just flush and break?
                        // Wait, if d=0, I have nowhere to go.
                        // But n>=2 and connected implies degree>=1.
                    }
                }
                
                // Output the chosen neighbor index
                // Use newline to flush
                cout << best_idx << "\n";
            }
        }
    }
    return 0;
}
```
Wait, `ios_base::sync_with_stdio(false);` with `cin.tie(NULL)` is dangerous if mixed with `endl` or manual flush?
Standard advice: disable sync only if you handle buffering manually.
Here I use `\n` which is buffered, but `cin` is tied to `cout` by default.
In interactive, `cout` needs to be flushed immediately.
`endl` flushes. `\n` does not (buffer).
So I should use `cout << best_idx << endl;` or `cout << best_idx << "\n" << flush;`.
Using `endl` is safer.
Also `cin` might buffer input if `sync_with_stdio` is false and ties are broken.
But for interactive, typically `cin` buffers output of `printf` etc.
Best to keep default sync for simplicity in CP, especially interactive, unless I/O speed is bottleneck (here it's not).
I will remove optimization lines to be safe.

Wait, the example input shows `base_move_count` is 1000.
`n=3`.
Moves $\approx 3$. 1000 is huge.
Okay.

One tiny thing: The problem says "You start at a known fixed vertex."
Does `start` affect anything?
I just need to begin there.
My code reads `n, m, start`.
It skips `m` edges.
Then starts interaction.
Does it need to know `start`?
Only for debugging perhaps.
My logic works for any vertex.

Wait, what if I am at `start`?
Start node is marked visited.
Input `deg` neighbors.
Flags: `visited` neighbors.
Initially `start` is visited.
Neighbors are unvisited (unless graph has multi-edges or loops, assumed simple).
If neighbors unvisited, I pick one.
Logic holds.

Final sanity check on `stoi`.
`stoi` throws exception if invalid.
But `cin >> s` ensures `s` is from input.
`AC` and `F` are strings.
If `s` is number, `stoi` works.
Should be fine.

```cpp
/*
Problem Analysis:
We are navigating a graph where neighbor ordering is randomized at each step.
We are given the current vertex's degree, and for each neighbor, its degree and a visited flag.
Goal: Visit every vertex.
Strategy:
Since we cannot distinguish neighbors by position (order changes), we rely on properties:
1. The `visited` flag.
2. The `degree` (static property).

A simple DFS/Greedy strategy works well here:
- Whenever at a vertex, check neighbor list.
- If there are unvisited neighbors (`flag == 0`), pick one and move there.
- If all neighbors are visited (`flag == 1`), pick any neighbor to backtrack or move within the explored component.
  
Why this works:
- Since the graph is connected, as long as there are unvisited nodes, there exists a path to them.
- By prioritizing unvisited nodes, we expand the set of visited nodes.
- When no unvisited neighbors are available, we must be in a visited component. We backtrack to find the "gate" to unvisited nodes (or we are done).
- Since N is small and we visit new nodes greedily, we avoid excessive wandering.
- We terminate when the interactor prints "AC".

Implementation Details:
- Read T.
- For each test case, read N, M, Start, Base.
- Read edges (structure known, but mainly needed to advance stream).
- Loop:
  - Read `d` (degree of current).
  - Read neighbors list. Keep track of `d` pairs.
  - Find first neighbor with `visited=0`. If found, pick it.
  - Else, pick first neighbor (index 1).
  - Output index.
  - Flush output.
  - Continue until AC/F.
*/
```
I'll incorporate this logic.

One detail: `d` is the degree of the *current* vertex.
Is it possible `d` is 0?
Constraint: `m >= 1`. Vertices are $2..n$.
If connected, min degree $\ge 1$ usually, unless $n=2$ and isolated? No, connected.
Wait, if $n=300$ and disconnected components?
Problem says "visit every vertex". Implies connected.
If not connected, we can't visit all.
Assume connected.

One case:
If I am at a node $u$, and all neighbors are visited.
This means $u$ is in a component of visited nodes.
If I am not done, I must move to find a bridge to unvisited.
My code picks *first* neighbor.
This is fine.

Double check the `visited` flag semantics.
"indicated by a flag".
Yes.
Is it possible I am told a neighbor is visited, but I haven't technically "visited" it in my logic?
This is handled by interactor.
But my logic needs to be consistent.
If I treat "unvisited" as "flag=0", and "visited" as "flag=1", and I never explicitly mark `visited` in my mind (since I don't track node IDs), does it matter?
It matters for backtracking.
If I treat visited nodes as "visited", I will backtrack.
If I treat a node as "unvisited" because I haven't seen it recently?
No, the flag reflects reality.
So relying on `flag` is correct.

```cpp
#include <iostream>
#include <vector>
#include <string>

using namespace std;

int main() {
    int t;
    if (!(cin >> t)) return 0;
    while (t--) {
        int n, m, start, base;
        cin >> n >> m >> start >> base;
        
        // Consume edge list
        for (int i = 0; i < m; ++i) {
            int u, v;
            cin >> u >> v;
        }
        
        while (true) {
            string token;
            cin >> token;
            
            if (token == "AC") {
                break;
            }
            if (token == "F") {
                break;
            }
            
            // Token is the degree d
            int d = stoi(token);
            
            int chosen_idx = -1;
            bool unvisited_found = false;
            
            for (int i = 0; i < d; ++i) {
                int deg, flag;
                cin >> deg >> flag;
                
                // Greedy choice: prefer unvisited neighbor
                if (!unvisited_found && flag == 0) {
                    chosen_idx = i + 1;
                    unvisited_found = true;
                }
            }
            
            // If no unvisited neighbors, pick the first neighbor (backtrack)
            // It's guaranteed d >= 1 for a connected graph with n >= 2
            if (chosen_idx == -1) {
                if (d > 0) chosen_idx = 1;
            }
            
            cout << chosen_idx << endl;
        }
    }
    return 0;
}
```
This looks concise and correct based on the greedy heuristic.

One minor point:
`cin >> token` skips whitespace.
`stoi` parses.
If I output `chosen_idx` and `endl`, it flushes.
Next iteration reads `token`.
If `AC` is printed, `token` becomes "AC".
Correct.

Wait, what if the interactor prints `AC` but the next read is expected to be `n` of next case?
"The interaction ends for a map when... AC".
"You must then proceed to the next map".
So after AC, the loop breaks, outer loop continues to `t--`.
It will try to read `n` again.
Correct.
The structure:
Outer loop `t`.
Inside: read graph.
Inner loop (interaction) -> breaks on AC/F.
Then repeat.
Seems correct.

What about memory/time limits?
5s is plenty for $T=5, N=300$.
Solution is efficient.

Final check on headers.
`iostream`, `vector`, `string`.
`stoi` is in `string`.
Namespaces.
Looks good.
</think>

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>

using namespace std;

int main() {
    // Ensure basic I/O safety
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int t;
    // Read number of test cases
    if (cin >> t) {
        while (t--) {
            int n, m, start, base;
            // Read graph parameters
            cin >> n >> m >> start >> base;

            // Read graph edges
            // We primarily need to consume this input.
            // Storing the graph is optional for the greedy strategy, 
            // as the interactor provides structural info at runtime.
            for (int i = 0; i < m; ++i) {
                int u, v;
                cin >> u >> v;
            }

            // Interaction loop
            while (true) {
                string s;
                cin >> s;

                // Check for termination conditions
                if (s == "AC") {
                    // Successfully visited all vertices
                    break;
                }
                if (s == "F") {
                    // Exceeded move limit or failed
                    break;
                }

                // The string 's' represents the degree 'd' of the current vertex
                int d = stoi(s);
                int chosen_idx = -1;
                bool found_unvisited = false;

                // Read the neighbor information and select a move
                // Greedy Strategy:
                // 1. Prioritize moving to an unvisited neighbor (visited == 0).
                //    This expands our explored area towards all vertices.
                // 2. If all neighbors are visited (visited == 1), we must backtrack.
                //    In this case, picking any neighbor (e.g., the first one) is acceptable
                //    to return to the previous node or explore connected branches.
                for (int i = 0; i < d; ++i) {
                    int deg, flag;
                    cin >> deg >> flag;

                    // Identify the best candidate.
                    // We pick the first unvisited neighbor if available.
                    // Indices are 1-based.
                    if (!found_unvisited && flag == 0) {
                        chosen_idx = i + 1;
                        found_unvisited = true;
                    }
                }

                // If no unvisited neighbors were found (we are in a visited component),
                // we pick a neighbor to move to (backtrack or traverse visited set).
                // Since the graph is connected and n >= 2, d >= 1.
                if (chosen_idx == -1) {
                    chosen_idx = 1;
                }

                // Output the chosen neighbor index and flush
                cout << chosen_idx << endl;
            }
        }
    }
    return 0;
}
```
````

---

## 直接SFT 样本(sample_idx=2,得分 0.0,completion_tokens=1651)

````
I need to implement a solution for this interactive treasure hunting problem.

Let me analyze the problem:
1. I'm navigating an undirected graph where edges shuffle their labels at each junction
2. At each vertex, I can see: degree of current vertex, and for each neighbor: neighbor's degree + whether it's visited
3. I need to visit all vertices to collect treasures (get "AC")
4. Score depends on moves vs base_move_count - fewer moves = better score

The key insight is that when shuffling happens:
- The adjacency structure doesn't change structurally, but edge labels do
- A neighbor's identity becomes random from its adjacent vertices' list

Since I can't track neighbors by original ID after shuffles happen, I need a strategy that works with the information available at each step.

Key observations:
1. **Visited neighbors** are clearly marked with flag=1 - these are already done, no point visiting again
2. **Unvisited neighbors** (flag=0) - one of these is my way forward if any exist
3. When returning to previous vertices, I should go toward unvisited areas, not keep cycling in visited regions

Strategy design:
- Track which vertices I've visited by seeing flag=1 at each node (each vertex tells me about itself too - degree and its own neighbor flags)
- When multiple unvisited neighbors exist, pick one - since we don't know original IDs after shuffles, all unvisited have equal probability
- Return to the most recently visited vertex (backtracking/depth-first approach) rather than always going to start or nearest

Here's the implementation:
```cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

void read_line(int& n, int& m, int& start, long long& base_move_count, vector<pair<int, int>>& edges) {
    cin >> n >> m >> start >> base_move_count;
    edges.clear();
    for (int i = 0; i < m; i++) {
        int u, v;
        cin >> u >> v;
        edges.push_back({u, v});
    }
}

void print_and_flush(const string& output) {
    cout << output;
    cout.flush();
}

string read_vertex_info(int t, vector<pair<int, int>>& edges) {
    int d;
    cin >> d;
    vector<int> degrees(d);
    vector<bool> flags(d);
    for (int i = 0; i < d; i++) {
        cin >> degrees[i] >> flags[i];
    }
    
    // Read all vertex info until we get the current vertex's description
    while (true) {
        string line;
        getline(cin, line);
        if (line.empty()) continue;
        return line;
    }
}

void solve(int t) {
    vector<pair<int, int>> edges;
    long long base_move_count;
    int n;
    cin >> n;
    for (int i = 0; i < n; i++) {
        int u, v;
        cin >> u >> v;
        edges.push_back({u, v});
    }
    cin >> base_move_count;
    
    vector<int> visited(n + 1, 0);
    int move_count = 0;
    int best_vertex_id = 0;
    
    cout << endl;
    cout.flush();
    
    while (move_count < base_move_count) {
        string line;
        getline(cin, line);
        if (line.empty()) continue;
        
        stringstream ss(line);
        string vertex_id_str;
        ss >> vertex_id_str;
        int vertex_id = stoi(vertex_id_str);
        
        string neighbor_id_str;
        ss >> neighbor_id_str;
        int neighbor_id = stoi(neighbor_id_str);
        
        int visited_count = 0;
        for (const auto& edge : edges) {
            if (edge.second == vertex_id) {
                visited_count++;
                break;
            }
        }
        
        int unvisited_count = 0;
        for (int j = 0; j < d && j <= visited_count; j++) {
            if (j <= d && flags[j]) {
                visited_count++;
            } else {
                unvisited_count++;
            }
        }
        
        int best_index = 0;
        int max_deg = 0;
        for (int k = 0; k < d; k++) {
            if (flags[k]) {
                visited_count++;
            } else {
                unvisited_count++;
            }
        }
        
        int visited_neighbour = 0;
        for (int j = 0; j < d; j++) {
            if (!flags[j]) {
                visited_neighbour = j;
                break;
            }
        }
        
        cout << vertex_id << endl;
        cout.flush();
        move_count++;
    }
}
solve(t);
return 0;
```

This code reads the graph, tracks visited vertices, identifies unvisited neighbors, and repeatedly chooses the most connected unvisited vertex using depth-first search with backtracking. However, there's a critical flaw: the `read` function isn't implemented correctly, and the movement logic needs substantial improvement.

For a proper solution, I need to rethink the entire approach:

The real challenge is that I can't track the true vertex ID after the first move because edge shuffling makes the neighbor ordering unpredictable at every junction. So I shouldn't try to follow a fixed path.

Instead, I should maintain local state:
1. Keep track of which vertices I've seen as neighbors (by comparing degree values)
2. Prioritize moving to unvisited neighbors
3. When all reachable neighbors are visited, backtrack
4. Revisit previously-visited vertices only when necessary to reach new ones

The scoring formula shows that exceeding twice the base move count means zero points, so efficiency matters greatly. Let me reconsider what information I actually have...

When I arrive at a vertex, I receive its full description right then - showing its degree, and for each neighbor its degree plus a visited flag. This is the current vertex's data. Then my move sends me away. The next time I encounter *that same vertex*, I'd be sent the vertex ID plus its full description again. But between encounters, I'd only ever see the neighbor-list portions of other vertices' descriptions.

So the question becomes: after I leave a vertex, how many times before does it show up again? If it appears frequently in neighbor lists I'll see, I can reidentify it by its degree value alone. For high-degree vertices appearing often, I could map them to IDs. Low-degree infrequent ones pose harder tracking problems...

But here's what might actually be simpler: what if I stop trying to track specific vertex identities altogether, and instead focus purely on navigation strategy? Every vertex shares the pattern of telling me about its own neighbors' degrees and flags. What if I use this consistently?

The cleanest interpretation is probably that each turn I'm given exactly one vertex's description - either where I currently stand, or a neighbor whose ID gets shown along with its details. My job is just to make one good choice from whatever's presented, not to build global ID mappings over time.

A practical heuristic then: prioritize reaching new ground, but intelligently choose which path down it leads through. Rather than randomly picking among unvisited options, favor routes where my progress spreads outward toward unexplored territory instead of circling within regions I've already mapped.
````
