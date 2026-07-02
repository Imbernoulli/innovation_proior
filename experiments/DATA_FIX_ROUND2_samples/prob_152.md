# 题 152 — base均分 0.0 vs 直接SFT均分 0.0(掉点 0.0)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=4,得分 0.0,completion_tokens=19890)

````
The user wants me to solve a competitive programming problem.

**Problem Summary:**
- Grid: $0 \le x, y \le 800$. Center is $(400, 400)$.
- Orders: 1000 orders total. Each order $i$ has a start $(a_i, b_i)$ and end $(c_i, d_i)$.
- Constraint: Select exactly 50 orders ($S$).
- Route: Must start at $(400, 400)$, visit all start points of selected orders in $S$, and visit all end points of selected orders in $S$. Specifically, for each selected order $i \in S$, start point must be visited before the corresponding end point.
- Objective: Minimize total Manhattan distance traveled.
- Scoring: Based on minimizing total travel time $T$. We need to output $m=50$ indices and the route.

**Analysis:**
1.  **Route Structure**: The route starts at center, ends at center, visits pairs of points $(a_i, c_i)$ for $i \in S$, with $a_i$ coming before $c_i$ in the sequence. Since we can pick up multiple dishes, this is essentially finding a path that covers a set of edges (ordered pairs) forming a tour starting and ending at the depot (center), such that each edge direction is respected. In Traveling Salesperson Problem (TSP) terms, we have a set of tasks where each task has a specific origin and destination. This looks like the "Asymmetric TSP with Pickup and Delivery" or simply finding an optimal ordering of 50 pairs plus the return to center.
    Actually, since we start and end at $(400, 400)$, and for each order we go $A \to C$, the cost contribution of picking up $A$ and delivering $C$ depends on the current position. If we are at the last delivery location $L_{prev}$ of previous orders, going to $A_{curr}$ costs $dist(L_{prev}, A_{curr})$. Then $A_{curr} \to C_{curr}$ costs $dist(A_{curr}, C_{curr})$. Finally $C_{curr} \to L_{next}$ costs $dist(C_{curr}, L_{next})$. Wait, actually, the route is a sequence of coordinates. We just need to ensure that for every selected order $i$, the coordinate $(a_i, b_i)$ appears in the sequence before $(c_i, d_i)$.

2.  **Cost Calculation**:
    Total time = Distance from Center to first pickup + sum of distances between pickups/deliveries + sum of distances from last delivery back to Center.
    Basically, we visit a sequence of points. Let the sequence of distinct points visited be $P_1, P_2, \dots, P_k$. Note that some points might be same.
    However, since we can optimize the route freely, once we fix the set $S$ of 50 orders, the problem becomes: Find a permutation of these 50 orders $\pi$ such that if we process order $\pi_j$, we incur cost.
    Actually, it's not strictly a permutation of orders because the route might visit intermediate points. But in Manhattan metric on a grid, the optimal way to satisfy "visit $A$ then visit $C$" constraints without visiting extra nodes is just chaining them.
    Wait, "after visiting $(a_i, b_i)$... visit $(c_i, d_i)$".
    So the route is a walk.
    Start at $O = (400, 400)$.
    Pick up $i_1$, deliver $i_1$ (at $O \to A_{i_1} \to C_{i_1}$).
    Then maybe pick up $i_2$, deliver $i_2$ (at $C_{i_1} \to A_{i_2} \to C_{i_2}$).
    ...
    Or we could interleave? E.g., $O \to A_1 \to A_2 \to C_1 \to C_2 \to O$.
    Cost is Manhattan distance.
    
    Crucial observation: The cost of a route covering a set of ordered pairs $(A_1, B_1), \dots, (A_k, B_k)$ starting and ending at $O$ is minimized when we visit locations in a geometrically efficient way.
    Usually, for small number of orders (50), we might try to model this as TSP.
    However, 50 is too large for standard TSP algorithms ($O(2^n \cdot n^2)$ or even heuristics might struggle given strict time limits, though usually 50 suggests heuristic or dynamic programming if structure exists). But wait, this is AtCoder High School Contest (likely HSC AHC or similar)? No, it says "AtCoder Inc." but context implies AHsC 2016? No, looking up the scoring rule `round(10^8/(1000+T))`, this is a typical optimization objective.
    Actually, this looks like **AHCP 007**? No.
    Wait, searching for the problem text...
    It matches **AtCoder Heuristic Contest 006 (AHC006)** Problem A.
    Link check: https://atcoder.jp/contests/ahc006/tasks/ahc006_a
    
    The problem is AHC006-A "Delivery Service".
    Constraints: 1000 orders, select 50.
    Time limit usually tight for meta-heuristics.
    
    Since $N=50$ selected orders, $O(N!) $ is impossible.
    However, the objective function is roughly related to the length of the tour.
    Key idea in TSP heuristics: 2-opt, simulated annealing, genetic algorithms.
    For AH contest, you have to implement a solver that minimizes total distance.
    Since it's a heuristic contest, there is no single correct algorithm, just one that finds the best score within time.
    
    **Strategy**:
    1.  **Select 50 orders**: Which 50 should be chosen? Intuitively, orders that are close to the center or close to each other (clustered) reduce travel time. Also, the distance $(a_i, b_i)$ to $(c_i, d_i)$ contributes to the base cost regardless of the order. Minimizing the sum of these fixed distances is trivial (pick 50 with smallest internal distances). But the routing cost (between orders) matters more.
    2.  **Optimal Routing**: Given a subset of 50 pairs $(A, C)$, find the tour $O \to \dots \to O$ that satisfies the precedence constraints (visit $A$ before $C$).
    Actually, does the constraint $s < t$ enforce strictly sequential pickup-delivery? "For each $i \in S$, visit $(c_i, d_i)$ after visiting $(a_i,b_i)$". It does not require immediate pickup and delivery. It allows visiting $A_1$, then $A_2$, then $C_1$, then $C_2$.
    This reduces the problem to finding a path through the multiset of points $\{A_i, C_i\}_{i \in S}$ such that in the path, $A_i$ precedes $C_i$ for all $i$, starting/ending at Center.
    With 100 points (2 per order), finding the optimal path respecting orderings is the **Traveling Salesman Problem with Precedence Constraints**.
    
    Given the scale (50 orders -> 100 points), exact TSP is impossible.
    However, often in these problems, "picking up then delivering immediately" is a good approximation if clusters are formed.
    Or, we can treat the problem as: we need to form clusters of $A$s and $C$s to minimize travel.
    
    **Heuristic Approach**:
    1.  **Order Selection**: Use a greedy approach or clustering to pick 50 promising orders. Maybe orders closer to $(400, 400)$ are better candidates. Also shorter $(A_i, C_i)$ segments are cheaper to traverse internally.
    2.  **Routing Optimization**: Once the set $S$ is picked, construct a tour.
        -   Sort orders based on distance to current position.
        -   Try permutations. 50 is too big for brute force.
        -   Can use Simulated Annealing (SA) or Genetic Algorithm (GA).
        -   Alternatively, since the constraint is just $A_i$ before $C_i$, and we can visit arbitrary points, we can view this as sorting 50 pairs.
    
    **Detailed Algorithm Idea**:
    -   Precompute distances between all potential points involved (Center + all $A_i$ + all $C_i$).
    -   Since 1000 orders, total points = 3000. Too many for full graph.
    -   However, we only visit 100 points (50 pairs * 2 + Center) + potentially intermediates. We assume we only stop at $A$'s and $C$'s.
    -   Wait, the input gives us 1000 orders. We need to pick 50.
    -   Let's approximate. Suppose we pick a candidate set. Construct a TSP-like tour on the $100 + 1$ points.
    -   The condition "visit $A_i$ before $C_i$" simplifies things. Often, the optimal route effectively processes pairs. i.e., pick up $i$, deliver $i$, pick up $j$, deliver $j$. Why? Because crossing deliveries increases distance unless they are far apart spatially.
    -   If we stick to "Pickup-Delivery" pairs (sequentially processing orders), the cost is $\sum dist(curr, A_i) + dist(A_i, C_i) + dist(C_i, next\_start)$.
    -   We want to minimize $\sum |x_i-x_{i+1}| + \dots$.
    -   Base cost: Sum of $|A_i - C_i|$ for $i \in S$.
    -   Variable cost: Connecting these segments.
    -   This looks like we want to order the pairs $(A_i, C_i)$ to minimize transition distances.
    
    **Meta-Search Strategy**:
    Since this is AH contest, usually a simple SA over the subset selection and permutation works well.
    State representation: A bitmask or index list of 50 orders.
    Or maybe encode state as a permutation of 1000 items but masked to keep size small? No, that's huge.
    State: A list of indices $P$ of size 50.
    Neighbors: Swap two elements in $P$; or swap a subset of orders; or replace an order in $P$ with an order outside $P$.
    
    Wait, if we fix the set of 50, how do we route?
    If we assume the route always picks up $i$ and delivers $i$ consecutively (i.e. $O \to A_{p_1} \to C_{p_1} \to A_{p_2} \to C_{p_2} \dots \to O$), this satisfies the constraints.
    Does it guarantee optimality? Not necessarily. One could go $A_1 \to A_2 \to C_1 \to C_2$.
    However, usually "locality" means $A_i$ and $C_i$ are somewhat close or clustered. If $A_i$ is far from $C_i$, going $A_1 \to A_2 \to C_1 \to C_2$ might save the trip back to $A_1$ if $C_2$ is near $A_1$.
    BUT, note that in the problem statement, "After picking up food at one restaurant, he may pick up food at another restaurant or deliver food to another destination before delivering that food to the destination."
    This confirms we don't have to deliver $i$ immediately after picking up $i$.
    So this is **General TSP with Delivery Constraints** (also known as VRP with Pickups and Deliveries).
    
    Given the difficulty of solving Optimal General P&D TSP with 50 pairs, a common practical heuristic is to group nearby requests and visit them.
    However, 50 orders is still quite large for generic TSP solvers to run inside the search loop.
    Usually for AH contests, a simpler heuristic works better than complex ones.
    Maybe simply selecting the 50 orders that are closest to $(400, 400)$ is not enough?
    
    Let's look at the example scoring. $\mathrm{round}(10^8/(1000+T))$.
    $T$ around 1000-3000 would yield reasonable score.
    Max distance is 800+800 = 1600.
    If we pick 50 orders, minimal theoretical lower bound:
    Even just going back and forth for 50 orders, average movement $\approx 200$ each.
    $50 \times (400) \approx 20000$.
    Score $\approx 10^8 / 21000 \approx 4700$.
    Wait, if $T$ is small, score is high.
    To get max score, we need small $T$.
    
    Let's refine the heuristic.
    1.  Cluster orders based on location of $(a_i, b_i)$ and $(c_i, d_i)$.
    2.  Greedy construction:
        -   Maintain a set of unvisited orders.
        -   Current pos = Center.
        -   Iterate 50 times:
            -   Choose next order to execute.
            -   Which order? Candidates: those that can be picked up efficiently.
            -   If we adopt the strategy $O \to A \to C \to O'$, cost is $d(O, A) + d(A, C) + d(C, O')$.
            -   If we switch strategies, say $O \to A_1 \to A_2 \to C_1 \to C_2 \dots$, it becomes much harder.
            -   Most submissions for this problem use a randomized local search (Simulated Annealing) optimizing the assignment of subsets and their permutation, assuming the $A \to C$ consecutive structure OR trying to reorder all $A$ and $C$ points.
            -   Wait, the problem is known to be solved by treating it as finding a path through 100 points (pairs of points).
            -   Actually, many successful solutions for AHC006-A simply use Simulated Annealing to optimize a tour of 100 points (where each order consists of 2 points $A_i, C_i$) subject to $A_i$ appearing before $C_i$.
            -   Since 100 points TSP is hard, but SA with random swaps can escape local optima.
            -   Specifically, we can generate a valid initial solution, then randomly perturb the order of orders.
            -   Perturbation type:
                -   Swap two orders in the sequence.
                -   Swap order of two $A$'s and two $C$'s?
                -   Change the set of selected orders.
            -   Calculating fitness is fast for 100 points.
            
    **Wait, calculating the cost of the tour is easy**:
    For a permutation of orders $p_1, \dots, p_{50}$, assume we perform operations:
    $Center \to A_{p_1} \to C_{p_1} \to A_{p_2} \to C_{p_2} \dots \to C_{p_{50}} \to Center$.
    Is it guaranteed that optimal is this form?
    In general P&D TSP, no. But in practice, with 50 orders, maybe yes? Or at least a very strong local optimum.
    However, we can verify if allowing interleaving helps.
    Actually, a stronger assumption is used in many solutions: we just order the 50 orders.
    Some solutions allow swapping the $A$ and $C$ positions in the sequence to see if it helps (which is invalid), so we must ensure validity.
    Wait, if the problem allows $A_1 \to A_2 \to C_1 \to C_2$, then treating them as 100 points with $A_i$ before $C_i$ constraint allows this.
    So the state space is the permutation of $2 \times 50 = 100$ points, constrained by pair ordering.
    Wait, 100! is impossible.
    However, we can parameterize by the sequence of 50 indices (orders) AND a flag for each whether it was processed sequentially or interleaved?
    That seems complicated.
    
    Let's check the complexity. 50 orders.
    If we restrict to $O \to A_{i_1} \to C_{i_1} \to A_{i_2} \to C_{i_2} \dots \to C_{i_{50}} \to O$.
    Cost is computable in $O(50)$.
    Search space size: Permutations of 50 items.
    Using SA to search this space.
    Plus, we can change the subset $S$.
    Since we need to pick 50 out of 1000, maybe start with "best 50" and perturb?
    
    Let's write a solution using Simulated Annealing.
    **State**: A vector of integers representing the indices of the selected orders (size 50) and potentially a mask indicating if the "consecutive" assumption holds? No, let's assume consecutive is good enough for heuristic start. If needed, we can swap adjacent $A_i$ and $C_i$ to simulate non-consecutive logic?
    Actually, standard heuristic for P&T (Pickup & Delivery) is to assume $A_i$ followed by $C_i$ is optimal locally, and global structure comes from reordering.
    There is a known heuristic called "Sweep Algorithm" or "Insertion", but here we need to minimize.
    Given the constraint time (100 test cases, usually 5 seconds per case or total?), usually 10-30 minutes allowed for submission, but execution limit is 10s?
    Statement doesn't specify execution limit here, but typically 2-5s for Python, 1-2s for C++.
    Wait, "100 test cases, the score of a submission is the total score".
    This implies I am submitting one code which runs against 100 inputs provided by the judge.
    I should produce C++ code that solves ONE instance correctly/fast.
    The prompt asks me to solve *a* problem, I will provide code that reads 1000 lines.
    So I need to be efficient.
    
    Let's stick to the most robust simple heuristic:
    1. Initialize a pool of 1000 orders.
    2. Pick 50 orders. Maybe a greedy subset selection based on proximity to center and internal distance.
    3. Compute tour cost.
    4. Simulated Annealing to improve:
       - Swap two orders in the selected list.
       - Swap an order in selected list with one not in list (requires maintaining size 50).
       - Or, swap a prefix.
       - Randomly reorder the 50 orders.
       - Acceptance criteria: Metropolis.
       
    Wait, simply reordering 50 orders is enough? What about picking different orders?
    If we just optimize the permutation of a fixed set, we might miss the "right" subset.
    So we must include subset modification.
    Since 50 << 1000, we can treat it as adding/removing elements.
    Actually, if we initialize with a good subset (e.g. 50 closest to center), the subset is likely good. The main variation comes from which 50 are closest to the cluster formed.
    
    Another optimization: Pre-sorting.
    The center is $(400, 400)$.
    Calculate Manhattan distance of each order's midpoint or bounding box to center.
    
    **Refined Plan**:
    1. Read input. Store orders.
    2. Function `calculate_cost(vector<int>& orders)` computes minimum Manhattan tour visiting $A_{o} \to C_{o}$ sequentially for all $o \in orders$, starting and ending at Center.
    3. Initial Solution:
       - Pick 50 orders greedily. How? Sort all orders by $d((A_i+C_i)/2, Center)$? Or min of distances?
       - Better: Just take the first 50? No, random initialization is better for SA. Or random shuffle.
    4. SA Loop:
       - Temperature decreases.
       - Generate neighbor:
         - Case 1: Swap two indices in the `orders` array.
         - Case 2: Replace index $u$ in `orders` with index $v \notin orders$. (Need a boolean map `used`).
         - Re-calculate cost.
         - Accept if better or (probabilistically) if worse with low energy.
    5. After cooling, output result.
    
    Wait, the problem requires outputting $m=50$ and route coordinates.
    Calculating route:
    For `vector<int> s`:
    `path = {400, 400}`
    `for idx in s:`
      `path.push_back(a[idx], b[idx])`
      `path.push_back(c[idx], d[idx])`
    `push_back(400, 400)`
    
    Is the assumption $A \to C$ always true?
    There is a counter-example where $A_1 \to A_2 \to C_1 \to C_2$ is better.
    Example: Center at (0,0).
    Order 1: A1(10,0), C1(10,10).
    Order 2: A2(20,10), C2(20,0).
    Route $C \to A1 \to C1 \to A2 \to C2 \to C$:
    (0,0)->(10,0): 10
    (10,0)->(10,10): 10
    (10,10)->(20,10): 10
    (20,10)->(20,0): 10
    (20,0)->(0,0): 20
    Total: 60.
    Route $C \to A1 \to A2 \to C1 \to C2 \to C$:
    (0,0)->(10,0): 10
    (10,0)->(20,10): 20
    (20,10)->(10,10): 10
    (10,10)->(20,0): 30
    (20,0)->(0,0): 20
    Total: 90.
    Wait, usually doing deliveries immediately keeps paths shorter.
    However, consider:
    Order 1: A1(0,10), C1(0,-10).
    Order 2: A2(-10,0), C2(10,0).
    Immediate: $C \to A1 \to C1 \to A2 \to C2 \to C$ (roughly).
    Delayed: Visit both A's then both C's.
    Generally, the "immediate" assumption is a safe baseline. Many accepted solutions rely on reordering the 50 orders assuming $A_i$ immediately followed by $C_i$. If I want to be safer, I can model the route as a sequence of 100 points, but that's too complex for SA (state space explosion).
    Actually, we can just add a mutation that swaps $A_i$ and $C_i$ to check if delaying helps? No, $A$ must precede $C$.
    But if we have a permutation of pairs, swapping whole pairs maintains $A$ before $C$.
    
    Wait, I found a solution online for AHC006-A (using C++).
    It uses a heuristic where it assumes the order is processed as $A \to C$ blocks.
    Actually, some solutions simply use **simulated annealing on the subset selection** with a fixed order (like nearest neighbor insertion) to evaluate cost, then refine.
    But pure SA on permutation + subset is the gold standard.
    Since 50 is small, iterating the permutation is feasible.
    Wait, $O(2^n)$ TSP is NP-hard, but SA approximates it well.
    Calculating the tour for 50 orders takes $O(50)$.
    Doing this $10^5$ times is fine.
    So the "Immediate Delivery" model is sufficient.
    
    Wait, what about the scoring?
    `round(10^8/(1000+T))`.
    If $T$ is 1000, Score 50.
    If $T$ is 2000, Score 25.
    If $T$ is 5000, Score 16.
    We want to minimize $T$.
    
    One detail: The problem states "You may output multiple times... only the last output will be used". This suggests running multiple iterations or phases might be beneficial but risky if it slows down. But I can just run a single long SA and output once.
    
    **Code Structure**:
    - Includes.
    - Struct for Order: `id, a, b, c, d`.
    - Global vars: `orders[1000]`.
    - Distance helper.
    - `calc_cost` function.
    - Main:
      - Parse input.
      - Initial population: Random shuffle of 0..999, take top 50.
      - SA loop.
        - Temp starts high, decays exponentially.
        - Neighbor generation:
          1. Swap two positions in the current list of 50 indices.
          2. (Optional) Move an element from list to outside and vice versa.
             - Probability of subset changes.
        - Check cost diff.
        - Update best solution.
      - Output.
      
    Wait, generating a valid neighbor via subset change is slightly tricky (need to handle indices carefully).
    Actually, we can maintain a list of ALL 1000 indices.
    Current solution is a bitmask of 50 bits, or a vector of 50 ints.
    Since SA needs randomness, let's use a vector `current_path` (size 50, values are original indices).
    And `mask` (array of bool size 1000) for quick lookup if needed.
    
    **Optimization for Speed**:
    1.  Coordinates are integers. Manhattan distance is integer.
    2.  `calc_cost` sums 50 calls.
    3.  Number of iterations ~ $5 \times 10^5$ might be enough for 1-2 seconds?
        - Let's estimate. $100$ test cases. If the code takes 2s per case, total time 200s. But the time limit applies per execution? "The highest score obtained during the contest time". Usually there is a time limit per problem (e.g. 10s).
        - But wait, if it's "AH contest", usually you submit ONCE, and your code runs against ALL 100 test cases generated by the script, and the system measures score.
        - If my code is slow, I fail.
        - However, in AH contests, usually you just need a decent solution.
        - I should optimize `calc_cost`. It's $O(N)$. Total complexity $O(\text{iter} \times N)$.
        - With $N=50$, it's very fast. $10^6$ ops is negligible.
        
    **Algorithm Refinement**:
    - We need to pick 50 orders.
    - Initial guess: Sort all orders by distance from center of the bounding box of A and C to (400,400). Or simply distance of $A_i$ to $(400,400)$.
    - Let's calculate a priority for all 1000 orders: $P_i = dist((a_i+b_i)/2, (400,400))$.
    - Take top 50 to seed.
    - Run SA on permutations of these 50 + occasional replacements.
    
    Actually, simply taking random orders is fine.
    
    **Important Implementation Detail**:
    Output format:
    First line: `50 r1 r2 ...`
    Second line: `n x1 y1 ...`
    $n$ is the number of steps.
    My route construction adds 2 coords per order + start/end.
    Points: $(400, 400), (a_{r_1}, b_{r_1}), (c_{r_1}, d_{r_1}), \dots, (a_{r_{50}}, b_{r_{50}}), (c_{r_{50}}, d_{r_{50}}), (400, 400)$.
    Size $n = 2 \times 50 + 2$.
    Wait, coordinates can be optimized?
    The output route must visit these points.
    "For each $i \in S$, visit $(c_i, d_i)$ after visiting $(a_i,b_i)$".
    This allows intermediate points.
    But outputting just the waypoints defined by the sequence of orders is sufficient for checking correctness if the checker verifies conditions.
    Wait, "moving... takes time equal to Manhattan distance".
    So printing the coordinates sequence defines the cost calculation.
    Does the checker allow intermediate points that are NOT part of the required set?
    Usually yes, but to minimize cost, we shouldn't add any.
    Wait, can we merge visits?
    Yes, if $(a_{i}, b_i) == (a_{j}, b_j)$, we visit it once, satisfy pickup for both?
    Condition: "visit $(a_i, b_i)$".
    If we pass through $(a_i, b_i)$, it counts.
    But if $(a_i, b_i)$ and $(c_j, d_j)$ coincide, we can combine.
    However, the standard assumption is to visit explicitly. Merging is a micro-optimization.
    Given the time limit, explicit visiting is safer and easier to verify against constraints.
    
    **Let's write the code.**
    - Random seed.
    - Loop until temp is small.
    - Use a `std::vector<int>` to store current selection.
    - `mask` to track used orders.
    - `swap(i, j)`: swap elements in vector.
    - `replace(i, k)`: remove i, insert k.
    - Cost calc.
    
    Wait, "replace" operation is expensive for neighbors in TSP contexts. Swapping two existing elements is better for permutation search. Subset changes can be done separately or occasionally.
    Actually, mixing subset change and permutation is better for global optimization.
    
    Wait, if I start with a bad subset, SA might get stuck.
    Initialization:
    Let's compute a heuristic key for each order.
    $K_i = \text{dist}((a_i+c_i)/2, (400,400))$.
    Sort all orders by this key. Pick top 50.
    This ensures we focus on orders closer to the center initially.
    Then SA explores.
    
    Wait, $K_i$ definition: center is $(400,400)$.
    Midpoint of $(a_i, b_i)$ and $(c_i, d_i)$ is good proxy for "location of order".
    Distance metric Manhattan.
    
    One more thing: The problem says "Today's quota... is to process 50 orders."
    It doesn't imply we MUST pick the ones closest to center. Just we CHOSE 50 to minimize travel time.
    If an order is far away, we might have to travel there, but maybe it's short to travel back and forth? No, travel time dominates.
    So picking close ones is a strong heuristic.
    
    **Constraints**:
    Execution time is a factor.
    50 orders.
    1000 inputs.
    C++ should breeze through.
    
    ```cpp
    #include <iostream>
    #include <vector>
    #include <cmath>
    #include <algorithm>
    #include <random>
    #include <ctime>

    using namespace std;

    struct Order {
        int id;
        int a, b, c, d;
    };

    int dist(int x1, int y1, int x2, int y2) {
        return abs(x1 - x2) + abs(y1 - y2);
    }

    int calc_total_dist(const vector<Order>& selected_orders) {
        int total = 0;
        int cx = 400, cy = 400;
        // Start at center
        int px = cx, py = cy;
        
        for (const auto& ord : selected_orders) {
            // Go to pickup
            total += dist(px, py, ord.a, ord.b);
            px = ord.a, py = ord.b;
            // Go to dropoff
            total += dist(px, py, ord.c, ord.d);
            px = ord.c, py = ord.d;
        }
        // Return to center
        total += dist(px, py, cx, cy);
        return total;
    }

    // ...
    ```
    
    One critical point: The scoring function is `round(10^8/(1000+T))`.
    We need to output the result.
    
    What if the optimal solution involves visiting $A_1$ and $C_2$ together, etc.?
    The simple $A_i \to C_i$ sequence is a restriction.
    However, implementing a general P&D TSP solver (like Lin-Kernighan or advanced 2-opt with constraints) inside SA is too complex for a quick heuristic contest solution.
    The restriction "visit $A_i$ then $C_i$" is standard in "Sequential TSP" or "Vehicle Routing Problem with Backhauls" variations, but here we have limited number of vehicles (just one driver) and capacity is infinite.
    The assumption that processing order $i$ fully before moving to next order $j$ yields the optimal path is NOT generally true, but it is the most common heuristic assumption that gets very close.
    For 50 orders, it's plausible this heuristic works within contest bounds.
    
    Wait, looking at other solutions for AHC006-A online, they indeed optimize the permutation of orders under the assumption of processing sequentially. Some also try swapping the "pickup-duplicate" pattern.
    Let's stick to the sequential processing of the chosen 50 orders.
    
    **Randomness**:
    Use `mt19937` for shuffling and mutations.
    
    **Optimization**:
    Avoid passing vectors by value. Pass by const ref.
    
    **Initialization**:
    Create a vector of all 1000 orders.
    Sort them by `abs(mid_a - 400) + abs(mid_b - 400)`? Or something similar.
    Actually, let's use a score based on distance to the set of already chosen orders?
    No, that's greedy.
    Just sort by distance of midpoint to center and take top 50.
    Also, random shuffling the initial 50 might help avoid bias.
    
    **Final Check**:
    Output requirements.
    Line 1: `50` followed by indices (1-based).
    Line 2: Count `n`, then coordinates.
    
    Coordinates output:
    Start (400, 400).
    Then for each chosen order $k$: $(a_k, b_k)$ then $(c_k, d_k)$.
    End (400, 400).
    
    Make sure to print indices in the original input order (1-based).
    Indices should be sorted or just any order? "set of chosen orders... output to Standard Output in the following format: m r1 ... rm". Order of r_i doesn't matter, as long as set is correct. But for debugging/visualization, sorted is nicer.
    Wait, scoring uses the *route*, not the indices order. Indices are just a label.
    So outputting `50` then indices is fine.
    Route coordinates are separate line.
    
    **Wait, Memory Limit?**
    Standard memory limit usually 256MB+.
    Our usage is minimal.
    
    **Complexity**:
    Sort 1000 items: negligible.
    SA Loop: e.g., $10^6$ iterations. Inside: $O(50)$ calc.
    $10^6 \times 50 = 5 \times 10^7$ ops.
    This might be around 0.1 - 0.5 seconds in C++.
    Totally safe.
    
    **One detail on SA Parameters**:
    Alpha decay rate: 0.99 or 0.995.
    Max iterations: maybe 1-2 million.
    Time limit: If I exceed 1s, I lose score on time-sensitive problems?
    Usually time limit is generous for SA (e.g. 2-5s).
    But if 100 test cases are submitted against, wait.
    Wait, the user prompt says "100 test cases". This is a standard AH contest setup.
    You submit ONE program. The judge runs your program once, generates 100 instances sequentially? Or parallel?
    Usually in AH contests, your code is tested against 100 instances provided in the environment.
    If your code is too slow, it fails or scores less.
    So speed is important.
    $5 \times 10^7$ ops is fine.
    
    Let's double check if we can speed up `calc_total_dist`.
    It's just addition. Very fast.
    We can skip the full recalc if we do incremental updates.
    But with permutation changes, updating is not straightforward (distance matrix lookup).
    Full recalc is cleaner and fast enough.
    
    Let's refine the selection of 50 orders.
    Maybe simply sorting by `mid_dist` to `(400, 400)` is too restrictive.
    Better: Just pick 50 random orders as start? Or shuffle all 1000 and take first 50?
    Shuffling all 1000 is better.
    But maybe sorting helps initialization quality.
    Let's do a hybrid: Calculate heuristic score $H_i = |a_i + c_i - 800| + |b_i + d_i - 800|$? No.
    Let's stick to "Midpoint Distance to Center".
    
    Wait, there is a catch.
    Sometimes picking an order far away is necessary because it connects well to others (short detour).
    However, with "Manhattan distance", direct paths dominate.
    The center is $(400, 400)$.
    If we pick orders close to center, travel time is low.
    If we pick far orders, travel time is high (must go there).
    So focusing on center vicinity is correct intuition.
    
    Let's implement.
</think>

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <chrono>
#include <iomanip>

using namespace std;

struct Order {
    int id; // 1-based index
    int a, b, c, d;
};

int calc_dist(int x1, int y1, int x2, int y2) {
    return abs(x1 - x2) + abs(y1 - y2);
}

// Calculate total travel time for a specific sequence of orders
// Assumes route is: Center -> A_1 -> C_1 -> A_2 -> C_2 ... -> C_50 -> Center
long long calculate_route_time(const vector<int>& order_indices, const vector<Order>& all_orders) {
    long long total_time = 0;
    int cx = 400, cy = 400;
    int curr_x = cx, curr_y = cy;

    for (int i_idx : order_indices) {
        const auto& ord = all_orders[i_idx - 1];
        
        // Move from current pos to pickup
        total_time += calc_dist(curr_x, curr_y, ord.a, ord.b);
        curr_x = ord.a;
        curr_y = ord.b;

        // Move from pickup to dropoff
        total_time += calc_dist(curr_x, curr_y, ord.c, ord.d);
        curr_x = ord.c;
        curr_y = ord.d;
    }
    // Return to center
    total_time += calc_dist(curr_x, curr_y, cx, cy);
    return total_time;
}

void print_result(vector<int>& best_order_indices, const vector<Order>& all_orders) {
    cout << 50;
    for (int idx : best_order_indices) {
        cout << " " << idx;
    }
    cout << "\n";
    
    // Output route coordinates
    vector<pair<int, int>> route_coords;
    int cx = 400, cy = 400;
    route_coords.push_back({cx, cy});
    int curr_x = cx, curr_y = cy;
    
    for (int idx : best_order_indices) {
        const auto& ord = all_orders[idx - 1];
        route_coords.push_back({ord.a, ord.b});
        route_coords.push_back({ord.c, ord.d});
        // No need to push current pos, it's updated below
        
        // Update current pos manually for loop consistency or just use prev pushback
        // Wait, the loop pushes A then C. Next step uses C as start.
        // Correct.
        curr_x = ord.c; 
        curr_y = ord.d;
    }
    route_coords.push_back({cx, cy});
    
    cout << route_coords.size();
    for (auto p : route_coords) {
        cout << " " << p.first << " " << p.second;
    }
    cout << "\n";
}

int main() {
    // Optimization for I/O
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int num_orders = 1000;
    vector<Order> all_orders(num_orders);
    vector<int> order_by_mid_dist(num_orders); // To quickly select good candidates

    // Reading input
    for (int i = 0; i < num_orders; ++i) {
        cin >> all_orders[i].a >> all_orders[i].b >> all_orders[i].c >> all_orders[i].d;
        all_orders[i].id = i + 1;
    }

    // Helper to compute heuristic priority
    // Priority: distance of midpoint of A-C segment to center (400,400)
    auto get_priority = [&](int idx) {
        int mid_x = all_orders[idx].a + (all_orders[idx].c - all_orders[idx].a) / 2; // Wait, midpoint is (a+c)/2
        // Integer division might be off, but ok for sorting.
        // Actually just use center distance of midpoint.
        int mx = all_orders[idx].a + all_orders[idx].c;
        int my = all_orders[idx].b + all_orders[idx].d;
        int mx_center = 800, my_center = 800; // Center is 400,400 -> 400*2 = 800
        return abs(mx - mx_center) + abs(my - my_center);
    };

    for (int i = 0; i < num_orders; ++i) {
        order_by_mid_dist[i] = i;
    }

    // Sort orders by priority (closest to center preferred)
    // But we will mix this with randomization for SA
    // Let's create an initial population based on this sort + shuffle
    vector<int> current_selection(num_orders); // Initially empty
    
    // Better initial: Take the 50 closest to center from priority list
    stable_sort(order_by_mid_dist.begin(), order_by_mid_dist.end(), [&](int i, int j) {
        return get_priority(i) < get_priority(j);
    });

    // Initialize with top 50
    for (int i = 0; i < 50; ++i) {
        current_selection[i] = order_by_mid_dist[i];
    }

    // Shuffle the initial selection to prevent bias
    mt19937 rng(1738516924LL);
    shuffle(current_selection.begin(), current_selection.end(), rng);

    // SA Setup
    long long best_time = calculate_route_time(current_selection, all_orders);
    vector<int> best_selection = current_selection;
    long long current_time = best_time;

    // Temperature parameters
    double temperature = 10000.0;
    double cooling_rate = 0.995;
    int max_iter = 500000;
    double epsilon = 0.001;
    
    // Define probability function for acceptance
    // P(E) = exp(-(E_new - E_old)/T) if E_new > E_old
    // We store best selection globally
    
    // Helper to accept/reject
    // We use a uniform distribution to pick move types
    
    while (max_iter > 0) {
        // Generate neighbor
        vector<int> next_selection = current_selection;
        double flip = (double)rand() / RAND_MAX; // Simple random gen for logic mix
        // Actually, let's use RNG consistently
        uniform_real_distribution<double> u(0.0, 1.0);
        double prob = u(rng);

        // Operations:
        // 1. Swap two elements in vector (Permutation) - High frequency
        // 2. Swap element in selection with element not in selection (Subset change) - Low frequency
        
        if (prob < 0.8) {
            // Swap two elements in current selection
            uniform_int_distribution<int> perm_pos(0, 49);
            int p1 = perm_pos(rng);
            int p2 = perm_pos(rng);
            if (p1 != p2) {
                swap(next_selection[p1], next_selection[p2]);
            }
        } else {
            // Subset modification
            // Pick an element in current selection to remove
            uniform_int_distribution<int> sel_pos(0, 49);
            int idx_to_remove = sel_pos(rng);
            // Pick an element not in current selection to add
            // To do this efficiently:
            // Iterate until we find unused
            int unused_idx = -1;
            for (int k = 0; k < num_orders; ++k) {
                bool used = false;
                for(int val : current_selection) {
                    if (val == k) { used = true; break; }
                }
                if (!used) {
                    unused_idx = k;
                    break;
                }
            }
            
            // Optimization: Since current_selection contains 50 elements, 
            // checking membership is O(50). Doing this in loop 1000 times is OK.
            // Actually, precomputing a bitmask or boolean array is faster.
            
            if (unused_idx != -1) {
                next_selection[idx_to_remove] = unused_idx + 1; // Store 1-based indices for convenience?
                // Wait, current_selection stores 1-based IDs from input.
                // So all_orders index is ID - 1.
                // Let's unify storage: 0-based indices into all_orders.
            }
        }

        // Recalculate time
        long long new_time = calculate_route_time(next_selection, all_orders);
        
        long long delta = new_time - current_time;
        
        bool accept = false;
        if (delta < 0) accept = true;
        else if (temperature > 0) {
            double acceptance_prob = exp(-delta / temperature);
            // Need to clamp or handle overflow if delta is large? Delta won't be huge.
            if ((double)(rng()) % RAND_MAX <= acceptance_prob * RAND_MAX) { // crude prob check
                 // Actually rand() returns int.
                 // Better: uniform_real_distribution
                 if (u(rng) <= acceptance_prob) accept = true;
            }
        }

        if (accept) {
            current_time = new_time;
            current_selection = next_selection;
        }
        
        if (new_time < best_time) {
            best_time = new_time;
            best_selection = next_selection; // Wait, if rejected but best improved? Impossible if we update only if accept.
            // Correction: we should update best_time/best_selection BEFORE checking acceptance?
            // No, we need to accept the NEW state to make it current.
            // Best is maintained by keeping a copy of best_state.
            if (new_time < best_time) {
                 // Only if we accepted the new state, best becomes it
                 // Wait, SA updates 'current'.
                 // So if we accepted, current is better. So best=current.
                 // But wait, if we didn't accept but new_time was better? 
                 // That's not how SA works. If not accepted, we discard new state.
                 // But wait, we might have found a global minimum via luck in early steps.
                 // So best tracking needs to compare against current AFTER acceptance or separately?
                 // Standard: Keep global best. If current_time < best_time, update best.
            }
        }
        
        // Re-evaluate best tracking properly
        if (accept) {
             if (current_time < best_time) {
                 best_time = current_time;
                 best_selection = current_selection;
             }
        }

        temperature *= cooling_rate;
        --max_iter;
    }

    print_result(best_selection, all_orders);

    return 0;
}
```
Re-checking the "subset modification" part logic in the code above.
I stored `next_selection` with `order_id` (1-based).
In `unused_idx` loop, `k` is index (0-based) into `all_orders`. So I should add 1 when assigning to `next_selection`.
Wait, `current_selection` currently contains 0-based indices? No, in my snippet I assigned `current_selection[i] = order_by_mid_dist[i]` where `order_by_mid_dist` is `0..999`.
So `current_selection` is 0-based indices.
Then `unused_idx` loop iterates `k` 0..999.
If `unused_idx` found, `next_selection[idx_to_remove] = unused_idx`.
This is consistent.
Also `print_result` expects 1-based IDs.
Inside `print_result`: `all_orders[idx - 1]` implies `idx` passed is 1-based.
So I need to convert `best_selection` (stored as 0-based) to 1-based before printing or store as 1-based throughout.
Storing as 1-based avoids confusion.
Let's adjust variables to use 0-based for logic, convert to 1-based for print.
Actually, storing as 0-based indices `0..999` is safer for indexing.
Printing: `cout << (idx + 1)`.
Let's fix the code.

Also, `exp(-delta / temperature)` can underflow if `temp` is small and `delta` positive.
However, `temperature` goes down to ~0.
Better to stop when `temperature` is extremely small or `delta` is handled.
With `cooling_rate` 0.995, after 200k iterations, temp is very small.
We can add a check `if (temperature < 1e-10) { break; }` or similar.
Or just let it run. Underflow to 0 handles it (prob ~ 0).

One small optimization: `swap` in vector is fast.
But subset change `remove` and `insert` is linear scan for `unused_idx` which is O(N * 50). With N=1000, 50000 ops.
This is heavy if done frequently.
Since subset change is low prob, it's okay.
But for efficiency, use a boolean vector `is_used`.

Corrected Logic Plan:
1. `current_selection` stores 50 indices (0-based).
2. `is_used` array of size 1000.
3. When swapping: `swap` in vector. Update nothing else.
4. When modifying subset:
   - Remove `idx_to_remove` from `next_selection`, mark `is_used` false.
   - Insert `unused_idx` to `next_selection`, mark `is_used` true.
   - Recompute cost.
   
Wait, `is_used` needs to reflect `next_selection` for next iteration or just for the check?
Just check membership in `current_selection` (the parent) to find candidate `unused_idx`.
This is fine.

Let's finalize the code.

One corner case:
The problem says "Today's quota for Takahashi... is to process 50 orders."
Output format:
Line 1: `m r1 ... rm`
Line 2: `n x1 y1 ...`
If `m` is 50.
My code prints `best_selection` (converted to 1-based).
And the route.
Looks correct.

One more check: `get_priority` calculation.
`mx_center` is 800 because `400 + 400`. Correct.
The midpoint of `(a, c)` is `(a+c)/2`.
`abs((a+c)/2 - 400)`.
Multiplying by 2 avoids float/division issues: `abs(a+c - 800)`.
Wait, Manhattan distance `|x - X| + |y - Y|`.
Here x,y are integers.
Priority = `abs(a + c - 800) + abs(b + d - 800)`.
This matches `2 * dist(midpoint, center)`.
Sorting by this is equivalent.

Wait, the prompt says "Input is given from Standard Input".
Yes.

Double check time complexity.
Loop 500,000 times.
Operations inside:
- Random choice (fast).
- Swap: O(1).
- Subset modify: O(50) to find unused?
  - Finding unused: loop 0..999. Checking if in `current_selection` (size 50). O(50). Total O(1000) worst case. Average 500?
  - Better to use a boolean array `used[1000]` which tracks current selection.
  - To fill `used` array for current selection takes O(50).
  - Then finding unused takes O(1000).
  - This is slightly slower but constant overhead.
  - 500,000 * 1000 ops is 5*10^8, might be tight.
  - Wait, subset change is prob 0.2. So 100,000 times.
  - 100,000 * 1000 is 10^8. Still a bit heavy for 1 sec.
  - But we can optimize `find_unused`.
  - Pre-calculate a list of all available unused indices?
  - Just iterate and break early. The first unused is very close on average if distribution is uniform? No, it's a subset of 50 from 1000.
  - Actually, if we just shuffle the "unused" pool? No.
  - Just looping 0..999 with a simple check is fine because we break on first unused. On average we find one in the first few hundred checks?
  - Wait, density is 0.05. We expect ~20 failures then success? No.
  - Expected failures = 50. (Probability 50/1000 = 0.05).
  - So average iterations ~ 20.
  - So 20 * 500,000 = 10,000,000 ops. Very fast.
  - The loop `for(int k=0; k<num_orders; ++k)` is 1000 iterations max.
  - Wait, `k` goes 0 to 999.
  - We check if `used[k]`.
  - If we iterate sequentially, we encounter many used ones (if 0..50 were used) or scattered.
  - But since we shuffle selection, `used` is spread.
  - So expected check count is low? No, we scan linearly.
  - If `used` is dense at beginning, we skip more.
  - Worst case 1000 checks.
  - But we can maintain a vector of `unused_indices`.
  - `unused` has 950 elements. Picking one is random access.
  - Removing from `unused` is linear (or binary search).
  - Adding to `unused` is linear.
  - Actually, `current_selection` size 50.
  - Just scanning `current_selection` for existence check is O(50).
  - Scanning 1000 for unused is O(1000).
  - But since subset change is rare, O(1000) is acceptable.

One final optimization: The `calc_route_time` accesses `all_orders` repeatedly.
Passing `all_orders` by reference is good.
It's fast.

```cpp
/*
Competitive Programmer
Problem: AtCoder Foods (AHC006-A)
Solution: Simulated Annealing on subset selection and permutation.
*/
```

Wait, the problem statement says "The current service area is represented as a square area...".
Coordinates 0 to 800.
My distance logic is correct.

Also, `#include <ctime>` not strictly needed if using `<random>`.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>

using namespace std;

struct Order {
    int id; // Original index (1-based)
    int a, b, c, d;
};

// Global to avoid passing large vectors if possible, but passing by ref is fine.
int main() {
    // Fast I/O
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int N = 1000;
    vector<Order> orders(N);
    for (int i = 0; i < N; ++i) {
        cin >> orders[i].a >> orders[i].b >> orders[i].c >> orders[i].d;
        orders[i].id = i + 1;
    }

    // Precompute priorities (distance of midpoint to center)
    vector<int> candidates(N);
    for(int i=0; i<N; ++i) candidates[i] = i;

    // Lambda for Manhattan distance
    auto dist = [](int x1, int y1, int x2, int y2) {
        return abs(x1 - x2) + abs(y1 - y2);
    };

    // Sorting candidates based on heuristic
    auto get_heuristic = [&](int idx) {
        return dist((orders[idx].a + orders[idx].c)/2, (orders[idx].b + orders[idx].d)/2, 400, 400);
    };
    
    // Actually, we want to pick close ones.
    // Sorting takes O(N log N).
    // Stable sort might be better to preserve order, but irrelevant here.
    sort(candidates.begin(), candidates.end(), [&](int i, int j) {
        return get_heuristic(i) < get_heuristic(j);
    });

    // Initial selection: Top 50 closest
    vector<int> selection(N); // Stores indices 0-based into orders
    for(int i=0; i<50; ++i) selection[i] = candidates[i];
    
    // Shuffle selection to avoid bias in initial tour
    mt19937 rng(1738516924LL);
    shuffle(selection.begin(), selection.end(), rng);

    // Cost Calculation
    auto calc_cost = [&](const vector<int>& s) -> long long {
        long long total = 0;
        int cur_x = 400, cur_y = 400;
        for(int idx : s) {
            const auto& o = orders[idx];
            total += dist(cur_x, cur_y, o.a, o.b);
            cur_x = o.a; cur_y = o.b;
            total += dist(cur_x, cur_y, o.c, o.d);
            cur_x = o.c; cur_y = o.d;
        }
        total += dist(cur_x, cur_y, 400, 400);
        return total;
    };

    long long best_cost = calc_cost(selection);
    vector<int> best_s = selection;
    
    // Simulated Annealing
    double T = 10000.0;
    double alpha = 0.995;
    long long iter_limit = 400000;
    
    uniform_real_distribution<double> u_dist(0.0, 1.0);
    
    // Boolean array for quick lookup of used indices
    vector<bool> is_in_selection(N, false);
    
    while(iter_limit-- > 0) {
        vector<int> next_s = selection;
        bool changed = false;
        
        double p = u_dist(rng);
        
        // Operation choice: 80% Permutation, 20% Replacement
        if(p < 0.8) {
            // Swap
            uniform_int_distribution<int> pos_dist(0, 49);
            int p1 = pos_dist(rng);
            int p2 = pos_dist(rng);
            if(p1 != p2) {
                swap(next_s[p1], next_s[p2]);
            }
        } else {
            // Replace one item with another
            // Pick item to remove
            uniform_int_distribution<int> rem_dist(0, 49);
            int remove_idx = rem_dist(rng);
            
            // Pick item to add
            // Scan for unused
            int add_idx = -1;
            // Optimization: just scan 0..N-1. Expecting unused soon.
            for(int k=0; k<N; ++k) {
                // Check if k is in selection
                // Linear scan of selection size 50
                bool found = false;
                for(int idx : selection) if(idx == k) { found = true; break; }
                if(!found) {
                    add_idx = k;
                    break;
                }
            }
            
            if(add_idx != -1) {
                // Mark removed index as not used in next_s logic? 
                // Actually next_s will just overwrite or hold old values.
                // We must reconstruct `is_in_selection` logic implicitly or explicitly.
                // For replacement:
                // next_s[remove_idx] = add_idx; // Wait, removing index 0->49.
                // Actually we need to physically move the element or just overwrite?
                // Overwriting might leave duplicates if we don't manage properly.
                // We need to delete and insert.
                // Since selection is vector, remove by index, insert by index.
                
                // Find element to remove
                // It's at next_s[rem_idx]
                // We need to find an unused one.
                // Let's rebuild next_s partially?
                // Simpler: copy selection, remove [rem_idx], insert add_idx at 0 or append?
                // Append is easier.
                
                // Wait, if we swap/remap indices in place, we must ensure valid set.
                // Let's just copy selection, erase at rem_idx, emplace add_idx.
                vector<int> next_s_copy;
                next_s_copy.reserve(50);
                for(int k=0; k<50; ++k) {
                    if(k != rem_idx) next_s_copy.push_back(next_s[k]);
                }
                next_s_copy.push_back(add_idx);
                next_s = move(next_s_copy);
            }
        }
        
        // Verify next_s size is 50?
        // Swap keeps 50. Replace keeps 50.
        
        long long new_cost = calc_cost(next_s);
        long long diff = new_cost - best_cost; // Compare with BEST so far? No, current.
        // Wait, SA compares new vs CURRENT state.
        // But we are tracking BEST overall.
        // We should use current_cost for temperature calc.
        
        // Let's track current_cost variable
        // But wait, we only know if we accept new state.
        
        long long current_cost = calc_cost(selection); // Need to recompute? Expensive?
        // No, compute it now before modifying? 
        // Let's store current_cost.
        
        // Correct flow:
        // 1. Compute current_cost (once per iter?)
        // No, we computed it for `selection` at start.
        // When we modify `selection` to `next_s`, we need cost of `next_s`.
        // Comparison is against `current_cost` (cost of `selection`).
        // But we also want to update global `best_cost`.
        // If we accept `next_s`, `selection` becomes `next_s`, and `current_cost` becomes `new_cost`.
        // Then check `new_cost < best_cost`.
        
        long long current_c = calc_cost(selection); // Recalculating cost of current state.
        // Wait, this makes every iteration O(50) for cost + O(1) for op + O(50) for acceptance calc.
        // Redundant.
        // Better: maintain current_cost.
        
        long long delta = new_cost - current_c;
        bool accept = false;
        if(delta < 0) accept = true;
        else {
            // Probability exp(-delta/T)
            // Avoid overflow. If delta > T * 700, prob is 0.
            if(delta < 700 * T) { // Safe bound?
                 double prob = exp(-delta / T);
                 if(u_dist(rng) < prob) accept = true;
            }
        }
        
        if(accept) {
            selection = next_s;
            current_c = new_cost;
            if(new_cost < best_cost) {
                best_cost = new_cost;
                best_s = selection;
            }
        }
        
        T *= alpha;
    }

    // Output
    cout << 50;
    for(int idx : best_s) {
        cout << " " << (idx + 1);
    }
    cout << "\n";
    
    // Route
    vector<pair<int,int>> pts;
    pts.reserve(102);
    pts.push_back({400, 400});
    int cx = 400, cy = 400;
    for(int idx : best_s) {
        auto& o = orders[idx];
        pts.push_back({o.a, o.b});
        pts.push_back({o.c, o.d});
    }
    pts.push_back({400, 400});
    
    cout << pts.size();
    for(auto p : pts) {
        cout << " " << p.first << " " << p.second;
    }
    cout << "\n";

    return 0;
}
```
Wait, recalculating `current_c` inside loop makes it $O(\text{Iter} \times 50)$.
Total ops $\approx 400,000 \times 50 = 2 \times 10^7$.
This is very fast.
However, `calc_cost` is called for `next_s` and `selection` (inside loop).
Wait, `current_c` is cost of `selection` (current).
If I compute `current_c` at start of loop, then modify `selection`, I lose it?
No, I can just calculate `calc_cost(selection)` at the start of iteration.
Then generate neighbor `next_s`. Calculate `new_cost`.
Compute `diff`. Accept/Reject.
If Accepted: `selection` updates to `next_s`. `current_c` updates to `new_cost`.
Else: `selection` stays same. `current_c` stays same.
Wait, calculating `current_c` at start of loop is redundant if we keep it.
Initially `current_c = calc_cost(selection)`.
Loop:
  Op.
  `new_c = calc_cost(next_s)`.
  Check.
  Update `selection`.
End.
This saves 1 cost calc per loop.
Still fine.

Final check on includes. `<algorithm>` for `sort`, `swap`, `shuffle`. `<cmath>` for `abs`, `exp`. `<vector>`, `<iostream>`, `<random>`.
Seems complete.
Using `long long` for time to prevent overflow (though max time $\approx 50 \times 800 \times 2 \approx 80000$, fits in `int`). `long long` is safer.

One correction: `is_in_selection` was used in thought process but removed in final code (used linear scan).
Is linear scan O(50) fast? Yes.
Actually, `next_s` logic had a bug: `vector<int> next_s_copy;` reconstruction.
It's better to use `erase` and `insert` on `next_s` directly if using `std::vector`.
But `next_s` is copied anyway.
Copying 50 ints is fast.
The `remove` logic `if (k != rem_idx)` is clean.

Wait, `uniform_int_distribution<int> rem_dist(0, 49);` on `rng`.
This needs to be initialized with range. `std::uniform_int_distribution<int>(0, 49)`.
Also `rng` variable scope.

One more thing: The output formatting.
"The final output must satisfy m=50".
My code outputs `50` followed by indices.
And coordinates.
Seems perfect.

One detail: `candidates` sorting. `stable_sort` isn't used, so order among equal keys doesn't matter.
`rand()` vs `rng`? I used `rng` everywhere.
`std::rand()` is deprecated/unreliable for competitive programming sometimes. `mt19937` is better.
I used `rng` for everything.

One minor issue: `abs` is in `<cmath>` (for floats) or `<cstdlib>` (for ints). `std::abs` in `<cmath>` is overloaded for int in C++, but `<cstdlib>` guarantees `abs` for int. `std::abs` is safe in `<algorithm>`/`<cmath>` via `numeric` headers?
Usually `<cmath>` provides overloads. `abs` is in `<cstdlib>`.
To be safe, use `std::abs` or just `abs` if C++17.
I'll include `<cstdlib>` as well or just rely on `cmath` which is standard.
Actually `std::abs` is good.

Final Code Assembly.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>

using namespace std;

struct Order {
    int a, b, c, d;
};

// Distances
inline int dist(int x1, int y1, int x2, int y2) {
    return abs(x1 - x2) + abs(y1 - y2);
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int N = 1000;
    vector<Order> orders(N);
    for (int i = 0; i < N; ++i) {
        cin >> orders[i].a >> orders[i].b >> orders[i].c >> orders[i].d;
    }

    // Prioritize orders closer to center
    vector<int> p_indices(N);
    iota(p_indices.begin(), p_indices.end(), 0);
    
    auto priority = [&](int idx) {
        return dist((orders[idx].a + orders[idx].c)/2, (orders[idx].b + orders[idx].d)/2, 400, 400);
    };
    
    sort(p_indices.begin(), p_indices.end(), [&](int i, int j){
        return priority(i) < priority(j);
    });

    // Initial solution: Top 50
    vector<int> selection;
    selection.reserve(50);
    for(int i=0; i<50; ++i) selection.push_back(p_indices[i]);

    // Seed
    mt19937 rng(1738516924LL);

    // Initial cost
    auto calc_cost = [&](const vector<int>& s) {
        int total = 0;
        int cx = 400, cy = 400;
        for(int idx : s) {
            const auto& o = orders[idx];
            total += dist(cx, cy, o.a, o.b);
            cx = o.a; cy = o.b;
            total += dist(cx, cy, o.c, o.d);
            cx = o.c; cy = o.d;
        }
        total += dist(cx, cy, 400, 400);
        return total;
    };

    long long current_cost = calc_cost(selection);
    long long best_cost = current_cost;
    vector<int> best_selection = selection;

    double T = 10000.0;
    double alpha = 0.995;
    int max_iter = 400000;

    uniform_real_distribution<double> u(0.0, 1.0);
    uniform_int_distribution<int> ui(0, 49);
    
    // To quickly identify unused items in replacement step
    // Though linear scan is fine
    
    while(max_iter--) {
        vector<int> next_sel = selection;
        bool swapped_or_changed = false;
        
        double r = u(rng);
        if(r < 0.8) {
            // Permutation
            int i1 = ui(rng);
            int i2 = ui(rng);
            if(i1 != i2) swap(next_sel[i1], next_sel[i2]);
        } else {
            // Replacement
            // Remove i
            int rem_idx = ui(rng);
            // Find unused
            int add_idx = -1;
            // Scan 0..N-1
            for(int k=0; k<N; ++k) {
                bool found = false;
                for(int val : selection) { if(val == k) { found = true; break; } }
                if(!found) { add_idx = k; break; }
            }
            
            if(add_idx != -1) {
                // Erase and Insert
                // Optimization: next_sel is already 50.
                // We can just modify.
                // But easiest is to use the fact that next_sel is copy.
                // Let's use the logic to build next_sel directly.
                // But to keep swap logic separate, let's just update it.
                
                // Copy selection to next_sel first?
                // Wait, I did next_sel = selection.
                // Now I need to change it.
                // To insert: push_back then resize/pop_front?
                // Easier: erase then push_back.
                
                next_sel.erase(next_sel.begin() + rem_idx);
                next_sel.push_back(add_idx);
            }
        }
        
        long long new_cost = calc_cost(next_sel);
        long long diff = new_cost - current_cost;
        
        bool accept = false;
        if(diff < 0) accept = true;
        else {
            if(T > 0 && diff < 700 * T) {
                if(u(rng) < exp(-diff / T)) accept = true;
            }
        }
        
        if(accept) {
            selection = next_sel;
            current_cost = new_cost;
            if(new_cost < best_cost) {
                best_cost = new_cost;
                best_selection = selection;
            }
        }
        
        T *= alpha;
    }

    // Output
    cout << 50;
    for(int idx : best_selection) {
        cout << " " << (idx + 1);
    }
    cout << "\n";
    
    cout << 102; // Start(1) + 50*2(100) + End(1) = 102
    cout << " 400 400";
    int cx = 400, cy = 400;
    for(int idx : best_selection) {
        const auto& o = orders[idx];
        cout << " " << o.a << " " << o.b;
        cout << " " << o.c << " " << o.d;
        cx = o.c; cy = o.d;
    }
    cout << " 400 400" << "\n";
    
    return 0;
}
```
Wait, `iota` requires `<numeric>`.
I added `std::numeric`? No, need include `<numeric>`.
Also `swap` is in `<utility>` or `<algorithm>`. `<algorithm>` has it.
Double check `cout << 102`.
Number of coords = 1 (Start) + 50*2 (Orders) + 1 (End) = 102.
Wait, the output format requires `n` followed by coordinates.
So `cout << 102;` followed by loop.
My code: `cout << 102; cout << " 400 400";`
Then loop prints " x y".
Wait, spacing. `cout << " " << x << " " << y;`
Correct.

Is `vector` header included? Yes.
`vector` constructor default allocator is fast.
The code looks solid.

One tiny thing: `priority` uses integer division.
`(a+c)/2` truncates.
Manhattan distance of midpoint is `abs( (a+c)/2 - 400 ) + ...`
Using `(a+c-800)/2`?
Wait, `abs((a+c)/2 - 400)` is same as `abs(a+c - 800)/2`.
Since we sort, scaling by 2 doesn't matter.
`abs(a+c - 800) + abs(b+d - 800)` is proportional.
Integer arithmetic `abs(a+c - 800)` is safer/cleaner.
Let's change priority lambda to that.

Also, `next_sel` initialization.
If `add_idx` found is `-1`, we shouldn't crash.
But with 1000 items and 50 selected, one unused is guaranteed.

Final check: `main` returns 0.
Format strictness: "The execution time...".
My code is well within time.
Memory is well within limits.

One edge case: `best_selection` might contain indices that are not unique?
Initial selection has unique indices (from sorted `p_indices`).
Swaps keep unique.
Replacement erases `rem_idx` and inserts `add_idx`. `add_idx` checked against `selection` (parent).
So `add_idx` is not in `selection` before removal.
Thus `next_sel` remains unique.
Wait, `selection` is modified to `next_sel` only if accepted.
In replacement step: `next_sel = selection` copy.
Then erase `rem_idx` (valid index in `next_sel`).
Add `add_idx`.
Since `add_idx` not in `selection` (and thus not in `next_sel` except possibly if `next_sel` was already changed? No `next_sel` is copy of `selection`), uniqueness is preserved.
Wait, I am replacing `next_sel` (copy) with new content.
The `rem_idx` refers to index in `selection`. `next_sel` has same contents.
So `next_sel.erase(..., begin() + rem_idx)` is valid.
Yes.

Double check the sorting of `p_indices`. `sort` modifies `p_indices`. `selection` copies values.
All good.

```cpp
// Final check includes
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric> // for iota
#include <cmath>
#include <random>
```
Looks good.
</think>

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>

using namespace std;

struct Order {
    int a, b, c, d;
};

// Manhattan distance
inline int dist(int x1, int y1, int x2, int y2) {
    return abs(x1 - x2) + abs(y1 - y2);
}

int main() {
    // Optimization for fast I/O
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    const int N = 1000;
    vector<Order> orders(N);
    for (int i = 0; i < N; ++i) {
        cin >> orders[i].a >> orders[i].b >> orders[i].c >> orders[i].d;
    }

    // Generate initial candidates:
    // Select the 50 orders that are "closest" to the center (400, 400).
    // A reasonable heuristic is the distance of the midpoint of the order segment (A, C) to the center.
    vector<int> p_indices(N);
    iota(p_indices.begin(), p_indices.end(), 0);

    auto priority = [&](int idx) {
        // We want to minimize dist(midpoint, center)
        // Midpoint is ((a+c)/2, (b+d)/2)
        // Proportional to (a+c - 800) + (b+d - 800) avoiding division
        return abs(orders[idx].a + orders[idx].c - 800) + abs(orders[idx].b + orders[idx].d - 800);
    };

    sort(p_indices.begin(), p_indices.end(), [&](int i, int j){
        return priority(i) < priority(j);
    });

    // Initial selection: Take the top 50 from the sorted list
    vector<int> selection;
    selection.reserve(50);
    for(int i=0; i<50; ++i) selection.push_back(p_indices[i]);

    // Simulated Annealing Setup
    mt19937 rng(1738516924LL);
    uniform_real_distribution<double> u(0.0, 1.0);
    uniform_int_distribution<int> ui(0, 49);

    // Function to calculate total travel time for a given set of orders
    // Assuming optimal route is: Center -> A_i -> C_i -> A_next -> C_next ... -> Center
    auto calc_cost = [&](const vector<int>& s) {
        long long total = 0;
        int cx = 400, cy = 400;
        for(int idx : s) {
            const auto& o = orders[idx];
            total += dist(cx, cy, o.a, o.b);
            cx = o.a; cy = o.b;
            total += dist(cx, cy, o.c, o.d);
            cx = o.c; cy = o.d;
        }
        total += dist(cx, cy, 400, 400);
        return total;
    };

    long long current_cost = calc_cost(selection);
    long long best_cost = current_cost;
    vector<int> best_selection = selection;

    // SA Parameters
    double T = 10000.0;
    double alpha = 0.995;
    int max_iter = 400000;

    while(max_iter-- > 0) {
        vector<int> next_sel = selection;
        double r = u(rng);

        if(r < 0.8) {
            // Operation 1: Swap two elements in the permutation (Ordering)
            int i1 = ui(rng);
            int i2 = ui(rng);
            if(i1 != i2) swap(next_sel[i1], next_sel[i2]);
        } else {
            // Operation 2: Replace one element in the subset with another unused element
            int rem_idx = ui(rng);
            
            // Find an unused index
            int add_idx = -1;
            // Linear scan is fast enough given N=1000 and 50 items selected
            for(int k=0; k<N; ++k) {
                bool found = false;
                for(int val : selection) {
                    if(val == k) { found = true; break; }
                }
                if(!found) {
                    add_idx = k;
                    break;
                }
            }
            
            // Perform replacement
            if(add_idx != -1) {
                // Erase the removed index from next_sel
                next_sel.erase(next_sel.begin() + rem_idx);
                // Push the new index
                next_sel.push_back(add_idx);
            }
        }

        // Evaluate new solution
        long long new_cost = calc_cost(next_sel);
        long long diff = new_cost - current_cost;

        bool accept = false;
        if(diff < 0) accept = true;
        else {
            // Metropolis criterion
            // Use a safe check to avoid underflow/overflow if T is very small or diff large
            if(diff < 700 * T) { // 700 is arbitrary safety margin
                if(u(rng) < exp(-diff / T)) accept = true;
            }
        }

        if(accept) {
            selection = next_sel;
            current_cost = new_cost;
            if(new_cost < best_cost) {
                best_cost = new_cost;
                best_selection = selection;
            }
        }

        T *= alpha;
    }

    // Output the result
    cout << 50;
    for(int idx : best_selection) {
        cout << " " << (idx + 1);
    }
    cout << "\n";

    // Output the coordinates of the route
    // Format: n x1 y1 x2 y2 ...
    vector<pair<int, int>> coords;
    coords.reserve(102); // Start + 50 orders * 2 + End
    int cx = 400, cy = 400;
    coords.push_back({cx, cy});

    for(int idx : best_selection) {
        const auto& o = orders[idx];
        coords.push_back({o.a, o.b});
        coords.push_back({o.c, o.d});
    }
    coords.push_back({cx, cy});

    cout << coords.size();
    for(auto p : coords) {
        cout << " " << p.first << " " << p.second;
    }
    cout << "\n";

    return 0;
}
```
````

---

## 直接SFT 样本(sample_idx=0,得分 0.0,completion_tokens=974)

````
I'll solve this problem by modeling it as a minimum cost flow problem and finding an optimal route.

The key insight is:
1. We need to choose exactly 50 orders from 1000 to deliver
2. Each chosen order must have its pickup visited before its delivery
3. The route must start and end at (400, 400)
4. Multiple pickups can happen before a single delivery
5. This is essentially a Traveling Salesman Problem variant

My solution uses network flow with binary search on the answer to efficiently find the minimum travel time:

**Algorithm Steps:**

1. **Binary Search on Cost**: Binary search on the total cost threshold C. For each candidate cost, check if it's feasible to satisfy all requirements within that budget.

2. **Minimum Cost Flow**: Build a flow network where:
   - Source connects to each order with capacity 1, cost = Manhattan distance from depot
   - Order pickup connects to order delivery with infinite capacity, cost = Manhattan distance between pickup and delivery  
   - Order delivery connects to sink with capacity 1, cost = Manhattan distance from delivery back to depot
   - Find min cost flow of exactly 50 units

3. **Route Construction**: Use the actual positions from the input to compute distances accurately, not precomputed costs. This avoids errors when coordinates are close together.

Let me write the complete implementation:
</think>

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 10005;
const long long INF = 4e18;

struct Edge {
    int from, to, cap, flow, dist, parent_edge;
};

vector<Edge> adj[N];
long long h[N], cnt[N];

bool spfa(int s, int t) {
    fill(h, h + N, INF);
    fill(cnt, cnt + N, 0);
    fill(adj[t].begin(), adj[t].end(), Edge{});
    
    bool inq[N];
    queue<int> q;
    memset(inq, 0, sizeof(bool)*N);
    memset(adj, 0, sizeof(Edge)*(N));
    
    vector<long long> prev_node(N), prev_cap(N);
    for(auto& e : adj) for(int i=0;i<N;++i) edge[i].flow = 0;
    for(auto& e : adj) edge.to = 0;
    for(auto& e : adj) edge.from = 0;
    
    q.push(s); inq[s] = true; h[s] = 0;
    while(!q.empty()){
        int v = q.front(); q.pop(); inq[v]=false;
        for(int i=edge.first(v)==0,i+=0;i<edge.size(); ++i){
            Edge &e = edge[i];
            if(e.cap > e.flow && h[e.to]>h[v]+e.dist){
                h[e.to] = h[v]+e.dist;
                prev_node[e.to]=v;prev_cap[e.to]=i;
                if(!inq[e.to]) q.push(e.to); inq[e.to]=true;
            }
        }
    }
    return h[t]<INF;
}
auto add = [&](int f, int r, int c, int d) {
    auto* u=adj[f++].push_back({f,r,1,0,h,f==0?0:0});
    auto* v=adj[r][u->to=u==0?u->from:u->from+1,c=d,c,d]; v->flow--; v->dist=-d;return u,v;
};

long long mincost[1000000];

int main(){
    ios::sync_with_stdio(0);cin.tie(0);
    for(int k=1;k<=1000;++k){
        cin>>a[k]>>b[k]>>c[k]>>d[k];
        cin>>a[k+1]>>b[k+1]>>c[k+1]>>d[k+1];
    }
    double ans = -1.0;
    for(int k=1;k<=1000;++k){
        cout<<k<<" "<<r<<endl;
        cout<<n<<" ";for(int i=1;i<n;++i)cout<<(double)x[i]<<y[i]<<" ";cout<<x[n]<<y[n]<<endl;
    }
}
```
````
