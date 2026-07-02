# 题 181 — base均分 7.4 vs 直接SFT均分 0.0(掉点 7.4)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=1,得分 33.5,completion_tokens=24233)

````
The problem asks us to solve a binary Quadratic Assignment Problem (QAP). We are given:
- $n \times n$ distance matrix $D$ where $D[i][j]$ represents the distance between location $i$ and location $j$.
- $n \times n$ flow matrix $F$ where $F[i][j]$ represents the flow from facility $i$ to facility $j$.
- We need to find a permutation $p$ of $\{1, 2, ..., n\}$ such that if we place facility $i$ at location $p[i]$, the total cost is minimized.
- Cost = $\sum_{i=1}^n \sum_{j=1}^n D[p(i)][p(j)] \times F[i][j]$

Wait, let me check the notation carefully.
$p_i = j$ means facility $i$ is assigned to location $j$. So $p$ is a permutation array where $p[i]$ is the location index for facility $i$ (1-indexed).
The formula says $Cost = \sum_i \sum_j D[p(i), p(j)] * F[i, j]$.
Here $p(i)$ denotes the location assigned to facility $i$.
So if $P$ is the permutation array where $P[i]$ is the location for facility $i$ (0-indexed in implementation), then:
$Cost = \sum_{i} \sum_{j} D[P[i]][P[j]] \times F[i][j]$

The constraints are $n \le 2000$.
This is the Binary Quadratic Assignment Problem. Generally, QAP is NP-hard.
However, $n \le 2000$ suggests that an exact solution using branch-and-bound or dynamic programming is not feasible ($O(n!)$ or exponential).
Usually, for $n \approx 2000$, this implies either:
1. A greedy/heuristic approach is expected.
2. There's a specific property in the "Binary" case that makes it easier.
3. The time limit allows for something close to $O(n^2)$ or maybe $O(n^2 \log n)$.
4. Actually, looking at the scoring function: `score = 1 - Cost / TotalFlow`. Minimizing Cost is equivalent to maximizing score.

Wait, if $n \le 2000$, maybe there's a simplification?
Let's look at the matrices. They are binary (0s and 1s).
Usually, standard QAP solvers cannot handle $n=2000$ within reasonable time limits unless the structure is very specific.
Is this a trick question? Or is it asking for an approximation?
The prompt says "Generate solution code". It doesn't explicitly say "exact solution". But usually, competitive programming problems require exact solutions unless specified otherwise.
However, solving general QAP for $N=2000$ exactly is impossible within typical time limits (seconds/minutes) because it's NP-complete.
Maybe the constraints on the matrices allow simplification?
Actually, if $n$ is up to 2000, maybe $N^2$ operations are acceptable, which suggests an $O(n^2)$ algorithm.
If I just need to output *a* valid assignment (permutation), does it need to be optimal?
"Your task is to assign n facilities to n locations (a permutation) to minimize the total cost." -> This implies finding the global minimum.
But finding the global minimum for QAP with N=2000 is practically impossible.
Perhaps there's a misunderstanding of the problem difficulty class in my context.
Let's consider if the problem is actually simpler.
Is it possible that the matrices have special properties? No info given.
Is it possible that the expected complexity is high, like $O(n^3)$ or even $O(2^n \cdot poly(n))$ but $N$ is small enough? No, 2000 is too big.
Wait, could it be that I need to implement a heuristic or an approximate solution, but frame it as the best effort?
Or perhaps the test cases are weak?
Actually, looking at similar problems on platforms like Codeforces or LeetCode, sometimes they ask for QAP approximations or the N is small (like 10-20). Here $N=2000$ is huge.

Let's re-read carefully.
"You are given two n×n binary matrices... Your task is to assign... to minimize the total cost."
If it's a real QAP instance, $N=2000$ is definitely intractable for exact solution.
However, in some contexts (like specific contest environments), there might be a constraint relaxation or a specific intended solution path.
Could it be that the distance matrix has only 0s and 1s and flow also 0s and 1s? Yes.
Does that change anything structurally? Not really, just values.

Wait, is there a chance that this is a "maximum weight matching" type reduction? No, QAP involves products of decision variables $x_{ik} x_{jl}$ (binary variable indicating facility $i$ at location $k$ times facility $j$ at location $l$). If $n=2000$, that's $n^2$ terms.
With $n=2000$, $O(n^3)$ is $8 \cdot 10^9$, which is too slow for 1 second (usually $\sim 10^8$ ops/sec).
Maybe the time limit is generous? Or the intended complexity is $O(n^2)$.
If it's $O(n^2)$, we can probably use a greedy strategy or sorting.
What if we treat this as a maximum bipartite matching problem? No, QAP is quadratic.

Hypothesis 1: The problem expects a simple randomized local search or greedy approach, and the "minimize" is for correctness on random/small test cases provided by the judge, but not strict worst-case guarantee.
Hypothesis 2: There is a transformation making it linear.
If $D$ was diagonal or something... but it's a full matrix.
Hypothesis 3: Maybe I should implement a very good heuristic (e.g., Simulated Annealing or Genetic Algorithm) and hope it passes.
Hypothesis 4: Maybe $n$ is small in actual test cases despite the statement saying 2000 (typo in problem?).
Hypothesis 5: Is there a specific algorithm mentioned in literature for binary QAP with large N? Usually no.

Let's assume the standard greedy strategy works reasonably well or simply sorts based on some heuristic.
For example, sort rows/columns of F/D to align them?
There is a known heuristic for QAP called the "Koopmans-Beckmann" bound or coordinate descent.
Given the constraints and the nature of LLM prompts for coding tasks, often if $N$ is this large, a greedy or heuristic approach is the intended solution, possibly utilizing some structure.
However, without more info, providing a complex meta-heuristic like Simulated Annealing inside C++ without optimization might TLE if implemented naively.
Wait, calculating the cost takes $O(n^2)$. Swapping takes $O(n^2)$.
Simulated Annealing iterates many times. $T \times n^2$ would be huge.
We need something closer to $O(n^2)$.

Let's reconsider the problem statement source. It looks like a variation of a problem from a platform like AtCoder or Codeforces.
Wait, if $N \le 2000$, maybe there is a randomized solution that gets accepted.
Or maybe, specifically, **the problem is unsolvable exactly**, so the prompt implies generating a valid code for *this* setup, possibly accepting heuristics.
BUT, the prompt asks me to "Implement a solution". Usually, this means "correct" solution.
If exact solution is impossible, I should provide the best possible standard algorithm.
However, looking at the constraints again: $N=2000$. $N^2 = 4 \times 10^6$.
An $O(N^2)$ pass is perfectly fine.
Can we formulate the cost calculation as:
$\text{TotalCost} = \sum_{i,j} D_{loc[i]}[loc[j]] \times F_{ij}$.
Let $L$ be the permutation matrix representing the assignment. Then $Cost = Trace(D P L^T F L)$? No.
Let $X$ be the assignment where $X_{il} = 1$ if facility $i$ at loc $l$.
Cost = $\sum_{i,j,l,m} D_{lm} F_{ij} X_{il} X_{jm}$.
This is bilinear in $X$.

Wait, is it possible that the distance matrix represents distances where $D[i][j]=1$ iff $i \neq j$ (or similar)? No, "distance matrix D" is general.
Actually, let's check if there's a trick.
If $N$ is up to 2000, maybe the time limit is 5-10 seconds?
Even so, Exact QAP is hard.
Wait, could it be that the input guarantees something? "Binary matrices".
Maybe we can use bitsets? In C++, `std::bitset` or manual bitmasks.
Since $N \le 2000$, we can fit $N$ bits into roughly $2000/64 \approx 32$ `long long`s.
Operations become $O(N^2 / 64)$.
But how to solve QAP?
Maybe a Greedy assignment based on correlation?
Sort rows of F and D?
Often, a simple heuristic for QAP when N is large is to sort the rows of F according to column sums (or vice versa) and try to match them.
Or, construct the assignment based on sorting indices.

Let's step back. What if the problem is actually simpler?
Is it possible that the cost function can be rewritten?
No.
Okay, I must write a solution. Since exact is likely impossible, I will write a solution that tries to optimize it efficiently.
Common strategies for large QAP:
1. **Greedy Initialization**: Sort facilities by row sums of F (total demand from facility $i$ to others) and locations by column sums of D. Assign highest demand to shortest path?
   Specifically, if facility $i$ sends a lot of traffic to other nodes, put it near nodes that receive a lot?
   Or rather: Facility $i$ communicates with $j$. Distance matters.
   We want to place highly communicating facilities close to each other (small $D$).
   
   Standard Heuristic:
   Iterate through all pairs $(i, j)$ with non-zero flow.
   This suggests a "sorting" approach.
   
   Actually, there's a simpler interpretation. Maybe this is not a general QAP instance but one where $D$ and $F$ are related or sparse? No info.
   
   Let's check if there's any existing algorithm that works for N=2000 in reasonable time.
   For $N \ge 20$, Branch & Bound is slow.
   For $N=2000$, heuristics are the only way.
   Given I am an AI assistant, I should provide a robust heuristic solution that performs well on average/test cases, while handling the input format correctly.
   
   One effective simple heuristic for QAP is the **Correlation Heuristic** or just random restart local search.
   However, Local Search (Swap-based) requires evaluating new cost.
   Cost update when swapping facilities $i$ and $k$ (locations $u$ and $v$):
   Old Cost contributions involving $i$ and $k$ with others $j$:
   $C_{new} = C_{old} + \sum_{j} F_{ij}(D[u][loc[j]] - D[v][loc[j]]) + \dots$ ? No, it's symmetric.
   Calculating delta cost for a swap takes $O(n)$.
   With $N$ swaps per iteration, one iteration is $O(N^2)$.
   With multiple iterations, it might be okay.
   Wait, $\Delta$ cost calculation:
   If we move facility $i$ from $u$ to $v$:
   Change in cost depends on $\sum_j F_{ij} (D[v][loc[j]] - D[u][loc[j]])$.
   This is $O(N)$.
   Doing this for all $N$ swaps is $O(N^2)$.
   Repeating this process $T$ times gives $T \cdot N^2$.
   With $N=2000$, $N^2 = 4 \cdot 10^6$.
   Even $T=100$ gives $4 \cdot 10^8$ operations. This is borderline but might pass in 2-3 seconds if operations are simple.
   
   Optimizations:
   Precompute row sums?
   Use `unsigned long long` or `vector<int>` for speed?
   Input reading takes $O(N^2)$.
   Calculating initial cost takes $O(N^2)$.
   
   Wait, is it possible to solve this using Maximum Flow or Matching?
   If $D$ was a metric space or graph, maybe. But here $D$ is arbitrary binary.
   
   Wait, I found a similar problem context.
   "Binary Quadratic Assignment Problem" with $N=2000$.
   There is a specialized algorithm called "Simulated Annealing" or "Genetic Algorithm" used for this.
   However, writing a full GA/Simulated Annealing with careful complexity analysis is safer than guessing a greedy.
   
   BUT, writing a full SA/GA might be too much code or risk timeout due to heavy overhead.
   Let's focus on the most efficient exact-ish check or a strong heuristic.
   Wait, if the scoring is `1 - Cost / TotalFlow`, minimizing Cost is the goal.
   Since $N$ is large, we definitely don't expect a perfect solution.
   However, I should try to make the code as fast as possible.
   
   Algorithm choice:
   1. Read input.
   2. Initialize permutation (e.g., identity).
   3. Compute current cost.
   4. Perform Local Search: Try to find a neighbor permutation that reduces cost.
      - Neighbor defined by swapping two assignments $p[i]$ and $p[k]$.
      - Check all pairs. Pick one that improves cost.
      - Repeat until no improvement.
   - To optimize Local Search:
     - Calculate delta cost for a swap $(i, k)$ in $O(N)$ time.
     - Maintain current cost.
     - Use `std::vector` for permutation.
     - Iterate $N$ times, each time scanning for better swaps.
     - Stop when stuck.
     - Repeat from scratch few times (randomized restarts) to avoid local minima.
   
   Complexity:
   - Initial Cost: $O(N^2)$.
   - One Iteration of Local Search (scanning all pairs): $O(N^2)$ deltas.
     Delta calc: $\sum_j F_{ij}(D[loc_k][loc[j]] - D[loc_i][loc[j]])$. Wait.
     Let $u = p[i]$, $v = p[k]$.
     Swap $i$ and $k$. Facilities at $u$ and $v$ exchange roles.
     Cost components involving $i$ and others ($j$): $D[u][p[j]] F_{ij}$.
     New cost involves $D[v][p[j]] F_{ij}$.
     Change: $\sum_j (D[v][p[j]] - D[u][p[j]]) F_{ij}$.
     Similarly for $k$ and others.
     Also self-interaction term $F_{ii}$ (doesn't matter if binary, but $D[u][u]$ term exists).
     And interaction between $i$ and $k$:
     Term in cost is $D[u][u] F_{ii} + D[u][v] F_{ik} + D[v][u] F_{ki} + D[v][v] F_{kk}$.
     After swap: $D[v][v] F_{ii} + D[v][u] F_{ik} + D[u][v] F_{ki} + D[u][u] F_{kk}$.
     Note $D[u][v] = D[v][u]$ if symmetric. Problem doesn't state symmetric. Assuming symmetric usually safe or we compute both.
     Formula for swap $(i,k)$ costs $O(N)$.
     Scanning all $N(N-1)/2$ pairs takes $O(N^2)$.
     With $N=2000$, $N^2 = 4 \times 10^6$.
     Inner loop operations are simple additions/multiplications.
     This should take ~100ms per iteration in C++.
     Running 10-20 iterations is fine.
     
     Randomization helps avoid local optima.
     We can shuffle the order of facilities to initialize the permutation differently, or perform swaps.
     
   Wait, there is a catch. $O(N^2)$ memory for matrices is $4 \times 10^6 \times 4$ bytes $\approx 16$ MB. Fits easily.
   Reading input is fast.
   
   Can we improve the cost update?
   Precompute prefix/suffix sums? No.
   Just stick to naive $O(N^2)$ Local Search with limited iterations.
   Wait, iterating over all pairs to find the BEST swap is expensive ($O(N^2)$ per outer loop).
   But we can just pick a few random pairs or the first pair that improves cost.
   Since we want to optimize, hill-climbing usually picks the best or first improvement. First improvement is faster to find but slower to converge. Best improvement is slower ($O(N^2)$) per step.
   Given $N=2000$, doing $O(N^2)$ per step for 10 steps is $10^7 \dots 4 \cdot 10^7$ ops $\approx 0.04$s.
   Wait, calculating ALL deltas for one swap takes $O(N)$. There are $N^2$ pairs. Total $O(N^3)$?
   Ah!
   Wait.
   Number of swaps to consider is $N(N-1)/2 \approx O(N^2)$.
   For each swap, we calculate delta in $O(N)$.
   So scanning all potential swaps is $O(N^3)$.
   $N=2000 \implies N^3 = 8 \times 10^9$. Too slow.
   We MUST NOT scan all pairs.
   
   Optimization:
   Greedy Local Search (steepest ascent/descent is too slow).
   We can just sample a subset of pairs.
   OR, we can use a randomized swap strategy.
   Pick a random pair $(i, k)$. Swap. Recompute change? No, recompute whole cost or delta.
   If we fix the cost maintenance, computing delta for ONE swap is $O(N)$.
   If we do $K$ random swaps, cost is $O(K \cdot N)$.
   If we iterate this $R$ times (restarts), total is $R \cdot K \cdot N$.
   If $K=2000$, $R=100$, total $2 \cdot 10^3 \cdot 100 \cdot 2000 = 4 \cdot 10^8$. Still tight but maybe OK.
   Wait, we can reduce $K$ (number of attempts per iteration) to something manageable if we prioritize.
   However, hill climbing requires finding *some* improvement.
   
   Wait, there is a faster way to calculate cost changes if we maintain row sums of the distance matrix weighted by flows.
   Let $W[l] = \sum_{i} F_{pi} \times (\text{something})$.
   Actually, let's define the objective value more cleanly.
   Cost = $\sum_{i,j} F_{ij} D_{p(i)p(j)}$.
   Let $p$ be the assignment.
   This looks like a scalar product.
   
   Alternative Idea:
   If we sort facilities based on their "flow profiles", maybe we can match them to locations sorted by "distance profiles".
   But since it's quadratic, sorting alone isn't sufficient.
   
   However, for the purpose of generating code in this setting, I should implement a randomized greedy/local search that fits the time limits.
   Key insight: We don't need to explore all permutations.
   Strategy:
   1. Start with random permutation.
   2. Run Iterated Local Search with Tabu Search or simple cooling (Simulated Annealing).
   3. To avoid TLE, limit the number of iterations/swaps.
   
   Wait, is it possible to calculate the cost in $O(1)$ after updates?
   Not easily without storing intermediate structures.
   But $O(N)$ update for one swap is standard.
   
   Let's refine the Local Search complexity.
   We can pick a random facility $i$. Try to find best location for it among unused locations? No, we have a full permutation.
   Try to move facility $i$ to another occupied location.
   This is essentially trying to find a pair $(i, j)$ to swap.
   If we pick a random $i$, and try all $k \neq i$ (swapping $p[i]$ and $p[k]$).
   Checking all $N-1$ neighbors for one $i$ takes $O(N^2)$.
   Doing this for one $i$ is fast. Doing this for all $i$ is $O(N^2)$ checks? No.
   If we pick random $i$, try $k=1..N$.
   That's $O(N)$ checks, each taking $O(N)$ -> $O(N^2)$ for this round.
   We repeat this round $N$ times? $O(N^3)$. Too slow.
   
   Better approach: **Randomized 2-Opt**.
   Pick two indices $i$ and $k$ uniformly at random. Swap $p[i]$ and $p[k]$.
   Compute delta cost.
   Accept if better.
   Repeat this process many times.
   Each step is $O(N)$.
   If we do $S$ swaps, total time $O(S \cdot N)$.
   If $N=2000$, $S=200000$, total ops $\approx 4 \cdot 10^8$.
   This might be on the edge but simpler logic.
   Simulated Annealing adds acceptance probability.
   
   Another perspective:
   The problem asks to minimize cost.
   If we interpret the scoring: $1 - \text{Cost}/\text{TotalFlow}$.
   This is equivalent to minimizing Cost.
   
   Wait, is there a constraint I missed?
   "Output exactly one line".
   Maybe the constraints are looser.
   Actually, if $N \le 2000$, an $O(N^2)$ approach for initialization is fine.
   An $O(N^2)$ approach for optimization is also fine (one pass).
   Can we achieve $O(N^2)$ total optimization?
   
   Consider the sorting idea again.
   Sort facilities by row sum of $F$ (demand sent).
   Sort locations by... what?
   If we map facilities with high outflow to locations with low connectivity?
   This is intuition.
   
   Given the constraints and the problem type, implementing a Simulated Annealing (SA) solution is the safest bet for quality vs time balance.
   It's easy to implement.
   Steps:
   1. Initialize cost.
   2. Loop while temp > min_temp.
      a. Pick random swap $(i, j)$.
      b. Calculate delta.
      c. Probabilistic accept.
      d. Update cost.
   3. Print result.
   
   Wait, picking random swap $(i,j)$ means swapping facility $i$'s assignment with facility $j$'s assignment.
   Calculation of delta:
   Let $L[u]$ be the location currently held by facility $u$.
   Let $S[u]$ be the current assignment for facility $u$?
   Let's store `p[i]` = location of facility `i`.
   Current Cost $C$.
   Swap facilities $i$ and $k$.
   New locations: $p'[i] = p[k], p'[k] = p[i]$. Others unchanged.
   Difference in cost comes from:
   - Interactions between $i$ and any other facility $x$.
   - Interactions between $k$ and any other facility $y$.
   - Interaction between $i$ and $k$.
   Change = $\sum_x (F_{ix} (D_{p[k], p[x]} - D_{p[i], p[x]})) + \sum_x (F_{kx} (D_{p[i], p[x]} - D_{p[k], p[x]})) + \text{CrossTerm}$.
   CrossTerm involves $i, k$:
   Original: $F_{ik} D_{p[i], p[k]} + F_{ki} D_{p[k], p[i]}$.
   New: $F_{ik} D_{p[k], p[i]} + F_{ki} D_{p[i], p[k]}$.
   (Note $D$ might not be symmetric, but $p[k]$ and $p[i]$ are just labels).
   Wait, $p[i]$ is the location for $i$. So old distance is $D_{p[i], p[k]}$.
   New distance is $D_{p[k], p[i]}$.
   So cross change is $F_{ik}(D_{p[k], p[i]} - D_{p[i], p[k]}) + F_{ki}(D_{p[i], p[k]} - D_{p[k], p[i]})$.
   Which is $(F_{ik} - F_{ki})(D_{p[k], p[i]} - D_{p[i], p[k]})$.
   Summing over $x$:
   $\sum_{x \neq i, k} F_{ix} (D_{p[k], p[x]} - D_{p[i], p[x]}) + F_{ix} \text{ terms}$.
   Basically iterate $x$ from $1$ to $n$.
   If $x=i$ or $x=k$, handle carefully.
   Wait, if $x=i$, term is $F_{ii} D_{p[i], p[i]}$. It becomes $F_{ii} D_{p[k], p[k]}$.
   So yes, iterating all $x$ covers everything.
   
   Complexity per swap: $O(N)$.
   Number of swaps:
   If we do $10^5$ swaps, total $2 \cdot 10^8$ ops. C++ handles $\sim 10^8$ in < 1 sec.
   If we use SA, we need cooling.
   We can restrict max iterations to e.g., $2000 \times 100 = 2 \cdot 10^5$.
   Actually, for $N=2000$, even $N \times N$ swaps is risky.
   However, we can stop early if no improvement found.
   
   Wait, I can optimize the cost calculation using precomputed arrays.
   Instead of iterating all $x$, notice that for a fixed $i$, the contribution is $\sum_x F_{ix} D_{p[i], p[x]}$.
   Let $RowSum[i][loc] = \sum_x F_{ix} D_{loc, p[x]}$.
   When we change $p[i]$, we recalculate $RowSum$ for $i$.
   This still seems $O(N)$ to query.
   Updating RowSum when swapping $p[i]$ and $p[k]$:
   $RowSum[i][new\_loc] = RowSum[i][p[i]]$ (wait, $RowSum$ depends on locations of others).
   No, simpler to just re-evaluate the cost difference.
   
   Let's refine the solution to be "Competitive Programmer" quality.
   - Use fast I/O.
   - Use 1D vector for permutation to cache locality.
   - Implement Efficient Cost Delta.
   - Limit iterations to ensure pass.
   - Maybe add 3 restarts.
   
   Wait, is it possible that the time limit is extremely generous?
   Standard limits are 1s or 2s.
   Operations $\approx 10^8$.
   $N=2000 \implies N^2 = 4 \cdot 10^6$.
   If we can do constant work per element, we can do $O(N^2)$ total.
   How?
   Initialize permutation.
   Calculate Cost $O(N^2)$.
   Now, we want to optimize.
   What if we perform ONLY $N$ swaps?
   Each swap takes $O(N)$. Total $O(N^2)$.
   This is likely sufficient for a heuristic.
   Or maybe $O(N)$ swaps per phase, and do a few phases.
   Given the problem statement "minimize total cost", I will implement a robust local search that runs for a fixed budget of time (iterations).
   I'll use a counter for operations to ensure it doesn't exceed a hardcoded safe limit (e.g. 0.5s worth of cycles).
   
   Actually, looking at the provided scoring, minimizing cost is the priority.
   I'll implement a randomized local search with a temperature schedule.
   
   Wait, is there a library or header required? Just `<iostream>`, `<vector>`, `<numeric>`, `<random>`, `<algorithm>`, `<cmath>`.
   Include `chrono` if I want to control time limit strictly, but better to limit iterations.
   
   Wait, the problem is from a contest (likely).
   If it's a "hard" QAP variant, maybe there's a trick.
   But usually, for generic inputs, exact solution is impossible.
   The instruction "You are a competitive programmer" implies solving it optimally *within the constraints*.
   If $N=2000$ is given, heuristics are the way.
   
   Let's check the constraints on Time/Memory.
   Not provided in the text ("The execution time and memory limit are also stated in the statement").
   Usually default 1-2s, 256MB.
   
   Plan:
   1. Read N.
   2. Read D ($N \times N$).
   3. Read F ($N \times N$).
   4. Store D and F as flattened vectors for cache efficiency.
   5. Permutation P initialized to 0..N-1 (facilities i to loc i).
   6. Calculate Initial Cost.
   7. Loop:
      - Pick random swap.
      - Calc Delta.
      - If Delta < 0, apply.
      - Else, accept with prob.
      - Repeat X times.
      - Restart if needed.
   8. Output P+1 (1-based).
   
   To optimize speed:
   - Use `int` arrays instead of `std::vector` inside loops?
   - Flatten matrices to 1D: `D[i*n + j]`.
   - Avoid modulo in inner loop: `(i << shift)` or pointer arithmetic.
   
   Implementation details:
   - $N=2000$. Arrays size $2000 \times 2000 = 4,000,000$.
   - Memory usage: $2 \times 4M \times 4$ bytes = 32MB. Well within 256MB.
   - Speed: Reading $4M$ integers takes some time. Fast IO is mandatory (`cin.tie(NULL)`).
   - Cost calculation: double nested loop.
   
   One small detail: The problem uses 1-based indexing for output.
   
   Wait, if I just implement a pure random local search without cooling, it might get stuck quickly.
   But adding cooling (SA) increases complexity slightly (need random doubles).
   Since time is critical, maybe just Hill Climbing with multiple random starts is enough?
   Let's go with Hill Climbing with 3 random restarts.
   Max iterations per start: $N \times 200$. (Total $2000 \times 200 \times 3 \approx 1.2 \times 10^6$ steps? No, wait.)
   Wait, cost calculation is $O(N)$ per swap.
   If I do $N^2$ swaps total (which is $O(N^3)$ overall), it's too slow.
   I must limit the number of swaps significantly.
   Say, perform exactly $N \times 50$ swaps per restart.
   Wait, if I only do $N$ improvements, it might not converge.
   How about:
   While (best_swap < 0 && iterations < Limit) {
       find best swap
       swap
       count++
   }
   Finding best swap takes $O(N^2)$ (iterate all pairs, calculate delta in $O(N)$).
   Wait, earlier I said finding best swap is $O(N^3)$. Yes.
   Calculating delta for *one* swap is $O(N)$.
   Checking *all* $N(N-1)/2$ pairs is $O(N^3)$ total per step.
   So "Steepest Descent" is $O(N^3)$. We can't afford that.
   
   Solution: **First Improvement** with Sampling.
   Don't check all pairs. Check a set number of random pairs.
   Say, 1000 random pairs per iteration.
   Pick the best one found. Apply.
   This reduces the step cost to $O(Sampling \times N)$.
   If $Sampling \times N$ is around $10^5$, and we do it 20 times...
   Total ops: $20 \times 10^5 \approx 2 \times 10^6$.
   Plus reading input $O(N^2) \approx 4 \times 10^6$.
   Total is dominated by input reading and basic ops.
   This is perfectly safe for 1 second.
   
   So:
   1. Read input.
   2. Initial Permutation (Identity).
   3. Calc Cost.
   4. For a few rounds (e.g., 5 rounds):
      a. Try $K$ random swaps (where $K \approx 5000$). Keep track of best delta.
      b. If best delta < 0, apply it.
      c. Reset $p$ if no improvement (to escape local optima).
   
   Wait, resetting $p$ costs computation.
   Let's refine:
   Do $M$ iterations total.
   In each iteration:
     Pick random $i, j$.
     Calc Delta.
     If Delta < 0, apply and update current best cost.
     Else, don't apply.
     If we did this $M$ times, total time $O(M \cdot N)$.
     If $M = 200,000$, $200000 \times 2000 = 4 \cdot 10^8$. A bit risky for 1s.
     Maybe $M = 100,000$.
     
   Better yet:
   Implement `calc_delta` efficiently.
   Using 1D arrays.
   Use `__builtin_popcount` if relevant? No.
   
   Let's stick to:
   - Multiple restarts (3).
   - In each restart, run Simple Hill Climbing for $N$ iterations.
   - In each iteration, pick random pair $(i, j)$, check swap. If beneficial, swap.
   - If after $N$ iterations no swap improved, break.
   
   Why $N$ iterations? $N=2000$. Cost calculation is $O(N)$. Total $O(N^2)$.
   Wait, checking one swap is $O(N)$. $N$ iterations is $O(N^2)$.
   So 3 restarts is $3 N^2 \approx 1.2 \times 10^7$ ops.
   Plus initialization $O(N^2)$.
   Total is very low.
   Wait, why $N$ iterations?
   If I only do 2000 iterations, I might miss a huge valley.
   But I need to stay under time limit.
   Let's increase to $K$ iterations per restart.
   Say $K = 5000$.
   Total ops $\approx 3 \times 5000 \times 2000 = 3 \cdot 10^7$.
   This is absolutely safe.
   
   Wait, can I verify the correctness of this approach?
   Yes, it's a standard heuristic for QAP.
   
   One optimization for `calc_delta`:
   Instead of recomputing sum every time.
   But for the complexity $O(N^2)$ total budget, simple summation is fine.
   Just ensure loops are tight.
   
   Also, to prevent TLE on input reading, disable sync.
   
   Double check the Cost formula.
   Cost = $\sum_i \sum_j D[p(i), p(j)] * F[i, j]$.
   $p(i)$ is location of facility $i$.
   Indices are 1-based in problem, 0-based in code.
   $D$ and $F$ input are $n \times n$.
   
   Let's draft the code.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>

using namespace std;

// Global variables to reduce stack usage and object overhead
int N;
// Flattened matrices
// Using vector<int> for int is slightly better performance-wise than char/short if dense
// but int consumes more RAM. 2000*2000 = 4M ints = 16MB. Safe.
// Actually, 0/1 matrices can be stored as vector<int> or vector<vector<int>>.
// Flattened is faster.
// vector<int> D_rows[N]; // Too fragmented.
// D[N*N]
// F[N*N]

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    if (!(cin >> N)) return 0;

    // Resize flat vectors
    // Use vector<int> to save memory allocation calls compared to fixed size if stack is issue
    // Heap is fine.
    // D[i*N + j]
    
    auto D = vector<int>(N * N);
    auto F = vector<int>(N * N);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            cin >> D[i * N + j];
        }
    }
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            cin >> F[i * N + j];
        }
    }

    // Helper to access D[i][j]
    auto getD = [&](int r, int c) { return D[r * N + c]; };
    auto getF = [&](int r, int c) { return F[r * N + c]; };

    // Permutation: p[i] is the location assigned to facility i.
    // Initialized to identity: facility i goes to location i.
    vector<int> p(N);
    iota(p.begin(), p.end(), 0);

    // Function to calculate current cost
    // Cost = sum_{i,j} D[p[i]][p[j]] * F[i][j]
    // Optimized calculation:
    // We iterate i and j.
    
    // To optimize, we can precompute row sums? No, cost is specific to permutation.
    // Direct calculation is O(N^2).
    
    // Let's calculate initial cost
    long long current_cost = 0;
    for (int i = 0; i < N; ++i) {
        int loc_i = p[i];
        for (int j = 0; j < N; ++j) {
            int valF = F[i * N + j]; // If 0, skip multiplication
            if (valF == 0) continue;
            int dist = getD(loc_i, p[j]); // p[j] is location of facility j
            current_cost += (long long)dist * valF; // dist is 0 or 1
        }
    }

    // Random seed
    mt19937 rng(time(0));
    uniform_int_distribution<int> dist(0, N * (N - 1) / 2 - 1); // Number of swaps
    
    // Helper for delta calculation
    // Swaps p[i] and p[k].
    // Returns delta = new_cost - old_cost.
    // We want to minimize cost, so we look for negative delta.
    auto calc_delta = [&](int i, int k) -> int {
        int loc_i = p[i];
        int loc_k = p[k];
        
        // Delta consists of interactions of i and k with all others, plus self interactions.
        // Terms involving i changing from loc_i to loc_k:
        // For all x != i: F[ix] * (D[loc_k][p[x]] - D[loc_i][p[x]])
        // Terms involving k changing from loc_k to loc_i:
        // For all x != k: F[kx] * (D[loc_i][p[x]] - D[loc_k][p[x]])
        // Note: x can be i or k.
        
        // Let's combine logic.
        // Iterate x over all facilities.
        // Term for facility i: contribution is F[i][x] * D[p[i]][p[x]].
        // If p[i] changes to loc_k, new contrib is F[i][x] * D[loc_k][p[x]].
        // Change_i = sum(F[i][x] * (D[loc_k][p[x]] - D[loc_i][p[x]])).
        // Similar for k.
        
        // Be careful not to double count?
        // Cost = sum_{u,v} D[loc(u)][loc(v)] * F[u][v].
        // If u=i, v=x: D[loc_i][loc_x] * F[i][x].
        // If we swap i and k, loc_i becomes loc_k, loc_k becomes loc_i.
        // Only terms involving i (as source) and i (as destination) change.
        // Terms involving k change similarly.
        
        // Cross terms between i and k:
        // Old: F[i][k]*D[loc_i][loc_k] + F[k][i]*D[loc_k][loc_i].
        // New: F[i][k]*D[loc_k][loc_i] + F[k][i]*D[loc_i][loc_k].
        // Diff = F[i][k]*(D[loc_k][loc_i] - D[loc_i][loc_k]) + F[k][i]*(D[loc_i][loc_k] - D[loc_k][loc_i]).
        
        int diff = 0;
        
        // Calculate contribution from i's movement
        // i goes loc_i -> loc_k
        for (int x = 0; x < N; ++x) {
            if (F[i * N + x]) {
                int loc_x = p[x];
                // Contribution change for pair (i, x)
                // Old: D[loc_i][loc_x] * F[i][x]
                // New: D[loc_k][loc_x] * F[i][x]
                // Add D[loc_k][loc_x] - D[loc_i][loc_x]
                
                // Check if loc_x is same as loc_i (when x=i)
                // If x=i, loc_x is loc_i.
                // D[loc_k][loc_i] - D[loc_i][loc_i]
                
                // To avoid redundant checks, iterate x.
                // Optimization: if F[i][x] is 0, skip.
                // Accessing F[i][x] is linear scan.
                
                // Accessing D[loc_k][loc_x] vs D[loc_i][loc_x]
                int d_new = getD(loc_k, loc_x);
                int d_old = getD(loc_i, loc_x);
                diff += (d_new - d_old); // multiply by 1 later
            }
        }
        
        // Calculate contribution from k's movement
        // k goes loc_k -> loc_i
        for (int x = 0; x < N; ++x) {
            if (F[k * N + x]) {
                int loc_x = p[x];
                int d_new = getD(loc_i, loc_x);
                int d_old = getD(loc_k, loc_x);
                diff += (d_new - d_old);
            }
        }
        
        // Wait, the above logic for 'i' includes x=k.
        // If x=k, term is F[i][k] * (D[loc_k][loc_k] - D[loc_i][loc_k]).
        // The 'k' loop includes x=i.
        // If x=i, term is F[k][i] * (D[loc_i][loc_i] - D[loc_k][loc_i]).
        
        // Plus the direct interaction i-k?
        // My logic above calculates:
        // i's terms: sum_x F[i][x] * (D[loc_k][loc_x] - D[loc_i][loc_x])
        // k's terms: sum_x F[k][x] * (D[loc_i][loc_x] - D[loc_k][loc_x])
        // Notice for x=k in i's loop: F[i][k] * (D[loc_k][loc_k] - D[loc_i][loc_k])
        // For x=i in k's loop: F[k][i] * (D[loc_i][loc_i] - D[loc_k][loc_i])
        
        // Are these correct?
        // Original cost contains:
        // sum_{a,b} D[loc[a]][loc[b]] F[a][b]
        // When swapping p[i] and p[k]:
        // Fac i moves loc_i -> loc_k.
        // Fac k moves loc_k -> loc_i.
        // Any other facility m stays at loc_m.
        // Cost changes due to:
        // 1. Pairs (i, m) and (m, i).
        //    i->loc_k, m->loc_m. D[loc_k][loc_m] instead of D[loc_i][loc_m].
        //    Sum over m: F[i][m] * (D[loc_k][loc_m] - D[loc_i][loc_m]) + F[m][i] * (D[loc_k][loc_m] - D[loc_i][loc_m]).
        // 2. Pairs (k, m) and (m, k).
        //    k->loc_i, m->loc_m. D[loc_i][loc_m] instead of D[loc_k][loc_m].
        //    Sum over m: F[k][m] * (D[loc_i][loc_m] - D[loc_k][loc_m]) + F[m][k] * (D[loc_i][loc_m] - D[loc_k][loc_m]).
        // 3. Pair (i, k).
        //    i->loc_k, k->loc_i. D[loc_k][loc_i] instead of D[loc_i][loc_k].
        //    D[loc_i][loc_k] instead of D[loc_k][loc_i] (for the other direction).
        //    Term: F[i][k]*(D[loc_k][loc_i] - D[loc_i][loc_k]) + F[k][i]*(D[loc_i][loc_k] - D[loc_k][loc_i]).
        
        // Wait, the sum over m logic in (1) and (2) handles x=m where m != i, k.
        // What about x=i or x=k?
        // If x=i in loop 1 (pairs (i,i)):
        // m=i. F[i][i] * (D[loc_k][loc_i] - D[loc_i][loc_i]).
        // Wait, pair (i,i) means facility i interacts with itself.
        // loc[i] was loc_i. Becomes loc_k.
        // D[loc_k][loc_k] * F[i][i] ?? No.
        // Facility i is at loc_k. Its distance to itself?
        // D[loc_k][loc_k] * F[i][i].
        // Change: F[i][i] * (D[loc_k][loc_k] - D[loc_i][loc_i]).
        
        // In my code:
        // Loop 1 (for i): x runs all.
        // If x=i: loc_x = loc_i.
        // Term added: F[i][i] * (D[loc_k][loc_i] - D[loc_i][loc_i]).
        // But it should be F[i][i] * (D[loc_k][loc_k] - D[loc_i][loc_i]).
        // Because when i moves to loc_k, its own location is loc_k, not loc_i.
        // My code used p[x] which for x=i is loc_i. WRONG.
        // When i moves to loc_k, for the term (i, i), distance is D[loc_k][loc_k].
        
        // Correct Logic:
        // We need to iterate all facilities x.
        // For facility i: New Loc is loc_k.
        // Contribution change: F[i][x] * D[loc_k][p[x]].
        // Wait, p[x] is the location of x. Does x move?
        // If x is k, p[x] is loc_i.
        // If x is i, p[x] (old) was loc_i, but now i is at loc_k.
        // This is getting confusing because x refers to facility ID, not position.
        
        // Let's denote:
        // p_new is the vector AFTER swap.
        // p_new[i] = loc_k.
        // p_new[k] = loc_i.
        // p_new[x] = p[x] for x not i, k.
        
        // Change = Cost(p_new) - Cost(p_old).
        // Cost(p_new) = sum_{a,b} D[p_new[a]][p_new[b]] * F[a][b].
        // Cost(p_old) = sum_{a,b} D[p[a]][p[b]] * F[a][b].
        // Terms where neither a nor b is i or k are unaffected.
        // Terms where a=i, b=k: F[i][k]*D[p_new[i]][p_new[k]] = F[i][k]*D[loc_k][loc_i].
        //              vs F[i][k]*D[p[i]][p[k]] = F[i][k]*D[loc_i][loc_k].
        // Terms where a=k, b=i: F[k][i]*D[p_new[k]][p_new[i]] = F[k][i]*D[loc_i][loc_k].
        //              vs F[k][i]*D[p[k]][p[i]] = F[k][i]*D[loc_k][loc_i].
        // Terms where a=i, b!=i,k: F[i][x]*D[loc_k][p[x]].
        //              vs F[i][x]*D[loc_i][p[x]].
        // Terms where a=k, b!=i,k: F[k][x]*D[loc_i][p[x]].
        //              vs F[k][x]*D[loc_k][p[x]].
        // Terms where a=i, b=i: F[i][i]*D[loc_k][loc_k].
        //              vs F[i][i]*D[loc_i][loc_i].
        // Terms where a=k, b=k: F[k][k]*D[loc_i][loc_i].
        //              vs F[k][k]*D[loc_k][loc_k].
        
        // Consolidating:
        // Iterate a over {i, k}. Iterate b over {0..N-1}.
        // But note p[a] changes.
        
        // Simpler logic:
        // Cost change = 
        //   (F[i][i]*(D[loc_k][loc_k] - D[loc_i][loc_i]) + F[k][k]*(D[loc_i][loc_i] - D[loc_k][loc_k]))
        // + sum_{x != i, k} [ F[i][x]*(D[loc_k][p[x]] - D[loc_i][p[x]]) + F[x][i]*(D[loc_k][p[x]] - D[loc_i][p[x]]) ]
        // + sum_{x != i, k} [ F[k][x]*(D[loc_i][p[x]] - D[loc_k][p[x]]) + F[x][k]*(D[loc_i][p[x]] - D[loc_k][p[x]]) ]
        // + (F[i][k]*(D[loc_k][loc_i] - D[loc_i][loc_k]) + F[k][i]*(D[loc_i][loc_k] - D[loc_k][loc_i]))
        // This splits into:
        // Self terms:
        //   S_self = F[i][i] * (D[loc_k][loc_k] - D[loc_i][loc_i])
        //          + F[k][k] * (D[loc_i][loc_i] - D[loc_k][loc_k])
        // Cross terms between i and k:
        //   S_cross = F[i][k]*(D[loc_k][loc_i] - D[loc_i][loc_k]) + F[k][i]*(D[loc_i][loc_k] - D[loc_k][loc_i])
        // Other terms involving i:
        //   S_other_i = sum_{x != i, k} F[i][x] * (D[loc_k][p[x]] - D[loc_i][p[x]])
        //   S_other_i_inv = sum_{x != i, k} F[x][i] * (D[loc_k][p[x]] - D[loc_i][p[x]])
        // Other terms involving k:
        //   S_other_k = sum_{x != i, k} F[k][x] * (D[loc_i][p[x]] - D[loc_k][p[x]])
        //   S_other_k_inv = sum_{x != i, k} F[x][k] * (D[loc_i][p[x]] - D[loc_k][p[x]])
        
        // Combine terms involving x (excluding i, k):
        // For each x != i, k:
        //   diff_x = F[i][x] + F[x][i]  (coeff for D[loc_k][p[x]] - D[loc_i][p[x]])
        //           + F[k][x] + F[x][k]  (coeff for D[loc_i][p[x]] - D[loc_k][p[x]])
        //           = (F[i][x] + F[x][i] - (F[k][x] + F[x][k])) * (D[loc_k][p[x]] - D[loc_i][p[x]])
        //           Wait, signs.
        //   Term 1: F[i][x](D[loc_k] - D[loc_i]) = (F[i][x])(+delta_D)
        //   Term 2: F[x][i](D[loc_k] - D[loc_i]) = (F[x][i])(+delta_D)
        //   Term 3: F[k][x](D[loc_i] - D[loc_k]) = (F[k][x])(-delta_D)
        //   Term 4: F[x][k](D[loc_i] - D[loc_k]) = (F[x][k])(-delta_D)
        //   Net Coeff: (F[i][x] + F[x][i] - F[k][x] - F[x][k]) * (D[loc_k][p[x]] - D[loc_i][p[x]])
        
        // Finally, if we include i and k in the loop, it's cleaner but requires care.
        // Let's just implement the explicit split to avoid confusion.
        
        int change = 0;
        
        // Self interactions
        if (F[i*N + i]) change += (D[loc_k][loc_k] - D[loc_i][loc_i]);
        if (F[k*N + k]) change += (D[loc_i][loc_i] - D[loc_k][loc_k]);
        
        // Cross interaction
        change += F[i*N + k] * (D[loc_k][loc_i] - D[loc_i][loc_k]);
        change += F[k*N + i] * (D[loc_i][loc_k] - D[loc_k][loc_i]);
        
        // Others
        for (int x = 0; x < N; ++x) {
            if (x == i || x == k) continue;
            
            int coeff = 0;
            if (F[i*N + x]) coeff++;
            if (F[x*N + i]) coeff++;
            
            int neg_coeff = 0;
            if (F[k*N + x]) neg_coeff++;
            if (F[x*N + k]) neg_coeff++;
            
            if (neg_coeff == 0 && coeff == 0) continue;
            
            int d_diff = getD(loc_k, p[x]) - getD(loc_i, p[x]);
            
            if (coeff > neg_coeff) {
                change += (coeff - neg_coeff) * d_diff;
            } else {
                change += (neg_coeff - coeff) * (-d_diff); // equivalent
            }
        }
        
        return change;
    }
```
This explicit split logic is $O(N)$ and robust.
Complexity:
Calc cost: $O(N^2)$.
One iteration of local search (swap): $O(N)$.
Total complexity: $O(\text{Iterations} \times N)$.
With Iterations $\approx 5000$, Total ops $\approx 10^7$, well within limits.

I will use this logic.
Also, I will use `long long` for cost accumulator. `int` for delta.
Since Cost can be up to $N^2 \times N^2$ (max distance 1, max flow 1, $N^2$ terms?), no, max cost is $N^2 \times 1$. $4 \cdot 10^6$. Fits in int.
Wait, cost sum is $\sum \sum$. Max $N^2$ terms. Each term max 1. Sum $\le N^2$. Fits in int.
However, using `long long` is safer.

Wait, check constraints again.
$D$ and $F$ are binary (0 or 1).
Max Cost = $N^2$.
Wait, if $D$ entries were larger, yes. But binary.
Actually, `p[i]` are locations.
The problem states `D` and `F` contain only 0s and 1s.
Wait, is Cost bounded by $N^2$? Yes, $N^2$ pairs of $(i,j)$, each contributes at most 1.
Wait, $D[p[i]][p[j]]$ is 0 or 1.
So Max Cost is $N^2$.
For $N=2000$, Max Cost $\approx 4 \times 10^6$.
Integers are sufficient.
I'll use `int` for cost and delta to keep math simple and faster.

Wait, the "cross terms" part in my code logic had a mistake in the simplified loop version I wrote.
Specifically `if (coeff > neg_coeff) ...`.
It's simpler to just accumulate directly:
`change += (coeff - neg_coeff) * d_diff;`
Because `(F[i][x] - F[k][x])` etc.
Wait.
Coeff of `D[loc_k][p[x]]` is `F[i][x] + F[x][i]`.
Coeff of `D[loc_i][p[x]]` is `F[k][x] + F[x][k]`.
Term in `Change` is `(D[loc_k] - D[loc_i]) * (F[i][x] + F[x][i])`.
Plus `(D[loc_i] - D[loc_k]) * (F[k][x] + F[x][k])`.
`= (F[i][x] + F[x][i]) * d_diff + (F[k][x] + F[x][k]) * (-d_diff)`
`= ((F[i][x] + F[x][i]) - (F[k][x] + F[x][k])) * d_diff`.
Yes.
So `net_coeff = (F[i][x] + F[x][i]) - (F[k][x] + F[x][k])`.
`change += net_coeff * d_diff`.
This is $O(N)$ inside the loop.
Inside the `calc_delta` function.
My previous snippet had a conditional `if (neg_coeff == 0 ...)`.
Direct accumulation is fine.
Just need to make sure `x` iterates all.
Optimization: if `net_coeff` is 0, skip.
Precomputing `row_sum_F` could help?
No, `net_coeff` depends on `x`.
Iterating `x` and accessing arrays is fast enough.

Wait, `getD` calls `D` array.
Access pattern is row-major.
`D[loc_k][p[x]]` -> `D[loc_k*N + p[x]]`.
`D[loc_i][p[x]]` -> `D[loc_i*N + p[x]]`.
These are in the same row of D, separated by stride 1 (since p[x] varies).
Wait, `p[x]` varies randomly. This causes cache misses if `p` is not sorted.
But `p` is a permutation.
If `loc_k` and `loc_i` are distinct indices in `D`.
We jump between rows `loc_k` and `loc_i`.
Accessing `D` with `p[x]` (column) jumps across columns.
This might be cache inefficient but necessary.
Given constraints, it should pass.

Final check on time limit.
If $N=2000$, `vector<int>` resize is fast.
Input reading dominates if done poorly.
Use `cin` with sync off.

I will wrap the solution in `solve` function and call from `main`.

Wait, the problem asks to output `n` integers $p_1, \dots, p_n$.
This corresponds to locations.
In code `p[i]` is location for facility `i`.
So printing `p[0]+1 ... p[n-1]+1` is correct.

One detail: `random_device` might fail on some judges. `time(0)` is safer fallback.

Wait, should I use `static` arrays?
Vectors are fine. Static arrays are faster and avoid heap fragmentation.
`int D[2005][2005]` -> Size $2005^2 \times 4$ bytes $\approx 16$ MB. Stack size might be limited on Windows (often few MB).
Better use global arrays or static inside main.
Global arrays are safest in CP.
`int D[2005 * 2005];`
`int F[2005 * 2005];`
This puts them in BSS segment (global data), which counts towards memory limit but is usually unlimited/very large (256MB).
Heap `vector` is also fine.
I'll use vectors to avoid large static array declarations which can cause issues with certain stack limits if declared inside main (though moved to global is fine).
Wait, $2000 \times 2000$ is $4 \times 10^6$.
`int` is 4 bytes. Total 16MB.
Declaring `vector<int> D(N*N)` allocates on heap.
I'll stick with vector for safety against stack overflow.

Wait, one more thing.
The input says `Line 1: one integer n`.
Then lines.
Reading is straightforward.

```cpp
/*
Binary Quadratic Assignment Problem - Heuristic Solution
Constraints: N <= 2000. Time Limit likely ~1-2s.
Approach:
1. Read Input.
2. Initial Permutation: Identity.
3. Calculate Initial Cost O(N^2).
4. Perform Iterated Local Search / Randomized Search.
   - Perform M iterations.
   - In each iteration, pick random swap (i, j).
   - Calculate cost change in O(N).
   - If improvement, swap.
5. Output permutation.
*/
```
The code should be complete.

One corner case: $N=2000$, loop $5000 \times 2000$ is $10^7$ ops.
Each op involves several array lookups.
On modern CPU $10^8$ ops take ~0.1-0.2s.
So this is extremely safe.

Wait, should I add a cooldown to the random selection?
No, random swaps are good enough for escaping local minima.
Just repeating the search process is key.
Maybe run multiple independent searches and keep the best result?
Running 5 searches with 1000 iterations each is $5 \times 10^6$ ops. Very cheap.
Let's add 3 restarts with different seeds.
This is a standard technique.

Restart logic:
Keep global best permutation.
Reset current permutation.
Loop.

Wait, copying vectors is slow.
I'll maintain `current_p` and `best_p`.
If `current` improves, update `best`.
Wait, to avoid keeping `best` separate and swapping logic being complex, I'll just keep one `best` copy.
Actually, updating `current` to `best` (cloning) is $O(N)$. Fine.
Or I can just run the loop, update `best` if `cost < best_cost`.
And reset `current` to random permutation occasionally?
Simplest:
Loop `restarts` times:
  Init random perm.
  Cost.
  Iterate steps:
    Find best swap? (Expensive $O(N^2)$)
    Or Single Swap (Cheaper).
    Try 1000 random swaps.
    Update cost.
  If `cost < best_cost`, `best_perm = current_perm`, `best_cost = cost`.

This seems solid.

Let's refine the single swap delta calculation again.
`net_coeff` method.
`coeff` for `D[loc_k][p[x]]` is `F[i][x] + F[x][i]`.
`coeff` for `D[loc_i][p[x]]` is `F[k][x] + F[x][k]`.
Diff is `(coeff1 - coeff2)`.
If `coeff1 > coeff2`, we want `D[loc_k][p[x]]` to be 1 (and `D[loc_i]` 0) ideally.
Actually, `change += (coeff1 - coeff2) * (D[loc_k][p[x]] - D[loc_i][p[x]])`.
Correct.

Double check: `i` and `k` indices.
If `i` and `k` are facility IDs. `p[i]` is location of `i`.
`loc_i = p[i]`. `loc_k = p[k]`.
When we swap `i` and `k` in `p` array: `p[i]` becomes `loc_k`, `p[k]` becomes `loc_i`.
This effectively exchanges the positions.
So `calc_delta` assumes `p` remains unchanged? No.
It calculates the difference assuming we perform the swap.
The `loc_i` and `loc_k` are read from `p` BEFORE swap.
The logic holds.

```cpp
#include <iostream>
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>

using namespace std;

int main() {
    // Optimize I/O
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int N;
    if (!(cin >> N)) return 0;

    // Use 1D vector for matrices to improve cache locality and simplicity
    // Dimensions N*N
    vector<int> D(N * N);
    vector<int> F(N * N);

    // Read D
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            cin >> D[i * N + j];
        }
    }
    // Read F
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            cin >> F[i * N + j];
        }
    }

    // Lambda to fetch elements
    auto getD = [&](int i, int j) { return D[i * N + j]; };
    auto getF = [&](int i, int j) { return F[i * N + j]; };

    // Permutation p: p[i] is the location for facility i
    vector<int> p(N);
    iota(p.begin(), p.end(), 0);

    // Helper to compute cost
    // Since we run this often, make it efficient
    auto compute_cost = [&]() -> long long {
        long long cost = 0;
        for (int i = 0; i < N; ++i) {
            int loc_i = p[i];
            for (int j = 0; j < N; ++j) {
                int f_val = F[i * N + j];
                if (f_val) {
                    int loc_j = p[j];
                    cost += getD(loc_i, loc_j);
                }
            }
        }
        return cost;
    };

    // Initial cost
    long long min_cost = compute_cost();

    // Setup Random Generator
    mt19937 rng(1337); // Fixed seed for reproducibility in testing, or time(0)
    // Let's use time for randomness, but consistent behavior is better for debugging if needed.
    // In CP, time(0) is standard.
    // I will stick to time(0) but ensure headers are there.
    // To ensure compilation, #include <ctime>
    // But for the solution, I'll just use a seed derived from time.
    // Wait, I can't access time(0) if not included.
    // I'll use 1337 to be deterministic if running locally, but problem implies any valid min cost.
    // Actually, for online judge, time is preferred to avoid anti-cheat on fixed seeds? 
    // Usually fine.
    // Let's use mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
    
    // Wait, chrono needs header. I'll include <chrono>.
    // For simplicity in snippet, I'll just use static mt19937(1) + loop counter or similar 
    // if I can't import headers easily? No, I can.
    
    // Let's just create a simple RNG.
    unsigned int seed = 12345; 
    mt19937 gen(seed);
    uniform_int_distribution<int> uid(0, N - 1);

    // Local search parameters
    int num_restarts = 3;
    int iterations_per_restart = 2000; // 2000 * 2000 ops is low risk
    
    // Actually, to maximize chances, let's increase iterations relative to N.
    // Let's do 5000 iterations.
    // Total swaps = 5000. Cost update per swap = O(N). Total 5000*N = 10^7.
    // This is very safe.
    
    vector<int> best_p = p;
    long long best_min_cost = min_cost;

    for (int restart = 0; restart < num_restarts; ++restart) {
        // Shuffle or init new random p
        // Generating a random permutation
        p.assign(N, 0);
        iota(p.begin(), p.end(), 0);
        shuffle(p.begin(), p.end(), gen);
        
        long long current_cost = compute_cost();
        
        for (int iter = 0; iter < iterations_per_restart; ++iter) {
            int i = uid(gen);
            int k = uid(gen);
            if (i == k) continue; // Must be distinct
            
            // Calculate delta
            // loc_i and loc_k are positions before swap
            int loc_i = p[i];
            int loc_k = p[k];
            
            // We want to check if swapping p[i] and p[k] reduces cost
            // Delta = Cost(new) - Cost(old)
            // We need delta < 0 for acceptance
            
            int delta = 0;
            
            // Self interactions
            if (F[i * N + i]) delta += (getD(loc_k, loc_k) - getD(loc_i, loc_i));
            if (F[k * N + k]) delta += (getD(loc_i, loc_i) - getD(loc_k, loc_k));
            
            // Cross interaction
            delta += getF(i, k) * (getD(loc_k, loc_i) - getD(loc_i, loc_k));
            delta += getF(k, i) * (getD(loc_i, loc_k) - getD(loc_k, loc_i));
            
            // Others
            // Precompute coeff difference
            // Coeff of D[loc_k][p[x]] is F[i][x] + F[x][i]
            // Coeff of D[loc_i][p[x]] is F[k][x] + F[x][k]
            // Net coeff is (F[i][x] + F[x][i]) - (F[k][x] + F[x][k])
            
            for (int x = 0; x < N; ++x) {
                if (x == i || x == k) continue;
                
                int diff_f = 0;
                int pos_val = 0; // Value of F[i][x] + F[x][i]
                int neg_val = 0; // Value of F[k][x] + F[x][k]
                
                if (getF(i, x)) pos_val++;
                if (getF(x, i)) pos_val++;
                
                if (getF(k, x)) neg_val++;
                if (getF(x, k)) neg_val++;
                
                if (pos_val == neg_val) continue;
                
                int d_diff = getD(loc_k, p[x]) - getD(loc_i, p[x]);
                delta += (pos_val - neg_val) * d_diff;
            }
            
            if (delta < 0) {
                // Accept
                p.swap(i, k); // Wait, p is index->location map?
                // Yes, p[i] is location of facility i.
                // Swapping p[i] and p[k] means facility i moves to loc_k, k moves to loc_i.
                // This is exactly what we checked:
                // Old: i@loc_i, k@loc_k. New: i@loc_k, k@loc_i.
                // Swapping elements in vector p achieves this.
                // BUT, we calculated delta based on loc_i=p[i], loc_k=p[k] BEFORE swap.
                // So yes, swap p[i], p[k].
                
                current_cost += delta;
                if (current_cost < best_min_cost) {
                    best_min_cost = current_cost;
                    best_p = p; // Copy is O(N)
                }
            } else {
                // Reject
                // Revert
                // Actually, simply undoing delta is hard (we didn't store p change state)
                // So swap back manually
                // Or just don't modify p.
                // My code above modifies p inside 'if'. 
                // Wait, I need to revert if rejected? 
                // Wait, 'p.swap(i,k)' modifies p. If I reject, I must swap back.
                // But swapping twice is O(1).
                // Or just check `if (delta < 0)` then swap.
                // My `calc_delta` assumed NO swap yet.
                // If delta < 0, I should swap. If > 0, do nothing.
                // So:
                /*
                if (delta < 0) {
                    swap(p[i], p[k]);
                    current_cost += delta;
                }
                */
            }
        }
    }

    // Output results
    // p is the best permutation found in best_p
    // Need to print best_p
    // Wait, I need to capture `best_p` inside the loop.
    // I assigned `best_p = p` inside the loop.
    // So outside, `best_p` holds the best.
    
    for (int i = 0; i < N; ++i) {
        cout << (best_p[i] + 1) << (i == N - 1 ? "" : " ");
    }
    cout << "\n";

    return 0;
}
```
Wait, inside the loop `p` changes.
If I don't revert `p` when rejected, the next iteration operates on modified `p`.
That's correct for Local Search (Hill Climbing / Simulated Annealing).
Wait, but `calc_delta` relies on `p`.
If I change `p` in the loop but don't revert, it continues on new state.
If I reject, I don't change `p`.
So my logic `if (delta < 0) { swap(p[i], p[k]); ... }` is correct.
However, `compute_cost` at the beginning of restart calculates cost from scratch.
Inside iteration, I update `current_cost` by adding delta.
This is correct assuming `p` hasn't changed (except for successful swaps).
Wait, I need to verify `delta` calculation correctness.
My `delta` function computes the cost difference between state A (before swap) and state B (after swap).
If I swap, state A becomes B, and cost changes by `delta`.
If `delta < 0` (improvement), I accept.
So yes, logic is sound.

Wait, I should check `current_cost` consistency.
Initial `compute_cost` is precise.
Subsequent updates are additive.
Should be fine.

However, `compute_cost` is $O(N^2)$. Calling it every restart is $3 \times O(N^2)$.
Calling it inside loop?
If `delta` accumulates, I don't need `compute_cost` inside loop.
Wait, I call `compute_cost` once per restart.
Correct.

One issue: `best_p` needs to be updated only when `best_min_cost` updates.
Also, if a better state is found mid-restart, it should be saved.
I did `best_min_cost = current_cost; best_p = p;`.
Correct.

Wait, one small bug in logic:
Inside the inner loop, `uid(gen)` generates index.
`shuffle` shuffles `p` initially.
Seems okay.

The time limit constraint:
Is $O(iterations \times N)$ definitely fast enough?
$iterations = 2000$?
$N=2000$.
Total ops $\approx 4 \times 10^6$.
Wait, earlier I said $4 \times 10^7$.
Ah, `delta` loop is $O(N)$.
Inner loop iterates $N$ times.
So one iteration is $O(N)$.
Outer loop runs $iterations\_per\_restart$ times.
So total $iterations\_per\_restart \times N$.
If $2000 \times 2000 = 4 \times 10^6$.
This is extremely fast.
Why did I worry about $10^8$?
Ah, because reading input is $4 \times 10^6$ ints.
Total operations $\approx 10^7$.
This will run in < 0.1s.
So I can increase `iterations_per_restart` to 10000.
Total $2 \cdot 10^7$ ops.
Still very safe.

Final check on logic:
`calc_delta` logic for `F[i][x]` etc.
Indices `i` and `k` are facility IDs.
`F` is accessed as `F[i * N + x]`.
`D` is accessed as `getD(loc_k, p[x])`.
This matches the derivation.

Wait, `getF(i, k)` helper in `delta` calculation.
I defined `auto getF`.
I used it.
Correct.

Wait, inside delta loop:
`int pos_val = 0; if (getF(i, x)) pos_val++; ...`
Since `F` is 0/1, `getF` returns int.
`pos_val` becomes count.
Correct.

One tiny optimization:
If `getF` returns 0, we can skip adding to `pos_val`?
Yes. `if (getF(i, x) && getF(x, i))`?
Wait, `pos_val` sums `F[i][x] + F[x][i]`.
It's just counting ones.
So `pos_val` is either 0, 1, 2.
Checking `if (pos_val == 0)` inside loop is good.

I will put the includes properly.

One correction:
`delta` calculation for cross term:
`getF(i, k) * (getD(loc_k, loc_i) - getD(loc_i, loc_k))`
My code: `getF(i, k) * (getD(loc_k, loc_i) - getD(loc_i, loc_k))`
Wait, `loc_k` and `loc_i` are swapped in arguments relative to my previous thought.
Previous thought: `F[i][k] * (D[loc_k][loc_i] - D[loc_i][loc_k])`.
Code: `getF(i, k) * (getD(loc_k, loc_i) - getD(loc_i, loc_k))`.
Matches.
Logic:
Old cost term for pair (i,k): `D[loc_i][loc_k] * F[i][k]`.
New cost term: `D[loc_k][loc_i] * F[i][k]`.
Difference (New - Old) = `F[i][k] * (D[loc_k][loc_i] - D[loc_i][loc_k])`.
Correct.

Looks good.

```cpp
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <chrono>

using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int N;
    if (!(cin >> N)) return 0;

    vector<int> D(N * N);
    vector<int> F(N * N);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            cin >> D[i * N + j];
        }
    }
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            cin >> F[i * N + j];
        }
    }

    auto getD = [&](int r, int c) { return D[r * N + c]; };
    auto getF = [&](int r, int c) { return F[r * N + c]; };

    vector<int> p(N);
    iota(p.begin(), p.end(), 0);

    long long min_cost = 0;
    // Initial cost calculation
    for (int i = 0; i < N; ++i) {
        int loc_i = p[i];
        for (int j = 0; j < N; ++j) {
            if (F[i * N + j]) {
                min_cost += getD(loc_i, p[j]);
            }
        }
    }

    mt19937 gen((uint64_t)chrono::steady_clock::now().time_since_epoch().count());
    uniform_int_distribution<int> dis(0, N - 1);

    long long best_total_flow = -1; // Not needed
    long long current_best_cost = min_cost;
    vector<int> best_perm = p;

    // Parameters
    int RESTARTS = 5;
    int ITERATIONS = 2000; // Sufficiently large given N=2000

    for (int r = 0; r < RESTARTS; ++r) {
        p.resize(N);
        iota(p.begin(), p.end(), 0);
        shuffle(p.begin(), p.end(), gen);

        long long current_cost = 0;
        for (int i = 0; i < N; ++i) {
            int loc_i = p[i];
            for (int j = 0; j < N; ++j) {
                if (F[i * N + j]) {
                    current_cost += getD(loc_i, p[j]);
                }
            }
        }

        if (current_cost < current_best_cost) {
            current_best_cost = current_cost;
            best_perm = p;
        }

        // Local Search
        for (int iter = 0; iter < ITERATIONS; ++iter) {
            int i = dis(gen);
            int k = dis(gen);
            if (i == k) {
                iter--; // Retry this iteration
                continue;
            }

            // Calculate delta
            int loc_i = p[i];
            int loc_k = p[k];

            int delta = 0;

            // Self
            if (F[i * N + i]) delta += (getD(loc_k, loc_k) - getD(loc_i, loc_i));
            if (F[k * N + k]) delta += (getD(loc_i, loc_i) - getD(loc_k, loc_k));

            // Cross
            int cross_term = getF(i, k) * (getD(loc_k, loc_i) - getD(loc_i, loc_k));
            cross_term += getF(k, i) * (getD(loc_i, loc_k) - getD(loc_k, loc_i));
            delta += cross_term;

            // Others
            // To optimize inner loop, use simple vars
            // We iterate x, skip i and k.
            // Accessing F and D arrays.
            // Access F[i][x] etc.
            
            // Optimization: Pre-calculate diffs if possible? No.
            // Just loop.
            for (int x = 0; x < N; ++x) {
                if (x == i || x == k) continue;
                
                int f_ik_x = 0, f_xi_x = 0; // Wait, notation clash.
                // F[i][x], F[x][i]
                int f_pos = 0;
                if (F[i * N + x]) f_pos++;
                if (F[x * N + i]) f_pos++;

                int f_neg = 0;
                if (F[k * N + x]) f_neg++;
                if (F[x * N + k]) f_neg++;

                if (f_pos != f_neg) {
                    int d_diff = getD(loc_k, p[x]) - getD(loc_i, p[x]);
                    delta += (f_pos - f_neg) * d_diff;
                }
            }

            if (delta < 0) {
                // Apply swap
                // Swap locations for facility i and k
                // p stores location of facility i at index i
                // So swapping p[i] and p[k] achieves moving fac i to loc_k, fac k to loc_i
                int t = p[i];
                p[i] = p[k];
                p[k] = t;
                
                current_cost += delta;

                if (current_cost < current_best_cost) {
                    current_best_cost = current_cost;
                    best_perm = p;
                }
            }
        }
    }

    for (int i = 0; i < N; ++i) {
        cout << (best_perm[i] + 1) << (i == N - 1 ? "" : " ");
    }
    cout << "\n";

    return 0;
}
```
One small error in my thinking:
`best_perm = p;` copies the vector. It's correct.
But `best_perm` should reflect the global minimum found across restarts.
The `best_perm` is updated whenever `current_best_cost` improves.
This is handled.

Wait, `best_perm` is updated with `p`.
Since `p` is modified in place during iterations.
If `p` finds a better cost, we save it to `best_perm`.
If later in another iteration `p` degrades but another restart finds better, `best_perm` might be stale?
No, `best_perm` is overwritten with `p` (which is the current state) whenever `current_cost < current_best_cost`.
So `best_perm` always tracks the best state seen.
Correct.

Final verification of problem statement.
"Output exactly one line: n integers".
Space separated.
My code outputs space separated.
Format is correct.

Wait, `iter--` logic:
If `i==k`, I retry.
If `dis(gen)` produces same `i` twice, `continue` skips loop body but decrements `iter`.
Actually `if (i==k) { ...; continue; }` inside the loop.
Wait, if `continue` is hit, loop body finishes immediately, but `iter` increments?
No. `continue` goes to next iteration of `for`.
So if I encounter a collision, I increment `iter`. That means wasted iteration.
I should decrement `iter`? No, `for` loop increments automatically.
If I want to retry *without* consuming an iteration count (from fixed budget), I can `iter--`.
But `dis(gen)` takes time. Just ignore it?
Better to not decrement but accept the cost.
But wasting time on invalid swaps reduces efficiency.
With $ITERATIONS = 2000$, waste is negligible.
I'll remove `iter--` to simplify.
Or just `do { ... } while (i == k)` loop.
`while (i == k) { i = dis(gen); k = dis(gen); }`
Then proceed.

Revised logic:
```cpp
            int i = dis(gen);
            int k = dis(gen);
            while (i == k) {
                i = dis(gen);
                k = dis(gen);
            }
```
This ensures distinct `i, k` without impacting iteration count logic (just skips bad draws immediately).
However, inside `for`, `iter` still counts down.
This is fine.

Memory: `vector<int>` uses heap.
Headers: `iostream`, `vector`, `numeric` (iota), `algorithm` (shuffle), `random` (mt19937), `chrono` (steady_clock).
All standard.

One final check: `min_cost` variable used.
Wait, `best_perm` initialization.
At start `best_perm` = `p` (identity).
Then restarts loop.
Inside restarts, `best_perm` is updated.
Initially `best_perm` is identity.
The first restart sets `p` to random, computes cost, compares with `current_best_cost` (init `min_cost`).
Wait, I initialized `current_best_cost = min_cost`.
Wait, `min_cost` was computed on identity `p`.
So `current_best_cost` starts with cost of identity.
Then inside restart, `current_cost` is computed on random `p`.
If `current_cost < current_best_cost`, update `best_perm`.
Correct.
However, inside loop, if `current_cost` drops below, update.
If a better `p` is found, it is stored in `best_perm`.
The logic holds.

Another small thing:
`getD(loc_k, loc_k)` access.
`loc_k` is index in `0..N-1`.
Valid.

Seems ready.

One concern: `long long` for `cost`.
Is `int` sufficient?
Max cost $N^2$. $4 \cdot 10^6$.
`int` is fine (up to $2 \cdot 10^9$).
But `long long` prevents overflow if I'm paranoid about something weird.
Used `long long` everywhere. Safe.

The code looks good.

```cpp
// Final check of code structure
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <chrono>
// ...
```
Okay, generating response.

Wait, I should confirm the `shuffle` seed.
Using `chrono` seed is better for variety in tests.

Final check on scoring: `score = 1 - Cost / TotalFlow`.
Minimizing Cost maximizes score.
We output the permutation.
Everything consistent.

One edge case: `N=2`.
Loops run fine.
Matrix access fine.

Time complexity: $O(R \cdot I \cdot N)$. $R=5, I=2000, N=2000 \to 2 \cdot 10^7$.
Input reading $O(N^2) = 4 \cdot 10^6$.
Total time roughly $0.1 - 0.2$s.
Memory $16 + 16 = 32$MB.
Limits satisfied.

Double check the `calc_delta` logic again.
It assumes that `loc_i` is the current location of `i` and `loc_k` is current location of `k`.
Before swap: `p[i]` is `loc_i`, `p[k]` is `loc_k`.
After swap: `p[i]` is `loc_k`, `p[k]` is `loc_i`.
My code:
`delta += ... getD(loc_k, p[x]) - getD(loc_i, p[x]) ...`
This corresponds to `p[x]` remaining at location `p[x]`, but `p[i]` moved to `loc_k`.
If `p[x]` is the *same* location as `p[i]` (i.e., `x` is already at `loc_i`? No `x` is facility index).
Wait, `p[x]` is the location of facility `x`.
If we swap `i` and `k`, facilities `i` and `k` swap locations.
Facility `x` stays at its location.
So `loc_x` (location of `x`) is `p[x]` (value in `p` before swap).
My code uses `p[x]`.
Is `p[x]` modified?
No, `p` is modified ONLY after swap is accepted.
So inside the delta calculation block (before modifying `p`), `p[x]` reflects the current configuration.
Correct.
So `p[x]` is indeed the location of facility `x`.
Correct.

Wait, inside the loop for `x`, if `x` happens to be `i` or `k`, we skip.
But what if `p[x]` (location of `x`) equals `loc_i` or `loc_k`?
This doesn't matter. `loc_i` and `loc_k` are the locations being swapped.
`p[x]` is the target location for the interaction term.
If `x=i`, we skip.
Wait, I skipped `x=i` and `x=k`.
I treated `i` and `k` interactions separately (Self and Cross terms).
Is it possible that `x` is some facility, but `p[x]` coincides with `loc_i`?
Yes, multiple facilities might be at same location?
NO. It's a permutation. Each location is unique.
So `p[x]` is unique for each `x`.
`p[i]` and `p[k]` are distinct locations.
So `p[x]` will never equal `loc_i` or `loc_k` for `x` distinct from `i` and `k` (assuming `x` is facility index).
Wait, `p[x]` is the location assigned to facility `x`.
Since it's a permutation, all `p[x]` are distinct.
So `p[x]` for `x != i, k` is definitely not `loc_i` (which is `p[i]`) or `loc_k` (which is `p[k]`).
So skipping `x=i` and `x=k` is logically sufficient to cover all interactions involving `loc_i` and `loc_k` except self/cross which are explicit.
Logic holds.

Final check: `getF(i*k)` syntax?
`getF(i * N + k)`?
Yes, `getF` takes `r, c`.
My lambda: `[&](int r, int c)`.
Call: `getF(i * N + k)`?
Wait, inside `delta`, I used `getF(i, k)`.
Lambda signature in `main` was `getF(int r, int c)`.
But I passed `i * N + k`? No.
In `main`, `auto getF = [&](int r, int c)`.
In loop: `getF(i, k)`.
Wait, in `getF` I implemented `return F[r * N + c]`?
Yes.
Wait, I defined `auto getF = [&](int r, int c) { return F[r * N + c]; };`.
But earlier in `main` code snippet I wrote `getF(i * N + j)` in loop?
Wait, if `getF` takes `r, c`, I should pass `i, k`.
My code: `int cross_term = getF(i, k) * ...`
Wait, `F` is flattened.
So `F[i*N+k]` is correct access.
If `getF(r,c)` implements `return F[r*N+c]`, then `getF(i,k)` is correct.
Wait, earlier in my thought trace I had `getF(i*k)`. No.
The final code block has `getF(i, k)`.
However, I need to make sure `getF` lambda is consistent.
In my drafted code: `auto getF = [&](int r, int c) { return F[r * N + c]; };`.
This takes two args.
Wait, `cin >> F[i * N + j]` uses index.
So `getF` should unpack or just use index?
Better to use index directly in `delta` to avoid passing args and doing mult inside lambda?
Actually `auto getF_idx = [&](int idx) { return F[idx]; };`.
This is faster and simpler.
Same for `D`.
Let's redefine helpers to take single index.
`auto getD_flat = [&](int idx) { return D[idx]; };`
Then `delta += getD_flat(loc_k * N + loc_k) - ...`
Wait, `getD` access `D[r*N+c]`.
Let's just use `D` directly.
It avoids function call overhead? Negligible but cleaner.
`int get_loc_d = D[loc_k * N + loc_k]`.
I will use direct array access in `delta` calculation.
In `compute_cost` loop, I used `D[loc_i * N + p[j]]`.
So I will just use indices.

Refined helpers:
None needed if I trust compiler.
But `getD` lambda is fine.

I'll stick to the helpers for readability.

Wait, `uniform_int_distribution` constructor.
Need `#include <random>`.
Included.

Ready to generate.
</think>

```cpp
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <chrono>

using namespace std;

int main() {
    // Optimize I/O operations for speed
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int N;
    if (!(cin >> N)) return 0;

    // Flattened matrices for cache efficiency
    // Size N*N. N <= 2000 => 4,000,000 elements.
    // Using vector<int> ensures they are allocated on heap.
    vector<int> D(N * N);
    vector<int> F(N * N);

    // Read Distance Matrix D
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            cin >> D[i * N + j];
        }
    }
    // Read Flow Matrix F
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            cin >> F[i * N + j];
        }
    }

    // Helper lambdas to access matrix elements
    // Using index arithmetic directly avoids overhead, though lambdas are clear
    auto getD = [&](int r, int c) { return D[r * N + c]; };
    auto getF = [&](int r, int c) { return F[r * N + c]; };

    // Permutation p: p[i] stores the location assigned to facility i (0-indexed)
    vector<int> p(N);
    iota(p.begin(), p.end(), 0);

    // Function to calculate total cost
    auto compute_cost = [&]() -> long long {
        long long cost = 0;
        for (int i = 0; i < N; ++i) {
            int loc_i = p[i];
            for (int j = 0; j < N; ++j) {
                if (F[i * N + j]) {
                    int loc_j = p[j];
                    cost += getD(loc_i, loc_j);
                }
            }
        }
        return cost;
    };

    // Initial cost calculation (Identity permutation is a starting point)
    long long current_best_cost = compute_cost();

    // Random Number Generators
    mt19937 gen((uint64_t)chrono::steady_clock::now().time_since_epoch().count());
    uniform_int_distribution<int> dist(0, N - 1);

    // Local Search Parameters
    const int NUM_RESTARTS = 5;
    const int ITERATIONS_PER_RESTART = 2000; // Balance between exploration and time limit

    vector<int> best_perm = p;

    for (int r = 0; r < NUM_RESTARTS; ++r) {
        // Initialize current permutation with a shuffled version
        p.assign(N, 0);
        iota(p.begin(), p.end(), 0);
        shuffle(p.begin(), p.end(), gen);

        // Calculate cost for the new random permutation
        long long current_cost = compute_cost();

        // Track best solution found across all restarts
        if (current_cost < current_best_cost) {
            current_best_cost = current_cost;
            best_perm = p;
        }

        // Hill Climbing / Local Search
        for (int iter = 0; iter < ITERATIONS_PER_RESTART; ++iter) {
            // Select two distinct facilities to swap
            int i = dist(gen);
            int k = dist(gen);
            while (i == k) {
                i = dist(gen);
                k = dist(gen);
            }

            // Current locations of the two facilities
            int loc_i = p[i];
            int loc_k = p[k];

            // Calculate the change in cost (delta) if we swap p[i] and p[k]
            // Swapping i and k means:
            // Facility i moves to loc_k (formerly k's location)
            // Facility k moves to loc_i (formerly i's location)
            
            int delta = 0;

            // 1. Self-interaction terms (diagonal elements F[i][i] * D[i][i])
            if (F[i * N + i]) {
                delta += getD(loc_k, loc_k) - getD(loc_i, loc_i);
            }
            if (F[k * N + k]) {
                delta += getD(loc_i, loc_i) - getD(loc_k, loc_k);
            }

            // 2. Cross-interaction between i and k (terms F[i][k] and F[k][i])
            // Original: F[i][k] * D[loc_i][loc_k] + F[k][i] * D[loc_k][loc_i]
            // New: F[i][k] * D[loc_k][loc_i] + F[k][i] * D[loc_i][loc_k]
            int diff_cross = F[i * N + k] * (getD(loc_k, loc_i) - getD(loc_i, loc_k))
                           + F[k * N + i] * (getD(loc_i, loc_k) - getD(loc_k, loc_i));
            delta += diff_cross;

            // 3. Interactions with all other facilities x (x != i, k)
            // For facility x at loc_x = p[x]:
            // Change in contribution due to i moving: F[i][x]*(D[loc_k][loc_x] - D[loc_i][loc_x])
            // Change in contribution due to k moving: F[k][x]*(D[loc_i][loc_x] - D[loc_k][loc_x])
            // And symmetric terms F[x][i], F[x][k].
            // Combined coefficient for D[loc_k][loc_x] is (F[i][x] + F[x][i])
            // Combined coefficient for D[loc_i][loc_x] is (F[k][x] + F[x][k])
            
            for (int x = 0; x < N; ++x) {
                if (x == i || x == k) continue;

                int coeff_k = 0;
                if (F[i * N + x]) coeff_k++;
                if (F[x * N + i]) coeff_k++;

                int coeff_i = 0;
                if (F[k * N + x]) coeff_i++;
                if (F[x * N + k]) coeff_i++;

                if (coeff_k == coeff_i) continue;

                int d_diff = getD(loc_k, p[x]) - getD(loc_i, p[x]);
                delta += (coeff_k - coeff_i) * d_diff;
            }

            // If swap reduces cost, apply it
            if (delta < 0) {
                p[i] = loc_k;
                p[k] = loc_i;
                current_cost += delta;

                // Update global best
                if (current_cost < current_best_cost) {
                    current_best_cost = current_cost;
                    best_perm = p;
                }
            }
        }
    }

    // Output the best permutation found
    // Convert 0-based index to 1-based for output
    for (int i = 0; i < N; ++i) {
        cout << (best_perm[i] + 1) << (i == N - 1 ? "" : " ");
    }
    cout << "\n";

    return 0;
}
```
````

---

## 直接SFT 样本(sample_idx=4,得分 0.0,completion_tokens=1025)

````
Let me analyze this problem:

1. We have an n×n distance matrix D and a flow matrix F
2. We need to find a permutation that assigns facilities to locations to minimize cost
3. Cost = Σᵢ Σⱼ D[p(i), p(j)] * F[i, j]
4. Score = 1 - Cost / TotalFlow (so we want to maximize score, which means minimizing Cost)

This is a Quadratic Assignment Problem (QAP). The general QAP is NP-hard. However, the scoring formula gives us a clue about the scale of costs.

Let me think about what constraints would make this tractable:

For a valid binary solution (score between 0 and 1), we need Cost ≤ TotalFlow since score ≥ 0.

In the worst case where all distances are 1 and flows could be up to n²:
- Max possible Cost with all 1s in D: Each term D[p(i),p(j)]*F[i,j] ≤ F[i,j], summing over all i,j: max Cost = TotalFlow when all distances contribute fully
- But D contains only 0s and 1s, and each distance value contributes at most once per position

Actually, let me reconsider: for the optimal assignment, the maximum cost should satisfy Cost/TotalFlow < 1, so Cost < TotalFlow.

The total number of pairs is n*(n+1)/2 distinct location assignments for each of the n facility pairs, giving ~n³ terms, each bounded by F[i,j].

For the algorithm to work within time limits (typically ~1 second), if we use a greedy approach iterating through positions:
- Permuting takes O(n!) worst case, but we'd optimize with backtracking/greedy
- Each evaluation is O(n²) for computing one cost function


- To stay under 10⁸ operations, evaluating around 500,000 full cost functions suggests each should run in roughly 256 ns, which aligns with simple operations counting ~1M ops per second.

I can construct test cases to verify the algorithm's behavior - like a small 2x2 instance where D and F have opposite patterns, forcing either facility ordering to minimize the cost product. Let me try implementing it in C++.
</think>

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    cin >> n;
    vector<vector<int>> d(n, vector<int>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            cin >> d[i][j];

    vector<vector<int>> f(n, vector<int>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            cin >> f[i][j];

    int bestPerm[2000];
    int bestCost = INT_MAX;
    int totalFlow = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            totalFlow += f[i][j];

    sort(bestPerm, bestPerm + n);

    for (int i = 0; i < n; i++)
        bestPerm[i] = i + 1;

    do {
        int curCost = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (!f[i][j]) continue;
                int li = upper_bound(bestPerm, bestPerm + n, i) - bestPerm;
                int lj = upper_bound(bestPerm, bestPerm + n, j) - bestPerm;
                curCost += d[li - 1][lj - 1];
            }
        }
        if (curCost < bestCost) {
            bestCost = curCost;
            memcpy(bestPerm + i, bestPerm + i, sizeof(int));
        } else {
            break;
        }
    } while (next_permutation(bestPerm, bestPerm + n));

    for (int i = 0; i < n; i++) {
        cout << (bestPerm[i] == bestPerm[n - 1] ? bestPerm[i] : bestPerm[n - 1]);
        if (i + 1 != n) cout << " ";
    }
    cout << "\n";
    return 0;
}
```
````
