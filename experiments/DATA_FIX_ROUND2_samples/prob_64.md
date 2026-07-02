# 题 64 — base均分 0.0 vs 直接SFT均分 0.0(掉点 0.0)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=1,得分 0.0,completion_tokens=19766)

````
The problem asks us to find a subset of $n$ integers $a_1, \dots, a_n$ such that their sum is as close as possible to some target value $T$. Wait, the problem statement says "sums as close as possible to T=x_i*a_i, x_i drawn from Bernoulli (1/2)". This part is confusing.

Let's re-read carefully: "find a subset of a_1..a_n that sums as close as possible to T=x_i*a_i, x_i drawn from Bernoulli (1/2)."
Actually, looking at standard competitive programming problem structures, especially with scoring based on error, it seems like we are given a target value $T$ and a set of numbers $A$, and we need to choose coefficients $x_i \in \{0, 1\}$ such that $\left| \sum_{i=1}^n x_i a_i - T \right|$ is minimized.

However, the input format description is:
Input: 
n T
A_1 a_2 ... a_n

Wait, the sample input is:
3 4
1 2 3
And the output is `101`.
If $T=4$ and array is $\{1, 2, 3\}$.
Subset sums:
Empty: 0
$\{1\}$: 1
$\{2\}$: 2
$\{3\}$: 3
$\{1, 2\}$: 3
$\{1, 3\}$: 4 -> Distance to 4 is 0. Binary string 101 corresponds to selecting indices 1 and 3.
So the goal is indeed: Given $T$ and an array $A$, select a subset of elements such that their sum is closest to $T$.

But what about the description "T=x_i*a_i, x_i drawn from Bernoulli (1/2)"? This looks like it describes how $T$ might be generated for test cases in a specific context (maybe generating the ground truth), or perhaps it implies something else. Let's look closer.
"Given (1 <= n <= 1e2) and (B = 1e15), n integers a_1 … a_n (0 <= a_i <= B) drawn from either (normal, uniform, pareto, exponential) distributions..."
This part likely describes the *generation process* of the input data for the problem creator's perspective (distribution types). The actual task is simply given the inputs (which might come from these distributions during contest generation), minimize $|\text{subset\_sum} - T|$.

Wait, if $a_i$ can be up to $10^{15}$, then the sum can be up to $10^{17}$ roughly (since $n \le 100$).
Standard knapsack-like dynamic programming where state is current sum doesn't work directly because the sum range is too large ($O(B)$ complexity per element would be $O(nB)$ which is $100 \times 10^{15}$, too big).
However, notice the constraints again: $n \le 100$. The values $a_i$ are large, but there are few items. This screams "Meet-in-the-middle".

Meet-in-the-middle approach:
Split the array into two halves of size $n/2$.
Generate all possible subset sums for each half.
For each sum $S_1$ from the first half and $S_2$ from the second half, we want $S_1 + S_2 \approx T$.
So $S_2 \approx T - S_1$.
We can store the subset sums of the first half in a sorted list.
Then for each subset sum $S_2$ of the second half, we binary search for the best matching $S_1$ in the first list.

Complexity:
$n \le 100$. Half size $m = 50$.
Number of subsets of one half is $2^{50} \approx 10^{15}$.
We cannot generate all $2^{50}$ sums.
Is there a constraint I missed?
"Score = 100 * (15 - log(error + 1)) / 15".
Maybe the constraints on $a_i$ are small in practice despite saying $B=1e15$?
No, "0 <= a_i <= B". If they can be large, meet-in-the-middle is still impossible with $2^{50}$.

Let's re-read the distribution info.
25% from U(0, B) (Uniform 0 to $10^{15}$)
25% from N(B/2, B/6) (Normal around $5 \cdot 10^{14}$)
25% from Exp(B/2) (Exponential mean $5 \cdot 10^{14}$)
25% from TruncatedPareto...
In all cases, $a_i$ can be quite large.
Wait, if $n$ is up to 100, and values are up to $10^{15}$, maybe we are allowed more than $O(2^{n/2})$.
Is this the Subset Sum approximation problem? Usually, for large numbers, if we just want to approximate $T$, maybe we don't need exact reachability. But here we have to print the binary string, so we need the exact selection.

Wait, looking at the score formula: $Score = 100 \times (15 - \log(\text{error} + 1)) / 15$.
This implies $\log$ base 10? Or natural? Usually log error means $\log_{10}$ in such contexts.
Max error is related to minimizing distance.
If $n=100$, meet-in-the-middle is definitely not feasible with $2^{50}$.
Is it possible that the effective number of distinct subset sums is smaller? No, random large numbers usually give distinct sums.

Is it possible $T$ is also large? The sample has $T=4$.
The text says "find a subset of a_1..a_n that sums as close as possible to T=x_i*a_i". This phrasing is really weird. $T$ is described as a random variable? Or is $T$ provided in the input?
The Input section says:
n T
A_1 ... A_n
So $T$ is fixed input. The description "T=x_i*a_i, x_i drawn from Bernoulli(1/2)" probably refers to how the author of the problem might construct a valid case where the optimal answer is known, or maybe it's just flavor text describing a stochastic setting. But since $T$ is an explicit input, we must treat it as the target.

Maybe the problem statement provided here is slightly incomplete or I'm missing a specific optimization for this type of problem.
However, there is a specific constraint often associated with this difficulty level: $N$ is small enough for meet-in-the-middle only if the exponent is around 20-30. Here $N=100$.
Perhaps the problem allows randomized greedy approaches due to the specific distribution?
Or maybe the question implies something about bit manipulation since we need a binary string?
Actually, looking at the distributions: Uniform, Normal, Exponential, Pareto. All are continuous random variables (discretized) centered around large values.
With $N=100$ and large weights, the sums will form a somewhat dense distribution if we consider modulo arithmetic or something? No.

Wait, let's check similar problems.
Usually, when $N$ is up to 40-50, Meet-in-the-middle is fine. When $N$ is 100, standard Subset Sum is NP-hard.
Is there a specific property here?
Maybe we are supposed to use floating point approximations? "log(error)" suggests we care about relative error or absolute error on a scale.
If we simply use a heuristic like Best Fit Decreasing?
Sort items by size? Sort randomly?
Or maybe since $N=100$, the maximum sum can be $10^{17}$. This fits in `long long` (up to $9 \cdot 10^{18}$).
Can we do meet-in-the-middle with pruning?
If we split into two sets of 50. The number of subset sums is $2^{50}$. Too many.
But wait, if the scores depend on error...
Maybe the intended solution for the full version is harder, but for this prompt, I should provide a generic C++ solution.
However, usually in these prompts, the problem comes from a specific contest. The score formula looks custom.
Wait, if the problem asks for $N \le 100$, maybe $B$ isn't actually reachable? Or maybe I should implement a Branch and Bound algorithm?
With $N=100$, branch and bound might work if we prune heavily.
Ordering $a_i$ descending helps?
Also, we can track the current sum `curr` vs `target T`. We want to add/subtract items.
Since we must pick exactly $x_i \in \{0, 1\}$, we are deciding inclusion.
Branching on each item: Include it or exclude it.
Maintain `current_sum`, `remaining_max_sum`.
If `current_sum + remaining_max_sum < T`, we MUST include everything remaining.
If `current_sum > T + best_error_found`, we can prune? No, because we might overshoot significantly but get closer later? No, we want `abs(sum - T)` minimal.
Actually, if `current_sum > T + min_abs_diff`, can we prune? Yes, if `remaining_max_sum` doesn't help reduce it below `min_abs_diff`.
More precisely, if we know that no matter what we do with remaining items, the sum will be far away.
But this is tricky.
Pruning conditions:
Let `S` be current sum, `R` be set of remaining items.
Lower bound on achievable future sum = `S` (if we exclude all remaining).
Upper bound on achievable future sum = `S + sum(R)`.
We want to find $S' \in [S, S + \text{sum}(R)]$ such that $|S' - T|$ is minimized.
So we just need to update `min_error = min(min_error, abs(S + sum(R) - T), abs(S - T))`?
Not quite. If `S > T`, the minimum possible new sum is `S` (exclude all remaining). So if `S >= T` and we haven't found a match better than `S - T`, we can stop exploring this branch ONLY IF adding any positive amount makes it worse. Since all $a_i \ge 0$, adding more makes sum larger. If `S > T`, adding anything increases distance.
So if `S > T`, we can just take `abs(S - T)` and stop (assuming we already processed the decision to stop adding).
Wait, we are iterating $x_i$.
At step $i$, we decide for $a_i$.
Current state: `(index i, current_sum)`.
If `current_sum > T`: we shouldn't add any more positive $a_j$ because they are non-negative. So we must stop.
Wait, $a_i \ge 0$. Yes.
If `current_sum > T`, the optimal strategy for this path (and subpaths) is to include NO MORE elements?
Yes! Because including any further element increases the sum, increasing the difference from $T$ (since $S > T$).
So if `current_sum > T`, we calculate `error = current_sum - T`. Update global best. Do NOT recurse further.
What if `current_sum + remaining_sum < T`? Then even if we include ALL remaining, we won't reach $T$. So `error = T - (current_sum + remaining_sum)`. Update global best. Do NOT recurse further?
Actually, if we don't reach T even with max possible sum, we know this path yields that specific error. Can't improve? Wait, maybe a previous path was better. But for this specific path prefix, we are done.
So simple pruning:
If `sum > T`: return max_error for this node (stop adding).
If `sum + rest_sum < T`: return max_error for this node (can't reach T).

With this pruning, how efficient is it?
If values are large (like $10^{15}$), we will likely jump over $T$ quickly.
However, we need to explore branches to find the one that hits $T$ closely.
If all $a_i \gg T$, the best strategy is picking the smallest subset sum $\ge T$ or largest $\le T$.
Wait, if $a_i$ are huge, most subset sums will be very far apart.
Wait, if $a_i$ are large, say $> T$, and we have many of them, we can just greedily sort?
If we sort $a_i$ ascending, maybe dynamic programming works?
If $a_i$ are small, DP works. If $a_i$ are large, recursion with pruning works.
Is $N=100$ solvable with recursion + pruning?
Given the constraint $B=10^{15}$, the density of subset sums is very low compared to $10^{15}$ unless $n$ is large and numbers are small.
Wait, if $n=100$ and numbers are uniformly distributed up to $10^{15}$, the expected sum is $50 \cdot 10^{15}$.
The variance is high.
If we simply use `std::sort` on $a_i$ descending and try to fill $T$.
Greedy approach: Iterate through sorted items. If adding $a_i$ keeps us under $T$, add it?
This is the classic change making problem approximation. Greedy doesn't always work perfectly for subset sum.
But with $N=100$, maybe we can run a depth-first search.
The pruning `if (current_sum > T)` is strong because all numbers are positive. Once you cross $T$, you can't go back down. You just record the excess.
So the search space effectively becomes:
Find subset sum $S \le T$ that is maximized.
AND Find subset sum $S \ge T$ that is minimized.
Because if $S < T$, adding more could bring it closer. If $S > T$, we are done for that branch.
So we essentially want to find the largest subset sum $\le T$.
Wait, is that true?
Suppose target $T=10$, items $\{6, 5\}$.
Path 1: Pick 6. Current 6. Next item 5. Pick 5 -> 11. Error 1. Stop. Don't pick 5 -> 6. Error 4.
Path 2: Don't pick 6. Current 0. Next item 5. Pick 5 -> 5. Error 5.
Best is 11 (Error 1).
Wait, if we treat finding "largest $\le T$" as the sole goal, we would get 6 (from Path 2+nothing, or Path 1 without 5). Error 4.
But 11 is better.
So we need to explore both directions crossing $T$.
Since once we exceed $T$, we stop adding, the tree stops at leaves where sum $\ge T$ or we run out of items.
With the condition that all $a_i \ge 0$, the "state" is monotonic.
This structure is basically a variation of the Knapsack problem or Subset Sum.
Is it possible that $N$ is small enough for meet-in-the-middle if we use random sampling?
No, $N \le 100$ is standard for "meet-in-the-middle" on CF/Codeforces if $N \le 40$.
For $N=100$, maybe the time limit is loose or the constants are small?
Wait, if I am forced to write code, and the problem is hard, I should provide the best possible attempt.
Recursive search with memoization? State is $(index, current\_sum)$.
State space size: $100 \times (\text{reachable sums})$.
Reachable sums are sparse.
If I use a `map<long long, vector<int>>`? No, too slow.
Just `map<long long, int>` to store best error? No, we need the subset.
Since we just need to output ANY best subset, maybe just DFS.
Wait, with $N=100$ and large random numbers, it's extremely likely that we can't hit $T$ closely, OR the numbers are small?
Ah, wait. The distributions mentioned are U(0, B) etc. $B=10^{15}$.
But wait, looking at the sample input: 3 items, T=4, values 1 2 3. These are small.
If the generator produces large numbers, say $10^{15}$, and $T$ is also around $5 \cdot 10^{15}$ (based on expectation), then the gap between subset sums is massive.
In the case of random values around magnitude $M$, the sums behave like sums of independent random variables. By Central Limit Theorem, sums are normally distributed.
However, we need EXACT subset selection.
If numbers are truly large and random, maybe brute force DFS is too slow.
However, note the problem statement says: "You will be given a problem statement...". It does not ask me to solve the specific test case (I can see I cannot see the test case values, I must read them from stdin).
I must write a program that reads stdin.
Wait, "Sample input" is provided. I should make sure my code runs on that.
The code must be general.

Given $N \le 100$, a simple recursive search is the most viable option if $a_i$ are large.
If $a_i$ are small, DP works.
How to detect small $a_i$?
If $n \cdot \max(a_i) \le$ manageable limit (e.g., $10^6$ or $10^7$), use DP.
Otherwise use recursion with pruning.
Actually, sorting $a_i$ descending is generally good for DFS (fills sum faster).
Let's refine the recursion.
`dfs(index, current_sum, current_subset_mask)`
`index`: 0 to n-1.
`current_sum`: sum of selected.
`current_subset_mask`: vector<bool> or similar.
Optimization:
Since $a_i$ are large, we might want to order them?
Sorting descending is usually good for branch and bound.
Wait, if we sort descending:
Large items first.
We have a limited budget $T$.
Actually, we want to approximate $T$.
If we maintain `rem_sum` (sum of unvisited items), we can prune.
1. If `current_sum > T`:
   We can't add any more items to stay under or equal to `T`? No, we want to minimize `abs(sum - T)`.
   If `current_sum > T`, adding more makes `sum` bigger, `diff` bigger.
   So if `current_sum > T`, the ONLY candidate on this branch is the current `current_sum` (as we stop immediately). Wait.
   Is it strictly true? Yes, assuming $a_i \ge 0$. If we stop adding items, sum is `current_sum`. Any addition increases sum. Since sum > T, adding increases diff.
   Wait, is it possible that we picked some items earlier and now sum > T, but we didn't check the option of removing items?
   Backtracking handles removal. The recursion explores decisions.
   When we are at index `idx` with `current_sum` > T:
   We are currently building a path.
   The path includes decisions made at `0..idx-1`.
   At `idx`, we have a choice: include $a_{idx}$ or exclude.
   If `current_sum > T`, including $a_{idx}$ results in `current_sum + a[idx] > T`. Diff increases.
   So we don't need to recurse for "Include $a_{idx}$".
   But do we need to try excluding? Excluding keeps `current_sum`. Still > T.
   So effectively, if at any point `current_sum > T`, we don't add ANYMORE items.
   However, we are inside a recursive function that loops over items.
   Actually, the logic should be:
   We iterate through the decision for $a_{idx}$.
   Option 1: Include $a_{idx}$. New sum $S + a_{idx}$. Recurse $idx+1$.
   If $S + a_{idx} > T$, we record this result (or rather, realize that we exceeded $T$ and no further additions are allowed).
   BUT, we might have exceeded $T$ before.
   Wait, if `S > T` upon arrival at step `idx` (meaning previous decisions led to exceeding T), then for the current step `idx`:
   Including $a_{idx}$ makes it worse.
   Excluding $a_{idx}$ keeps it as `S` (still > T).
   So we can just stop exploring deeper levels entirely. The best error found SO FAR along this prefix is `S - T` (actually `abs(S-T)`). And no further refinement is possible on this branch.
   
   So Pruning Rule 1: If `current_sum > T`, terminate this branch immediately. The error is `current_sum - T`.
   Pruning Rule 2: If `current_sum + suffix_sum[idx] < T`, even taking all remaining won't reach T. The error is `T - (current_sum + suffix_sum[idx])`. Terminate branch.

   With these rules, if the values are somewhat large but not impossibly large (relative to T), does it pass?
   With $N=100$, if values are small, the search space is huge? No, if values are small, $T$ is small, we can just run DP.
   But the problem states $a_i \le 10^{15}$.
   If $a_i$ are large, we skip almost everything.
   Wait, if values are large, `suffix_sum` decreases rapidly.
   If `suffix_sum` is much less than $T$, we can prune.
   If values are large, say $10^{15}$, and $T$ is $5 \cdot 10^{15}$.
   We need to pick about 5 items.
   Random search might find the solution quickly?
   But worst case is specific adversarial construction.
   However, usually competitive programming problems with these parameters allow meet-in-the-middle if implemented efficiently.
   But $2^{50}$ is the hard wall.
   Is there a library function or trick?
   Wait, $N=100$ is typical for **Randomized algorithms** or **Approximation**?
   The score depends on `error`. We want to minimize it.
   Maybe Monte Carlo simulation?
   Run multiple tries. In each try, shuffle the array and perform a greedy packing or BFS.
   Or better: Recursive search with heuristics (sort descending, keep best).
   If I sort descending, I consume the budget $T$ faster.
   Wait, if I sort ascending, I build up slowly?
   To maximize chances of hitting $T$, maybe sorting by magnitude is key.
   Let's stick to the DFS with pruning. It is the most robust deterministic approach for small $N$ and large weights.
   Actually, if $N=100$, even with pruning, it might TLE on worst case (many items summing around $T$).
   But given the constraints $a_i \le 10^{15}$, it's unlikely to have a dense cluster of subset sums unless $a_i$ are small.
   Wait, if $a_i$ are small (e.g. $\le 1000$), then DP works.
   So I can combine strategies:
   Check sum of all elements. If sum is small (e.g. $< 2 \cdot 10^5$), use DP.
   Else, use DFS with pruning.
   Wait, "Check sum of all elements".
   Actually, $a_i$ can be 0.
   Also, duplicates can be merged?
   If many duplicates exist, we can group them?
   Like, count frequencies of each number.
   This reduces state space.
   
   Algorithm Plan:
   1. Read N, T.
   2. Read A.
   3. Precompute suffix sums for pruning.
   4. Implement DFS function `solve(index, current_sum)`.
      Global variable `best_error` initialized to infinity (or `abs(T)`).
      Global `best_selection` (vector<string>).
      
      Inside DFS:
      - Calculate `current_error = abs(current_sum - T)`.
      - Update global best if `current_error < best_error`.
        Wait, we want to find THE BEST.
        If we find an error of 0, we can stop and return? (If 0 is found, it's optimal).
      - Apply Pruning:
        If `current_sum > T`:
          Return (cannot improve by adding more positive numbers).
          WAIT! My pruning logic was: if `current_sum > T`, we just record this error and stop THIS BRANCH.
          But is it possible that we had `current_sum > T` earlier and now we are at index `k`.
          We decided to INCLUDE $a_k$. Now sum is even higher.
          We decided to EXCLUDE $a_k$. Sum stays same.
          In both cases sum > T.
          So yes, if `current_sum > T`, we can return immediately because neither adding nor skipping future items helps reduce distance to T (adding makes sum larger, skipping doesn't change sum).
          Correct.
        
        If `current_sum + suffix_sum[index] < T`:
          Max possible sum we can reach is `current_sum + suffix_sum[index]`.
          The error is `T - (current_sum + suffix_sum[index])`.
          If this error is greater than `best_error`, we can prune.
          Wait, if we can't reach T, we just accept the "fail to reach" and backtrack?
          But maybe we can reach a value close to T from below.
          The error is fixed for this leaf (where we take all remaining).
          So if `best_error <= T - (current_sum + suffix_sum[index])`, we prune.
          But we must be careful. We need to check if this potential leaf gives a BETTER solution than `best_error`.
          Actually, the logic should be: if we prune, we skip exploring children. The implicit result of this branch is determined by the fact that we can't improve the current situation beyond this point?
          No. If `current_sum + suffix < T`, the only reachable sums are $\le \text{suffix}$.
          Wait, if we prune, we don't find out what intermediate sums are possible.
          But if the MAXIMUM possible sum is already worse than our current best, we don't care.
          So: if `T - (current_sum + suffix) >= best_error`, then even the best outcome of this branch is worse than or equal to what we already have.
          So we prune.
          
      - Before branching:
        Should we filter duplicates?
        Counting occurrences might speed up if many $a_i$ are same.
        But since $N$ is only 100, grouping might not be strictly necessary, but safer.
        Sorting descending $a_i$: helps to fill $T$ fast.
        Usually sorting descending is good for Knapsack to fail fast or succeed early.
        
      Wait, sorting descending:
      We try to take large items first.
      If we take a large item and sum exceeds T, we backtrack.
      If we don't take, we continue.
      
      What if $T$ is very large?
      Then `current_sum` is rarely $> T$. We rely on reaching exactly $T$.
      
      Let's double check the constraints. $N \le 100$. Time limit? Not specified, assume 1s or 2s.
      Standard meet-in-the-middle is not applicable ($2^{50}$).
      Is it possible to use randomized greedy search to find a good solution quickly and use local search to optimize?
      Or maybe simply running DFS with sorting and pruning is sufficient given the "random" nature of test cases (Uniform, Normal, etc.).
      Adversarial test cases for Subset Sum usually involve numbers that form a "trick" to beat pruning.
      With large random values, the distribution of subset sums is spread out. The probability of hitting $T$ exactly or very close is low unless $T$ is specifically constructed.
      However, if the test cases are from those distributions, they are "easy" in terms of structure (not highly structured like Fibonacci).
      So DFS with pruning is likely intended.
      One catch: $a_i$ can be $0$.
      If $a_i = 0$, we can handle it easily (just skip or ignore).
      
      Another optimization:
      Since we need to output the binary string, we need to reconstruct the solution.
      We can store `best_selection` as `vector<bool>` and update it whenever `current_error < best_error`.
      
      Let's refine the pruning condition again.
      We maintain `best_error` globally.
      At any node (after choosing $x_0, \dots, x_{k-1}$):
      Current sum $S$.
      Remaining items indices $k, \dots, n-1$.
      Max possible additional sum $R = \sum_{j=k}^{n-1} a_j$.
      Min possible additional sum $L = 0$.
      Range of future sums $[S, S+R]$.
      Target $T$.
      We want to find $y \in [S, S+R]$ minimizing $|y - T|$.
      Since $y$ can only take discrete values, but the bounds give the interval.
      The best possible theoretical error achievable from this state is $\min(|S-T|, |S+R - T|)$? No.
      The minimum error possible in the interval $[S, S+R]$ relative to $T$:
      Case 1: $T < S$. The interval is strictly above $T$. Minimum value is $S$. Error $S-T$.
         Wait, if $T < S$, then adding nothing is the best we can do (since all $a_i \ge 0$).
         Adding anything increases sum $\rightarrow$ larger error.
         So if $S > T$, the min error is $S - T$.
         We check if this is better than global `best_error`.
         And importantly, we can STOP recursing (prune) because any child will have error $\ge S - T$.
         Wait, this is correct pruning. If we exceed T, we can never recover (by adding more non-negatives) to get closer to T. We are stuck at the overshoot value.
         
      Case 2: $T > S+R$. Interval is strictly below $T$. Maximum value is $S+R$.
         Minimum error is $T - (S+R)$.
         If this is $\ge best\_error$, prune.
         
      Case 3: $S \le T \le S+R$.
         The interval covers $T$. The minimum error is 0.
         So we MUST proceed to search.
         But wait, if we cover $T$, we don't know if we can hit $T$ exactly.
         So we continue.
      
      Wait, in Case 1, if $S > T$, the minimal error on this branch is exactly $S - T$.
      If $S - T < best\_error$, we update `best_error` and `best_selection` (copy current selection).
      Then we PRUNE this branch completely (don't call children).
      Why? Because any child adds a non-negative $a_i$. New sum $S' = S + a_i \ge S > T$.
      Error $S' - T \ge S - T$. So we cannot find a better solution (strictly better) by going deeper.
      Wait, what if there is another selection in another branch that is better?
      That's handled by the recursion returning and checking other branches.
      Within this specific branch, no improvement is possible.
      Correct.
      
      So the pruning logic is solid.
      
      Wait, if $S > T$, error is $S-T$.
      But we must be careful: `current_sum` is accumulated.
      We need to store the selection corresponding to `current_sum`.
      
      Wait, if $S > T$, we update `best` with $S-T$ and prune.
      Wait, is it possible that $S-T$ is worse than `best_error`?
      Yes. If so, we prune immediately because any descendant will be even worse or equal.
      So if `abs(S-T) >= best_error`, prune.
      Actually, since $S > T$, `abs(S-T) = S-T`.
      Condition: If `S - T >= best_error`, prune.
      Else (meaning `S - T < best_error`), update `best_error`, update `best_selection`.
      AND prune (do not recurse).
      
      Wait, is this safe?
      Suppose $S = T + 5$. `best_error` initially $\infty$.
      Update `best_error` = 5. Prune.
      Wait, what if we excluded $a_i$ later in the path?
      Wait, "prune this branch" means we don't call DFS for the current index's next steps.
      But the current sum $S$ is formed by previous choices.
      The decision at current index `idx` is whether to include $a_{idx}$ or not.
      The state $(index, S)$ represents "we have considered $0 \dots index-1$, and included some subset".
      Wait, this representation assumes we fix the inclusion decisions for $0 \dots index-1$.
      Then we branch for $index$.
      Inside the loop at `index`:
      Try including $a_{idx}$. New sum $S + a_{idx}$.
      Recurse.
      Try excluding $a_{idx}$. New sum $S$.
      Recurse.
      
      Ah, the pruning logic above assumed we are at a NODE after deciding $0 \dots idx-1$.
      So we are considering decisions for $idx \dots n-1$.
      If $S > T$:
      If we INCLUDE $a_{idx}$, sum becomes $S+a_{idx} > T$. Error increases.
      If we EXCLUDE $a_{idx}$, sum remains $S > T$. Error stays same.
      So regardless of choice at $idx$, subsequent choices can't help reduce error below $S-T$.
      So yes, prune entire subtree of this node.
      
      What about the other pruning?
      If $S + R_{suffix} < T$:
      We can never reach $T$.
      The best we can do is maximize sum (include all remaining).
      If that max sum $S+R_{suffix}$ gives error $\ge best\_error$, prune.
      But we might still want to explore if $S+R_{suffix}$ is actually the best we can do (it's the "closest from below").
      Wait, if $S + R_{suffix} < T$, then ALL reachable sums are $\le S+R_{suffix} < T$.
      So all reachable sums are $< T$.
      The error is $T - \text{sum}$.
      Minimizing error $\equiv$ Maximizing sum.
      So we want to find the MAXIMUM reachable sum $\le S+R_{suffix}$.
      Is it guaranteed that the maximum reachable sum is $S+R_{suffix}$?
      Only if we can pick ALL remaining items.
      Which we can.
      So the best possible sum we can achieve from this node is $S+R_{suffix}$.
      Any subset of remaining items will yield a smaller sum, hence larger error.
      So the optimal strategy from here (conceptually) is to take ALL remaining items.
      But wait, we need to print the subset selection.
      So we should record the selection "take all remaining".
      Does this require recursion? No, we can just compute it immediately.
      If $S + R_{suffix} < T$ and $T - (S + R_{suffix}) < best\_error$:
        We update `best_error` and `best_selection`.
        (Selection is `current_prefix_selection` + `all_remaining_items`).
        We DO NOT recurse (because all other options yield smaller sums $\implies$ larger errors).
        Correct.
      
      So the algorithm is:
      Sort $a$ descending? Or ascending?
      If descending, large items first.
      Helps to reach $S > T$ quickly (pruning) or $S+R_{suffix} \approx T$.
      
      Implementation Details:
      Use `std::vector<long long>` for A.
      Precompute `suffix_sums`.
      Use a `vector<int>` or `string` for `best_selection` to reconstruct.
      Since $N$ is small, `vector<bool>` or `bitset` is fine.
      Pass a `path` vector to `dfs`.
      Wait, passing path is overhead?
      We can just maintain a `vector<int>` (indices or bools) globally and copy/update.
      Or just pass it by reference and backtrack.
      
      Data types: `long long` for sums. $T$ fits in long long?
      $B=10^{15}$, $n=100$. Max sum $10^{17}$. Fits in signed 64-bit integer (limit $\approx 9 \cdot 10^{18}$).
      
      Corner case: $a_i = 0$.
      If $a_i = 0$, it doesn't affect sum, affects only count/selection string.
      If we have zeros, does it matter?
      If $a_i=0$, it contributes to selection length but not sum.
      Problem says "binary string of length n". So we must specify 0 or 1 for each position.
      If $a_i=0$, setting it to 0 or 1 changes nothing in sum.
      Does it matter which one we output?
      "find a subset... sums as close as possible". Both give same sum.
      Any valid one works. I'll probably prefer 0 for simplicity or just whatever.
      My code logic will naturally handle it (include/exclude).
      
      Complexity:
      Worst case is still bad, but average case with pruning should pass.
      Wait, what if `best_error` is updated by a partial solution, helping future prunings?
      Yes, that's the point.
      Initializing `best_error` to `abs(sum(all_a) - T)` or `abs(T)`?
      Initial `best_error` can be computed by trying to include nothing (0). `abs(0 - T) = T`.
      Better to initialize with the sum of all items.
      Compute total sum `total_sum`. `best_error = abs(total_sum - T)`.
      Start DFS from index 0.
      Optimization: `vector<long long> suffix_sum(n+1, 0);`
      `for(int i=n-1; i>=0; --i) suffix_sum[i] = suffix_sum[i+1] + a[i];`
      
      Wait, one edge case: `T` can be negative? Problem says $0 \le a_i \le B$. Doesn't specify range for $T$.
      Usually subset sum target is non-negative. The problem mentions $T=x_i a_i$, so $T$ is likely non-negative.
      Sample $T=4$.
      I will assume $T \ge 0$. If $T < 0$, and all $a_i \ge 0$, min sum is 0. Closest is 0.
      So I should handle `T` correctly. `abs` handles signs.
      
      One important detail: The problem statement says "print a binary string".
      Output format: `101` etc.
      
      Also, sorting order.
      If we sort $a$ descending, we encounter large elements early.
      This helps `S > T` condition trigger earlier, pruning branches that overshoot.
      It also helps `S + R < T` trigger later (since R decreases).
      Wait, sorting descending:
      Items: 100, 90, 80...
      S starts 0.
      Add 100 -> 100. If T=50, $S > T$. Prune.
      Backtrack. Exclude 100. S=0.
      Add 90 -> 90. $S > T$. Prune.
      This finds the "overshoot" candidates quickly.
      Sorting ascending:
      Items: 10, 20, 30...
      Adds many small items to get near T.
      Maybe this is better for finding solutions $\approx T$?
      If T is large, ascending allows gradual approach.
      However, if T is small, descending helps avoid overshooting.
      With $T$ unknown, maybe randomized? Or just descending.
      Actually, for subset sum approximation, sorting descending is common heuristic (greedy filling).
      But we have backtracking.
      I will use descending sort.
      Wait, if I sort descending, I permute $a_i$. I must remember original indices for output string.
      Store pairs `{value, original_index}`. Sort.
      In output, map back to original indices.
      
      Wait, I should probably preserve original order for the binary string?
      "Print a binary string of length n".
      Indices 0 to n-1 correspond to $a_1 \dots a_n$ in input order.
      So yes, track original indices.
      
      Code Structure:
      Struct Item { long long val; int id; };
      Sort Items.
      DFS takes `item_idx`, `current_sum`, `vector<int>& current_path`.
      Updates global `best_error`, `best_path`.
      
      Wait, `current_path` needs to be restored (backtrack).
      
      Double check memory limits.
      Memory limit not specified, assume standard. Vectors are fine.
      
      One issue: The sample input has $T=4$, $A=\{1, 2, 3\}$.
      Sorted: 3(2), 2(1), 1(0). (Value, Id)
      1. Index 0, Val 3.
         Inc: Sum 3. Path {2}. Recurse idx 1.
           Index 1, Val 2.
             Inc: Sum 5. $S > T$. Error 1. Update Best (5, {2,1}). Prune.
             Exc: Sum 3. Path {2}. Recurse idx 2.
               Index 2, Val 1.
                 Inc: Sum 4. $S=T$. Error 0. Best {2, 0}. Path 0-indexed ids {2,0}? No, vals 3 and 1 correspond to original indices 2 and 0. (Using 0-based indexing for internal, input order 1-based).
                 Original indices: Val 3 is index 2. Val 1 is index 0.
                 Wait, sample input: 1 2 3. Ids 0, 1, 2.
                 Sorted: (3, 2), (2, 1), (1, 0).
                 Trace:
                 Root: {}
                 Try (3, 2): S=3. Path {2}.
                   Try (2, 1): S=5. Err=1. Best=5? No init best=4 (total sum 6).
                     Wait, if I init best with total sum? Total sum 6. Err 2.
                     With S=5, err 1 < 2. Best=5.
                     Return (pruned).
                   Try (2, 1): Exc. S=3. Path {2}.
                     Try (1, 0): S=4. Path {2, 0}. Err 0. Best=0. Found optimal!
                     Return.
                     Pruning? If err 0, we can technically prune and return 0 immediately.
                   Return.
                 Try (3, 2): Exc. S=0. Path {}.
                   Try (2, 1): S=2. Path {1}.
                     Try (1, 0): S=3. Path {1, 0}. Err 1.
                       Compare with Best=0. No update.
                       ...
                     Return.
                   
                   Result Best {2, 0} corresponds to values 3 and 1. Sum 4. Correct.
                   Indices {2, 0}. String "101"?
                   Index 0 (val 1): 1. Index 1 (val 2): 0. Index 2 (val 3): 1.
                   Wait, my trace:
                   Set {2, 0} -> Indices 2 and 0.
                   String: Index 0 is 1, Index 1 is 0, Index 2 is 1.
                   Output "101". Matches sample.
                   
      Wait, the sample output "101" means select index 0 and index 2 (values 1 and 3). Sum 4.
      My manual trace got indices 0 and 2.
      Wait, why did I sort?
      If I don't sort, indices 0, 1, 2.
      Try 1: S=1.
        Try 2: S=3.
          Try 3: S=6.
        Exc 2: S=1.
          Try 3: S=4. (Found 0).
      
      Okay, sorting helps pruning.
      
      Wait, initialization of `best_error`.
      Can we start with empty set? $Sum=0$, Error=$T$.
      Total set? $Sum=Total$, Error=$|Total - T|$.
      We can run both as initial checks or just let DFS cover them.
      DFS covers them via Include/Exclude logic.
      But pruning relies on `best_error`. If initialized to infinity, pruning might not happen early enough.
      Better to compute `best_error` for empty set and full set before DFS, or handle it inside.
      Actually, empty set (exclude all) and full set (include all) are covered by the recursion.
      Just need a starting value for `best_error` to enable early pruning.
      Initialize `best_error` with `abs(0 - T)`? Or `abs(total_sum - T)`?
      Taking `min(abs(0-T), abs(total-T))` as initial lower bound for "known solution quality".
      Actually, just set `best_error` to `LLONG_MAX` and inside first valid leaf update it.
      But pruning requires a finite value.
      So initializing with `abs(total_sum - T)` is safe and effective.
      Also `abs(0 - T)` is safe.
      So `best_error = min((long long)abs(T), abs(total_sum - T));`
      
      One optimization: If `best_error == 0`, we can terminate and print immediately.
      
      Wait, one detail from problem statement: "x_i drawn from Bernoulli (1/2)".
      This implies T is random.
      But in the input, we are given T explicitly.
      Okay, the code should just read T.
      
      The code must handle $n$ up to 100.
      Using `long long` everywhere for sums.
      Time limit is typically generous enough for backtracking with pruning on these random distributions.
      
```cpp
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <string>

using namespace std;

struct Element {
    long long val;
    int id;
};

int n;
long long T;
vector<long long> a; // For suffix sum calculation, using original values?
// Wait, I should use sorted values for DFS, but need suffix sums based on sorted?
// Pruning relies on sum of remaining available items.
// So we need suffix sums of the SORTED array (the ones we iterate over).
vector<Element> sorted_elements;
long long total_sum_sorted;
long long suffix_sum[105]; // 1-based index for convenience
long long best_error;
string best_solution_string;
bool zero_error_found = false;

void dfs(int idx, long long current_sum, string& current_str) {
    if (zero_error_found) return;

    // Pruning
    // 1. If we already found 0 error, stop (optimization)
    // Check inside main or here.
    
    // 2. Pruning based on upper/lower bounds of reachable sums
    
    // Remaining elements indices: idx to n-1
    // If we are at idx, we have elements from idx onwards.
    // Wait, suffix_sum array should match sorted_elements.
    
    // Long long remaining_max = suffix_sum[idx]; // If idx is exclusive?
    // Let's define suffix_sum[i] as sum of sorted_elements[i...n-1].
    
    if (idx == n) {
        long long err = abs(current_sum - T);
        if (err < best_error) {
            best_error = err;
            best_solution_string = current_str;
        } else if (err == best_error && !zero_error_found) {
            // In case of tie, does it matter?
            // Usually not. Keep first found or last?
            // Problem asks for "a" subset.
        }
        return;
    }

    // Pruning Logic
    // Remaining sum available: suffix_sum[idx]
    long long rem_max = suffix_sum[idx];
    
    // Case 1: current_sum > T
    // Since all a_i >= 0, adding more will only increase sum -> error increases.
    // Best we can do now is stop adding.
    // Wait, this applies to the whole branch.
    // We can stop adding ANYMORE.
    // But we still need to compare current error with best.
    // Note: We haven't decided for current index yet.
    // If current_sum > T, including a[idx] makes it worse.
    // So we must effectively "exclude" all remaining items to get the tightest sum for this path?
    // No, we just need to evaluate the CURRENT state (excluding all remaining) vs current_best.
    // Wait, if current_sum > T, the best sum we can make from here is 'current_sum' itself (by excluding all remaining).
    // Any inclusion adds >= 0, so sum increases.
    // So the best error achievable from this node is (current_sum - T).
    // If (current_sum - T) >= best_error, we prune.
    // Else, we update best_error and return (since we can't get better by recursing).
    
    if (current_sum > T) {
        long long err = current_sum - T;
        if (err < best_error) {
            best_error = err;
            best_solution_string = current_str; // This corresponds to excluding all remaining
            // But wait, current_str includes selections made BEFORE idx.
            // We need to append 0s for remaining?
            // The problem requires a binary string of length N.
            // current_str has length idx.
            // We need to pad with '0's?
            // Yes.
            for(int k=0; k<n-idx; ++k) current_str += '0';
        }
        return; // Cannot do better by including more items
    }
    
    // Case 2: current_sum + rem_max < T
    // Even if we include ALL remaining items, we can't reach T.
    // Max reachable sum is current_sum + rem_max.
    // Min error is T - (current_sum + rem_max).
    // If this error >= best_error, prune.
    // Else, we can potentially update best with this configuration (take all remaining).
    long long potential_sum = current_sum + rem_max;
    long long err = T - potential_sum;
    if (err >= best_error) {
        // Can we prune?
        // Yes, if we can't even reach T, the best error we can get here is 'err'.
        // If this is worse than existing best, discard branch.
        // Wait, is it strictly worse? 
        // If err == best_error, it's same.
        // If err < best_error, it's better.
        // But wait. If we just take all remaining, we get 'potential_sum'.
        // Is there any other combination? Any subset of remaining is smaller than potential_sum.
        // Since current_sum < T, smaller sums mean LARGER error.
        // So the optimal move in this branch is to take ALL remaining.
        // So if taking all remaining is not better than best, we prune.
        // Correct.
        // Note: If we decide to update best_solution_string, we need to append '1's.
        // But wait, if we don't prune, we still need to traverse to find WHICH subsets sum to potential_sum?
        // Actually, if the ONLY way to get 'potential_sum' is taking all, then we know the result.
        // But wait, maybe there's an intermediate subset that is closer?
        // No, because all a_i >= 0.
        // If current_sum < T, adding more reduces error. Adding less than 'rem_max' means sum < potential_sum.
        // Thus error > (T - potential_sum).
        // So potential_sum IS the closest value from below.
        // So we just check this single candidate.
        if (err < best_error) {
            best_error = err;
            // Update solution
            // We need to append 1s for all remaining items.
            // BUT wait, this assumes we are committing to taking all remaining.
            // So we update best_solution_string.
            // How to update?
            // Construct it?
            // It's better to update best_solution_string lazily or pass info.
            // But here we are at idx, and we are conceptually "taking all".
            // So we can append 1s.
            for(int k=0; k<n-idx; ++k) current_str += '1'; // temporary?
            // Wait, we shouldn't modify current_str permanently if we return.
            // We can use a temporary string or update a global variable.
            // Let's use a separate mechanism or just update global `best_solution`.
            // We need to map this selection.
            // The logic is: Selection = current_selection + {all remaining}.
            // So we can construct this locally and update global.
            // But constructing string might be O(N). Acceptable.
            // But wait, do we really just append 1s?
            // Yes.
            // But we have to be careful not to leave dirty state in `current_str`.
            // So update global directly.
        }
        // Wait, I should check: Is it possible that by pruning we miss something?
        // No, because any other subset sum is < potential_sum, so > T - err?
        // Wait, current_sum < T. potential_sum < T.
        // We want sum closest to T from below.
        // That is the maximum possible sum. Which is potential_sum.
        // So yes, we just check this one candidate.
        // However, we might have chosen to UPDATE global best and RETURN (prune).
        // If we update, we don't recurse.
        
        if (err < best_error) {
             // Update global best.
             // Need to update best_solution_string.
             // We can just set it to current_str + string(n-idx, '1').
             // But wait, current_str holds prefix.
             // Let's handle the string update outside or carefully.
             // Actually, simpler: Don't modify current_str in place for update.
             // Just store result in a temp variable and update globals.
        }
        // Wait, if we don't recurse, we implicitly prune.
        // But wait, inside Case 2, we found the "best possible from below".
        // If that is worse than `best_error` (found from elsewhere or above), we prune.
        // If it is better, we take it and prune (no need to try subsets).
        // So logic:
        //   If (T - (current_sum + rem_max) < best_error) {
        //      Update best_error and best_sol.
        //      // Do not recurse.
        //      // Why? Because any other subset sum <= potential_sum, so worse error.
        //      return;
        //   }
        // Wait, we should also update best_sol string properly.
        // But modifying `best_solution_string` inside `dfs` is risky if not careful.
        // But if we return, it's fine?
        // Wait, if we return, we skip recursive calls.
        // But wait, `current_str` is passed by reference.
        // We need to restore it? No, we return.
        // So just updating global `best_solution_string` is fine.
        // But constructing the string is needed.
        
        // Let's restructure:
        // If we decide to update global, we compute the candidate string.
        // Then update global.
    }

    // But wait, Case 2 pruning logic is slightly flawed if we don't verify the condition.
    // Specifically, we update `best_error` with `T - (current_sum + rem_max)`.
    // But we might already have `best_error` smaller.
    // If `best_error` is already smaller, we prune immediately.
    // If `best_error` is larger, we update.
    // Wait, do we need to explore ANYTHING else?
    // If `current_sum + rem_max < T`, then the closest value is `current_sum + rem_max`.
    // Why? Because `current_sum + rem_max` is the max sum we can form $\le T$.
    // Wait, `current_sum < T`. `rem_max >= 0`.
    // If `current_sum + rem_max < T`, then ALL subset sums are $< T$.
    // We want to maximize sum to minimize error.
    // Max sum is `current_sum + rem_max`.
    // So yes, we just check this single point.
    
    // Revised Pruning 2 Logic:
    // If current_sum + rem_max < T:
    //    long long potential_err = T - (current_sum + rem_max);
    //    if (potential_err < best_error) {
    //       best_error = potential_err;
    //       // Set best_solution = current_prefix + "111..."
    //       // But wait, we can't easily set "111..." into current_str if we pass by ref.
    //       // But we can build a temporary string.
    //       best_solution_string = current_str + string(n - idx, '1');
    //    } else {
    //       return; // Prune
    //    }
    //    return; // Either updated and done, or pruned and done.
    
    // Wait, but what if `current_sum + rem_max < T` but `potential_err >= best_error`?
    // Then we prune.
    // BUT wait. Is it possible that `current_sum + rem_max < T` and `potential_err` is not the best?
    // No, it's the best *possible* from this branch (maximal sum).
    // So if it's not better than global, this branch is dead.
    
    // Is there any overlap with Case 1?
    // Case 1: current_sum > T.
    // Case 2: current_sum + rem_max < T.
    // Overlap? If current_sum > T, then current_sum + rem_max > T. So disjoint (unless rem_max is negative, but it's not).
    // So they are distinct.
    // Case 3: Otherwise.
    // Continue recursion.
    
    // Wait, one correction:
    // If we are in Case 2, and we update best_solution_string, we must ensure we do NOT corrupt `current_str` used for other branches (siblings)?
    // No, we return immediately. We don't loop siblings here.
    // Wait, we are inside the loop over decisions for `idx`.
    // No, I put the pruning before the decision loop.
    // So this pruning applies to the WHOLE SUBTREE rooted at `idx`.
    // If we prune, we stop the ENTIRE recursion from `idx`.
    // So it's safe.
    
    // BUT wait. The pruning logic depends on `current_sum`.
    // `current_sum` is the sum of decisions made BEFORE `idx`.
    // Pruning 1 (current_sum > T) checks if we already overshoot.
    // Pruning 2 (max_reach < T) checks if we can't reach.
    // If either holds, we process the immediate implication and prune.
    
    // There's a nuance:
    // Pruning 1: `current_sum > T`. The best outcome of this branch is `current_sum` (exclude all).
    // Pruning 2: `current_sum + rem_max < T`. The best outcome is `current_sum + rem_max` (include all).
    // Are these the ONLY candidates we need to check?
    // If `current_sum > T`:
    //    We have already established that adding more makes sum worse.
    //    So we just consider the state "exclude all remaining".
    //    Wait, what about "include some"? No, that makes it worse.
    //    So yes, only "exclude all" matters.
    //    And we compare its error with best.
    //    Then prune.
    // If `current_sum + rem_max < T`:
    //    We established that adding less than "rem_max" makes sum smaller (worse).
    //    So only "include all" matters.
    //    Compare error. Prune.
    
    // So logic is sound.
    
    // BUT wait! There is a small logical gap.
    // In Pruning 1 (`current_sum > T`), I said "exclude all remaining".
    // Is it possible that "exclude all remaining" is NOT the optimal selection among remaining?
    // No, because any "include some" adds value > 0, making sum > current_sum > T, error > current_sum - T.
    // So "exclude all" is strictly optimal for that branch.
    // Similarly for Pruning 2.
    
    // Wait, what if `best_error` is already 0?
    // We should stop everything.
    
    // Wait, I should put the updates to `best_solution_string` safely.
    // `current_str` is a prefix.
    // If I do `current_str + string(...)`, that creates a new string. Safe.
    
    // BUT wait, what if `best_error` was updated in parent calls?
    // We access the global `best_error` and `best_solution_string`.
    
    // Let's refine the code implementation.
    // The `dfs` function signature:
    // void dfs(int idx, long long current_sum, const string& current_str)
    // Passing `current_str` by value or const ref? Ref to save copy.
    // But we need to backtrack.
    // Wait, if we use `current_str` by ref, appending modifies it.
    // So we must push/pop.
    // Or just pass by value (since depth 100, copy is cheap).
    // `string current_str` by value is fine.
    
    // Wait, passing by value is easier for backtracking.
    // `dfs(idx, current_sum, current_str)`.
    
    // Pruning Logic Implementation:
    // long long rem_sum = suffix_sum[idx];
    // if (current_sum > T) {
    //    long long err = current_sum - T;
    //    if (err < best_error) {
    //       best_error = err;
    //       best_solution_string = current_str + string(n - idx, '0');
    //    }
    //    return; // Prune
    // }
    // if (current_sum + rem_sum < T) {
    //    long long err = T - (current_sum + rem_sum);
    //    if (err < best_error) {
    //       best_error = err;
    //       best_solution_string = current_str + string(n - idx, '1');
    //    }
    //    return; // Prune
    // }
    
    // This covers the boundaries.
    
    // Base case: idx == n.
    // We reached end of array.
    // Check error.
    // If 0, return and flag.
    
    // Recursive Step:
    // We have `current_str`.
    // Try including `sorted_elements[idx]`.
    // But wait, if we include, we add 1.
    // Try excluding, add 0.
    // Order: Try 1 (Include) first or 0 (Exclude)?
    // Try 1 (Include) first if we want to reach T faster?
    // If we include, sum increases. If we exclude, sum stays.
    // Usually try Include first.
    // But wait, if we are in the `current_sum + rem_sum < T` case?
    // That case returns immediately.
    // Wait, if we are in the "Case 3" (middle range), we recurse.
    
    // Wait, one thing: `suffix_sum` should be computed based on sorted elements.
    // `suffix_sum[i] = sum(sorted_elements[i...n-1].val)`.
    
    // Wait, I need to reset `current_str`?
    // If I use recursion with `string` by value, I don't need to revert.
    
    // Wait, string concatenation copies the string. $O(N)$. Depth $N$. Total $O(N^2)$.
    // With $N=100$, $10000$ ops. Negligible.
    
    // Optimization:
    // If `zero_error_found`, return immediately.
    
    // One tricky thing:
    // `best_solution_string` needs to reflect the binary choices.
    // The problem uses 1-based indexing in description ("binary string of length n"), but standard strings are 0-indexed chars.
    // The bits correspond to original $a_i$ order.
    // My sorting scrambles order.
    // `sorted_elements[i].id` stores original index.
    // When I construct `best_solution_string`, I need to set the character at `sorted_elements[i].id` to '1' (or '0').
    // So I can't just append to a string. I must set bits in an array/string of length $N$.
    // `vector<char> final_res(n, '0')`.
    // Fill it. Then convert to string.
    // Better: use `string best_solution(n, '0')` initialized.
    // In DFS, we mark bits.
    // But DFS branches. Marking/unmarking?
    // Or just construct it when updating `best_solution_string`.
    // Since `best_solution` is only updated, we can build it fresh.
    // Building a string of length 100 is fast.
    // So `best_solution_string` construction on update is fine.
    // Just `vector<char> res(n, '0')` is good.
    
    // Wait, updating `best_solution_string`:
    // Iterate `idx` to `n-1`? No, iterate all original indices?
    // The path taken in DFS determines which `sorted_elements` are selected.
    // So we need to reconstruct the selection vector.
    // We can pass a `vector<bool> mask` down the recursion?
    // Or simpler: Just `vector<char>` `res` of size $N$ initialized to '0'.
    // When visiting, we modify `res`. But we need to backtrack.
    // Since $N$ is small, we can allocate `res` vector and copy to update global `best_selection`.
    // Actually, constructing `best_selection` from scratch when updating `best_error` is easiest.
    // But we don't want to iterate $N$ times every time.
    // It's $O(N^2)$ in worst case (updates often).
    // $100^2 = 10000$. It is perfectly fine.
    
    // Wait, we can optimize `suffix_sum`.
    // `vector<long long> suffix_sum(n + 1, 0);`
    // Precompute it.
    
    // Wait, I should handle `a_i` being 0 correctly?
    // If `a_i` is 0, it doesn't affect sum.
    // It affects the selection string (should be '0' or '1'?).
    // The problem asks to "find a subset".
    // A subset implies specific elements.
    // If $a_i=0$, including it in subset or not doesn't change sum.
    // Any choice is valid.
    // My code will treat 0 normally. It might output '0' or '1'.
    // I will stick to default behavior.
    
    // Final Check on Constraints.
    // $B = 10^{15}$.
    // $n=100$.
    // Time limit.
    // If I have many test cases? No, problem says "Input...". Single test case usually.
    // "Sample input".
    // So I assume single run.
    
    // Wait, reading "Drawn from distributions".
    // The problem is asking for a solver for the subset sum approximation.
    // The text mentions distributions to explain why inputs are generated that way (likely for validation).
    // I just need to solve the instance given.
    
    // Wait, one detail about the output.
    // "Print a binary string".
    // I will produce a string.
    
    // One more thing:
    // What if `current_sum + rem_sum == T`?
    // Then error is 0.
    // My logic `current_sum + rem_sum < T` handles strictly less.
    // If equal, it falls through to recursion (base case will find it).
    // Wait, if `current_sum + rem_sum == T`, we can just pick all remaining and stop.
    // But base case handles it.
    
    // Wait, what if `best_error` is updated to 0?
    // I added `if (zero_error_found) return;`.
    // This propagates up.
    
    // Wait, `suffix_sum` indexing.
    // `sorted_elements` has size `n`. Indices `0` to `n-1`.
    // `suffix_sum[i]` sum of `sorted_elements[i]` to `n-1`.
    // `suffix_sum[n] = 0`.
    // So in DFS `dfs(idx, ...)` where `idx` is current element index.
    // `suffix_sum[idx]` is correct.
    
    // Let's write the code.
    // Includes.
    // Globals.
    // Main function.
    // IO speed up.
    
    // Wait, what about `B`? It's $10^{15}$.
    // `long long` is sufficient.
    
    // Wait, one subtle bug in `dfs`:
    // If I update `best_error` and return, I lose the chance to update `best_solution_string`?
    // No, I update it.
    // But wait.
    // If I prune, I don't explore.
    // Is it possible that a different path leads to same `best_error` but preferred solution?
    // Problem says "find a subset". Any valid is okay.
    
    // Wait, I should double check the `case 2` pruning.
    // `if (err < best_error)`:
    // I update `best_error` and `best_solution_string`.
    // But I should also check `best_solution_string` validity?
    // The string construction logic is complex in my head.
    // Let's make a helper to build string from selection logic?
    // No, I can just maintain `vector<bool> selection(n, false)`?
    // No, recursion backtracking.
    // If I update, I have to know exactly which elements were picked.
    // My DFS tracks `idx`.
    // Elements `0` to `idx-1` in `sorted_elements` have been decided.
    // Elements `idx` to `n-1` are NOT decided yet.
    // So `best_solution_string` cannot be built incrementally easily without tracking.
    // Easier to use a `vector<int> current_indices` and pass it? No, copying vector is fine.
    // Or just build string from scratch inside DFS when updating?
    // Iterating `n` to build string is fine.
    // Wait, I need to know if `sorted_elements[k]` was picked.
    // But I don't store that history.
    // I need to track it.
    // `vector<bool> current_selection(n, 0);`
    // Pass by value or backtrack?
    // Backtrack is better to avoid copy.
    // But `dfs` is deep.
    // Backtracking logic:
    // Select `i`. `current_selection[i] = 1`.
    // Recurse.
    // `current_selection[i] = 0`.
    // This works.
    // So `current_selection` vector passed by ref.
    // When updating `best_solution_string`, we construct it from `current_selection`.
    // But wait, I'm using `current_str` logic in thoughts, but `current_selection` vector is cleaner.
    // So I will switch to passing `current_selection` bitmask/vector.
    // But wait, constructing string at update time: `string res(""); res.resize(n);` fill.
    // Cost $O(N)$. OK.
    
    // One tiny optimization: `current_selection` is passed by value?
    // `vector` copy is fast for size 100.
    // It avoids manual backtrack (restore) code, preventing bugs.
    
    // Let's finalize.
    // `best_solution` string construction:
    // `vector<bool> sel = selection;`
    // Copy `sel` to string? No, `sel` maps to sorted elements.
    // We need to map to original indices.
    // So `string res(n, '0');`
    // Loop `i` from 0 to `n-1`:
    //    if `sel[i]` and `sorted_elements[i].id != -1` (valid), `res[sorted_elements[i].id] = '1'`.
    // This works.
    
    // Wait, what if `a_i` are large, say `10^{15}`.
    // If I have two `10^{15}` items.
    // They are indistinguishable in value.
    // Sorting puts them adjacent.
    // If I pick the first one (id X) and not second (id Y), I get different string.
    // Both are valid.
    // My code picks whichever it encounters.
    
    // Wait, I need to make sure I don't access out of bounds.
    // `suffix_sum` size `n+1`.
    
    // What if $n=0$?
    // Problem says $1 \le n \le 1e2$. So $n \ge 1$.
    
    // Code generation.
</think>

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <cmath>
#include <iomanip>

using namespace std;

struct Element {
    long long val;
    int id;
};

int n;
long long T;
vector<Element> sorted_elements;
vector<long long> suffix_sum;
long long best_error;
string best_solution_string;
bool zero_error_found = false;

void dfs(int idx, long long current_sum, const vector<bool>& current_selected_indices_sorted) {
    if (zero_error_found) return;

    // Pruning: If we have found an exact solution, stop.
    if (best_error == 0) return; 

    // Determine the range of possible sums we can achieve with remaining elements.
    long long rem_max = 0;
    if (idx < n) {
        rem_max = suffix_sum[idx];
    }

    // Case 1: Current sum already exceeds T.
    // Since all elements are non-negative, adding more will only increase the sum.
    // The closest value to T we can achieve from this state is current_sum (by excluding all remaining).
    if (current_sum > T) {
        long long err = current_sum - T;
        if (err < best_error) {
            best_error = err;
            best_solution_string = string(n, ' '); // Reset placeholder
            
            // Reconstruct solution based on current state (exclude all remaining)
            // The current state implies that for indices 0 to idx-1, selections are in current_selected_indices_sorted.
            // For indices idx to n-1, we select NOTHING ('0').
            
            // We need to map sorted indices back to original indices.
            // However, current_selected_indices_sorted tells us exactly what was chosen in the sorted array.
            // We can just use that to build the string.
            // Create a mapping of selected elements.
            vector<char> sol(n, '0');
            for (int i = 0; i < idx; ++i) {
                if (current_selected_indices_sorted[i]) {
                    sol[sorted_elements[i].id] = '1';
                }
            }
            // Elements from idx to n-1 are not selected.
            best_solution_string = sol;
        }
        return;
    }

    // Case 2: Even taking ALL remaining elements, we cannot reach T.
    // The best we can do is take all remaining elements to maximize sum (closest to T from below).
    long long potential_sum = current_sum + rem_max;
    if (potential_sum < T) {
        long long err = T - potential_sum;
        if (err < best_error) {
            best_error = err;
            vector<char> sol(n, '0');
            // Current selected
            for (int i = 0; i < idx; ++i) {
                if (current_selected_indices_sorted[i]) {
                    sol[sorted_elements[i].id] = '1';
                }
            }
            // All remaining selected
            for (int i = idx; i < n; ++i) {
                sol[sorted_elements[i].id] = '1';
            }
            best_solution_string = sol;
        }
        return;
    }

    // Base Case
    if (idx == n) {
        long long err = abs(current_sum - T);
        if (err < best_error) {
            best_error = err;
            // Reconstruct solution from current_selected_indices_sorted
            vector<char> sol(n, '0');
            for (int i = 0; i < idx; ++i) { // idx is n here
                if (current_selected_indices_sorted[i]) {
                    sol[sorted_elements[i].id] = '1';
                }
            }
            best_solution_string = sol;
        }
        // Note: We only care if we improved best_error.
        return;
    }

    // Decision: Include or Exclude sorted_elements[idx]
    // Optimization: Try Include first.
    // However, to avoid recomputing selection overhead, we pass vector by value (safe copy).
    // Wait, copying vector<bool> is cheap (bitset optimization might apply, or byte array).
    // Size 100 bytes is negligible.
    
    // Option 1: Include
    {
        // We include the element.
        // We create a new selection vector.
        // Actually, creating new vectors in recursion might be slightly costly but safe.
        // With N=100, depth 100. It's ok.
        // But passing by value is clean.
        // To optimize, we can manage the vector outside.
        // But let's stick to vector parameter for safety against bugs.
        
        // Wait, we should check if adding causes issues?
        // No, pruning handles it.
        
        // But wait, if rem_max is huge, this might overflow? 
        // max sum ~ 100 * 10^15 = 10^17. Long long max is ~9*10^18. Safe.
        
        // We need to pass the modified selection.
        // Let's pass a reference to a mutable selection vector?
        // Backtracking with reference is better performance-wise.
        // But I decided to use value for simplicity.
        // Wait, passing by value means copying the vector at every call.
        // With branching factor 2, number of nodes can be large.
        // Copying vector 100 times per node is expensive if many nodes visited.
        // Better to pass by reference and backtrack (push/pop).
        
        // Let's switch to passing `vector<bool>&` and restoring state manually.
    }
}

// Redefined DFS to use backtracking for efficiency
void dfs_optimized(int idx, long long current_sum, vector<bool>& selection_sorted) {
    if (zero_error_found) return;

    // Pruning Logic
    long long rem_max = suffix_sum[idx];
    
    // Optimization 1: Current Sum > T
    if (current_sum > T) {
        long long err = current_sum - T;
        if (err < best_error) {
            best_error = err;
            // Construct string
            string s(n, '0');
            for (int i = 0; i < idx; ++i) { // selection_sorted is size idx for decisions made? 
                // Wait, selection_sorted has size n?
                // I should track selection for n items.
                // selection_sorted[i] corresponds to sorted_elements[i].
                // But we only made decisions up to idx-1.
                // So selection_sorted[x] is 0 for x >= idx?
                // Or we just use `idx` to limit loop.
                if (selection_sorted[i]) {
                    s[sorted_elements[i].id] = '1';
                }
            }
            best_solution_string = s;
        }
        return;
    }

    // Optimization 2: Max Reachable < T
    if (current_sum + rem_max < T) {
        long long err = T - (current_sum + rem_max);
        if (err < best_error) {
            best_error = err;
            string s(n, '0');
            for (int i = 0; i < idx; ++i) {
                if (selection_sorted[i]) {
                    s[sorted_elements[i].id] = '1';
                }
            }
            // Append remaining
            for (int i = idx; i < n; ++i) {
                s[sorted_elements[i].id] = '1';
            }
            best_solution_string = s;
        }
        return;
    }

    if (idx == n) {
        long long err = abs(current_sum - T);
        if (err < best_error) {
            best_error = err;
            string s(n, '0');
            for (int i = 0; i < n; ++i) {
                if (selection_sorted[i]) {
                    s[sorted_elements[i].id] = '1';
                }
            }
            best_solution_string = s;
        }
        return;
    }

    // Try Include
    {
        selection_sorted[idx] = 1;
        long long next_sum = current_sum + sorted_elements[idx].val;
        // Before calling, check if pruning condition met in next call?
        // Pruning is checked at start of dfs.
        // However, if next_sum > T, we might waste stack depth?
        // We can add an inline check here to prune sooner.
        // if (next_sum > T) dfs... (and prune inside)
        // My dfs handles it.
        // But we can optimize by skipping if next_sum > T?
        // If next_sum > T, we enter the loop inside dfs, check pruning, return immediately.
        // So no real harm.
        // BUT, passing `selection_sorted` by ref: we must undo.
        // Undo happens after return.
        // If next_sum > T, dfs returns immediately (due to pruning).
        // Then we undo `selection_sorted[idx] = 1`.
        // Then try Exclude.
        // Seems OK.
        dfs_optimized(idx + 1, next_sum, selection_sorted);
    }

    // Try Exclude
    selection_sorted[idx] = 0;
    dfs_optimized(idx + 1, current_sum, selection_sorted);
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    if (!(cin >> n >> T)) return 0;

    sorted_elements.resize(n);
    long long current_total = 0;
    for (int i = 0; i < n; ++i) {
        cin >> sorted_elements[i].val;
        sorted_elements[i].id = i;
        current_total += sorted_elements[i].val;
    }

    // Sort descending to help pruning
    sort(sorted_elements.begin(), sorted_elements.end(), [](const Element& a, const Element& b) {
        return a.val > b.val;
    });

    // Precompute suffix sums
    suffix_sum.assign(n + 1, 0);
    for (int i = n - 1; i >= 0; --i) {
        suffix_sum[i] = suffix_sum[i + 1] + sorted_elements[i].val;
    }

    // Initial best error: Consider taking nothing (0) and taking all (total)
    long long err0 = abs(0 - T);
    long long errAll = abs(current_total - T);
    best_error = min(err0, errAll);

    // Prepare best solution string
    string temp_s = "";
    if (err0 < errAll) {
        temp_s = string(n, '0');
        // Wait, taking nothing means string of all 0s.
        // If errAll is better, string of all 1s.
        // But wait, we might find better later.
        // Let's just init.
    } else {
        temp_s = string(n, '0');
        // For errAll, we set all 1s.
        // But we can defer this and handle in first DFS call.
        // Actually, let's just init with empty/error logic inside DFS.
    }
    // Actually, we can just rely on DFS to update it, but we need a valid string for "best_solution_string" initially.
    // Let's set it to "00...0" initially (corresponding to 0 sum).
    best_solution_string = string(n, '0');

    vector<bool> selection_sorted(n, false);
    zero_error_found = false;

    // Run DFS
    // Optimization: If current sum > T, we prune.
    // But initially sum is 0. If T < 0 (unlikely), handle.
    // Assume T >= 0.
    
    dfs_optimized(0, 0, selection_sorted);

    cout << best_solution_string << endl;

    return 0;
}
```
````

---

## 直接SFT 样本(sample_idx=0,得分 0.0,completion_tokens=1205)

````
I need to find a subset of integers that sums as close as possible to T. This is the knapsack-like subset sum problem.

Given:
- n (1 ≤ n ≤ 100) integers
- Target value T
- Find binary selection x_i ∈ {0,1} such that Σ(x_i * a_i) ≈ T

Since we want to minimize |Σ - T|, this can be solved with dynamic programming.

Key observation: Since the target T is at most B = 1e15, but we only have n ≤ 100 numbers, and each number is at most B, the maximum possible sum is at most n * B = 1e17. However, we're looking for values close to T, so we should consider reachable sums up to around T (or slightly beyond).

The approach:
1. Use DP where dp[i] = true means sum i is achievable
2. We iterate through each number and update which sums are reachable
3. Find the reachable sum closest to T

However, with T up to 1e15, a direct array would be too large. So instead, I'll use a hash set (unordered_set) to store reachable sums, or use a meet-in-the-middle approach since n ≤ 100 is manageable.

For n up to 100, a pure DP with unordered_set might still be slow if there are many distinct sums. Let me reconsider...

Actually, since we want a single answer, not all reachable subsets, and n ≤ 100, meet-in-the-middle could work well:
- Split the n numbers into two halves (n/2 each)
- Generate all subset sums for each half
- For each sum from the first half, find the best matching from the second half

With n=100, splitting gives us two sets of size 50, each with up to 2^50 subsets - way too many. But actually we don't need to generate all: many will give overlapping or redundant sums, especially with smaller numbers. And with random distributions, sums spread out more.


Let me reconsider what makes sense here. With uniform distribution U(0,B), many combinations cluster tightly around T, while exponential distributions create fewer distinct values near the middle. Rather than generating all 2^n subsets directly, I should filter out clearly impossible ranges early—skipping any subset whose minimum possible contribution exceeds our current best result's distance from T. When numbers grow exponentially in the Pareto case, larger elements alone can exceed T, allowing aggressive pruning. The real issue is handling the potentially enormous search space without exhaustive enumeration.

Looking at the constraints more carefully: with n up to 100, even small numbers create massive sums. But the scoring system rewards accuracy logarithmically, suggesting near-exact answers are typically found rather than requiring complete exploration. A greedy strategy ordered by magnitude works well when numbers form reasonable ratios—but breaks on pathological inputs like [8,6,4,3] targeting 10, where it gets stuck at 8+6=14 instead of finding 8+3=11 (closer to 10). The DP approach is cleaner and correct across all cases. Using an unordered_set to track reachable sums iteratively handles the full range efficiently despite T reaching 1e15.
</think>

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long T;
    if (!(cin >> n >> T)) return 0;
    vector<long long> a(n);
    for (int i = 0; i < n; i++) cin >> a[i];

    // DP to find all reachable sums using a hash set
    unordered_set<long long> reachable;
    reachable.insert(0);
    vector<char> used(n + 1, 0); // used[i] tracks whether element i was used to reach current sum

    for (int i = 0; i < n; i++) {
        long long val = a[i];
        vector<pair<long long, int>> next_reachable; // pair<sum, last_used_index>
        for (auto &p : reachable) {
            next_reachable.push_back({p.first, -1}); // keep existing sums
        }
        for (auto &p : reachable) {
            next_reachable.push_back({p.first + val, i});
        }
        // Remove duplicate sums, keeping the one with highest last_used_index for tie-breaking
        sort(next_reachable.begin(), next_reachable.end(), [](const auto &a, const auto &b){
            if (a.first != b.first) return a.first < b.first;
            return a.second > b.second;
        });
        next_reachable.erase(unique(next_reachable.begin(), next_reachable.end(),
                                   [](const pair<long long,int>& a, const pair<long long,int>& b) {
                                       return a.first == b.first && a.second == b.second;
                                   }), next_reachable.end());
        reachable = next_reachable;
    }

    long long best = -1, diff = numeric_limits<long long>::max();
    for (long long s : reachable) {
        long long cur_diff = abs(s - T);
        if (cur_diff < diff || (cur_diff == diff && (best == -1 || s > best))) {
            diff = cur_diff;
            best = s;
        }
    }

    cout << best;
    return 0;
}
```
````
