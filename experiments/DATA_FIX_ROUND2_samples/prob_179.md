# 题 179 — base均分 3.0 vs 直接SFT均分 0.0(掉点 3.0)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=0,得分 15.0,completion_tokens=24381)

````
The user wants a C++ solution for the "Subset Sum" problem.
The problem asks to select a subset of $n$ non-negative integers such that their sum is as close as possible to a target value $W$.
Constraints:
- $n \le 2100$
- Numbers (including $a_i$ and $W$) can be up to $10^{1100}$. This means we need big integer arithmetic.

Since the numbers are extremely large, we cannot use standard dynamic programming or bitsets because the state space would be too large. The values are not fitting in standard integer types (like `long long`).
However, usually competitive programming problems with such large inputs have specific tricks or constraints that allow optimization.
Wait, let's look at the score function:
`score = max(0, 1 - |W - S| / (W + M))`
where $M = \max(a_i)$.
If we can achieve $S = W$, score is 1.
We want to minimize $|W - S|$.

Let's look at the constraints again.
$n \le 2100$.
Values up to $10^{1100}$.

This looks like it might be solvable using some property of big integers or perhaps the constraints are looser than they appear? No, $10^{1100}$ is huge. Standard DP array size depends on the sum or $W$. Since $W$ is huge, we can't do DP based on sums.
Maybe there is a greedy approach? Or maybe the number of significant digits matters?
Actually, with arbitrary precision numbers, exact subset sum is NP-complete. However, if $n$ is small enough, maybe meet-in-the-middle works? But $n=2100$ is too big for meet-in-the-middle ($2^{1000}$ is impossible).

Wait, usually when input values are given as strings and $N$ is up to 2000-3000, it implies either:
1. There is a special mathematical structure (e.g., powers of 2).
2. We only need to output *any* valid subset if an optimal one exists, but here we need the closest.
3. Maybe the constraint on values is a distractor, and the actual relevant part is related to $n$?
4. Or maybe this is a variation where we just need to compare magnitudes.

Let's reconsider the score.
$|W - S|$ needs to be minimized.
With numbers up to $10^{1100}$, exact equality check requires big integers.
Is it possible that we simply treat them as strings?
Comparing two big integers takes time proportional to length. Addition also.
With $n=2100$, if we sort the numbers by magnitude, does that help?

Actually, looking at similar problems from platforms like CodeChef or AtCoder involving large numbers and subset sums...
Often, if numbers are very large compared to $n$, or vice versa, there are patterns.
But here both are huge.
Wait, is it possible the problem statement implies standard big integer libraries (like GMP) are allowed? Usually in CP, only STL is allowed unless specified. Implementing a BigInt class is necessary.

However, implementing a full BigInt class with division/subtraction logic for comparison purposes is feasible.
The core issue is the NP-hardness. Subset Sum is hard.
Is there any constraint I missed?
$a_i \ge 0$.
Target $W$.
Score definition involves $W + M$.
Maybe the values are sparse? Or maybe $n$ is actually smaller in practice for hard cases?
No, $n \le 2100$ suggests an $O(n^2)$ or $O(n \log n)$ algorithm if we can process elements efficiently.
Wait, if $a_i$ were small (like $\le 10^5$), we could use boolean array. Here they are large.

Could it be that we just select numbers greedily?
Like the change-making problem? That works if divisibility holds or coin system has specific properties. General integers don't work.
However, note that $n$ is up to 2100. If we pick the largest $a_i$ that doesn't exceed current remainder, it's a greedy heuristic. It won't always give optimal.
But maybe for the purpose of "scoring" where the denominator is $W+M$, maybe getting close on magnitude is enough? No, the score depends on absolute difference $|W-S|$.

Is there a specific trick?
Perhaps we can represent numbers as polynomials over some base?
Or, consider the constraint: "Large integer arithmetic may be required".
And "n <= 2100".
Usually, subset sum with large numbers and large N is hard. Is this problem actually solvable within time limits (likely 1-2 seconds)?
Maybe there is a misunderstanding of the problem complexity?
Ah, wait. Let's look at the problem source or context if possible. No external access.
Let's assume standard CP constraints. Time limit ~ 2s. Memory limit ~ 256MB or more.

If the values are truly arbitrary large numbers, finding the exact closest sum is likely NP-Hard. Unless there is a constraint I am missing or a simplification.
Wait, what if $n$ is small enough that we can process all $2^n$ combinations? No, $2^{2100}$ is impossible.
What if most numbers are small? The problem says "can be up to", meaning some can be small.
What if we just use a heuristic? The prompt asks for a solution code. In many CP contexts with "soft scoring", heuristics get partial points.
But usually, if it's a submission request, it expects an accepted solution.
Wait, is it possible that this is a "subset sum" problem where the number of elements is limited such that $2^n$ isn't the way, but since numbers are big, maybe the intended solution relies on the fact that numbers are distinct?
"All numbers ... can be up to".
Actually, if this is from a specific contest (e.g. HackerRank, LeetCode, etc.), sometimes there are specific constraints hidden or known issues.
Wait, let's search my internal knowledge base for "Subset Sum" + "large integers" + "n <= 2100".
This looks very specific.
Actually, if $N$ is 2100, maybe we can split the set into small parts?
If we can't use BigInt, we can't store these numbers. So we must implement BigInt.
Implementing a BigInt class that supports addition, comparison, and subtraction is required.
Given $N=2100$, a meet-in-the-middle is impossible. A pseudo-polynomial DP is impossible.
Greedy is wrong generally.
Knapsack-like greedy? Sort descending?
Actually, if the goal is to match $W$, and we have many items...
Wait! What if the number of items is large, but they follow a pattern? No info.

Is it possible the problem is actually about **finding IF** a subset sums to W, but the soft score implies approximate?
The prompt says "Your task is to choose a subset whose sum is as close as possible".
With Soft Scoring, usually you want to minimize $|W-S|$.
With large integers, comparing $|W-S|$ exactly is hard without full implementation.

Wait, could this be a case for randomized search or simulated annealing? With $n=2100$ and $10^{1100}$, the search space is massive. Random sampling might hit the solution if density allows.
But for CP, deterministic is preferred.

Let's step back. Is there any property that makes this tractable?
If we have numbers $a_1, \dots, a_n$.
Consider the most significant digits. We want to match the MSBs of $W$.
Suppose $W \approx 10^{1000}$. Most numbers will have fewer digits. Only those with high digits matter.
If we treat numbers as arrays of decimal digits (strings).
Let's implement a simple BigInt class supporting:
1. Parsing string to vector<int> (big int).
2. Add two BigInts.
3. Subtract two BigInts.
4. Compare two BigInts.
5. Store selected indices.

Now the algorithmic strategy.
Since $N$ is up to 2100, maybe we can use the fact that we want to minimize $|W - S|$.
If we sort $a_i$ in some order?
Or maybe this is a variant of the "Change Making Problem" which is solved greedily only for canonical coin systems. Not applicable here.

Is it possible that we are expected to solve this using **Branch and Bound** or some recursion with pruning?
If $|W-S|$ gets too large relative to remaining max sum, prune.
Given the score formula, maybe we don't need to be precise if $|W-S|$ is significantly larger than remaining possible sums?
Current Diff $D = |W - S_{current}|$.
Remaining Sum Cap $R = \sum a_j$.
If $R < D$, we definitely cannot reduce the diff to 0. The best we can hope is $|W - (S + R)|$ vs $|W - S|$.
Specifically, we want to reach $W$.
Current Sum $S$. Target $W$. Remainder needed $Rem = W - S$.
Max possible additional sum is $MaxRem = \sum a_k$.
If $Rem > MaxRem$, we are under. Best is $MaxRem$ (include everything).
If $Rem < -MaxRem$, we are over. Best is include nothing (exclude everything from current branch).
If $-MaxRem < Rem < MaxRem$, the target is reachable potentially.

This sounds like a search algorithm. With $N=2100$, we can't explore all subsets.
BUT, notice the score metric again.
$|W - S|$ needs to be minimized.
If we can't find exact $W$, we accept whatever is closest.
Given the constraints and the nature of the problem description (which looks like it might be generated or specific to a platform with large inputs), maybe a randomized greedy approach is acceptable?
Or perhaps there's a misunderstanding of the difficulty level.
However, usually with $N \approx 2000$, we expect $O(N \log N)$ or $O(N)$.
With $10^{1100}$, operations on BigInts take $L \approx 1100/9 \approx 122$ words.
So $O(N \cdot L)$ is acceptable ($2100 \cdot 120 \approx 2.5 \cdot 10^5$ ops).
The issue is NOT the cost of addition/comparison (BigInt is fine).
The issue is the combinatorial explosion of states.

Is there a constraint I missed?
"All numbers (including W and aᵢ) can be up to 10^1100."
Wait, "up to". They can be 0? Yes.
Are they distinct? "multiset". Can be duplicates.

Okay, if exact solution is NP-Hard, why is $N$ so high?
Usually, if $N$ is high and numbers are large, it implies the answer is found via some other method.
Maybe the problem assumes we can use floating point approximations for selection?
"Your task is to choose a subset whose sum is as close as possible".
If we select items greedily to fill $W$?
Sort $a_i$ descending. Pick $a_i$ if adding it doesn't overshoot $W$ too much?
A common heuristic for Knapsack/Subset Sum is the fractional knapsack relaxation or greedy by density.
But here we want sum = W exactly or close.
If we sort by magnitude?
Actually, a very strong heuristic for subset sum when values are large is to try to satisfy the most significant digits first.
But that's basically Greedy.

Wait! Is it possible that this problem is solvable because the judge provides weak test cases or specific types of numbers?
No, I should write code that tries to be optimal or near-optimal within reason.
Actually, given "soft scoring", heuristics are often used.
However, if I were writing a C++ solution, I should try to make it correct for solvable instances.
But Subset Sum is NP-Complete. For general inputs, no polynomial time algorithm exists (assuming P!=NP).
So there must be something specific here.
Could it be related to the fact that numbers are up to $10^{1100}$ but maybe the number of *distinct* values is small? Or many are 0?
The problem says "multiset", $n \le 2100$.
If I have 2100 items and values up to $10^{1100}$.
Maybe the intended solution is simply to use a greedy strategy with Backtracking/Recursion?
Let's try to write a recursive solver with memoization?
State: `(index, current_sum)`. Current sum is BigInt. Hash map `map<BigInt, index>`.
Memory will be the issue because `BigInt` keys are heavy. And state space explodes.
Memoization based on difference? `(index, diff)`. Still heavy.

Let's pause. Is there a solution that uses `std::bitset`? No, sum too big.
Maybe the time limit is generous?
What if we just implement a robust greedy strategy?
Sort $a_i$ in random order?
Better: Sort $a_i$ by value magnitude (largest first).
Why? Because to adjust the sum precisely, large numbers dominate the decision for higher orders of magnitude.
Algorithm:
1. Parse $W$ and all $a_i$.
2. Sort $a_i$.
3. Use Recursion with Pruning.
   `solve(index, current_sum)`
   Goal: Minimize $|W - current\_sum|$.
   Pruning condition:
   `remaining_capacity` = sum of all $a_i$ from `index` to end.
   `current_diff` = $|W - current\_sum|$.
   If `remaining_capacity` < `current_diff`, then even if we add ALL remaining, we can't bridge the gap better than staying as is?
   Wait, if we are under ($W > current\_sum$), adding remaining reduces diff.
   If we are over ($W < current\_sum$), adding remaining increases diff further.
   So if $W > current\_sum$ and $rem\_sum < W - current\_sum$, we can never catch up.
   Actually, we should track `min_abs_diff` found so far globally.
   Prune if `min_possible_abs_diff` >= `min_abs_diff_found`.
   
   However, `min_possible_abs_diff` estimation requires knowing how much we can change the sum.
   If we include all remaining: new sum = current + rem_sum. Diff = $|W - (current + rem\_sum)|$.
   If we exclude all remaining: new sum = current. Diff = $|W - current|$.
   Any subset sum lies between `current` and `current + rem_sum`.
   The closest to $W$ we can get from here is bounded.
   Specifically, if interval `[current, current + rem_sum]` does not contain $W$, we know the direction.
   If $W < current$, we should probably drop elements.
   If $W > current + rem_sum$, we should add all elements (or try to add all remaining if they fit?).
   
   This looks like a Branch and Bound.
   Given $N=2100$, pure B&B is risky. But if the numbers are large, the gaps between reachable sums are huge, except at lower bits?
   Wait, if numbers are large, the number of reachable sums near $W$ is sparse? No, if numbers are large, sums vary wildly.
   
   Let's assume the provided solution should be a greedy strategy or randomized local search since exact is impossible in poly-time.
   But as a Competitive Programmer, I should provide the best possible deterministic attempt or a library-based solution if allowed.
   Wait, I cannot use external libraries. I must implement BigInt.
   
   Let's check the constraints again.
   $n \le 2100$.
   Values $10^{1100}$.
   If I submit a simple Greedy approach (sort ascending, pick smallest until sum exceeds W?), it usually fails for subset sum.
   Standard heuristic:
   Try to reach W exactly.
   Sort $a_i$ descending.
   Iterate through $a_i$. If $current\_sum + a_i \le W$, take it.
   This is greedy.
   
   Is there a better one?
   What about sorting $a_i$ randomly and running a randomized greedy with restarts?
   Or, "meet-in-the-middle" is $2^{N/2}$. $2^{1050}$ is impossible.
   
   Wait, there is a possibility.
   Is it possible that the intended solution is simpler?
   "All numbers (including W and aᵢ) can be up to 10^1100."
   Maybe they fit in `__int128_t`? No, $10^{1100} \gg 2^{128}$.
   Maybe $n$ is small enough for bitset if numbers were small?
   
   Okay, let's look at the "Soft Scoring Rule".
   "Achieving S = W yields score = 1".
   "Any valid output is accepted and scored".
   This confirms that finding an exact match is ideal, but anything works.
   Since exact match is NP-hard, and $N$ is large, exact solutions for all inputs are unlikely.
   The test cases might be weak, or the constraints allow a specific optimization I'm missing.
   Could it be that we just output `0`s everywhere? Sum=0. Closest to $W$?
   Score $\propto 1/(W+M)$. If we pick nothing, score is low.
   Better to pick stuff.
   
   Let's implement a robust Greedy with Backtracking.
   To make it faster, group numbers?
   No, let's just implement a recursive search with aggressive pruning.
   State: `solve(idx, current_sum)`.
   Global `best_diff`.
   Pruning:
   Calculate `max_possible_sum` from remaining elements.
   If `current_sum + max_possible_sum < W`: We must add all remaining to get closer (if we want to increase sum). But we might still fall short.
   Wait, if `current_sum + max_possible_sum < W`, then even with all remaining, sum is $< W$.
   So any solution from this state will be $< W$. The diff is $W - (current + included)$.
   To minimize this, we want to maximize included.
   So if we are strictly below and can't catch up, adding all remaining is the ONLY path that improves?
   Actually, if we are below $W$, adding *more* decreases diff. Adding *less* (subsets) decreases sum, increasing diff.
   So if `current_sum + max_possible_sum < W`, we MUST take ALL remaining elements to minimize diff locally.
   Similarly, if `current_sum > W` AND `current_sum > max_possible_sum`? No.
   If `current_sum > W`, adding more increases diff. So we should NOT add any element.
   So:
   1. If `current_sum > W`:
      We shouldn't add anything further to keep sum small? No, we want sum close to $W$. If we are already above, adding makes it worse.
      So we stop taking. We record diff and backtrack (try excluding).
      Wait, in standard recursion `solve(idx, ...)`, we decide for `idx`.
      If `current_sum > W`, then including `a[idx]` will make it even larger, so diff grows.
      Excluding `a[idx]` keeps it same.
      So if we are already over, we effectively skip all future elements.
      Exception: Maybe later we find an element we excluded earlier was needed?
      No, standard recursion handles decisions one by one.
      Wait, if `current_sum > W`, adding ANY future positive integer makes `current_sum` larger, moving away from $W$ (since $W \ge 0$).
      So if we are over, we MUST stop selecting.
      Update `best_diff` with `abs(W - current_sum)`.
      Return.
      
   2. If `current_sum + remaining_max_sum < W`:
      Even if we take everything, we won't reach $W$.
      To minimize diff ($W - sum$), we need max sum.
      So we MUST take everything remaining.
      Update `current_sum` += remaining_all. Record diff. Return.
      
   3. If `remaining_max_sum` is roughly covering the gap:
      We branch.
      Take `a[idx]`: recurse.
      Don't take: recurse.
      
   Optimization:
   Sort $a_i$ descending. Large items processed first.
   This helps pruning early?
   If we process large items first, we quickly jump the range of achievable sums.
   Also, calculating `remaining_max_sum` requires prefix/suffix sums.
   Precompute suffix sums of sorted $a_i$.
   
   Heuristic to avoid deep branches:
   Use a stack limit or depth-first.
   Since we want *closest*, maintaining global `best_diff` and updating bounds.
   
   This is essentially "Branch and Bound" on the tree of inclusion/exclusion.
   Complexity: Worst case $2^n$. But with large numbers, the values spread out, so maybe few nodes satisfy the tight bound conditions.
   Actually, if numbers are large, `remaining_max_sum` drops quickly.
   
   Wait, if numbers are large, say $10^{100}$, and $W$ is around there.
   Taking $a_i$ changes sum by $10^{100}$.
   The decision is binary.
   This feels like it's designed to pass with this approach.
   
   One important detail:
   The problem asks for outputting $b_i$ (indices).
   We need to track which items were selected.
   With recursion, we can backtrack the choice.
   
   Implementation details:
   - BigInt class: Vector of `uint32_t` or `int64_t`. Base $10^9$ or $10^{18}$.
     Since we need to handle up to $10^{1100}$, that's $\approx 122$ blocks of $10^9$.
     Operations: Add, Subtract, Compare.
     Comparison is crucial for sorting and pruning.
     Subtraction for checking diffs.
     Absolute value logic.
   - Parsing strings.
   - Sorting indices based on values (descending).
   - Recursive search.
   
   Wait, `current_sum > W` case.
   If `current_sum > W`, can we ever improve?
   Only if we removed a previously added element?
   My pruning logic: "If we are currently over, we must stop adding".
   This assumes monotonicity. Yes, $a_i \ge 0$.
   So if current sum is already $> W$, adding more increases sum $\rightarrow$ further from $W$.
   So yes, we can stop. We just return.
   This significantly prunes the search space once we overshoot.
   
   Case 2: `current_sum + remaining_sum < W`.
   Here we are definitely under $W$. Even adding all remaining won't reach $W$.
   So adding all remaining minimizes the difference ($W - (S + R)$).
   Any subset would result in $S + S' < S + R \implies W - (S+S') > W - (S+R)$.
   So we should just take all remaining.
   This is another pruning cut.
   
   This strategy turns the exponential search into something much smaller, likely polynomial in practice for "typical" large number inputs where ranges are disjoint or cover gaps well.
   Actually, this is known as "Interval Pruning".
   
   Complexity of BigInt operations: $O(L)$ where $L \approx 120$.
   Sorting: $O(N \log N \cdot L)$.
   Recursion: Hard to bound, but pruning makes it fast.
   Total memory: $O(N \cdot L)$.
   
   Let's refine BigInt.
   Base $10^9$ fits in `unsigned int` easily, addition avoids overflow if careful.
   Actually `unsigned long long` (base $10^{18}$) is better for fewer words.
   $10^{1100}$ needs $\lceil 1100 / 18 \rceil \approx 62$ chunks. Very fast.
   
   Corner cases:
   - All $a_i = 0$.
   - $W = 0$.
   - No solution equals W.
   
   Input format:
   Line 1: n, W
   Line 2: a_1 .. a_n
   
   Output:
   Space separated b_i corresponding to original indices (wait, input gives a list, but we should output in order of input indices $1 \dots n$).
   "Output exactly one line: n integers b₁, b₂, ..., bₙ"
   "each bᵢ ∈ {0, 1}"
   "bᵢ = 1 means aᵢ is selected"
   So we need to preserve original indices for output, but processing order can be permuted.
   Store pairs `(value, original_index)`. Sort them.
   Keep track of selected state in a boolean vector `ans[n]`.
   
   Code Structure:
   1. BigInt Struct.
   2. Parse.
   3. Store `pair<BigInt, int>` items.
   4. Sort items descending.
   5. Compute suffix sums of values (BigInt) for pruning.
   6. DFS function.
   7. Print results.
   
   Wait, one thing. `current_sum` starts at 0.
   Target is $W$.
   DFS parameters: `(item_idx, current_sum, selected_mask_indices)`
   Pass `selected_mask_indices` by reference?
   Backtracking:
   Set `ans[item.original_idx] = 1`, recurse.
   Reset, recurse (exclude).
   Wait, sorting changes indices.
   `ans` array should be indexed by original position.
   
   BigInt implementation details:
   - Store digits in reverse order (least significant first).
   - Size type `size_t`.
   - Constructor from string.
   - `operator+`
   - `operator-` (assuming no borrow check or returns negative?)
   - `operator-` returns non-negative if $A \ge B$.
   - `compare`
   - Assignment.
   
   Since we deal with differences $|W - S|$, we need `diff(A, B)`.
   If $A > B$, $A - B$. Else $B - A$.
   
   Optimization:
   Use `std::vector<uint32_t>` with base $10^9$.
   String to BigInt: $O(len^2)$ naive, but length is small (1100 digits -> 110 chars). Wait, $10^{1100}$ means 1101 digits.
   Conversion is fast.
   
   Wait, $10^{1100}$ is a value. That has 1101 digits.
   So parsing is trivial.
   
   Wait, memory limit. "execution time and memory limit are also stated".
   Prompt says "Execution time and memory limit are also stated in the statement".
   I didn't see them in the prompt text provided, but usually defaults apply.
   Assuming standard.
   
   One subtle point:
   Is it possible `current_sum + remaining_sum == W`?
   If so, we found the solution (diff 0). We can terminate immediately.
   
   Another check:
   $N$ up to 2100.
   With the pruning logic (interval check), it behaves very efficiently for large numbers.
   Because `remaining_sum` drops by factors or amounts comparable to $a_i$.
   If $a_i$ are large, the "window" where valid subset sums lie is narrow compared to total sum space?
   Actually, if $a_i$ are random large numbers, the probability of hitting $W$ is low.
   But the pruning logic `current + rem < W` and `current > W` cuts off vast swathes of the tree.
   Specifically, if we overshoot, we stop. If we undershoot but `rem` is insufficient, we take all.
   This restricts the branching factor.
   Effectively, this finds the optimal subset in a reasonable time for these constraints.
   
   I will proceed with this implementation.
   
   Steps:
   1. Define BigInt.
   2. Read Input.
   3. Store (Value, Original Index).
   4. Sort Descending.
   5. Compute Suffix Sums (`suffix_sum[i]` = sum of values from `i` to `end`).
   6. Call DFS(0, 0, ans_vector).
   7. Inside DFS:
      - If `current_sum == target`: Found! Record `best_diff = 0`, `found = true`, return.
      - If `current_sum > target`:
          We cannot improve diff (adding increases sum).
          Update global best.
          Return.
        - If `current_sum + suffix_sum[idx] < target`:
          We cannot reach target even with all remaining.
          We should add all remaining to get closest to target.
          For all $k$ from `idx` to `end`:
             `ans[items[k].original_idx] = 1`.
          Update global best with diff($target, current + suffix[idx]$).
          Return.
      - Branch:
         - Include `idx`: DFS(idx + 1, current + val, ans)
         - Exclude `idx`: DFS(idx + 1, current, ans)
         Note: Order of branches? Include first usually better?
         - Need to update `ans` correctly (backtrack).
         - Optimization: Check if `best_diff` is 0 before returning.
         
   Wait, if `best_diff` becomes 0, we can exit everything.
   
   Wait, the "Include All Remaining" logic:
   When we take all remaining, we don't just stop; we record the result.
   Do we need to backtrack further?
   No, because that subtree leads only to sums $S \in [current, current + suffix[idx]]$.
   Since we know $W$ is outside this interval (on the higher side), the point closest to $W$ inside this interval is `current + suffix[idx]`.
   Any subset choice inside gives a smaller sum, hence larger diff.
   So `current + suffix[idx]` is the optimal choice for this subtree.
   So we commit all bits for the rest and return.
   
   Correctness check:
   What if $W$ is inside the interval?
   Then we cannot prune.
   
   What if `current_sum > target`?
   Interval is `[current, current + suffix]`.
   Target is below `current`.
   Closest point is `current`.
   Any addition increases sum $\to$ further from target.
   So `current` is optimal.
   Commit nothing (exclude all remaining).
   Return.
   
   So:
   Lower Bound of reachable from idx: $L = current$.
   Upper Bound of reachable from idx: $U = current + suffix[idx]$.
   Target $W$.
   If $W < L$: Optimal is $L$. We pick nothing.
   If $W > U$: Optimal is $U$. We pick everything.
   If $L \le W \le U$: Pruning not possible based on bounds alone.
   
   Wait, is it strictly "pick nothing" for $W < L$?
   Yes, because $a_i \ge 0$. Adding anything moves sum away from $W$.
   Is it strictly "pick everything" for $W > U$?
   Yes, because adding more is needed to approach $W$.
   (Assuming we want to minimize distance).
   
   This covers the "greedy" aspects dynamically.
   
   One edge case: `suffix_sum` calculation.
   Need BigInt suffix sums.
   
   Data structures:
   `struct Item { BigInt val; int id; };`
   `vector<Item> sorted_items;`
   `vector<BigInt> suffix_sums;` (Size n+1, suffix_sums[n] = 0).
   
   Memory:
   $2100 \times 122$ ints $\approx 250$ KB. Trivial.
   Recursion depth 2100. Stack usage ok.
   
   Time:
   BigInt op overhead small.
   The branching factor is heavily reduced.
   
   Final check on BigInt base.
   Using `unsigned int` (base $10^9$) ensures easy multiplication/addition.
   Actually, `unsigned int` is sufficient for storing digits $0..999999999$.
   Vector size $\approx 1100/9 \approx 123$.
   
   ```cpp
   #include <iostream>
   #include <vector>
   #include <string>
   #include <algorithm>
   #include <cmath>
   
   using namespace std;
   
   // BigInt struct
   struct BigInt {
       vector<unsigned int> digits; // Base 10^9
       
       BigInt() {}
       BigInt(long long n) {
           if (n == 0) digits.push_back(0);
           while (n > 0) {
               digits.push_back(n % 1000000000);
               n /= 1000000000;
           }
       }
   
       BigInt(string s) {
           if (s.empty()) { digits.push_back(0); return; }
           bool leading_zeros = true;
           string rev = "";
           // Handle potential negative signs? Problem says non-negative.
           if (s.back() == '-') { 
              // Should not happen per spec.
              return; 
           }
           // Pad with zeros to multiples of 9? Or parse char by char.
           for (int i = s.length() - 1; i >= 0; --i) {
               if (isdigit(s[i])) {
                   rev += s[i];
               }
           }
           
           // Process in chunks
           for (int i = 0; i < rev.length(); ) {
               int chunk_len = min(9, (int)rev.length() - i);
               unsigned int chunk = stoi(rev.substr(i, chunk_len));
               digits.push_back(chunk);
               i += chunk_len;
               if (digits.size() > 0 && digits.back() == 0 && chunk < 1000000000 && i < rev.length()) {
                   // Leading zero handling if needed, though logic handles 0 naturally
               }
               // Optimizing construction: direct push
               // Re-implementation for speed/correctness
           }
           
           // Clear and redo properly
           digits.clear();
           if (s.empty()) return;
           for (int i = s.length() - 1; i >= 0; i -= 9) {
               string seg = s.substr(max(0, i - 8), 9);
               BigInt temp(seg);
               if (!temp.digits.empty()) digits.insert(digits.end(), temp.digits.begin(), temp.digits.end());
           }
           digits.erase(remove(digits.begin(), digits.end(), 0), digits.end());
           if (digits.empty()) digits.push_back(0);
       }
   };
   ```
   Wait, the constructor is buggy. Let's fix it.
   
   Better BigInt logic:
   ```cpp
   BigInt(string s) {
       if (s == "0") { digits.push_back(0); return; }
       // Remove leading zeros?
       // Input string can be large.
       // We process from right to left.
       digits.clear();
       int n = s.length();
       for (int i = n; i > 0; i -= 9) {
           int start = max(0, i - 9);
           string segment = s.substr(start, i - start);
           unsigned int val = stoll(segment);
           digits.push_back(val);
       }
       // Ensure non-zero top or remove leading zeros if empty
       if (digits.size() > 0 && digits.back() == 0) {
           // Remove trailing zeros from vector (which are most significant)
           digits.pop_back();
           if (digits.empty()) digits.push_back(0);
       }
   }
   ```
   Actually `stoll` handles the substring conversion.
   
   Comparisons:
   Length of vector determines magnitude.
   Equal length: compare last element (MSB).
   
   Addition:
   Standard column addition.
   
   Subtraction:
   Borrow handling. Assume A >= B.
   
   Wait, `stoll` might throw if string is very long? No, max 9 chars. Safe.
   
   Global variable `min_diff` initialized to a very large number.
   We also track `best_selection`.
   
   One tricky part: `min_diff` initialization.
   Since numbers up to $10^{1100}$, initial diff is $W$ (assuming empty set sum 0).
   Or $|W|$.
   
   Also, printing the result.
   We need to construct the vector `b` where `b[i]` corresponds to input order.
   `sorted_items` stores original indices.
   
   Time Limit:
   With BigInt operations, every node does a few BigInt adds/comparisons.
   If depth is 2100, even with efficient pruning, it's safe?
   Usually yes for these problem types.
   
   Double check the problem source style.
   "Problem: Subset Sum".
   Often found in beginner contests or with specific constraints.
   With "Soft scoring", partial credit is key.
   Greedy with pruning is likely the intended solution or "Good Enough".
   
   Wait, if `current_sum + suffix_sum[idx] == target`, we can just take all and finish?
   Yes, `min_diff = 0`, break everything.
   
   Also, check $W + M$ constraint in score formula.
   Does it affect our logic? No, minimizing $|W-S|$ maximizes the term $1 - \dots$, maximizing score.
   
   I'll write the full code now.

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <limits>

using namespace std;

// BigInt implementation to handle up to 10^1100
struct BigInt {
    vector<unsigned int> digits;
    static const unsigned int BASE = 1000000000;

    BigInt() {}

    void clear() { digits.clear(); if (digits.empty()) digits.push_back(0); }

    // Construct from string
    BigInt(string s) : digits() {
        if (s.empty()) {
            digits.push_back(0);
            return;
        }
        // Skip leading spaces/zeros if any
        size_t start_pos = 0;
        while (start_pos < s.length() && isdigit(s[start_pos]) == false) {
             // Should not happen based on constraints
             start_pos++;
        }
        if (s.find_first_of("0-9") == string::npos) { digits.push_back(0); return; }
        
        // Process in reverse chunks of 9
        size_t len = s.length();
        size_t pos = 0;
        while (pos < len) {
            int chunk_len = min(9, (int)(len - pos));
            string sub = s.substr(pos, chunk_len);
            unsigned int val = stoll(sub);
            digits.push_back(val);
            pos += chunk_len;
        }
        
        // Remove leading zeros (most significant in vector)
        while (digits.size() > 1 && digits.back() == 0) {
            digits.pop_back();
        }
        if (digits.empty()) digits.push_back(0);
    }
    
    // Comparison operators
    bool operator>(const BigInt& other) const {
        if (digits.size() != other.digits.size())
            return digits.size() > other.digits.size();
        for (int i = digits.size() - 1; i >= 0; --i) {
            if (digits[i] != other.digits[i])
                return digits[i] > other.digits[i];
        }
        return false;
    }
    
    bool operator<(const BigInt& other) const {
        if (digits.size() != other.digits.size())
            return digits.size() < other.digits.size();
        for (int i = digits.size() - 1; i >= 0; --i) {
            if (digits[i] != other.digits[i])
                return digits[i] < other.digits[i];
        }
        return false;
    }
    
    bool operator==(const BigInt& other) const {
        return digits == other.digits;
    }
    
    // Addition
    BigInt operator+(const BigInt& other) const {
        BigInt res;
        int carry = 0;
        size_t n = max(digits.size(), other.digits.size());
        for (size_t i = 0; i < n || carry; ++i) {
            unsigned int d = carry;
            if (i < digits.size()) d += digits[i];
            if (i < other.digits.size()) d += other.digits[i];
            res.digits.push_back(d % BASE);
            carry = d / BASE;
        }
        return res;
    }

    // Subtraction (assumes *this >= other)
    void subtract(const BigInt& other) {
        int borrow = 0;
        for (size_t i = 0; i < other.digits.size() || borrow; ++i) {
            int d = borrow;
            if (i < digits.size()) d += digits[i];
            else d = 0; // Should not happen if this >= other
            
            if (other.digits.size() > i) d -= other.digits[i];
            
            if (d < 0) {
                d += BASE;
                borrow = 1;
            } else {
                borrow = 0;
            }
            digits[i] = d;
        }
        // Trim zeros
        while (digits.size() > 1 && digits.back() == 0) {
            digits.pop_back();
        }
        if (digits.empty()) digits.push_back(0);
    }
    
    // Get sum of all elements in this struct as vector
    void reset() { digits.clear(); if (digits.empty()) digits.push_back(0); }
};

class Item {
public:
    BigInt val;
    int id; // original index
    Item(BigInt v, int i) : val(v), id(i) {}
};

// Global variables
vector<Item> items;
vector<BigInt> suffix_sums;
vector<bool> ans;
BigInt W_val;
BigInt current_sum; // passed in recursion
BigInt min_diff; // passed/globally
bool solved = false;
vector<int> best_result_ids; 

// Function to copy current ans to result
void update_best(int idx) {
    if (abs_diff(current_sum, W_val) < abs_diff(best_solution_sum, W_val)) { // Wait, need stored best sum
    }
}
// We need to store the best diff value directly.
// Since BigInt comparison is slow, store as string? No, BigInt.

```
Wait, `solved` flag.
I need to store the `best_diff` globally or in a member.
And the `best_selection` vector.
When a better diff is found (or 0), update global best.
`min_diff` tracks the minimal difference found. Initialize with a dummy large value.
Initial state: Sum = 0. Diff = |0 - W|.
Since $a_i \ge 0$, empty set sum is 0.

Correct Logic for Pruning:
Inside `dfs(idx)`:
1. `current` is sum of selected so far.
2. `upper_limit` = `current` + `suffix_sums[idx]`.
3. If `current` == `W_val`: Found perfect match! `solved = true`, update `best_result`, return.
4. If `current` > `W_val`:
   Since `a_i >= 0`, any further addition increases sum $\rightarrow$ worse.
   So this branch is dead for improvement (unless we are forced to take more, but we want minimum diff).
   Actually, we already took decisions. `current` is the sum of a subset. If it's > W, adding more elements will make diff larger.
   Wait, is it possible we skipped a decision earlier that would have helped? No, this is backtracking.
   At this state, `current` is fixed by previous choices.
   Since `current > W`, any subset of `{items[idx], ...}` added to `current` will be $\ge current > W$.
   The value closest to $W$ in the range $[current, upper\_limit]$ is `current` itself.
   So we just record this diff as a candidate and prune this branch of additions (return after recording).
   However, we still need to record the result to ensure we capture the configuration.
   Wait, `ans` array reflects the choices made *so far*.
   We need to save the current `ans` to compare.
   
5. If `upper_limit` < `W_val`:
   Range is $[current, upper\_limit]$. $W$ is above this.
   Value closest to $W$ is `upper_limit`.
   To achieve `upper_limit`, we MUST select all remaining elements.
   So we update `ans` for remaining elements to true, calculate sum, record result, return.
   
6. If neither (i.e., overlap):
   We branch:
   - Include `items[idx]`. (Add `val`).
   - Exclude `items[idx]`.
   Order? Include first is often good for filling.
   
   Need a helper to compute BigInt difference: `abs(W - S)`.
   
   To handle updates safely:
   Global `vector<bool> best_b(n)`.
   Global `BigInt best_sum`.
   Global `BigInt best_diff`. (Compare using custom `operator<` on difference? Just `min_diff` logic).
   
   Initialization:
   `min_diff` = `W_val` (diff when choosing 0 elements).
   `best_b` = all zeros.
   `best_sum` = 0.
   
   Wait, if `W` is 0?
   Empty set sum 0. Diff 0. Done.
   
   Implementation Details:
   `subtract`: `current` >= `other`. Returns void modifying `*this`.
   `diff(A, B)`: returns `A - B` (assuming A>=B) or `B - A`.
   We need `abs_diff` to compare with `min_diff`.
   We don't need to return BigInt for comparison, just `bool is_smaller`.
   Actually, we just need to maintain `min_diff` (as BigInt).
   If `abs_diff(current, W) < min_diff`, update `min_diff` and `best_b`.
   
   Wait, copying BigInt vectors is cheap ($\sim 100$ ints).
   
   Optimization: `ans` is global?
   Backtracking modifies it.
   When recording best: copy current state to `best_b`.
   Wait, `best_b` needs to be size $n$.
   
   Let's refine `dfs`.
   Parameters: `idx`, `current_sum` (BigInt).
   
   ```cpp
   void dfs(int idx, BigInt current) {
       if (solved) return; // Optimization: if we found 0 diff
     
       // Calculate range
       // Range: [current, current + suffix_sums[idx]]
       
       BigInt current_plus_rem = current.clone_or_add(suffix_sums[idx]);
       
       // Check lower bound
       if (current > W_val) {
           // Already over. Closest is 'current'.
           if (abs_diff(current, W_val) < min_diff) {
               update_ans();
               min_diff = abs_diff(current, W_val);
           }
           return; // Cannot improve by adding more
       }
       
       // Check upper bound
       if (current_plus_rem < W_val) {
           // Even with everything, less than W.
           // Closest is current_plus_rem.
           if (abs_diff(current_plus_rem, W_val) < min_diff) {
               update_ans_for_current_plus_rem(idx);
               min_diff = abs_diff(current_plus_rem, W_val);
           }
           return; 
       }
       
       // Branching
       // Option 1: Include items[idx]
       // Check if adding it makes it > W_val significantly?
       // We branch regardless, but we can order.
       
       // Include
       items[idx].val.add_to(current, &current); // Helper
       ans[items[idx].id] = 1;
       dfs(idx + 1, current);
       ans[items[idx].id] = 0;
       
       // Exclude
       dfs(idx + 1, current); 
   }
   ```
   Wait, passing `current` by value copies the BigInt. Cheap enough.
   Helper `add_to` is unnecessary if we just call constructor with sum.
   But constructing new BigInt each time creates garbage.
   Maybe pass `current` by reference and modify? Backtrack modifications.
   Better: `BigInt next_sum = current + item.val`.
   
   Wait, `update_ans_for_current_plus_rem` needs to mark all items from `idx` to end as 1.
   This is $O(n)$. Since this happens at most once per valid node, it's fine.
   
   Wait, `update_ans` (for current over) needs to record state.
   But `ans` contains selections made up to `idx`.
   The items `idx` onwards were not selected yet.
   So for `current > W`, we record `ans` as is (meaning none of `idx..n-1` selected).
   Wait, if we are at `idx`, and `current > W`, we decided on `0..idx-1`.
   Now we want to see if this path is better than best.
   Since we are at the boundary, any further decisions are redundant.
   So we just check `ans` (representing choices `0..idx-1`) and update best.
   BUT, `ans` is global. We are iterating.
   Wait, `ans` represents the current configuration.
   If `current > W`, we should probably set `ans` to reflect that `idx..end` are EXCLUDED.
   Wait, `ans` is being modified by recursion.
   Before entering `dfs(idx)`, `ans` has `0..idx-1` filled.
   Wait, standard DFS:
   At `idx`, we make decision for `items[idx]`.
   If `current > W`, it means decisions made SO FAR lead to `current > W`.
   So we can just check this state.
   We don't need to add anything more.
   So we update `best` with current `ans`.
   
   Wait, if we are at `idx` and `current > W` is checked INSIDE the loop for decisions.
   Suppose we are at root (idx=0). `current`=0.
   If we decide to take `items[0]`, `current` might become `> W`.
   In `dfs(idx+1, new_current)`, we check bounds.
   If `new_current > W`, we record and return.
   The recorded state is `ans` (with `items[0]` set).
   Correct.
   
   What about `current + suffix < W`?
   This implies we need to select ALL `idx` onwards.
   So we should iterate `k` from `idx` to `n-1` and set `ans[items[k].id] = 1`.
   Update best. Return.
   After returning, the parent restores state.
   
   One correction:
   `min_diff` init.
   Initial state: empty set sum = 0.
   Diff = `W_val`. (Assume `W` non-negative).
   Is it possible `W < 0`? Problem says `non-negative integers`. So `W >= 0`.
   So `abs(0 - W) = W`.
   `best_diff` initialized to `W`.
   `best_b` all 0.
   
   One corner case: $W=0$.
   Init `min_diff` = 0. `solved` = true.
   DFS checks `current == W`.
   
   Efficiency of cloning BigInts:
   Copying `vector<unsigned int>` of size 120 is fast.
   
   Complexity check:
   $N=2100$.
   BigInt ops $\approx 100$ cycles?
   DFS depth 2100.
   If pruning works, nodes visited $\ll 2^{2100}$.
   Likely $O(N^2)$ or similar effective nodes.
   
   Wait, sorting items is descending.
   Does this help pruning?
   If items are large, adding one jumps over intervals.
   Pruning condition `current + suffix < W` or `current > W` is powerful if `suffix` is large.
   Yes, sorting descending ensures `suffix` decays relatively fast, making the window tighter?
   Or rather, if we process large items first, we quickly determine the magnitude.
   Yes.
   
   One detail:
   "Line 1: two integers n and W".
   Input reading.
   
   Wait, `W` can be $10^{1100}$.
   Reading $W$ needs to be parsed into `BigInt`.
   
   I will put the `update_ans` logic carefully.
   
   Also, `min_diff` comparison.
   Two BigInts comparison `is_less`.
   
   Let's assemble.
</think>

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

using namespace std;

// BigInt structure
struct BigInt {
    vector<unsigned int> digits;
    static const unsigned int BASE = 1000000000;

    BigInt() {}
    explicit BigInt(unsigned int n) {
        if (n == 0) digits.push_back(0);
        else {
            while (n > 0) {
                digits.push_back(n % BASE);
                n /= BASE;
            }
        }
    }
    
    // Parse string representation
    BigInt(string s) {
        if (s.empty()) {
            digits.push_back(0);
            return;
        }
        // Remove leading whitespace (just in case)
        size_t pos = 0;
        while (pos < s.length() && s[pos] == ' ') pos++;
        if (pos == s.length()) { digits.push_back(0); return; }
        
        size_t len = s.length();
        for (int i = 0; i < len; ) {
            int chunk_len = min(9, (int)(len - i));
            unsigned int val = stoll(s.substr(i, chunk_len));
            digits.push_back(val);
            i += chunk_len;
        }
        
        // Remove leading zeros (which are at the end of vector)
        while (digits.size() > 1 && digits.back() == 0) {
            digits.pop_back();
        }
        if (digits.empty()) digits.push_back(0);
    }

    // Comparison
    bool operator>(const BigInt& other) const {
        if (digits.size() != other.digits.size())
            return digits.size() > other.digits.size();
        for (int i = digits.size() - 1; i >= 0; --i) {
            if (digits[i] != other.digits[i])
                return digits[i] > other.digits[i];
        }
        return false;
    }
    bool operator<(const BigInt& other) const {
        if (digits.size() != other.digits.size())
            return digits.size() < other.digits.size();
        for (int i = digits.size() - 1; i >= 0; --i) {
            if (digits[i] != other.digits[i])
                return digits[i] < other.digits[i];
        }
        return false;
    }
    bool operator==(const BigInt& other) const {
        return digits == other.digits;
    }
    bool operator<=(const BigInt& other) const {
        return !(*this > other);
    }
    
    // Addition
    BigInt operator+(const BigInt& other) const {
        BigInt res;
        int carry = 0;
        size_t n = max(digits.size(), other.digits.size());
        for (size_t i = 0; i < n || carry; ++i) {
            unsigned int d = carry;
            if (i < digits.size()) d += digits[i];
            if (i < other.digits.size()) d += other.digits[i];
            res.digits.push_back(d % BASE);
            carry = d / BASE;
        }
        return res;
    }

    // Subtraction (destroys digits, assumes *this >= other)
    void sub(const BigInt& other) {
        int borrow = 0;
        for (size_t i = 0; i < other.digits.size() || borrow; ++i) {
            int d = borrow;
            if (i < digits.size()) d += digits[i];
            else d = 0; 
            
            if (i < other.digits.size()) d -= other.digits[i];
            
            if (d < 0) {
                d += BASE;
                borrow = 1;
            } else {
                borrow = 0;
            }
            digits[i] = d;
        }
        while (digits.size() > 1 && digits.back() == 0) {
            digits.pop_back();
        }
        if (digits.empty()) digits.push_back(0);
    }
    
    // Clone
    BigInt clone() const {
        BigInt res;
        res.digits = digits;
        return res;
    }
    
    // Equality check helper
    bool greaterThanZero() const {
        if (digits.empty()) return false;
        for (int i = digits.size()-1; i>=0; --i) {
            if (digits[i] != 0) return true;
        }
        return false;
    }
};

// Item structure
struct Item {
    BigInt val;
    int id;
    Item(BigInt v, int i) : val(v), id(i) {}
};

int n;
BigInt W;
vector<Item> items;
vector<bool> best_selection;
BigInt best_diff; // Minimal |W - S|
bool found_exact = false;

// Global temporary storage
vector<bool> current_selection;
BigInt current_sum;

// Difference between A and B (assume A >= B, returns A-B)
BigInt get_diff(const BigInt& A, const BigInt& B) {
    BigInt res = A.clone();
    res.sub(B);
    return res;
}

// Helper to compare BigInt differences (returns true if lhs < rhs)
bool diff_is_smaller(const BigInt& a, const BigInt& b) {
    return a < b;
}

// Helper to check if A == B
bool is_equal(const BigInt& a, const BigInt& b) {
    return a == b;
}

// Depth First Search
void dfs(int idx, BigInt cur_sum) {
    if (found_exact) return;

    // Pruning: Calculate upper bound of reachable sum
    BigInt remaining_max = items_remaining_sum(idx); // We need to precompute or calc on fly
    
    // We'll implement logic using global precomputed suffix sums to avoid recomputing
    // But accessing global inside lambda or function is okay.
}

// Precomputing suffix sums globally
vector<BigInt> suffix_sums;

// Initialize globals
void init_globals() {
    suffix_sums.resize(items.size() + 1);
    suffix_sums.back() = BigInt("0"); // 0 is empty set sum
    for (int i = n - 1; i >= 0; --i) {
        suffix_sums[i] = suffix_sums[i + 1] + items[i].val;
    }
    best_selection.assign(n, false);
    current_selection.assign(n, false);
    current_sum = BigInt("0");
    
    // Initial best: empty set sum = 0. Diff = W.
    best_diff = W.clone();
    current_selection.fill(false); // Default 0
    // Update best_diff logic
    // Actually, we can initialize best_diff inside main or here.
    // But we need to track current best solution state.
    // Let's rely on finding better during DFS.
    // Initially, if W=0, we found exact.
    if (is_equal(W, BigInt("0"))) {
        found_exact = true;
        // Output 0s
    } else {
        // Empty set is a candidate with diff = W
        // best_diff is W.
    }
}

BigInt abs_diff(const BigInt& A, const BigInt& B) {
    if (A > B) return A - B;
    return B - A;
}
// Wait, I overloaded - operator for BigInt but it's invalid syntax here (no subtraction defined as operator-(BigInt)).
// I should implement operator-.
// Let's rewrite BigInt methods to be clean.

// ... BigInt operator- ...
BigInt operator-(const BigInt& lhs, const BigInt& rhs) {
    BigInt res = lhs.clone();
    res.sub(rhs);
    return res;
}

// ... Updated BigInt structure ...

// Main logic rewritten for clarity within the snippet

BigInt subtract(const BigInt& A, const BigInt& B) {
    BigInt res = A.clone();
    res.sub(B);
    return res;
}

void solve_dfs(int idx, BigInt cur) {
    if (found_exact) return;
    
    // Bounds
    BigInt upper_bound = cur + suffix_sums[idx];
    
    // Case 1: Current sum > W
    if (cur > W) {
        BigInt d = subtract(cur, W);
        if (diff_is_smaller(d, best_diff)) {
            best_diff = d;
            // Update selection state (reset remaining to 0 implicitly by returning)
            // But we need to capture current state.
            // Since 'cur' came from previous decisions, current_selection holds valid choices.
            // Items idx..n-1 are 0.
        }
        // We cannot improve by adding more positive numbers
        return;
    }
    
    // Case 2: Upper bound < W
    if (upper_bound < W) {
        // We must take all remaining to get closest (highest possible sum)
        // Set all remaining selections to 1
        for (int k = idx; k < n; ++k) {
            current_selection[items[k].id] = true;
        }
        // Verify/update
        BigInt d = subtract(W, upper_bound);
        if (diff_is_smaller(d, best_diff)) {
            best_diff = d;
            // Copy current_selection to best_selection
            best_selection = current_selection;
        }
        return;
    }
    
    // Case 3: Exact match
    if (cur == W) {
        found_exact = true;
        // Record solution (though we just need to stop)
        // Actually, we record it if we haven't.
        // But logic below updates best_selection.
    }
    
    // Branch 1: Include items[idx]
    // Optimization: if taking it pushes us > W, we know we will handle it in next call or prune.
    // But we just recurse.
    // To avoid creating objects too often, pass reference or move? Move semantics available in C++11.
    // Since we use BigInt struct, moving is cheap if default ctor clears vector (it does).
    
    BigInt next_sum = cur + items[idx].val;
    current_selection[items[idx].id] = true;
    solve_dfs(idx + 1, next_sum);
    
    if (!found_exact) { // If found_exact became true, we return from inner calls
        current_selection[items[idx].id] = false;
        // Branch 2: Exclude items[idx]
        solve_dfs(idx + 1, cur);
    }
}
// Need to be careful with update logic.
// If we find exact, found_exact=true.
// In that case we might not have updated best_selection?
// We should update best_selection whenever best_diff improves.
// If we find exact, diff is 0.
```

One critical logic fix:
Updating `best_selection` inside the recursion is tricky because it's a global vector.
Copying it `best_selection = current_selection` inside the `if (diff_is_smaller)` block works fine.
However, `current_selection` reflects the current active decisions.
When we prune (Case 2), we modify `current_selection` to include the rest.
When we return from recursion, we must revert `current_selection` changes?
In Case 2, we iterated `k` from `idx` to `n`.
Wait, `current_selection` should reflect `idx..n`?
Yes. But we must ensure we don't mess up the parent's view?
Actually, `current_selection` is shared.
If we enter Case 2, we modify indices `idx..n-1` to 1.
Then return.
The caller (parent `idx-1`) expects `current_selection[items[idx].id]` to be 0 (or restored).
But we set it to 1 inside the case.
So we MUST revert changes to `current_selection` upon return.
Reverting logic for Case 2:
Iterate `k` from `idx` to `n`, set `current_selection[items[k].id] = false`.
Also need to revert the modification done in Branch 1 (setting to true) before trying Branch 2 (excluding).

Refined DFS:
```cpp
void dfs(int idx, BigInt cur) {
    if (found_exact) return;

    // Bounds
    BigInt remaining = suffix_sums[idx]; // Sum of items[idx]...items[n-1]
    // Note: suffix_sums[i] includes items[i]
    
    BigInt upper = cur + remaining;

    if (cur > W) {
        BigInt d = subtract(cur, W);
        if (diff_is_smaller(d, best_diff)) {
            best_diff = d;
            best_selection = current_selection;
        }
        return; // Adding more makes sum larger, diff larger.
    }

    if (upper < W) {
        // Need to add all remaining to minimize diff (closest from below)
        for (int k = idx; k < n; ++k) {
            current_selection[items[k].id] = true;
        }
        BigInt d = subtract(W, upper);
        if (diff_is_smaller(d, best_diff)) {
            best_diff = d;
            best_selection = current_selection;
        }
        // Revert
        for (int k = idx; k < n; ++k) {
            current_selection[items[k].id] = false;
        }
        return;
    }

    // Branch 1: Include
    current_selection[items[idx].id] = true;
    BigInt next_inc = cur + items[idx].val;
    dfs(idx + 1, next_inc);

    // Backtrack inclusion
    current_selection[items[idx].id] = false;

    // Branch 2: Exclude
    dfs(idx + 1, cur);
}
```
Need to be careful: if `found_exact` became true in child call, `dfs` returns.
We need to check `found_exact` inside loops to stop?
Actually, if `found_exact` is set, the children will propagate return.
We need to break out or rely on recursion.
The simplest is to check `found_exact` at the top of `dfs`.
For pruning cases: if `found_exact` is true, do nothing.
For Branch 2: if `found_exact` is true, return immediately.

One issue: `suffix_sums` computation.
It uses `items`.
We need to ensure `items` is populated and sorted.
Sorting should be descending.
`suffix_sums` logic is correct.

Input parsing: `cin >> n >> W_str` is safer than `cin >> n >> W` (BigInt needs string).
But `cin` skips whitespace.
Reading BigInt from string is good.
`n` is integer.

Wait, input says:
Line 1: `n` and `W`
`W` can be `10^1100`.
`cin >> n` then `cin >> ws` then read `W` string?
`cin >> string_var` reads a whitespace delimited word.
So `string w_str; cin >> w_str;` works.
Same for `a_i`. Loop `n` times.

Double check headers and namespace.
Use `<cstdio>` for speed? `cin/cout` is fine.
`ios::sync_with_stdio(0); cin.tie(0);`
Output format: `b_1 b_2 ... b_n` separated by space.

Final check on logic:
If `cur > W`, we prune adding. Correct.
If `cur + rem < W`, we force adding. Correct.
If `cur == W`, found exact. Correct.
Pruning ensures efficiency for large numbers.

Memory:
Vectors of BigInts (size 120).
$N=2100$. Total memory is negligible.
Time:
Operations are efficient.
Correctness seems solid for the intended constraints.

One small fix:
`current_selection` size is `n`.
Indices are `0` to `n-1`.
In `dfs`, we access `current_selection[items[k].id]`.
`items` stores original `id` from input (0-indexed).
Yes.

Code wrap:
Wrap in cpp block.
Remove comments to keep it clean? No, comments help understanding. But instructions say "ONLY contain the C++ code". Comments are part of code.
I will provide clean code.

```cpp
// Includes
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <functional>

using namespace std;

// BigInt class
struct BigInt {
    vector<unsigned int> digits;
    static const unsigned int BASE = 1000000000;
    
    BigInt() {}
    BigInt(unsigned int n) {
        if (n == 0) digits.push_back(0);
        else {
            while (n > 0) {
                digits.push_back(n % BASE);
                n /= BASE;
            }
        }
    }
    BigInt(string s) {
        digits.clear();
        if (s.empty()) { digits.push_back(0); return; }
        for (int i = 0; i < (int)s.length(); ) {
            int l = min(9, (int)s.length() - i);
            unsigned int val = stoll(s.substr(i, l));
            digits.push_back(val);
            i += l;
        }
        while (digits.size() > 1 && digits.back() == 0) digits.pop_back();
        if (digits.empty()) digits.push_back(0);
    }
    
    bool operator>(const BigInt& o) const {
        if (digits.size() != o.digits.size()) return digits.size() > o.digits.size();
        for (int i = digits.size() - 1; i >= 0; --i)
            if (digits[i] != o.digits[i]) return digits[i] > o.digits[i];
        return false;
    }
    bool operator<(const BigInt& o) const {
        if (digits.size() != o.digits.size()) return digits.size() < o.digits.size();
        for (int i = digits.size() - 1; i >= 0; --i)
            if (digits[i] != o.digits[i]) return digits[i] < o.digits[i];
        return false;
    }
    bool operator==(const BigInt& o) const {
        return digits == o.digits;
    }
    
    BigInt operator+(const BigInt& o) const {
        BigInt r;
        int c = 0;
        size_t m = max(digits.size(), o.digits.size());
        for (size_t i = 0; i < m || c; ++i) {
            unsigned int d = c;
            if (i < digits.size()) d += digits[i];
            if (i < o.digits.size()) d += o.digits[i];
            r.digits.push_back(d % BASE);
            c = d / BASE;
        }
        return r;
    }
    
    void sub(const BigInt& o) {
        int c = 0;
        for (size_t i = 0; i < o.digits.size() || c; ++i) {
            unsigned int d = c;
            if (i < digits.size()) d += digits[i];
            else d = 0;
            if (i < o.digits.size()) d -= o.digits[i];
            if (d < 0) { d += BASE; c = 1; }
            else c = 0;
            digits[i] = d;
        }
        while (digits.size() > 1 && digits.back() == 0) digits.pop_back();
        if (digits.empty()) digits.push_back(0);
    }
    
    BigInt clone() const { return *this; } // Implicit copy works for vector
};

// Globals
int n;
BigInt W;
vector<pair<BigInt, int>> items;
vector<bool> best_sol;
vector<bool> curr_sel;
BigInt min_diff;
bool exact_found = false;

// Precomputed suffix sums
vector<BigInt> suf;

// Helper for subtraction (returns |A-B|)
BigInt abs_diff(const BigInt& a, const BigInt& b) {
    BigInt res;
    if (a > b) {
        res = a.clone();
        res.sub(b);
    } else {
        res = b.clone();
        res.sub(a);
    }
    return res;
}

void dfs(int idx, BigInt cur_sum) {
    if (exact_found) return;
    
    // Optimization: Upper bound check
    // suf[idx] is sum of items[idx]...items[n-1]
    // But wait, suf should be computed based on sorted items.
    
    // Calculate upper bound
    // If idx == n, suffix sum is 0.
    BigInt rem_sum = (idx < n) ? suf[idx] : BigInt("0");
    BigInt ub = cur_sum + rem_sum;
    
    // Case: Already over
    if (cur_sum > W) {
        BigInt d = abs_diff(cur_sum, W);
        if (d < min_diff) {
            min_diff = d;
            best_sol = curr_sel;
        }
        return; 
    }
    
    // Case: Even with all remaining, cannot reach W
    if (ub < W) {
        // Must take all
        for (int k = idx; k < n; ++k) {
            curr_sel[items[k].second] = true;
        }
        BigInt d = abs_diff(W, ub);
        if (d < min_diff) {
            min_diff = d;
            best_sol = curr_sel;
        }
        // Backtrack
        for (int k = idx; k < n; ++k) {
            curr_sel[items[k].second] = false;
        }
        return;
    }
    
    // Case: Match exact
    if (cur_sum == W) {
        exact_found = true;
        best_sol = curr_sel; // Record solution
        return;
    }
    
    // Branch 1: Include
    curr_sel[items[idx].second] = true;
    dfs(idx + 1, cur_sum + items[idx].first);
    if (exact_found) return; // Propagate stop
    
    curr_sel[items[idx].second] = false;
    
    // Branch 2: Exclude
    dfs(idx + 1, cur_sum);
}
```
Wait, one bug in `dfs` logic.
If `exact_found` is set, we stop.
However, we need to ensure `best_sol` is fully populated.
Since `curr_sel` is global and we backtrack, if we found exact, we set `curr_sel` before returning.
`best_sol = curr_sel` copies the vector.
So it's correct.
The initial call should pass 0 and 0.
Initialization of `min_diff` should be `W` (case sum=0).
Initialize `curr_sel` to all false.

Wait, `items[k].second` is original index.
Items are sorted, so `idx` refers to sorted array.
`curr_sel` is mapped to original indices.
Correct.

Wait, `rem_sum` logic.
`suf` vector should be accessible.
Compute it before `dfs`.
`suf` size `n+1`. `suf[n]=0`.
Loop `i` from `n-1` down to `0`. `suf[i] = suf[i+1] + items[i].first`.

One more thing: `suf[idx]` access.
If `idx == n`, access `suf[n]`.
Check `idx < n`. Handled.

Complexity:
BigInt copy is small.
Vector assignment is fast.
Seems good.

Final Check: `items[k].second`
`items` stores `{val, original_index}`.
`curr_sel` is `vector<bool>(n)`.
When `items` sorted, `items[k].second` is unique $0 \dots n-1$.
Yes.

```cpp
/*
Solution for Subset Sum
*/
```
I'll put everything together.
Note: Need to handle `BigInt` header includes. `<vector>`, `<string>`, `<algorithm>`, `<iostream>`.

Double check `operator-` logic.
My `sub` method modifies `*this`. `abs_diff` clones and calls `sub`. Correct.

One detail: `stoll` might fail if string > 18 digits?
`min(9, ...)` limits to 9 digits. `stoll` takes `long long` (up to 19 digits).
So `stoll` is safe for chunks of 9.
`BASE` is $10^9$.
Digits are pushed to vector.
Construction logic:
`i` starts 0. `chunk_len` 9.
`s.substr(i, 9)`.
Puts into `digits` (LSB first).
Example: `123456789012`.
`s[0..8]` = `123456789` (last digit is `9`).
Wait, string `s` is written left to right. `s[0]` is MSD.
My loop:
`stoll(s.substr(...))`
`123` -> LSB?
If string is "123", `substr(0, 3)` is "123". `stoll` -> 123. Pushed to `digits`.
Vector: `[123]`.
Represents value 123. Correct.
If "123456789012".
1. `substr(0, 9)` -> "123456789". `val` = 123456789. Pushed.
2. `substr(9, 3)` -> "012". `val` = 12. Pushed.
Vector: `[123456789, 12]`.
Value = $123456789 \cdot BASE + 12$.
This is correct for Big Endian storage in vector?
Wait, standard BigInt usually stores LSB at index 0.
Here `push_back` adds to end.
Index 0 is pushed LAST in my loop?
No, `i` goes 0 to len.
First iteration: `i=0`. Pushes chunk from start (MSD).
This chunk ends up at `digits[0]` (smallest index).
Second iteration: `i=9`. Pushes chunk from end.
This chunk ends up at `digits[1]`.
This means `digits[0]` is the MOST SIGNIFICANT part?
If `digits` is printed `digits[0]` then `digits[1]` (separated by commas), it looks reversed?
Let's check addition logic.
Loop `i` from 0. Adds `digits[i]` and `o.digits[i]`.
If `digits[0]` is MSD, then addition must align MSDs.
Standard BigInt addition processes from LSB (least significant) to carry propagation towards MSD.
Here `digits[0]` holds the highest order chunk.
Wait.
String "123". `digits`=[123].
String "12". `digits`=[12].
`digits` size 1 vs 1.
Comparison:
Size check. Then compare `digits[i]` from `size-1` downwards.
If size equal, `digits[1]` is MSD. `digits[0]` is LSD?
If I push chunk 1 then chunk 2 (higher significance).
Chunk 1 (from index 0) is high significance.
Chunk 2 (from index 9) is low significance.
Wait.
String "123456789012".
Chunk 1: "123456789".
Chunk 2: "012".
"123456789" is $10^9$ place. "012" is unit place.
So `digits[0]` should be the unit place?
My loop: `i=0` (starts at front of string).
Pushes "123456789". This is the high part.
So `digits[0]` is high part.
This is WRONG for standard LSB-at-0 storage.
My `sub` logic iterates `i` from 0 to `size-1`.
If `digits[0]` is MSD, then `sub` borrows from LSD (at end).
Borrow propagates from LSD to MSD.
So I should iterate from `digits.size()-1` down to 0.
Similarly `add`: carry propagates from LSD (index 0?) to MSD.
So if `digits[0]` is MSD, carry should go from `size-1`? No, carry comes from low to high.
So LSB should be at index 0.
Let's fix construction.
Reverse string first? Or parse from end.
Correct way:
Parse from end of string (LSB).
`for (int i = s.length() - 1; i >= 0; i -= 9)`
Extract 9 chars starting from `i`.
This puts lowest chunk at `digits[0]`.
Then `digits` is LSB first.
Comparison loop: `for (int i = digits.size() - 1; ...)` -> Checks MSD first.
Wait, if LSB is at 0, MSD is at `size-1`.
Comparison loop `i` from `size-1` down to 0 is correct.
Subtraction loop:
Borrow flows from LSD (0) to MSD?
No, in manual subtraction:
Start at LSB (rightmost digit). Borrow propagates left.
So loop from 0 upwards.
If `digits` is LSB at 0, loop `i=0..size-1` is correct.
So I should fix BigInt constructor to parse LSB first.

Fixing BigInt Construction:
```cpp
    BigInt(string s) {
        digits.clear();
        if (s.empty()) { digits.push_back(0); return; }
        // Parse from right to left
        for (int i = (int)s.length() - 1; i >= 0; ) {
            int l = min(9, (int)(i + 1)); // length of current chunk
            // Extract substring ending at i
            // Wait, easier: take chunk of length 9 from end
            int start_idx = i - 8;
            if (start_idx < 0) start_idx = 0;
            // But this is messy with loop variable
            // Better: just substr from pos
        }
    }
```
Actually, just:
```cpp
    BigInt(string s) {
        digits.clear();
        int len = s.length();
        for (int i = len; i > 0; i -= 9) {
            int start = max(0, i - 9);
            string seg = s.substr(start, i - start);
            unsigned int val = stoll(seg);
            digits.push_back(val);
        }
        // Trim leading zeros?
        // Wait, this fills LSB first?
        // Example: s="12345". len=5.
        // i=5. start=0. seg="12345". val=12345. digits={12345}.
        // i=-4. stop.
        // Result {12345}.
        // This treats "12345" as single chunk. Correct.
        // Example: s="123456789012". len=12.
        // i=12. start=3. seg="456789012". val=456789012. digits={456789012}.
        // i=3. start=0. seg="123". val=123. digits={456789012, 123}.
        // Vector: {High, Low}.
        // So this is MSB first in vector.
    }
```
To get LSB first:
Loop `i` from 0 upwards.
Parse from start.
Wait, LSB is rightmost.
So `s[len-1]` is LSD.
If I parse from `len` down to 0.
First iteration `i=len`. Chunk `s[len-9..len-1]`.
This chunk corresponds to units place.
Next iteration `i=len-9`. Chunk `s[len-18..len-10]`.
Corresponds to $10^9$ place.
So pushing this way gives LSB at `digits[0]`.
Let's trace: `s="123456789012"` (High..Low).
Len=12.
1. `i=12`. Start `12-9=3`. `s[3..11]` = "456789012". (Last 9 digits).
Wait, `substr(3, 9)` gives "456789012".
Is "456789012" the LSD chunk?
Original string: `123` `456789012`.
Yes.
So first push is LSD.
2. `i=3`. Start `0`. `s[0..2]` = "123".
Second push is Next chunk.
Result `digits` = {LSD, ...}.
This is correct LSB-at-0.

So the construction logic:
```cpp
    BigInt(string s) {
        digits.clear();
        int len = s.length();
        for (int i = len; i > 0; i -= 9) {
            int start = max(0, i - 9);
            string seg = s.substr(start, i - start);
            unsigned int val = stoll(seg);
            digits.push_back(val);
        }
        // Remove leading zeros (trailing in vector)
        while (digits.size() > 1 && digits.back() == 0) digits.pop_back();
        if (digits.empty()) digits.push_back(0);
    }
```
Wait, `seg` logic:
`s` indices: `0` to `len-1`.
Iteration `i` (representing count of characters taken).
Take last `min(9, len)` chars.
Wait, if I take `i=12`, `start=3`. `seg`=`s[3..11]`.
These are the characters from index 3 to 11.
In "123456789012" (12 chars).
Indices 0,1,2 ("123"). Indices 3..11 ("456789012").
So `digits[0]` = "456789012" (Low).
`digits[1]` = "123" (High).
This means `digits` stores LSB at index 0.
Comparison loop:
Check size.
Check `digits[size-1]` (MSD) vs `other[size-1]`.
My `operator>` loop: `for (int i = digits.size() - 1; ...`
Accesses `digits[size-1]`. Correct.
Addition loop:
Sum `digits[i]` and `others.digits[i]`. `carry` propagates `digits[i]` to `digits[i+1]`.
Since `i` goes `0` upwards, this matches LSB->MSD flow.
Correct.

Logic verification:
`s = "10"`. Len 2.
i=2. start=0. seg="10". val=10. digits={10}.
i=-7. Stop.
Val 10.

Wait, `stoll` parses "10" as 10.
If `s="100...0"`.
i=9. start=0. seg="100...0" (9 zeros). val = 10^9? No `stoll` handles it.
Wait, if I split at $10^9$ boundary.
If `s="1" + 9 "0"s`. Len 10.
i=10. start=1. seg="000000000". val=0. digits={0}.
i=1. start=0. seg="1". val=1. digits={0, 1}.
This means number is $1 \cdot 10^9 + 0$.
Wait, `stoll` of "000000000" is 0.
So digits={0, 1}.
Wait, `digits[0]=0`, `digits[1]=1`.
This represents $1 \cdot 10^9$.
Wait, `s` was "1000000000".
If I treat `digits[0]` as units. `digits[0]=0` means units 0. `digits[1]=1` means $10^9$?
No. `digits[0]` holds value of `s` chars from index `len-9` to `len-1`.
In "1000000000". `len=10`.
Chars 1..9 are "000000000". Value 0.
Chars 0..0 are "1". Value 1.
So `digits[0]` (Units) = 0.
`digits[1]` ($10^9$) = 1.
Correct.

Wait, `substr` takes `(pos, len)`.
`i=10`. `start=1`. `seg = s.substr(1, 10-1)`.
`s` is "1000000000".
Indices: 0='1', 1='0'... 9='0'.
`s.substr(1, 9)` = "000000000".
Correct.

My construction code:
```cpp
        for (int i = len; i > 0; i -= 9) {
            int start = max(0, i - 9);
            string seg = s.substr(start, i - start);
            digits.push_back(stoll(seg));
        }
```
Wait, `s.substr(start, length)`.
If `i=10`, `start=1`, `len=9`. Correct.
If `i=1`, `start=0`, `len=1`. Correct.

One final check on `stoll`.
If `s` contains `0`s and `stoll` works fine.
Leading zeros in substring are fine.

Okay, ready to write.

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>

using namespace std;

// ... (BigInt struct) ...

// ... (globals) ...

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int n_int;
    string w_str;
    if (!(cin >> n_int >> w_str)) return 0;
    n = n_int;
    W = BigInt(w_str);

    items.reserve(n);
    for (int i = 0; i < n; ++i) {
        string a_str;
        cin >> a_str;
        BigInt a = BigInt(a_str);
        items.emplace_back(a, i);
    }

    // Sort items descending
    sort(items.begin(), items.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });

    // Compute suffix sums
    suf.assign(n + 1, BigInt("0"));
    for (int i = n - 1; i >= 0; --i) {
        suf[i] = suf[i + 1] + items[i].first;
    }

    // Initialize best
    min_diff = W.clone(); // Diff for sum=0 is |0-W| = W
    curr_sel.assign(n, false);
    best_sol.assign(n, false); // Default 0s

    dfs(0, BigInt("0"));

    // Output result
    for (int i = 0; i < n; ++i) {
        cout << best_sol[items[i].second];
        if (i < n - 1) cout << " ";
    }
    cout << "\n";

    return 0;
}
```
Wait, output loop needs to print `b_1 ... b_n` corresponding to original indices `1..n`.
In `main`, `items` is sorted.
But output requirement: `b_1` corresponds to `a_1`.
`best_sol` is indexed by original index (stored in `items[k].second`).
However, printing loop iterates `0` to `n-1`.
Inside loop: `cout << best_sol[items[i].second]`?
No. We need `b[original_index]`.
We have `best_sol` array indexed by `0..n-1` (original).
So just print `best_sol[0]`, `best_sol[1]`, ...
Wait, `items` stores mapping.
Wait, if `best_sol` is `vector<bool>(n)`, then `best_sol[0]` is result for `a_1`.
My DFS sets `curr_sel[items[idx].second]`. `items[idx].second` is original index.
So `best_sol` should hold the correct values.
Just iterate `i` from 0 to `n-1` and print `best_sol[i]`.
Wait, `best_sol` vector size is `n`.
`curr_sel` is size `n`.
Indices `0` to `n-1`.
Correct.
Wait, `items` vector contains `(val, original_index)`.
So `items[k].second` is the original index.
DFS sets `curr_sel[items[k].second]`.
So `best_sol` (copy of `curr_sel`) will have flags at correct positions.
Printing `best_sol[0]`, `best_sol[1]`, ... is correct.

Wait, I used `best_sol[items[i].second]` in my thought loop above.
No, `items` loop in `main` prints `b` for `a`.
So:
```cpp
    for (int i = 0; i < n; ++i) {
        // We want to know if a[i] was selected.
        // a[i] corresponds to item at index 'i' in original list.
        // Its original index is 'i'.
        // So we access best_sol[i].
        // Wait, items[i].second IS i? No, items are sorted.
        // But best_sol is indexed by original ID.
        // So best_sol[0] is for a_0, best_sol[1] for a_1.
        // Correct.
        cout << best_sol[i];
        if (i < n - 1) cout << " ";
    }
```
Yes. `best_sol` is indexed by original index. `items` is used for traversal logic.
So I don't need `items[i].second` in the print loop.
Wait, `items` is used to identify which original index to set in `curr_sel`.
Yes.
So print `best_sol[i]`.
Wait, if `n` is input variable, `best_sol` is size `n`.
Wait, `curr_sel` size is `n`.
Yes.

One edge case: `min_diff` initialization.
If `W` is 0.
Init `min_diff` = 0. `exact_found` logic handles.
But `min_diff` init should be `abs(0 - W)`.
My init `min_diff = W` (assuming `W >= 0`).
Yes.

Double check `best_sol` init.
If `dfs` never runs or finds better (only empty set), `best_sol` should be all 0.
My `dfs` will visit `dfs(0, 0)`.
`current_sum = 0`. `upper = 0`. `0 < W` (if W>0).
It will trigger Case 2 (force take all).
Wait, if `W` is large, `upper < W`. It forces take all.
So `best_sol` will be updated.
If `W=0`.
`cur=0`. `cur==W`. `exact_found` true.
Does it run `dfs`?
`dfs(0, 0)` called.
Check `cur==W`. True.
Sets `found_exact` true.
Returns.
Does it update `best_sol`?
I put `best_sol = curr_sel` inside exact block.
`curr_sel` is all false initially.
So `best_sol` is all false. Correct.

One edge case: `suf` vector.
`suf` is vector of BigInts.
Constructor `BigInt("0")`.
Works.

Final Code Assembly.

```cpp
// Includes
// BigInt struct
// Globals
// Functions
// Main
```
</think>

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>

using namespace std;

// BigInt structure to handle numbers up to 10^1100
struct BigInt {
    vector<unsigned int> digits;
    static const unsigned int BASE = 1000000000;

    BigInt() {}

    BigInt(unsigned int n) {
        if (n == 0) digits.push_back(0);
        else {
            while (n > 0) {
                digits.push_back(n % BASE);
                n /= BASE;
            }
        }
    }

    BigInt(string s) {
        digits.clear();
        if (s.empty()) {
            digits.push_back(0);
            return;
        }
        int len = (int)s.length();
        // Parse from right to left (LSB first)
        for (int i = len; i > 0; i -= 9) {
            int start = max(0, i - 9);
            string seg = s.substr(start, i - start);
            digits.push_back(stoll(seg));
        }
        while (digits.size() > 1 && digits.back() == 0) digits.pop_back();
        if (digits.empty()) digits.push_back(0);
    }

    bool operator>(const BigInt& o) const {
        if (digits.size() != o.digits.size()) return digits.size() > o.digits.size();
        for (int i = digits.size() - 1; i >= 0; --i)
            if (digits[i] != o.digits[i]) return digits[i] > o.digits[i];
        return false;
    }

    bool operator<(const BigInt& o) const {
        if (digits.size() != o.digits.size()) return digits.size() < o.digits.size();
        for (int i = digits.size() - 1; i >= 0; --i)
            if (digits[i] != o.digits[i]) return digits[i] < o.digits[i];
        return false;
    }

    bool operator==(const BigInt& o) const {
        return digits == o.digits;
    }

    BigInt operator+(const BigInt& o) const {
        BigInt r;
        int c = 0;
        size_t m = max(digits.size(), o.digits.size());
        for (size_t i = 0; i < m || c; ++i) {
            unsigned int d = c;
            if (i < digits.size()) d += digits[i];
            if (i < o.digits.size()) d += o.digits[i];
            r.digits.push_back(d % BASE);
            c = d / BASE;
        }
        return r;
    }

    void sub(const BigInt& o) {
        int c = 0;
        for (size_t i = 0; i < o.digits.size() || c; ++i) {
            unsigned int d = c;
            if (i < digits.size()) d += digits[i];
            else d = 0;
            if (i < o.digits.size()) d -= o.digits[i];
            if (d < 0) { d += BASE; c = 1; }
            else c = 0;
            digits[i] = d;
        }
        while (digits.size() > 1 && digits.back() == 0) digits.pop_back();
        if (digits.empty()) digits.push_back(0);
    }

    BigInt clone() const { return *this; }
};

// Global variables
int n;
BigInt W;
vector<pair<BigInt, int>> items;
vector<bool> best_sol;
vector<bool> curr_sel;
BigInt min_diff;
bool exact_found = false;
vector<BigInt> suf;

// Helper for subtraction (returns |A-B|)
BigInt abs_diff(const BigInt& a, const BigInt& b) {
    BigInt res;
    if (a > b) {
        res = a.clone();
        res.sub(b);
    } else {
        res = b.clone();
        res.sub(a);
    }
    return res;
}

// Depth First Search with Pruning
void dfs(int idx, BigInt cur_sum) {
    if (exact_found) return;

    // Calculate upper bound of reachable sum from this state
    // suf[idx] is the sum of all items from idx to end
    BigInt rem_sum = (idx < n) ? suf[idx] : BigInt("0");
    BigInt ub = cur_sum + rem_sum;

    // Case 1: Current sum > W. Adding more positive numbers will only increase sum further.
    // Thus, the current state is the best we can do on this path.
    if (cur_sum > W) {
        BigInt d = abs_diff(cur_sum, W);
        if (d < min_diff) {
            min_diff = d;
            best_sol = curr_sel;
        }
        return;
    }

    // Case 2: Even adding all remaining items, we cannot reach W (Upper bound < W).
    // To minimize difference (W - Sum), we should take the maximum possible sum.
    // So we must take all remaining items.
    if (ub < W) {
        for (int k = idx; k < n; ++k) {
            curr_sel[items[k].second] = true;
        }
        BigInt d = abs_diff(W, ub);
        if (d < min_diff) {
            min_diff = d;
            best_sol = curr_sel;
        }
        // Backtrack: unset the items we just set
        for (int k = idx; k < n; ++k) {
            curr_sel[items[k].second] = false;
        }
        return;
    }

    // Case 3: Exact match found
    if (cur_sum == W) {
        exact_found = true;
        // Record solution (though it's already implicit in curr_sel logic below if we weren't cutting short)
        best_sol = curr_sel;
        return;
    }

    // Branch 1: Include items[idx]
    curr_sel[items[idx].second] = true;
    dfs(idx + 1, cur_sum + items[idx].first);
    
    // If exact match found in child, return immediately
    if (exact_found) return;

    // Backtrack
    curr_sel[items[idx].second] = false;

    // Branch 2: Exclude items[idx]
    dfs(idx + 1, cur_sum);
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    string w_str;
    if (!(cin >> n >> w_str)) return 0;
    W = BigInt(w_str);

    items.reserve(n);
    for (int i = 0; i < n; ++i) {
        string a_str;
        cin >> a_str;
        BigInt a = BigInt(a_str);
        items.emplace_back(a, i);
    }

    // Sort items descending to optimize pruning
    sort(items.begin(), items.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });

    // Precompute suffix sums
    suf.assign(n + 1, BigInt("0"));
    for (int i = n - 1; i >= 0; --i) {
        suf[i] = suf[i + 1] + items[i].first;
    }

    // Initialize best solution with empty set
    // Initial sum = 0. Diff = |0 - W| = W
    min_diff = W.clone(); // Assuming W >= 0 as per problem
    curr_sel.assign(n, false);
    best_sol.assign(n, false);

    // Run DFS
    dfs(0, BigInt("0"));

    // Output result
    for (int i = 0; i < n; ++i) {
        cout << best_sol[i];
        if (i < n - 1) cout << " ";
    }
    cout << "\n";

    return 0;
}
```
````

---

## 直接SFT 样本(sample_idx=2,得分 0.0,completion_tokens=1838)

````
Let me analyze this problem carefully.

The key observations are:

1. We need to select a subset whose sum is as close to W as possible
2. The numbers can be up to 10^1100, so we definitely need big integer arithmetic
3. n is at most 2100 (about 21 elements) - small enough for exponential search!
4. Score = max(0, 1 - |W-S|/(W+M)) means the closer S is to W, the better

Since only ~20-21 elements, the brute force would be 2^21 = about 2 million combinations, which should be fine with efficient big integer handling. However, since each number is very large and there are only ~20 numbers, many combinations will produce the same or near-equal sums due to rounding. A greedy approach might work well here - sort by value and pick greedily. Let me try the sorted greedy approach first as it's much more efficient than checking all subsets.
</think>

```cpp
#include <bits/stdc++.h>
using namespace std;

// Custom Big Integer class supporting addition, subtraction, multiplication, comparison
class BigInt {
private:
    vector<long long> a; // stores digits in reverse order
    
public:
    BigInt() {}
    BigInt(long long v) {
        if (v == 0) a.push_back(0);
        else while (v > 0) { a.push_back(v % BASE); v /= BASE; }
    }
    
    static const int BASE = 9007199254740992; // 2^53
    
    void add(const BigInt& other) {
        size_t n = max(a.size(), other.a.size());
        a.resize(n, 0);
        for (size_t i = 0; i < n || carry; i++) {
            long long r = carry + (i < a.size() ? a[i] : 0) + (i < other.a.size() ? other.a[i] : 0);
            a[i] = r / BASE; carry = r % BASE;
        }
        while (!a.empty() && a.back() == 0) a.pop_back();
        if (a.empty()) a.push_back(0);
    }
    
    void sub(const BigInt& other) {
        size_t n = max(a.size(), other.a.size());
        size_t carry = 0;
        for (size_t i = 0; i < n; i++) {
            long long r = carry + (long long)(i < a.size() ? a[i] : 0) - (long long)(i < other.a.size() ? other.a[i] : 0);
            carry = -(r >= 0); a[i] = r;
        }
        while (!a.empty() && a.back() == 0) a.pop_back();
        if (a.empty()) a.push_back(0);
    }
    
    bool greaterThan(const BigInt& other) const {
        return a.size() > other.a.size() || 
               (a.size() == other.a.size() && 
                lexicographicalCompareGreater(a, other.a));
    }
    
    bool equals(const BigInt& other) const {
        return a.size() == other.a.size() && 
               a == other.a;
    }
    
    BigInt operator+(const BigInt& other) const {
        BigInt result = *this; result.add(other); return result;
    }
    
    BigInt operator-(const BigInt& other) const {
        BigInt result = *this; result.sub(other); return result;
    }
    
    // Convert to string for output
    string toString() const {
        stringstream ss;
        for (int i = a.size() - 1; i >= 0; i--)
            ss << a[i];
        return ss.str();
    }
};

bool lexicographicalCompareGreater(const vector<long long>& a, const vector<long long>& b) {
    for (size_t i = 0; i < min(a.size(), b.size()); i++)
        if (a[i] != b[i]) return a[i] > b[i];
    return a.size() > b.size();
}

BigInt abs_val(const BigInt& x) {
    return x.greaterThan(BigInt(0)) ? x : (-x);
}

void solve() {
    int n;
    string W_str; cin >> n >> W_str;
    vector<string> s(n);
    for (int i = 0; i < n; i++) cin >> s[i];
    
    vector<BigInt> a(n);
    BigInt W(W_str);
    int idx = 0, maxIdx = -1;
    BigInt maxVal;
    for (int i = 0; i < n; i++) {
        a[i] = BigInt(s[i]);
        if (maxIdx == -1 || a[i].greaterThan(maxVal)) {
            maxIdx = i; maxVal = a[i];
        }
    }
    
    string resultW = W.toString();
    vector<int> b(n); fill(b.begin(), b.end(), 0);
    
    for (int start = 0; start < n; ) {
        string targetStr = resultW;
        int end = start + 28; if (end > n) end = n;
        
        vector<vector<BigInt>> dp(end - start + 1, vector<BigInt>(W.toDigits() + 200, ZERO_VAL()));
        
        for (int i = 0; i <= end - start; i++)
            for (int j = 0; j <= W.toDigits() + 200; j++)
                dp[i][j] = ZERO_VAL();
        
        for (int i = start; i <= end; i++) {
            dp[i][W.toDigits()] = ONE_VAL();
        }
        
        int midStart = (start + end) / 2 + 1;
        
        for (int i = start; i < midStart; i++) {
            for (int j = 0; j <= W.toDigits(); j++) {
                if (dp[i][j] > ZERO_VAL())
                    BigInt d = a[i]; j -= d.toDigits();
                if (j >= 0 && dp[i + 1][j] < dp[i][j]) {
                    dp[i + 1][j] = dp[i][j]; b[j] = i - start;
                }
                if (j < W.toDigits()) {
                    BigInt c = ZERO_VAL(); a[i] += c; b[j + 1] = i - start;
                }
            }
        }
        
        string bestS = "0"; int bestScore = -1;
        
        for (int j = 0; j <= W.toDigits() + 200; j++) {
            BigInt val = dp[end][j]; if (val <= ZERO_VAL()) continue;
            
            BigInt scoreDenom = W + maxVal;
            BigInt absDiff = abs_val(dp[start][j].subtract(W));
            
            string score = "";
            for (size_t k = 0; k < absDiff.size(); k++) {
                char c = '0' + absDiff.a[k] + ((j < absDiff.size() ? absDiff.a[k] : 0) - (absDiff.a[k] + ((j < absDiff.size() ? absDiff.a[k] : 0) < 10)));
                if (c >= '0' && c <= '9') score += c;
            }
            
            if (score.length() > bestScore || (score.length() == bestScore && score.compare(resultW.substr(0, bestScore)) < 0)) {
                bestScore = score.length();
                bestS = score;
            }
        }
        
        resultW = bestS;
        idx = end; start = end;
    }
    
    cout << maxVal.toString().length();
}

int main() {
    cin.tie(NULL); ios_base::sync_with_stdio(false);
    cout << fixed << setprecision(6);
    solve();
    return 0;
}
```
````
