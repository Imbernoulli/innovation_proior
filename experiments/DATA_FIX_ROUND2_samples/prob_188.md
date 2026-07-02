# 题 188 — base均分 4.1 vs 直接SFT均分 0.0(掉点 4.1)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=0,得分 5.2,completion_tokens=25057)

````
The user wants a solution to the "LCS Challenge" problem.
I need to find a long common subsequence $Z$ of two strings $S1$ and $S2$.
The lengths $N, M \le 30,000,000$. However, the constraints section later says $N, M \le 10,000,000$. Let's assume up to $10^7$.
An exact LCS algorithm takes $O(NM)$ time, which is too slow ($10^{14}$ ops). We have 2 seconds.
This is a heuristic approximation problem. The score depends on how close we get to the optimal LCS length.

Common strategies for approximate LCS on large strings include:
1.  **Greedy matching:** Match the first character of S2 with the first occurrence in S1, etc. This finds *a* subsequence, not necessarily the longest. Often called the "greedy alignment" or just matching greedily.
2.  **Bitap / SIMD:** Fast bitset operations if alphabet is small. Here alphabet size $\Sigma = 36$ ('A'-'Z', '0'-'9'). Bitset approach might be feasible if implemented efficiently (e.g., using `std::bitset` or manual unrolling), but memory might be tight for $10^7$.
3.  **Two-pointers with heuristics:** If one string is sorted? No, order matters.
4.  **Randomized shuffling:** Sometimes helps avoid worst-case scenarios where greedy fails badly.
5.  **Divide and Conquer + Greedy:** Or specific algorithms designed for this.

Since it's an approximation challenge ("heuristic optimization"), usually a good greedy strategy works well enough, especially if randomized or if we prioritize frequent characters.
However, standard greedy (left-most match) often yields much worse than optimal LCS in adversarial cases, though it performs reasonably well on random data.
Given the constraints (2s, 512MB) and massive strings ($10^7$), I cannot allocate an $O(N)$ boolean array easily for every character. But wait, $10^7$ bits is only ~1.2MB. $O(N)$ space is fine.
Actually, $O(N+M)$ space is required anyway to store input.

Let's consider the standard $O((N+M)\Sigma)$ approach where we compute next occurrences.
For each position $i$ in $S1$, we want to know the index of the next occurrence of character $c$ at or after $i$. Precomputing this table takes $O(N\Sigma)$, which is $10^7 \times 36 \approx 3.6 \times 10^8$ integers/bytes. That fits in memory (maybe ~1GB if int, but ~144MB if char/short). Since limit is 512MB, precomputing the full "next occurrence" table for all characters might be borderline or possible depending on implementation (using `uint16_t` since indices fit in 32-bit but might overflow if $N > 65535$... wait $N=10^7$, so need 32-bit ints or compressed). 32-bit ints = 144MB. So `next_occurrence[36][10^7]` using `int32_t` is 144MB. This is safe.
With this table, finding the LCS via dynamic programming logic becomes faster, specifically the Hirschberg-Smithe-Larsen or similar linear-time greedy approaches.
Wait, knowing next occurrence allows us to run a DP that transitions in $O(1)$.
Specifically, let $dp[i]$ be the longest suffix of $S1[i..N-1]$ that matches a prefix of $S2[0..M-1]$? No.
Standard bitset optimization for LCS is $O(MN/W)$ where $W$ is word size. $10^7 \times 10^7$ bits is huge ($10^{14}$ bits $\approx 12$ TB). Too big.
We need something linear-ish in $N$ or $M$.

Wait, this problem description looks very similar to "Greedy Common Subsequence" or just finding *some* valid common subsequence, but trying to maximize length.
If we simply perform a greedy match from the beginning:
Pointer $i$ in S1, pointer $j$ in S2.
Find first occurrence of $S2[j]$ in $S1$ at or after $i$. Append $S2[j]$ to Z, update $i$. Repeat.
This is $O((N+M)\alpha)$ where $\alpha$ is search cost. With precomputed next occurrence arrays, this search is $O(1)$ amortized. Total time $O(N+M)$. Space $O(N\Sigma)$ for the table.
Is this the best? For random strings, yes. For worst case (like $S1=ABCD..., S2=DCBA...$), yes it's optimal if they share chars. But for things like $S1=A^M, S2=B^M$ (no overlap) it returns empty. If they are permutations, it might be bad.

Better heuristic:
Instead of fixing start positions, iterate through both strings?
There is a known result that the simple greedy strategy for LCS (match $S2[j]$ to the first occurrence in $S1$ after current pos) produces a subsequence of length roughly related to the optimal. It can be significantly shorter than optimal on some cases.

Another heuristic often used in competitive programming for LCS approximations when $N, M$ are large is:
1.  Randomize the order of processing?
2.  Use the fact that common characters should be prioritized.

Actually, there is a specific algorithm for large scale LCS approximation called "Linear LCS Approximation".
One effective simple approach for this specific problem context (heuristic optimization):
Precompute the positions of each character in S1 and S2.
To construct Z:
Iterate through characters of S2. For each character $c$ in S2, try to add it to Z such that it appears after the last added character in S1.
If multiple occurrences exist, pick the earliest valid one (standard greedy).
Why might this fail? Suppose S1 = "AAAA", S2 = "AA". Greedy gives "AA".
Suppose S1 = "ABCD", S2 = "ACBD". Greedy picks A (idx 0), C (idx 2). Then B in S2 needs to be after idx 2 in S1, but B is at idx 1. Fail. Result "AC". Optimal is "ABD" (len 3).
So left-to-right greedy on one string is risky.

We want to maximize $|Z|$.
Maybe we can use a greedy approach on both directions or a sampling approach?
Given the constraint $N, M \le 10^7$ and 2s, we must process everything quickly.
Reading the input takes time.
Using fast I/O is mandatory.

Let's look at the available strategies again.
Strategy 1: Greedy from $S1$'s perspective against $S2$ (pick $S1[i]$ if present in remaining $S2$).
Strategy 2: Greedy from $S2$'s perspective against $S1$.
Taking the maximum length of these two might help.

Strategy 3: Randomized permutation of characters. Shuffle $S2$ conceptually? No, relative order must be preserved.
Shuffle $S2$ means picking subsequences differently.
Actually, one very strong heuristic for large strings without DP is:
**Forward Greedy + Backward Greedy.**
Or even better:
Compute the LCS by scanning S1 and S2.
We can optimize the scan.
Map characters of S1 to their list of indices.
Store `vector<int> occ[S1]`.
For S2, for each char `c`, iterate through its list in S1. Pick the first one greater than `last_idx`.
This is exactly Strategy 1.
To improve, we can try starting from different "states". But state is just an integer (index in S1).

What if we treat the characters as nodes in a graph?
No, too complex.

Let's check the constraints again. Time limit 2.0s. $10^7$ operations per second is conservative. $10^8$ is likely safe in C++.
Input reading $10^7$ chars using `cin` might be TLE. Need `scanf` or custom parser.
`ios_base::sync_with_stdio(false); cin.tie(NULL);` is needed.

Is there a randomized greedy that works well?
Sort $S2$ based on frequency in $S1$? No, order matters.
What about the following approach:
Iterate $i$ from 0 to $N-1$ (indices of S1). If $S1[i]$ is present in $S2$ at or after $current\_pos\_in\_S2$, match them.
But $S2$ is fixed order. We can't iterate $S1$ freely unless we skip.
Wait, the classic Greedy LCS is:
Let $P_S2$ be the list of indices for each char in $S2$.
Wait, no.
Greedy Algorithm (Forward):
$i = 0, j = 0$
While $j < M$:
   Find smallest $k \ge i$ such that $S1[k] == S2[j]$.
   If found: append $S2[j]$, $i = k+1$.
Else return.

Time complexity: Finding $k$ requires searching S1. If we use precomputed Next_Occurrence table, finding $k$ is $O(1)$, but we consume the index.
Actually, if we don't backtrack, total work is proportional to number of matches + skips.
Matches = $|Z|$. Skips $\le N$.
Total time $O(N+|Z|)$. With precomputation $O(N\Sigma)$ setup.
As discussed, this is efficient.
Does it maximize $|Z|$? Not always.
However, often for this type of problem on judges (where you get partial scores), getting a reasonable length is better than nothing.
Can we do better?
There is a randomized strategy. Instead of taking the first occurrence in $S1$, take the $(p\%K)$-th occurrence?
Or iterate S2 randomly permuted characters?
Wait, if I just output the output of "Forward Greedy" (match S2 elements to first available in S1), is it sufficient?
Example where it fails badly:
S1: A B A B A B ...
S2: B A B A B A ...
Greedy S2->S1:
Match 'B': first B in S1 is idx 1.
Match 'A': first A after 1 is idx 2.
Match 'B': first B after 2 is idx 3.
It matches perfectly.

Counter example:
S1: A C B D E F G H
S2: A B C D
Greedy S2->S1:
'A': 0.
'B': After 0? Index 2.
'C': After 2? None (C is at 1, before B).
Result: "AB". Length 2.
Optimal: "ACBD"? No, S1 has C at 1.
S1: A C B D ...
S2: A B C D.
Possible CS: A (0), B (2). C (1) no. C (none after 2).
Actual LCS: A, C, B, D -> S1(A at 0, C at 1, B at 2, D at 3) -> Order in S1: A(0), C(1), B(2), D(3).
Order in S2: A(0), B(1), C(2), D(3).
Intersection: A, C, D?
S1: A(0), C(1), D(3). S2: A(0), C(2), D(3). Valid "ACD". Length 3.
S1: A(0), B(2), D(3). S2: A(0), B(1), D(3). Valid "ABD". Length 3.
Wait, my previous trace was flawed.
In S1: A C B D. Indices 0 1 2 3.
In S2: A B C D.
Subseq S1: A(0), C(1), D(3).
Subseq S2: A(0), C(2), D(3).
Valid CS: "ACD".
My greedy trace: S2='A' -> S1=0. S2='B' -> S1=2. S2='C' -> S1 search after 2 -> Fail. Result "AB". Length 2.
Optimal 3.
So simple greedy is definitely suboptimal here.

However, I can swap roles.
Run greedy S1->S2.
Match 'A' in S2 first?
S1: A C B D.
S2: A B C D.
Try greedy S1->S2.
Pick S1 char 'A'. Is 'A' in S2 after current? Yes.
Add 'A'. Last S2 pos = 0.
Next S1 char 'C'. Is 'C' in S2 after 0? Yes, at 2.
Add 'C'. Last S2 pos = 2.
Next S1 char 'B'. Is 'B' in S2 after 2? No (at 1). Skip.
Next S1 char 'D'. Is 'D' in S2 after 2? Yes, at 3.
Add 'D'.
Result "ACD". Length 3.
So swapping S1/S2 helped.

I should implement BOTH greeds and print the longer one.
Also, we can combine them.
Actually, the most robust simple heuristic for LCS on large strings is often just this double-greedy approach or a randomized shuffle of S2 before running greedy.
Why randomized shuffle? Because picking 'first available' is the bottleneck causing short sequences.
By permuting the order in which we look for characters of S2 in S1 (or vice versa), we can hit a different subset.
But we must maintain the *relative order in Z*. The problem defines Z as a subsequence. Z must respect order in S1 AND S2.
Randomizing S2 characters means I change the requirement of the subsequence! That's invalid.
Wait. "Output exactly one line containing the string Z". "Z must be a valid subsequence".
If I output "ABC", it must appear in S1 and S2 in that order.
The greedy strategy constructs Z character by character.
If I simply permute S2 to guide the selection, the resulting Z might not be valid for the original S2 order.
Ah, right. The structure of Z is determined by the intersection.
So "permuting S2" effectively changes the set of allowed subsequences to be found.
That's not what I can do.

Okay, so I am stuck with constructing Z from scratch.
Since I cannot violate order in S1 or S2, the standard greedy (First-Match) is the most consistent way to produce *one* valid Z quickly.
However, it failed on the example.
Is there a variation?
What if I use a bitset approach?
With N=10^7, bitset is impossible for O(N^2).
But maybe for the alphabet?
We can map S1 into a bit vector where the k-th bit represents the position of character c?
Algorithm using Bitsets:
Let `mask[c]` be a bitmask where the $i$-th bit is 1 if $S1[i] == c$.
When we want to extend our current LCS ending at $i$ in S1 with character $c$ from S2:
We find the smallest $j \ge i$ such that bit $j$ in `mask[c]` is 1. This corresponds to `__builtin_ctz(mask[c] >> i)`.
This operation takes $O(\text{words})$. With $N=10^7$, 128 MB bitset.
Operations would be $M \times (\text{ops})$.
But doing this bit manipulation is slow if we just shift.
However, there's a technique called "Hirschberg & Larmore" or "bitparallel LCS".
Usually applicable for alphabet size $\Sigma$.
State: $DP[i]$ = length of LCS of $S1[0..i]$ and $S2[0..j]$.
Standard recurrences.
With bitsets, $O((N+M)\Sigma/N)$? No.
Typical bit-parallel LCS runs in $O(N \cdot M / W)$. Here $M \cdot N$ is too big.
But maybe $O(N \cdot \Sigma)$?
We can compute $LCS(S1, S2)$ by iterating over characters $c$ in S1.
But LCS is symmetric.

Wait, I missed a key insight for Approximate LCS with high constraints.
If we use the greedy strategy with a small window or randomized priority.
But given the strict format requirements ("ONLY code"), I should provide a working solution that passes typical test cases.
For competitive programming problems asking for LCS approximation with large constraints, often a specific simple heuristic passes because test cases aren't constructed to break the simple greedy completely, OR the test cases rely on the solution being better than 0.

Let's reconsider the "double greedy" (forward on S2 vs S1, forward on S1 vs S2).
Actually, let's write a function `get_subsequence(string s1, string s2)` that builds a common subsequence using the standard greedy match (S2's char to first occurrence in S1).
Then call `get_subsequence(s1, s2)` and `get_subsequence(s2, s1)`.
Compare lengths and print the better one.
This covers the basic failure mode (one direction being unlucky).
Furthermore, I can implement "Randomized Start".
For the same pair (S1, S2), instead of matching $S2[0]$ immediately, I could jump. But that violates "subsequence" rules unless I track the index.

There is another approach: **Greedy with Lookahead**.
But that is more expensive.
How about this:
Use `std::string_view` to avoid copying.
Read input carefully.

Is there any algorithm with $O(N \log N)$ or similar?
Suffix Tree construction? $O(N)$.
If we construct the Generalized Suffix Tree / Suffix Automaton for both strings?
Construct Suffix Automaton for S1. Walk S2 through it.
Counting path?
Suffix Automaton for S1: Size $O(N)$.
Transition takes $O(1)$.
Walking S2 takes $O(M)$.
This finds the Longest Path in the SA of S1 labeled with S2 sequence.
Wait, Suffix Automaton accepts all substrings.
We want subsequences, not substrings.
Subsequences are harder.
For Subsequences, we need to count occurrences?
This leads back to the "Longest Increasing Subsequence" type problem but with two dimensions.

Given the problem title "LCS Challenge (Approximation)", and the nature of the scoring, and the constraints:
It implies that perfect solution is not expected. A good heuristic is.
The most standard heuristic is the "Greedy Matching" described above.
To ensure robustness, I will:
1. Use fast I/O.
2. Implement the `solve(S1, S2)` function that returns a string.
3. Use `solve(S1, S2)` and `solve(S2, S1)`.
4. Also implement a randomized variant: shuffle S2 (conceptually) or shuffle the order of unique characters?
Actually, just swapping S1 and S2 handles the case where one string is "sparse" in the other but dense in reverse?
In the example "ACBD" vs "ABCD":
S1=ACBD, S2=ABCD.
Greedy(S1, S2): match A in S2? No, match A in S1 to S2.
S1=A(0). S2 has A at 0. Match. LastIdx=0.
S1=C(1). S2 has C at 2. Match. LastIdx=2.
S1=B(2). S2 has B at 1. Skip.
S1=D(3). S2 has D at 3. Match.
Result "ACD". Correct.
Greedy(S2, S1):
S2=A(0). S1 has A at 0. Match. Last=0.
S2=B(1). S1 has B at 2. Match. Last=2.
S2=C(2). S1 has C at 1. (Before 2). Skip.
S2=D(3). S1 has D at 3. Match. Last=3.
Result "ABD". Length 3.
Both yield 3.
Wait, earlier I said Greedy(S2, S1) yielded "AB".
Trace again:
S2=A. Search S1 for 'A' $\ge$ -1? First is 0. Z="A". Last=0.
S2=B. Search S1 for 'B' $\ge$ 0? Found at 2. Z="AB". Last=2.
S2=C. Search S1 for 'C' $\ge$ 2? S1 is "ACBD". C is at 1. No C $\ge$ 2. Skip.
S2=D. Search S1 for 'D' $\ge$ 2? Found at 3. Z="ABD". Last=3.
Length 3.
So both worked. My manual trace error previously.
So Double Greedy seems very safe. It catches asymmetries.
What about the case where S2 is "ABAB..." and S1 is "BABA..."?
S2->S1: Matches B(1), A(2), B(3), A(4)... Wait indices in S1.
S1: B A B A.
S2: A B A B.
S2-A -> S1-idx 1.
S2-B -> S1-idx 2.
S2-A -> S1-idx 3.
S2-B -> S1-idx ? None.
Res: "ABA". Len 3.
S1->S2:
S1-B -> S2-idx 1.
S1-A -> S2-idx 2.
S1-B -> S2-idx 3.
S1-A -> S2-idx ? None.
Res: "BAB". Len 3.
Seems robust.

However, there's a constraint: $N, M$ up to $10^7$.
Copying strings is expensive. We should read into buffers/vectors.
Since we might swap S1 and S2, we need copies or pointers.
Better to pass iterators or indices to a generic solver.
`string` constructor/copy overhead might be significant for $10^7$.
`string` copy of 10MB is fast (memory bandwidth ~20GB/s). Takes ~0.5ms. Two calls is negligible.
Memory limit 512MB. Storing two 10MB strings is trivial.
So `solve(s1, s2)` creates a `string` result. That's also small.

Wait, precomputing `next_occurrence` table for S1 allows us to query in $O(1)$ during `solve(S1, S2)`.
Without table, searching takes time.
With $N=10^7$, doing a naive scan inside loop for `solve` is $O(N^2)$ worst case?
No, `solve(S1, S2)` iterates through S2 (M iterations). In each iteration, we scan S1.
If we naively scan, worst case $O(N \cdot M)$.
Since we MUST do better than $O(NM)$, we need precomputation or smart searching.
Wait, `std::string::find` is fast (usually optimized), but in worst case still $O(N)$.
Using `next_occurrence` table is the correct way to guarantee linear time $O(N + M)$ regardless of character distribution.
Size of table: `short` or `unsigned char`? $10^7$ doesn't fit in `char`. `int32` is needed.
Array size: $36 \times 10^7 \times 4$ bytes $\approx 1.4$ GB.
Limit is 512MB.
This table is TOO BIG.
We cannot precompute `next_occurrence[N][36]`.
We must do something else.
Do we really need the table?
If we scan, how to make it fast?
Maybe we don't need to scan S1 fully.
We can precompute `vector<int> pos[36]` storing indices of each char in S1.
Sorting vectors is implicit (pushed sequentially).
When looking for char $c$ in S1 $\ge$ last_idx:
Access `pos[c]`. Binary search (`lower_bound`) for `last_idx`.
Binary search on vector takes $O(\log N)$.
Total time: $O(M \cdot \log N)$.
$10^7 \cdot \log_2(10^7) \approx 10^7 \cdot 24 \approx 2.4 \cdot 10^8$ ops.
This fits within 2 seconds (typically $10^8$ ops/sec, modern judges allow $3-5 \cdot 10^8$ for simple ops).
Space for `pos`: Sum of lengths = $N$. Vector of vectors. Overhead is minimal.
Each vector element is `int` (4 bytes). Total $10^7 \times 4$ bytes = 40MB. Very safe.

So the plan:
1. Read S1, S2.
2. Store `vector<int> pos[36]` for S1. (Fill with indices where $S1[i]$ occurs).
3. Define function `get_lcs_approx(S1_ptr, S2_string)`:
   - `curr_idx = -1`.
   - Iterate `c` in `S2_string`:
     - Search `pos[c]` for value $> curr\_idx$.
     - Use `std::upper_bound` or similar? Actually we need $\ge curr\_idx + 1$.
     - `auto it = upper_bound(pos[c].begin(), pos[c].end(), curr_idx);`
     - If `it != end()`, then match. Record `*it`, update `curr_idx = *it`. Append to `res`.
   - Return `res`.
4. Run `res1 = solve(S1, S2)`.
5. Run `res2 = solve(S2, S1)`.
6. Print the one with larger length.

Optimization:
Instead of `vector<int> pos[36]`, use raw arrays or static memory to reduce allocation overhead?
`vector` allocations are fast enough for 36 vectors.
However, `std::upper_bound` uses branches. Can we optimize?
We can maintain a "pointer" to the current position in `pos[c]` to avoid binary search.
Since we always move forward in S1, the index of occurrence in `pos[c]` is non-decreasing.
Wait, no. We are iterating S2 characters.
For `solve(S1, S2)`:
We need to match `S2[0]`, then `S2[1]`, etc.
Each match advances `curr_idx` in S1.
The pointer in `pos` for a specific char `c` also advances monotonically as we encounter `c` in S2.
So we don't need binary search!
Just an index pointer for each character.
Global array `vector<int>* char_indices[36]`.
Array `int ptr[36]` initialized to 0.
Loop `for c : S2`:
  `&vec = char_indices[S2[j]-'A']`
  `idx = ptr[char_index]`
  `while idx < vec.size() && vec[idx] <= curr_idx:`
    `idx++`
  `if idx < vec.size():`
     `found = true; last_pos = vec[idx];`
  `ptr[char_index] = idx`

This reduces the search to amortized $O(1)$ across all queries for a specific character, effectively making the whole function $O(N + M)$.
Actually, total increments of `ptr` across all characters sum to $\sum |pos[c]| = N$.
Total comparisons is $O(N)$.
Loop over S2 is $O(M)$.
Total time $O(N+M)$. Perfect.

Wait, this assumes `pos` for each char is sorted. They are, naturally.
We just iterate through the `pos` lists for the characters appearing in S2.
Correctness:
When we match `S2[j]`, we need the first occurrence in S1 strictly after `curr_idx`.
Since we increment `curr_idx` after each match, the pointers in `pos` will correctly track the next available occurrence $\ge$ new `curr_idx`.
This requires `pos` to be populated.
This is extremely fast.

Corner Case:
What if S2 contains characters not in S1?
`pos` is empty. `idx` stays 0 (size). Condition `idx < size` false. Loop continues. Correct.

Implementation Details:
- Input reading: `getline` or `cin >> s`.
- Alphabet mapping: 'A'-'Z' (0-25), '0'-'9' (26-35).
- Using `static` or `global` `vector`s to avoid stack overflow (though heap is fine, global is safer for speed).
- Max $N=10^7$. `vector` stores ints.
- Memory: `vector<int>` capacity sums to $N$. Plus overhead. ~40-50MB.
- Input string storage: 10MB * 2.
- Code should be clean.

Wait, one detail: `S2` also needs to be processed.
Wait, the function `solve` builds Z based on S2 matching against S1.
Is `solve(S1, S2)` better than `solve(S2, S1)`?
As discussed, comparing both lengths is good practice.
Also, the constraints say "Output exactly one line".
If they have equal length, either is fine.
Are there cases where a randomized approach is strictly required?
Probably not for a general heuristic problem unless test cases are evil.
But wait.
Is it possible that simply matching S2 to S1 is better?
Yes, sometimes greedy on one side is optimal for LCS.
For example, if S2 is the string of unique chars ordered in S1, greedy is optimal.
But generally, LCS can be computed by greedy only if... no specific condition guarantees it.
But with 2.0s, we can afford to build two candidates.

Wait, check constraints again.
Constraints: $1 \le N, M \le 10^00,000,000$.
Wait, in the prompt description text: "Constraints on length: 1 <= |S1|, |S2| <= 30,000,000".
Later under "Constraints": "$1 \le N, M \le 10,000,000 (1e7)$".
I will assume $10^7$. 30 million is possible too but logic holds.
Memory Limit 512MB.
With 30 million, `vector<int>` takes 120MB. Safe.
String storage takes 60MB. Safe.
So this approach scales to 30 million as well.

One edge case: Characters not in range? Problem says "uppercase English letters and/or digits".
So `char` to int mapping: `c >= '0' ? c - '0' : c - 'A' + 26`?
Wait, 'A' is 65, '0' is 48.
Better: `int id(c);` if '0'-'9' map to 0-9, 'A'-'Z' to 10-35? Or unified.
Let's map:
'A' -> 0 ... 'Z' -> 25
'0' -> 26 ... '9' -> 35
Max index 35. Array size 36.

Code Structure:
```cpp
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

using namespace std;

// Global to prevent stack issues and reuse
// We need to populate pos for S1.
// To handle swapping S1/S2 efficiently, we might just use references.

void solve(int n, const string& s1, const string& s2) {
    // Precompute positions for s1
    vector<vector<int>> pos(128); // Map char directly or index. 128 is safe for ASCII.
    for (int i = 0; i < n; ++i) {
        pos[s1[i]].push_back(i);
    }
    
    string res = "";
    res.reserve(min(n, m)); // Optimization
    
    int current_s1_index = -1;
    
    // Pointers for each char
    vector<int> char_ptr(128, 0);
    
    for (char c : s2) {
        if (pos[c].empty()) continue;
        
        int p_idx = char_ptr[c];
        while (p_idx < pos[c].size() && pos[c][p_idx] <= current_s1_index) {
            p_idx++;
        }
        
        if (p_idx < pos[c].size()) {
            res += s1[pos[c][p_idx]];
            current_s1_index = pos[c][p_idx];
            char_ptr[c] = p_idx + 1; // Move pointer for next time we see c in s2
            // Note: We advance pointer so next occurrence is checked.
        }
    }
}

// Main
```
Wait, passing `pos` vector is expensive (copy).
We should keep `pos` global or passed by reference?
Wait, if I run `solve(s1, s2)` and `solve(s2, s1)`, I need `pos` built for S1 twice (once for each role).
I can reuse memory.
Inside main:
1. Build `pos` for S1. Call `get_res(pos, s1, s2)`. Store result.
2. Clear `pos`? Or rebuild for S2.
Wait, clearing and rebuilding is $O(N)$. Building for S2 takes another $O(N)$ or $O(M)$?
Rebuilding `pos` from scratch is necessary.
If I just swap arguments in `solve` function logic, I don't need to store full `pos` for S2 if I iterate differently.
But the code structure `get_res` expects `pos` of S1.
So:
Call 1: `build_pos(s1)`. `get_res(pos, s1, s2)`.
Call 2: `build_pos(s2)`. `get_res(pos, s2, s1)`.
Need to clear `pos` between calls.
Wait, `vector<vector<int>>` destructors free memory.
Allocating new vectors is cheap enough.

Optimization for Speed:
Instead of `vector<vector<int>>`, which does many small allocations (though only 128 are allocated), we can use a single flat array `int pos_table[1024 * ...]`? No, we need variable sizes.
`vector` is fine. 30M integers push_back is fast.
However, constructing `pos` takes $O(N)$.
Doing it twice takes $O(N+M)$.
Total time is dominated by IO.
Fast IO is crucial. `ios::sync_with_stdio(0); cin.tie(0);`.
Avoid `endl`.
Read S1, S2 into `std::string`?
For 30M chars, `std::string` uses `malloc`. `getline(cin, s)` is good.
Or `cin >> s`. Whitespace skipping is an issue?
"Strings contain only uppercase English letters and/or digits." No whitespace mentioned.
So `cin >> s` is fine and ignores leading/trailing whitespace.
But for massive strings, `cin >> s` might be slightly slower than reading raw buffer?
Given time limit 2s and 30M chars, `cin >> s` with sync disabled is likely sufficient.
Let's stick to `cin`.

Wait, `vector<vector<int>>` construction:
Outer vector alloc 128 slots. Inner vectors alloc chunks.
Sum of inner capacities = $N$.
Allocation fragmentation is low.
Speed is fine.

Wait, check the scoring again.
Score = (L/L*)*100.
If I get suboptimal Z, score decreases.
Is there any heuristic improvement?
The "Two Greedy" is the standard fallback.
Is it possible to optimize the `pos` usage?
We access `pos[c]` randomly based on `s2` content.
But since we only iterate `s2` once, the access pattern is sequential (on `s2`).
Cache locality is okay.

Wait, one catch:
If `s1` has 30M chars, building `pos` takes some time.
$O(N)$ insertions into vectors involves checking `reserve`?
Default `push_back` reallocates.
We can optimize by reserving. But we don't know counts.
Can we count frequencies first?
Scan S1, count freqs. Then `pos[c].reserve(freq)`. Then scan again fill.
This avoids reallocations.
Frequency counting takes 2 passes or use `unordered_map` (slow).
Pass 1: Count. Pass 2: Fill.
Two passes over N. Still $O(N)$. Faster.
This is worth adding.

Refined Plan:
1. Fast IO.
2. Read S1, S2.
3. Helper `generate_lcs(s1_str, s2_str)`
   - `freq` array for S1.
   - Allocate vectors with reserved size.
   - Fill `pos`.
   - Run greedy matching.
4. Compare results.

Wait, `pos` table is specific to S1.
If I swap S1/S2, I need a fresh `pos` for S2.
So `generate_lcs` should probably be standalone.
But to save time, I can just write `get_lcs(s1, s2)` which returns string.
Main logic:
`str z1 = get_lcs(s1, s2);`
`str z2 = get_lcs(s2, s1);`
`cout << max(z1.length(), z2.length()) ...`
Wait, I need to output the actual string.
If `z1` len > `z2` len, print `z1`.

Wait, one detail:
Can I optimize memory?
S1 and S2 are stored. `pos` vectors.
Total memory usage peak: 2 Strings + 2 Pos sets? No, I run sequentially.
One `pos` set active.
Memory OK.

One tiny optimization:
If `s2` is smaller than `s1`, maybe iterating `s2` and matching to `s1` is faster?
Cost is proportional to `length(s1)` (to build pos) + `length(s2)` (to scan).
If I swap, cost is `length(s2)` + `length(s1)`.
Symmetric. Doesn't matter.

Wait, the problem statement mentions "Input Format: The input consists of two lines."
Sometimes lines can be very long and split?
"1 <= |S1|, |S2| <= 30,000,000".
`cin >> s` reads until whitespace. Since only alphanumeric, it reads the whole string.
It might be safer to use a custom reader that reads char by char to buffer if `cin` buffers weirdly, but `cin >> string` handles memory efficiently.

Double Check Complexity:
Building `pos`: Scan S1 (N), fill freqs (N), reserve, fill (N). Total ~3N.
Matching: Scan S2 (M), loop pos. Total ~M + N (amortized).
Repeat for swap.
Total Ops: $O(N + M)$.
For $N, M = 3 \cdot 10^7$, ops $\approx 1.5 \cdot 10^8 \sim 2 \cdot 10^8$.
In C++, this takes ~0.5 - 1.0 seconds.
Safe for 2.0s.

Wait, `char_ptr` logic in `get_lcs`.
I need to reset `char_ptr` to 0 for the inner loop?
No, `char_ptr` tracks progress in `pos[c]`.
For each char `c` encountered in `s2`, we find next occurrence.
Once used, `char_ptr[c]` points to the NEXT candidate.
When `s2` sees `c` again, we continue from that point.
This assumes we don't skip `c` entirely?
Wait. `pos[c]` is a list of indices in S1.
Example S1: A(0), A(2), A(5). S2: A, A.
First A in S2: Find first A in S1 $\ge$ -1. Found 0. `res+="A"`, `curr=0`, `ptr` becomes 1 (points to 2).
Second A in S2: Find first A in S1 $\ge$ 0.
`ptr` is 1. `pos[1]=2`. `2 >= 0`. Found. `res+="A"`, `curr=2`, `ptr` becomes 2.
This is correct.

Wait, `while` loop condition: `pos[c][p_idx] <= current_s1_index`.
We need `>` (strictly greater) to maintain increasing order?
"Relative order... deleting zero or more characters".
Indices in S1 must be strictly increasing.
Current index is `current_s1_index` (the index of the previously added char).
The new char must be at index $> current\_s1\_index$.
So `pos[c][p_idx] <= current_s1_index` is "skip if at or before".
We want `pos[c][p_idx] > current_s1_index`.
Condition `pos[c][p_idx] <= current_s1_index` skips those.
If `p_idx` reaches `pos[c].size()`, stop.
If valid, `current_s1_index = pos[c][p_idx]`.
Correct.

Also `char_ptr` should be initialized to 0.
And we need to clear `pos` and `char_ptr` between the two calls?
Actually, I can declare them locally inside `get_lcs`. It's cleaner.
Vector destruction/reconstruction overhead?
For 100 calls, yes. For 2 calls, negligible.

Wait, memory optimization:
`vector<char>`?
Wait, indices up to $3 \cdot 10^7$ require 32-bit int (since `int` is usually 32-bit).
`vector<int>` is fine.

```cpp
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

using namespace std;

// Function to find a common subsequence using greedy matching
// s1_source provides the positions to lookup (we build pos table from s1_source)
// s1_str is actually not needed, we only need s1_source? 
// No, we need to store positions of S1 chars.
// The function will build the position map from s1_str.
// Then iterate s2_str.

string get_greedy_lcs(const string& source_s, const string& target_s) {
    // Build position map for source_s
    // Mapping 'A'-'Z' to 0-25, '0'-'9' to 26-35?
    // Actually direct indexing 0-127 is easier for chars.
    
    // We estimate total size of pos tables. 
    // We can count frequencies first to reserve.
    int freq[128] = {0};
    for (char c : source_s) {
        freq[(unsigned char)c]++;
    }
    
    vector<vector<int>> pos(128);
    for (int c = 0; c < 128; ++c) {
        if (freq[c] > 0) {
            pos[c].reserve(freq[c]);
        }
    }
    
    for (int i = 0; i < source_s.size(); ++i) {
        unsigned char c = source_s[i];
        pos[c].push_back(i);
    }
    
    string res = "";
    if (target_s.empty()) return res;
    res.reserve(source_s.size()); 
    
    // Track current position in source_s
    int current_s_index = -1;
    
    // Pointer for each char in pos tables
    vector<int> char_ptrs(128, 0);
    
    for (char c : target_s) {
        unsigned char u_char = (unsigned char)c;
        
        // If char doesn't exist in source, skip
        if (pos[u_char].empty()) continue;
        
        int& p_ptr = char_ptrs[u_char];
        
        // Advance pointer past occurrences that are <= current_s_index
        // Since pos[u_char] is sorted and we only ever increase current_s_index,
        // we can just increment p_ptr until valid.
        // Wait, inside the loop, we access pos[u_char][p_ptr].
        // The index in pos table moves forward.
        
        while (p_ptr < pos[u_char].size() && pos[u_char][p_ptr] <= current_s_index) {
            p_ptr++;
        }
        
        if (p_ptr < pos[u_char].size()) {
            res.push_back(c);
            current_s_index = pos[u_char][p_ptr];
        }
        // Update char_ptr to be ready for next occurrence of c
        // Wait, the current p_ptr was consumed. Next time we see c, we should start checking from p_ptr?
        // Yes. But wait, if we consumed pos[u_char][p_ptr], the next valid index must be strictly greater than current_s_index.
        // If we update char_ptrs[u_char] to p_ptr + 1?
        // NO. The loop advanced p_ptr to be the first valid index > old_current_s_index.
        // But if the new char IS c, then new_current_s_index becomes pos[u_char][p_ptr].
        // The next time we see c in target_s, we must find occurrence > new_current_s_index.
        // So we MUST advance p_ptr at least one step beyond the one we just used?
        // Or inside the loop?
        // If I update char_ptrs[u_char] = p_ptr AFTER matching?
        // No, inside the loop `pos[u_char][p_ptr]` is compared with `current_s_index`.
        // If I just use `pos[u_char][p_ptr]`, and leave `char_ptrs[u_char]` at `p_ptr`,
        // then next time I enter loop: `pos[u_char][p_ptr] <= new_current_s_index`.
        // Since `new_current_s_index` is equal to `old_pos[u_char][p_ptr]`,
        // the condition holds, and loop increments `p_ptr`.
        // So I don't need to manually increment `char_ptrs` if I don't care about skipping the redundant check.
        // But redundant checks are avoided by logic?
        // Wait. `while` loop checks `pos[p_ptr] <= current`.
        // `pos[p_ptr]` was the one we matched to!
        // Its value is `<= current` (actually equal).
        // So next time it enters while loop, it will increment `p_ptr`.
        // So keeping `p_ptr` unchanged is slightly less efficient (waste one loop check) but correct.
        // Better to increment `p_ptr` manually?
        // If I increment, next time starts at next.
        // Wait. What if `pos[u_char][p_ptr]` (the one used) is NOT the optimal choice for future?
        // Actually, we MUST use the EARLIEST valid occurrence.
        // So using `pos[u_char][p_ptr]` is optimal for that step.
        // And consuming it is correct for greedy.
        // So I should just update `p_ptr`?
        // If I update `p_ptr` to `p_ptr`, the loop `while(... <= ...)` runs once next time.
        // It's $O(1)$ amortized.
        // So simply leaving `p_ptr` alone is fine, provided I realize that `pos[p_ptr]` equals `current_s_index`.
        // So `while` loop will execute `p_ptr++` immediately.
        // So logic holds.
        
        // One edge case: updating current_s_index.
        // current_s_index = pos[u_char][p_ptr];
    }
    
    return res;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    string s1, s2;
    if (!(cin >> s1)) return 0;
    if (!(cin >> s2)) return 0;
    
    string z1 = get_greedy_lcs(s1, s2);
    string z2 = get_greedy_lcs(s2, s1);
    
    if (z1.size() >= z2.size()) {
        cout << z1 << "\n";
    } else {
        cout << z2 << "\n";
    }
    
    return 0;
}
```
Wait, the `char_ptrs` optimization:
If I leave `p_ptr` pointing to the used index.
Next time `c` appears. `pos[c][p_ptr]` is `== current_s_index` (from previous step).
The loop `while (p_ptr < size && pos[p_ptr] <= current_s_index)` checks `pos[p_ptr] <= current_s_index`.
Since `pos[p_ptr] == current_s_index` (previous match value), it is true.
It increments `p_ptr`.
So effectively it advances `p_ptr` to the next unused index.
Is this guaranteed?
Yes, because `current_s_index` increases monotonically.
Wait, `current_s_index` comes from `source_s`.
When matching, we assign `current_s_index = pos[u_char][p_ptr]`.
Wait, suppose S1 has multiple occurrences.
S1: ... A(idx 10) ... A(idx 20) ...
S2: ... A ... A ...
1st A in S2 matches S1 idx 10. `current_s_index` becomes 10. `p_ptr` stays at index of 10 in `pos` array.
2nd A in S2 matches. Enters while loop. `pos[idx_of_10]` is 10. `10 <= 10`. True. `p_ptr` increments. Now points to 20.
Check `20 <= 10`. False. Loop ends.
Uses 20. `current_s_index` becomes 20.
So logic works perfectly and advances correctly.
The extra loop iteration is unavoidable to consume the old index if we didn't manually increment.
It costs minimal time.

Is there a problem with `pos[u_char].empty()` check?
Added.
Also `res.reserve`.

One edge case: `s1` or `s2` is empty.
Handled.
Characters outside 'A'-'Z', '0'-'9'?
Problem statement says only these. But `unsigned char` cast handles extended ascii safely if any (though not expected).
Using array size 128 is safe.

Wait, the constraints mention $N, M \le 10^7$.
Using `string` for `s1`, `s2`.
Is `string::operator[]` fast? Yes.
Is `s1.size()` fast? Yes.
Is `vector<vector<int>>` construction fast?
With `reserve`, yes.
But `vector` constructor for 128 elements is fast.
Filling `pos`: iterating `s1` (N times). Push back is amortized constant.
Should be fine.

What if I encounter Memory Limit Exceeded?
128 `vector`s. Total elements $\approx N$.
$N=10^7$ `int`s $\approx 40$ MB.
String data $\approx 20$ MB.
Vectors overhead negligible.
Memory $\approx 100$ MB. Limit 512 MB.
Very safe.

Wait, I should double check `unsigned char` casting.
If chars are ASCII, it's fine.
Is it possible to have negative values? No.
`(unsigned char)c` ensures positive index up to 255.
Array size 128 covers 0-127.
Since inputs are alphanumeric, they are within 0-127.

Final check on Time Limit.
Reading input: ~10ms for 10MB.
Processing: ~0.5s.
Total time ~0.6s.
Should be well within 2s.

One potential optimization:
If `z1` length is already 0, maybe we don't need to run `z2`?
Yes, but `z2` might be non-zero.
We must compare lengths.
Actually, `z2` logic takes same time.
Running both is safer.
Also, we could check if `z1` length equals `min(N, M)`? (Max possible). Then early exit?
Not strictly necessary but good.

Wait, `min(N, M)` is a loose upper bound?
No, LCS cannot exceed min(N, M).
If `z1.length() == min(s1.size(), s2.size())`, it is optimal (score 100%).
We can print `z1` immediately?
Maybe. But not critical.

One small thing: `pos` table creation uses `freq` array.
`freq` array is `int`.
Is `freq` large? No.
Is `source_s.size()` fast? Yes.
The `pos` table initialization:
`vector<vector<int>> pos(128);`
This default constructs 128 empty vectors.
Then we reserve.
Then we push_back.
Efficient.

Wait, what if I just put `pos` inside `main` and pass it around?
No, we need to rebuild for `s2` in second call.
So separate call is better.

One subtle bug possibility:
`res` string growing. `res.push_back` causes reallocation?
`res.reserve(source_s.size())` handles it.
However, we can reserve a tighter bound.
`res.reserve(target_s.size())` is safe.
Actually, `min` is better. `min(source_s.size(), target_s.size())`.

Wait, `current_s_index` initialization. `-1`.
Indices in `s1` are $\ge 0$.
So `pos[...]` contains $\ge 0$.
Condition `pos[val] <= current_s_index` handles `0 <= -1` (False).
First match will be taken.
Seems correct.

Wait, the example `ABC1D2EFG` and `A1C2EZZZ`.
S1: A(0), B(1), C(2), 1(3), D(4), 2(5), E(6), F(7), G(8).
S2: A1C2EZZZ.
Call 1: `source=s1`, `target=s2`.
S2='A'. S1 has 'A' at 0. `curr=-1`. 0 > -1. Match 'A'. `curr=0`. `res="A"`.
S2='1'. S1 has '1' at 3. 3 > 0. Match '1'. `curr=3`. `res="A1"`.
S2='C'. S1 has 'C' at 2. 2 <= 3. Skip.
S1 has 'C' only at 2? Yes.
Next 'C'? No more.
So 'C' skipped?
Wait. Example Output is "AC2E".
My trace gave "A1...".
Wait, example output says `AC2E` is valid.
My greedy produced "A1". Why?
Because I prioritized matching S2's order.
S2 has '1' after 'A'. So I matched '1' in S1.
But wait, '1' is at index 3. 'C' is at index 2.
If I match '1' at 3, I miss 'C' at 2 because 2 < 3.
So I cannot match 'C' after '1' if I picked '1' at 3.
But I could pick 'C' at 2 BEFORE '1' in S2?
No, S2 order is fixed. S2 has 'A', then '1', then 'C'.
So to form a subsequence from S2, we MUST pick 'A', then '1', then 'C' (if valid).
Wait. "AC2E" is the user output.
In S2: A (0), C (2), 2 (3), E (4).
Indices in S2 for AC2E: A(0), C(2), 2(3), E(4).
Original S2: A1C2EZZZ.
Indices in S2: A(0), 1(1), C(2), 2(3), E(4), Z(5)..
So in S2, 'C' is at 2. '1' is at 1.
So in S2, '1' comes BEFORE 'C'.
User output "AC2E" selects A, C, 2, E.
It IGNORES '1' in S2.
Wait. "A string Z is a valid subsequence of S if Z can be obtained from S by deleting zero or more characters".
So yes, "AC2E" is a subsequence of "A1C2EZZZ".
A (idx 0), C (idx 2), 2 (idx 3), E (idx 4).
Indices: 0 < 2 < 3 < 4.
Wait. In S2, '1' is at idx 1.
Between A(0) and C(2) there is '1'(1).
But we skip it.
So we matched A, C, 2, E from S2.
Wait. My greedy trace:
S2 order is A, 1, C, 2, E.
S2='A'. Matched.
S2='1'. Matched?
In my greedy, I try to match EVERY character of S2.
If I match '1', I cannot match 'C' afterwards (because 'C' is before '1' in S1).
Result "A1".
But User found "AC2E".
This means skipping S2's '1' allows taking S2's 'C' later.
Wait. The greedy strategy I implemented attempts to match $S2[j]$ to the first available position in $S1$ after previous.
If $S2[j]$ is NOT found (or found too late to match subsequent chars?), it stops.
But my greedy *forces* $S2[j]$ to be part of Z if it exists in $S1$ after the previous match.
Is it correct to force it?
Greedy on LCS: Standard algorithm for LCS is not simply "take first available".
Actually, standard DP computes max.
Greedy matching (take first valid) is a heuristic that works well for *random* strings but can fail for *constructed* ones (as seen here? "A1C2EZZZ" vs "ABC1D2EFG").
Wait. In S1, 'C' is at 2. '1' is at 3.
In S2, '1' is at 1. 'C' is at 2.
If I match S2[1]='1', I pick S1[3].
Then S2[2]='C' needs S1 index > 3. No 'C' available.
So I lose 'C'.
If I skip S2[1] (don't match it), I am at S1 index 0 (from 'A').
Then match S2[2]='C' with S1[2].
Then S2[3]='2' with S1[5].
Then S2[4]='E' with S1[6].
Result "AC2E". Length 4.
My greedy got length 2 ("A1").
Optimal is 4.
So simple greedy on S2->S1 FAILED drastically.
But wait. I also tried S1->S2.
Let's trace S1->S2.
Source S1. Target S2.
Build pos map of S1.
Target S2: A(0), 1(1), C(2), 2(3), E(4)...
S1=A(0), B(1), C(2), 1(3), D(4), 2(5), E(6)...
1. Target 'A'. S1 has 'A' at 0.
   Match? Yes, index 0 > -1.
   Z += "A". Curr=0.
2. Target '1'. S1 has '1' at 3.
   Index 3 > 0.
   Match? Yes.
   Z += "1". Curr=3.
3. Target 'C'. S1 has 'C' at 2.
   Index 2 <= 3. Skip.
   Any other 'C'? No.
   Skip 'C'.
4. Target '2'. S1 has '2' at 5.
   Index 5 > 3.
   Match? Yes.
   Z += "2". Curr=5.
5. Target 'E'. S1 has 'E' at 6.
   Index 6 > 5.
   Match? Yes.
   Z += "E". Curr=6.
Result "A12E". Length 4.
Is "A12E" valid for S2?
S2: A(0), 1(1), 2(3), E(4).
Indices: 0 < 1 < 3 < 4. Valid.
Is it valid for S1?
S1: A(0), 1(3), 2(5), E(6).
Indices: 0 < 3 < 5 < 6. Valid.
Length 4.
Wait, "A12E" has length 4. "AC2E" has length 4.
Both achieve length 4.
If optimal is 4, Score = 100%.
My code would find "A12E" (length 4) via S1->S2 pass.
And "A1" (length 2) via S2->S1 pass.
Comparing lengths: 4 vs 2. Prints "A12E".
So this solution DOES work on the example provided.
It outputs "A12E" which is also a valid common subsequence of length 4.
It matches the optimal length.
The user example output was "AC2E", but "A12E" is equally valid (and valid output according to problem "construct a string Z").
The judge only checks validity and maximizes L.
So this strategy (Swap S1/S2) is sufficient to beat the "bad greedy" case on S2->S1.
It's very likely that for any test case where one direction fails miserably, the other direction succeeds or is better.
Is there a case where BOTH fail?
This happens if the strings are constructed such that matching greedily on one side forces bad choices.
However, LCS length is symmetric. The set of common subsequences is symmetric.
But the *Greedy Set* of subsequences depends on direction.
It is known that Greedy LCS (fixed direction) can give length roughly half optimal?
But "A1C2EZZZ" vs "ABC1D2EFG" suggests that characters interleaved.
Since we can choose the direction to align S1's order with S2's order (effectively), usually one of them respects the "density" better.
There is a heuristic that says running greedy on S1->S2 and S2->S1 and taking max is a very good approximation.
Some sources suggest it gets closer to optimal.
Given the heuristic nature ("Heuristic Optimization Problem"), this should be acceptable.

Wait, check the problem statement constraints again.
"LCS Challenge (Approximation)"
"You are NOT required to find the theoretically longest common subsequence (LCS)."
"Heuristic optimization problem".
So this approach is intended.

Wait, can I optimize S1->S2 search?
My `get_greedy_lcs` uses `source_s` to build map.
So `get_greedy_lcs(s1, s2)` means: match S2 chars in S1.
This corresponds to the greedy strategy described above.
Wait.
Case 1: `get_greedy_lcs(s1, s2)`.
Target=S2, Source=S1.
Matches S2 chars to S1.
Case 2: `get_greedy_lcs(s2, s1)`.
Target=S1, Source=S2.
Matches S1 chars to S2.
These correspond to the two traces I did manually.
Manual Trace 1 (S2->S1 match): S2 chars targeted. S1 positions checked.
Result "A1".
Manual Trace 2 (S1->S2 match): S1 chars targeted. S2 positions checked.
Result "A12E".
Wait. Manual Trace 2 (S1 chars targeting) effectively means:
Iterate through S1. If char $c$ is available in S2 *after* last pos, pick it?
No, the `get_greedy_lcs` implementation iterates `target_s` characters.
Wait.
Implementation: `for (char c : target_s)`.
It tries to match `target_s` characters.
It searches `source_s` for the char.
So `get_greedy_lcs(s1, s2)` searches for S2 characters in S1. (Target=S2, Source=S1).
This yielded "A1" in manual trace.
`get_greedy_lcs(s2, s1)` searches for S1 characters in S2. (Target=S1, Source=S2).
This yielded "A12E" in manual trace.
So yes, implementing both covers the directions.

Is it possible to have a better hybrid?
Maybe iterate through S1 and check availability in S2?
That would mean `get_greedy_lcs(s1, s2)` is not quite the "match S2 to S1" but rather "match S1 to S2"?
No. The function signature `get_greedy_lcs(source, target)` explicitly implements matching `target` against `source`.
The logic `for c in target` means we want to cover as much of `target` as possible using `source`.
So it prefers preserving the order of `target` (since we iterate S2) while fitting into `source` (by finding first available slot).
If we swapped arguments `get_greedy_lcs(s2, s1)`, we prioritize preserving `s1`'s order (iterating S1) while fitting into S2.
This matches the intuition: Try to follow S2's characters (preserving S2 order) -> Good for density in S2?
Try to follow S1's characters (preserving S1 order) -> Good for density in S1?
This seems balanced.

One final check: Can I modify `char_ptrs` to optimize further?
Currently, `char_ptrs` array is used to remember where we were in `pos` lists.
Since `target` is processed once, `char_ptrs` effectively keeps a cache.
Correct.

Wait, one detail about `pos` building.
`pos` stores indices.
Indices are `int`.
If string is very large, `pos` consumes RAM.
Is there any other way?
If I don't build `pos` for S2?
For S1->S2 (iterate S1, match S2):
I need to know where S2 has character `S1[i]`.
If I build `pos` for S2 (Target), I can search S2 for S1's chars.
Wait. `get_greedy_lcs(source, target)`:
We iterate `target`.
For each `c` in `target`, we need `pos[c]` (indices in `source`).
So yes, we need `pos` for `source`.
So we build `pos` for `S1` (Source=S1) when solving for Target=S2.
And we build `pos` for `S2` (Source=S2) when solving for Target=S1.
Correct.

The code seems solid.

One small optimization: `res.reserve`.
`min(s1.size(), s2.size())` is a safe bet.
Also, `ios_base::sync_with_stdio(0); cin.tie(0);`.
Include `<cstdio>` not needed if using `cin`.
Include `<vector>`, `<string>`, `<algorithm>`, `<iostream>`.

One tricky thing: `string z1`.
If S1 and S2 are large, creating `z1` string takes memory.
If `z1` is huge (up to min(N,M)), it's fine (fits in 512MB).
Wait, if L is close to 10^7, string takes 10MB. Fine.

Final check: The problem asks to "wrap the code in ```cpp and ```".
"Your response should ONLY contain the C++ code".

```cpp
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

using namespace std;

// Helper function to find LCS approximation using greedy strategy
// Builds a position map for 's1_source', then tries to match characters of 's2_target'
// s1_source is the string we search in.
// s2_target is the string we try to preserve order of.
string get_approx_lcs(const string& s1_source, const string& s2_target) {
    // Frequency count to optimize vector resizing
    int freq[256] = {0};
    for (char c : s1_source) {
        freq[(unsigned char)c]++;
    }

    // pos maps character to a sorted list of indices in s1_source
    vector<vector<int>> pos(256);
    for (int c = 0; c < 256; ++c) {
        if (freq[c] > 0) {
            pos[c].reserve(freq[c]);
        }
    }

    for (int i = 0; i < s1_source.size(); ++i) {
        pos[(unsigned char)s1_source[i]].push_back(i);
    }

    string res = "";
    if (s2_target.empty()) return res;
    // Heuristic reservation
    res.reserve(min(s1_source.size(), s2_target.size())); 

    // Tracks the last index added to res (must be strictly greater than previous index in s1_source)
    int last_idx = -1;

    // Pointer to track the current position in pos[c] for each char c
    vector<int> ptrs(256, 0);

    for (char c : s2_target) {
        unsigned char uc = (unsigned char)c;
        
        // If char not present in source, skip
        if (pos[uc].empty()) continue;

        // Find the first occurrence of c in source after last_idx
        // We start searching from where we left off for this character
        int p_idx = ptrs[uc];
        
        while (p_idx < pos[uc].size() && pos[uc][p_idx] <= last_idx) {
            p_idx++;
        }

        if (p_idx < pos[uc].size()) {
            // Found a match
            res += c;
            last_idx = pos[uc][p_idx];
            // Advance pointer for next time we encounter this char in s2_target
            ptrs[uc] = p_idx; 
            // Note: we don't increment ptrs[uc] manually because the next time we enter loop,
            // the condition pos[uc][p_idx] <= last_idx will catch it (since pos[uc][p_idx] == old_last_idx)
            // But wait, last_idx is updated to pos[uc][p_idx].
            // So pos[uc][p_idx] is now <= new last_idx (equal).
            // The loop will skip it.
            // If we don't increment ptrs[uc], it remains at p_idx.
            // Inside loop, it checks <= last_idx. It matches.
            // Then p_idx++.
            // So logic is sound without manual increment, but manual increment saves one check?
            // Wait. If I leave ptrs[uc] at p_idx.
            // Next time we see 'c'.
            // We check pos[p_idx] <= last_idx (which is pos[p_idx]).
            // True. Increment p_idx.
            // So one wasted check.
            // Better to update ptrs[uc] = p_idx;
            // Wait, what if I update ptrs[uc] = p_idx + 1?
            // Then I must check if pos[p_idx+1] <= last_idx?
            // pos[p_idx+1] > pos[p_idx] = last_idx.
            // So it would be > last_idx.
            // So ptrs[uc] = p_idx + 1 would be valid next start?
            // YES.
            // So setting ptrs[uc] = p_idx + 1 skips the useless check.
            // However, I need to make sure I am not skipping a valid match if I shouldn't.
            // We used the *earliest* valid match.
            // We want to find the *next* match for the NEXT occurrence of 'c'.
            // The next match must be > current_match_idx.
            // Since we consumed the match at p_idx, the next valid candidate is at >= p_idx + 1.
            // So ptrs[uc] = p_idx + 1 is safer/cleaner.
            // But wait, if I set it to p_idx + 1, I must ensure that `pos[uc][p_idx+1]` (if exists)
            // satisfies `> last_idx` (current).
            // Since indices are sorted and distinct, `pos[p_idx+1] > pos[p_idx]`.
            // And `pos[p_idx] == last_idx`.
            // So `pos[p_idx+1] > last_idx`.
            // So yes, we can safely set `ptrs[uc] = p_idx + 1`.
            // Wait, the loop condition `pos[uc][p_idx] <= last_idx` is meant to skip occurrences that are too early.
            // Since we matched exactly `last_idx` at `p_idx` (wait, `last_idx` was *before* match).
            // Ah.
            // Before match: `last_idx` was previous match.
            // We found `p_idx` such that `pos[p_idx] > last_idx` and it is first such.
            // New `last_idx` becomes `pos[p_idx]`.
            // Next time we see `c`. We need occurrence `> new_last_idx`.
            // Occurrences at `pos[p_idx]` is `== new_last_idx` (too early/equal).
            // So we need to skip it.
            // So yes, advancing `ptrs[uc]` to `p_idx + 1` means we won't check it again.
            // We will check `p_idx + 1`.
            // And since `pos[p_idx+1] > pos[p_idx]`, it satisfies `> new_last_idx`.
            // So manual increment is correct and efficient.
            
            // WAIT. What if we used the loop to skip?
            // Loop skips occurrences <= last_idx.
            // If I set ptrs[uc] = p_idx + 1, I skip checking `p_idx` manually.
            // Is it safe? Yes, because `pos[p_idx]` is now `<= last_idx` (specifically `==`).
            // So it would be skipped anyway.
            // So manual increment is valid.
            // BUT. What if `p_idx` was incremented by loop?
            // `while (...) p_idx++;`
            // `p_idx` ends up pointing to valid index.
            // So `pos[p_idx] > last_idx` (old).
            // Then `last_idx` updates.
            // Next time, we need `> new last_idx`.
            // `pos[p_idx] == new last_idx`. So it is skipped.
            // So `ptrs[uc]` should be `p_idx`.
            // Next time `ptrs` points to `p_idx`. Loop `while` executes, checks `==`, increments.
            // This is correct.
            // If I set `ptrs[uc] = p_idx + 1` manually.
            // Next time `ptrs` points to `p_idx+1`. Loop condition checked for `p_idx+1`.
            // Is it possible `pos[p_idx+1]` is `> last_idx`? Yes.
            // Is it possible `pos[p_idx+1]` is `<= last_idx`? No, strictly greater.
            // So we don't need the loop.
            // This implies `ptrs[uc] = p_idx + 1` is optimization.
            // But wait. `p_idx` is `pos[uc].begin` iterator? No index.
            // Yes.
            // So I will use `ptrs[uc] = p_idx + 1`?
            // Wait. `p_idx` is the index in the vector `pos[uc]`.
            // `ptrs[uc]` also stores index in `pos[uc]`.
            // Initial `ptrs` is 0.
            // Loop advances `p_idx` to valid.
            // Match found.
            // If I set `ptrs[uc] = p_idx + 1`.
            // Next time, `p_idx` starts at `ptrs[uc]`.
            // It checks `pos[p_idx] <= last_idx`.
            // If `pos[p_idx+1]` (the one we pointed to) is valid, loop doesn't run.
            // Wait. `pos[p_idx+1]` is definitely `> pos[p_idx] == last_idx`.
            // So `pos[p_idx+1] <= last_idx` is False.
            // So loop terminates immediately.
            // So yes, `p_idx + 1` is safe.
            // BUT wait. Is it possible `pos` has gaps? No, it stores all indices.
            // Wait, what if I didn't match anything?
            // `if (p_idx < pos.size())` block executes.
            // So `p_idx` is valid.
            // What if `p_idx` was the last element?
            // `ptrs` becomes size. Loop terminates next time.
            // Correct.
            
            // WAIT. What if I didn't match? (i.e. `p_idx >= pos.size()`).
            // Then `if` block not entered. `ptrs[uc]` remains as is.
            // If `ptrs` reached end, it stays end.
            // If `ptrs` was at middle but not matched yet?
            // Wait. `p_idx` advances.
            // If `p_idx` reached end. We don't match.
            // `ptrs` stays at end.
            // Next time, `p_idx` starts at end. Loop doesn't run. `if` fails.
            // Correct.
            
            // There is a risk.
            // If I use `ptrs[uc] = p_idx + 1`, I am assuming `p_idx` is the *consumed* index.
            // Is it? Yes.
            // So `ptrs[uc] = p_idx + 1` seems safer/faster than leaving it.
            // Let's verify.
            // S1: A(0), A(10). S2: A.
            // `ptrs[A]` = 0.
            // Loop `pos[0]` (0) <= -1 (False).
            // Match `0`. `last=0`.
            // `ptrs[A]` becomes 1.
            // S2: A.
            // `p_idx` = 1.
            // `pos[1]` (10) <= 0 (False).
            // Match `10`. `last=10`.
            // `ptrs[A]` becomes 2.
            // Works.
            
            // S1: A(0), A(10). S2: A, A.
            // 1st A: `p_idx`=0. Match 0. `ptrs`=1. `last`=0.
            // 2nd A: `p_idx`=1. Loop check `pos[1]`(10) <= 0? False.
            // Match 10. `ptrs`=2. `last`=10.
            // Works.
            
            // Wait, what if `while` loop ran?
            // S1: A(0), A(2). S2: A, A.
            // prev `last`=10.
            // Wait, if we had processed some other char in S2 before?
            // S2: B...
            // S1: A(0), B(5), A(2)? Impossible, indices must increase.
            // Assume S1: A(0), A(10). S2: B(0), A(0).
            // `last` for A is initially -1.
            // Process B. Skip. `ptrs[B]` stays 0.
            // Process A. `p_idx`=0. `pos[0]=0` <= -1 False.
            // Match A(0). `last`=0. `ptrs`=1.
            // Next A. `p_idx`=1. `pos[1]=10` <= 0 False.
            // Match A(10).
            
            // Is it possible `pos[p_idx] <= last_idx` is true when `p_idx` points to a valid future match?
            // No, because `pos` is sorted. `pos[p_idx]` must be `> last_idx` to be valid.
            // If `pos[p_idx] <= last_idx`, it is invalid (out of order).
            // So we must skip.
            // Since `ptrs` starts at 0 and grows, and `last_idx` grows.
            // The skipped indices are permanently skipped for THIS char.
            // Because `last_idx` never decreases.
            // So once skipped, they stay skipped.
            // Thus, `ptrs[uc] = p_idx + 1` is perfectly correct because `pos[p_idx]` (the consumed one) is `> last_idx_prev`.
            // But `pos[p_idx] == last_idx_curr`.
            // So `pos[p_idx] <= last_idx_curr` is True.
            // So it would be skipped if we kept it.
            // So skipping it is just optimization.
            
            // Wait. Is `pos[p_idx]` the one we used?
            // Yes, `res += c; last_idx = pos[uc][p_idx];`
            // So `pos[uc][p_idx] == last_idx`.
            // So next time, `pos[uc][p_idx] <= last_idx` is True.
            // So `p_idx++` happens.
            // So `p_idx` moves to `p_idx + 1`.
            // So effectively `ptrs[uc]` becomes `p_idx + 1`.
            // So either way works.
            // But if I set `ptrs[uc] = p_idx + 1` manually, I avoid the `while` loop execution for that element?
            // Actually, `while` loop condition is checked. `p_idx` is `ptrs`.
            // If I set `ptrs` to `p_idx + 1`.
            // Next time `p_idx` starts at `p_idx + 1`.
            // Check `pos[p_idx+1] <= last_idx`.
            // Since `pos[p_idx+1] > pos[p_idx] == last_idx`, it is False.
            // Loop terminates instantly.
            // So yes, saving 1 comparison per match.
            // Not critical, but nice.
            // Wait. `ptrs[uc] = p_idx + 1`.
            // Does `p_idx` variable go out of scope? No, local.
            // I must update `ptrs[uc]`.
            // Okay.
            ptrs[uc] = p_idx + 1; 
            // Wait. If I update `ptrs` INSIDE the if block.
            // And initialize `p_idx = ptrs[uc]` OUTSIDE.
            // If I increment `p_idx` inside while.
            // At end of loop, `p_idx` points to matched element.
            // Then I update `ptrs[uc]` to `p_idx + 1`.
            // This is consistent.
            // Wait. If `p_idx` was already at `pos.size()` (not matched), we don't enter if.
            // `ptrs` remains at `p_idx` (which was `ptrs`).
            // Correct.
            
            // Wait. `p_idx` is local copy of `ptrs[uc]`.
            // I use `ptrs[uc]` for `p_idx`.
            // `p_idx` is incremented in loop.
            // If matched: `ptrs[uc] = p_idx + 1`.
            // Next time `p_idx` starts at new `ptrs[uc]`.
            // Seems good.
            
            ptrs[uc] = p_idx + 1; 
        }
    }
    return res;
}
```
Wait, one small fix.
`pos[uc].empty()` check is there.
If `pos` is not empty, we enter.
We initialize `p_idx = ptrs[uc]`.
If `ptrs[uc]` is 0 (start).
Loop runs.
Match found.
`ptrs[uc]` updated.
Wait, `ptrs` should be declared outside loop?
Yes, `vector<int> ptrs(256, 0);` declared before loop.
Yes.

Wait, the manual increment optimization:
If I have `p_idx` inside `while`.
After while, `p_idx` is valid.
If I set `ptrs[uc] = p_idx + 1`.
Wait. If `p_idx` is `pos.size()`, I don't match.
Then `if (p_idx < pos.size())` is false.
`ptrs` is NOT updated.
It retains its old value (which was `< size` maybe?).
Wait. If `ptrs[uc]` was `size - 1`.
Loop `while` increments to `size`.
`p_idx == size`. No match.
`if` false. `ptrs` stays `size - 1`.
Next time `c` comes. `p_idx = size - 1`.
Loop runs. `pos[size-1] <= last`. True. Increments to `size`.
Still no match.
`ptrs` stays `size-1`.
Infinite loop?
NO. `p_idx` is a local variable `int p_idx`.
But `ptrs[uc]` is NOT updated.
Wait. If I update `ptrs` inside the IF.
If no match, I don't update `ptrs`.
So `ptrs` remains at `size - 1`.
Next time, loop runs again, increments `p_idx` to `size`.
It repeats forever!
YES.
I MUST update `ptrs` if I reach the end, OR use `ptrs` as the `p_idx` directly and not reset it?
No. `ptrs` acts as cursor.
If we exhaust cursor (reach end), we should not leave `ptrs` at `size - 1`.
We should leave it at `size`?
If I leave it at `size`, next time `p_idx = size`. Loop condition `size < size` false.
Terminates instantly.
So if no match (i.e. cursor reached end), I should update `ptrs` to `pos.size()`.
So `ptrs[uc]` should be updated whenever we finish looking for `c`.
Logic:
We search `c` starting from `ptrs[uc]`.
Eventually `ptrs[uc]` will point to something that is too late, or end.
We must advance `ptrs[uc]` to the position where we left off.
If we found a match at `matched_idx`. Next start is `matched_idx + 1`.
If we didn't find a match, it means all indices from `ptrs[uc]` to end were skipped (invalid due to `last_idx`).
So we should advance `ptrs[uc]` to end?
Wait. If we skip indices `x` such that `pos[x] <= last`.
Since `last` increases, eventually we might reach end.
So yes, we should update `ptrs[uc]` to `pos.size()`.
So `ptrs[uc]` should ALWAYS be updated to `p_idx`?
Wait. `p_idx` at end of `while` is the first valid index?
No. `p_idx` after `while` is the first index satisfying `pos[p_idx] > last`?
Wait. `while (condition) p_idx++`.
Loop terminates when `p_idx == size` OR `pos[p_idx] > last`.
So `p_idx` is indeed the first valid index (or `size`).
So `ptrs[uc] = p_idx`?
Wait. If `pos[p_idx] > last`, we matched it.
Then we use it.
Then next time we need to start AFTER `pos[p_idx]`.
So `ptrs` should be `p_idx`.
Wait. `ptrs` is updated to `p_idx`.
Wait. If I update `ptrs` to `p_idx`.
Next time `p_idx` starts at `p_idx` (value).
We check `pos[p_idx] > last`.
But `last` is now `pos[p_idx]`.
So `pos[p_idx] <= last` is TRUE.
So loop runs again.
So `ptrs` must be `p_idx + 1` (since we used `p_idx`).
Wait. If `pos[p_idx]` was used, we need to skip it next time.
So `ptrs` should point to next.
If we didn't use it (skipped it because `pos[p_idx] <= last`), then `p_idx` became `p_idx + 1` in loop.
So `ptrs` should point to `p_idx` (new value).
So in ALL cases, `ptrs` should be updated to `p_idx` (after loop)?
Wait. If we matched. `p_idx` is valid.
If we use `p_idx`. Next start should be `p_idx + 1`.
So `ptrs` should be `p_idx + 1`.
Wait. What if `p_idx` came from loop?
Say loop ended at `p_idx` (valid).
We used it.
Next start `p_idx + 1`.
What if loop terminated because `p_idx == size` (didn't match).
Next start `size`.
So `ptrs` should be `p_idx` (final value)?
Wait. If `p_idx` is `size`.
We didn't match.
Next time `ptrs` is `size`. Loop terminates.
So `ptrs = p_idx` works if `p_idx` is the *result* of the loop.
But if `p_idx` was `ptrs` initially, and we incremented it in loop.
Wait.
Initial `p_idx = ptrs[uc]`.
We check `while (p_idx < size && pos <= last) p_idx++`.
At exit `p_idx` is either `size` or first valid index.
If `p_idx < size` (Match found):
   We use `pos[p_idx]`.
   We need next search to start at `p_idx + 1`.
   So `ptrs[uc] = p_idx + 1`.
If `p_idx == size` (No match):
   We need next search to start at `size`.
   So `ptrs[uc] = size`.
   Which is same as `p_idx`.
   
   Wait. Is `p_idx + 1` always correct for matched case?
   Yes.
   Is `p_idx` (which is `size`) correct for non-matched case?
   Yes.
   
   BUT. Is it possible `p_idx` was valid but we chose not to match?
   No, greedy matches if found.
   
   So logic:
   Run loop.
   If match (i.e. `p_idx < size`):
      `ptrs[uc] = p_idx + 1`.
   Else:
      `ptrs[uc] = p_idx` (which is `size`).
      
   Actually, simpler: `ptrs[uc] = p_idx` IF we match?
   No. `ptrs[uc]` must point to next.
   If I use `ptrs[uc] = p_idx + 1` when match.
   When no match (`p_idx == size`), `ptrs[uc] = size + 1`? No `size`.
   So `ptrs[uc] = (match ? p_idx + 1 : p_idx)`.
   Wait. If `p_idx` was valid index, `p_idx` is used. Next is `p_idx+1`.
   If `p_idx` was `size`, next is `size`.
   Wait. What if `p_idx` was `size` initially?
   `ptrs` was `size`.
   `p_idx` starts `size`.
   `while` condition fails immediately.
   No match.
   `ptrs[uc] = size`.
   Consistent.
   
   Wait. Is there a case where `p_idx` is valid but we skipped it?
   No, if `pos[p_idx] > last` (valid), loop doesn't skip it.
   Wait. If `pos[p_idx] <= last`. Loop skips. `p_idx` increments.
   Eventually loop stops.
   If stopped at `size`. No match. `ptrs=size`.
   If stopped at valid `k`. Match. `ptrs=k+1`.
   
   Wait. Is it possible to have `ptrs[uc]` become larger than `pos.size()`?
   Only if I do `+1`.
   If `pos.size() == k`. Match at `k-1`. `ptrs` becomes `k`.
   Next time `p_idx=k`. Loop check `k<size` (false).
   Terminates. `if` check `k<size` (false).
   `ptrs` becomes `k`.
   Next time `ptrs` is `k`.
   Consistent.
   
   Wait. `ptrs[uc]` init to 0.
   Suppose `size=5`. Match at 4. `ptrs` becomes 5.
   Next `size`. Loop fails. No match.
   `ptrs` becomes 5.
   So correct.
   
   So logic:
   `p_idx = ptrs[uc];`
   `while(p_idx < size && pos <= last) p_idx++;`
   `if(p_idx < size) { match; ... ptrs[uc] = p_idx + 1; }`
   `else { no_match; ptrs[uc] = p_idx; }` // or just skip.
   Actually `ptrs[uc]` update is only needed if we want to optimize next search.
   If I don't update `ptrs[uc]` in `else` case.
   `p_idx` (local) was incremented to `size`.
   `ptrs[uc]` remains old value.
   Next time, `p_idx` starts at old value.
   Loop runs again. `while` increments to `size`.
   This repeats forever?
   NO. `p_idx` is incremented in `while`.
   `while` condition `p_idx < size`.
   If `ptrs` points to `pos` index that was skipped.
   `pos` index was skipped because `<= last`.
   Now `last` increased.
   Maybe `pos` index is now valid?
   Wait. If `pos[x] <= last_prev`.
   Now `last_new > last_prev`.
   It is still `pos[x] <= last_prev < last_new`.
   So `pos[x]` remains `<= last_new`.
   So it will still be skipped.
   So yes, once skipped, it is done.
   So `ptrs` must be advanced.
   So in `else` branch (no match), we MUST update `ptrs[uc]`.
   Since `p_idx` reached `size`, `ptrs[uc] = size` is correct?
   Wait. `p_idx` in loop reaches `size`.
   So `ptrs[uc] = p_idx` (which is `size`).
   So yes.
   So:
   `if (p_idx < size) { res += ...; ptrs[uc] = p_idx + 1; } else { ptrs[uc] = p_idx; }`
   Wait. If `p_idx` is `size`, `p_idx` is valid `int`.
   `ptrs` becomes `size`.
   Next time `p_idx` starts `size`. `while` check `size < size` false.
   `else` triggers. `ptrs` becomes `size`.
   Infinite?
   If `p_idx == size`. `else` triggered. `ptrs` becomes `size`.
   Next loop `p_idx` starts `size`.
   `if (size < size)` false. `else` `ptrs` becomes `size`.
   So `ptrs` stays `size`.
   Logic holds.
   
   Wait. Why do I need `ptrs` update in `else`?
   If I don't update `ptrs` in `else`.
   `ptrs` stays at some index `k < size` that was skipped.
   Next time, we try `k`.
   Check `pos[k] <= last`.
   It is true.
   So we increment `p_idx`.
   Eventually `p_idx` hits `size`.
   Loop terminates.
   Then we hit `else`.
   If `else` updates `ptrs` to `size`.
   Then next time `p_idx` starts `size`.
   So it converges.
   But if `else` DOES NOT update `ptrs`.
   `ptrs` stays `k`.
   Next time, we try `k`. Skip.
   Repeat until `k` runs out?
   If `ptrs` stays `k`.
   `while` loop runs, increments `p_idx` from `k` to `size`.
   `if` check fails.
   `else` does NOT update `ptrs`.
   So `ptrs` stays `k`.
   Next time `p_idx` starts `k`.
   Loop runs again.
   We repeat the check for `k` every time `c` appears in `s2`.
   If `c` appears 100 times in `s2`, and `k` is far behind.
   Each time we loop `pos[k] <= last`?
   Wait. `last` increases.
   So `pos[k] <= last` is true.
   So `p_idx` increments.
   Wait. If I don't update `ptrs`. `ptrs` is `k`.
   Loop starts. `p_idx` starts `k`.
   Inside loop, `p_idx` increments to `size`.
   So effectively we skipped all.
   BUT we spent time iterating from `k` to `size`!
   We waste time `size - k`.
   If we update `ptrs` to `size`, we spend $O(1)$ (loop check fails).
   So updating `ptrs` is CRITICAL for performance if `c` is repeated.
   
   So I must update `ptrs[uc] = p_idx` at end?
   Wait.
   If `p_idx < size` (Matched):
      Used `pos[p_idx]`. Next search must start at `p_idx + 1`.
      `ptrs[uc] = p_idx + 1`.
   If `p_idx == size` (Skipped all):
      Next search starts at `size`.
      `ptrs[uc] = size`.
   
   Wait. If I update `ptrs` to `size` in `else`.
   Is `size` a valid index?
   It's out of bounds for `pos` vector.
   But `while` check `p_idx < pos.size()` handles it.
   So yes.
   
   So refined code:
   ```cpp
   int p_idx = ptrs[uc];
   while (p_idx < pos[uc].size() && pos[uc][p_idx] <= last_idx) {
       p_idx++;
   }
   if (p_idx < pos[uc].size()) {
       res += c;
       last_idx = pos[uc][p_idx];
       ptrs[uc] = p_idx + 1; 
   } else {
       ptrs[uc] = pos[uc].size(); 
       // Wait. If I set ptrs to size. 
       // Next time p_idx starts at size. 
       // while loop condition p_idx < size fails.
       // else block runs again. Sets ptrs to size.
       // Loop doesn't run. Constant time.
       // GOOD.
   }
   ```
   Wait. If I use `ptrs[uc] = p_idx + 1` in `if`.
   Is it safe?
   Yes.
   
   However, there's a catch.
   `ptrs[uc]` is updated.
   Does `p_idx` variable need to be `long long`? No, `int` fine.
   
   Wait. `p_idx` in `while` loop.
   If `pos` is empty, size 0.
   `p_idx = 0`. `0 < 0` false.
   `else` branch `ptrs[uc] = 0`.
   Correct.
   
   So this logic is $O(1)$ amortized and handles repeated chars efficiently.

One minor issue:
If `p_idx` reaches `pos.size()` in `if` block (impossible since check `p_idx < size`).
Wait. In `if` block, `p_idx` is a valid index. `p_idx < size`.
So `p_idx + 1` is at most `size`.
If `p_idx` was `size - 1`. `ptrs` becomes `size`.
Next time `size < size` false.
OK.

Wait, I should check `ptrs` update logic in `else`.
`ptrs[uc] = pos[uc].size()`.
Is this strictly necessary?
If I don't update `ptrs` in `else`.
Next time `p_idx` starts at old value.
Since `old_value` led to skip of all remaining, it means `pos[old] <= last_old`.
Now `last` is `last_new`.
`pos[old]` might still be `<= last_new`.
Actually, if `p_idx` reached `size`. It means ALL remaining were skipped.
Meaning for all `k >= old_val`, `pos[k] <= last`.
Since `last` only increases, `pos[k] <= last` will always hold for `last >= last`.
So all these `k` are irrelevant for ANY future `last` (since `last` will be $\ge$ previous).
So yes, we can discard them.
So updating `ptrs` is essential to avoid re-scanning the tail.
So `ptrs[uc] = pos[uc].size()` is correct.

Wait. `p_idx` in `if` is `p_idx + 1`.
`p_idx` in `else` is `pos.size()`.
They are consistent if we think of "state after scanning".
If match found at `i`. State is `i+1`.
If no match found (all scanned to end). State is `size`.
Yes.
So `ptrs[uc] = (match ? p_idx + 1 : pos.size())`.
Wait. If `p_idx` is valid index (used). `pos[p_idx]` is used.
`last` becomes `pos[p_idx]`.
We need next occurrence `> last`.
Next occurrence must be at index `> p_idx` (since `pos` sorted).
Wait. `pos[p_idx+1]` is next value.
Is it possible `pos[p_idx+1] <= last`?
`pos[p_idx+1] > pos[p_idx] == last`.
So no. `pos[p_idx+1] > last`.
So `p_idx + 1` is a valid start point for search (and loop check will pass immediately).
So yes.

I'll put this logic into the code.

Double check the `ptrs` declaration.
`vector<int> ptrs(256, 0);`
`unsigned char` casting for array index is good.

Final check on includes.
`<iostream>`, `<vector>`, `<string>`, `<algorithm>`, `<climits>` (maybe not needed).
`min` is in `<algorithm>`.

One more thing. `s1_source` and `s2_target` strings are large.
Passing by `const reference` is done.
Good.

```cpp
// Final check of the code logic inside loop
// while (p_idx < pos[uc].size() && pos[uc][p_idx] <= last_idx)
// p_idx++;
// if (p_idx < pos[uc].size()) {
//    res += c;
//    last_idx = pos[uc][p_idx];
//    ptrs[uc] = p_idx + 1;
// } else {
//    ptrs[uc] = pos[uc].size();
// }
// This looks solid.
```
</think>

```cpp
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

using namespace std;

/**
 * Computes an approximate Longest Common Subsequence (LCS).
 * Uses a greedy strategy where it attempts to match characters of the target string
 * with the earliest valid occurrences in the source string.
 * 
 * @param source_s String to search in (S1).
 * @param target_s String to match patterns of (S2).
 * @return A common subsequence formed by greedy matching target_s onto source_s.
 */
string get_approx_lcs(const string& source_s, const string& target_s) {
    // Map each character to a vector of its indices in source_s
    // Using 256 to cover standard ASCII including extended if needed, though only [A-Z, 0-9] used.
    int freq[256] = {0};
    for (char c : source_s) {
        freq[(unsigned char)c]++;
    }

    vector<vector<int>> pos(256);
    // Reserve memory based on frequency to minimize reallocations
    for (int c = 0; c < 256; ++c) {
        if (freq[c] > 0) {
            pos[c].reserve(freq[c]);
        }
    }

    for (int i = 0; i < source_s.size(); ++i) {
        pos[(unsigned char)source_s[i]].push_back(i);
    }

    string res = "";
    if (target_s.empty()) return res;
    res.reserve(min(source_s.size(), target_s.size())); 

    // last_idx tracks the index in source_s of the character previously added to the result
    // Must be strictly greater than this index for the next match to maintain subsequence property.
    int last_idx = -1;

    // ptrs tracks the current search position in the pos vectors for each character
    // to avoid rescanning indices that are already ruled out.
    vector<int> ptrs(256, 0);

    for (char c : target_s) {
        unsigned char uc = (unsigned char)c;
        
        if (pos[uc].empty()) continue;

        int p_idx = ptrs[uc];
        
        // Skip occurrences that are at or before the last matched index
        while (p_idx < pos[uc].size() && pos[uc][p_idx] <= last_idx) {
            p_idx++;
        }

        if (p_idx < pos[uc].size()) {
            // Valid occurrence found
            res += c;
            last_idx = pos[uc][p_idx];
            // Move pointer to next occurrence for next time we see this character in target_s
            ptrs[uc] = p_idx + 1;
        } else {
            // No valid occurrence found, mark this character as exhausted in source_s
            ptrs[uc] = pos[uc].size();
        }
    }
    
    return res;
}

int main() {
    // Optimize I/O operations for large input
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    string s1, s2;
    // Reading input. Since strings are guaranteed alphanumeric and no spaces, cin >> s works.
    if (!(cin >> s1)) return 0;
    if (!(cin >> s2)) return 0;

    // Compute two greedy candidates:
    // 1. Match S2's characters against S1 (prioritize preserving S2 order)
    // 2. Match S1's characters against S2 (prioritize preserving S1 order)
    // Taking the longest result improves robustness against bad ordering cases.
    string z1 = get_approx_lcs(s1, s2);
    string z2 = get_approx_lcs(s2, s1);

    if (z1.length() >= z2.length()) {
        cout << z1 << "\n";
    } else {
        cout << z2 << "\n";
    }

    return 0;
}
```
````

---

## 直接SFT 样本(sample_idx=1,得分 0.0,completion_tokens=2720)

````
The problem is asking for a heuristic solution to find a common subsequence of two very large strings. Since the exact LCS algorithm would take O(N*M) time and space which is way too much, we need a greedy approach.

The key insight is that we can use the **Two-Pointer Greedy Method** on the character positions:
1. For each string, build a position map so we can quickly find where each character appears
2. Start from position 0 in both strings
3. Look at the characters at current positions - if they match, include them in our result and move both pointers forward
4. If they don't match, we try to advance one pointer while keeping the other fixed (to maximize alignment opportunities)

The trick is deciding which pointer to advance when characters don't match. The optimal strategy is to move whichever pointer has the character that appears more frequently in the remaining portion of its string - this gives us more chances to find matches later.

Let me implement this:

```cpp
#include <iostream>
#include <string>
#include <vector>
using namespace std;

const int MAXN = 10000000;
char str[MAXN + 5];
int pos[26][MAXN / 10];
int count[26];
bool present[26];
int next_pos[MAXN + 5];

// Find all positions of each character
void get_pos(const char* s, int n) {
    // Clear previous data
    for(int i = 0; i < 26; i++) {
        count[i] = 0;
        for(int j = 0; j <= n / 10; j++) pos[i][j] = -1;
        present[i] = false;
    }
    
    // First pass: count occurrences of each character
    for(int i = 0; i < n; i++) {
        int c = s[i] - 'A';
        count[c]++;
        present[c] = true;
    }
    
    // Second pass: store all positions for each character
    // We divide by 10 because there are at most ~10^7 chars per string
    int start = 0;
    for(int c = 0; c < 26; c++) {
        for(int j = start; j < start + count[c] && j < n / 10; j++) {
            pos[c][j] = -1;
        }
        start += count[c];
    }
}

// Main solution using two-pointer greedy approach
int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    cin >> str;
    
    cin >> str;
    
    int len1 = strlen(str);
    int len2 = strlen(str);
    
    get_pos(str, len1);
    get_pos(str, len2);
    
    // Initialize position tracking
    for(int i = 0; i < MAXN; i++) next_pos[i] = -1;
    
    // Two-pointer greedy matching
    int p1 = 0, p2 = 0;
    int ans_len = 0;
    
    while(p1 < len1 && p2 < len2) {
        char c1 = str[p1];
        char c2 = str[p2];
        
        if(c1 == c2) {
            next_pos[ans_len] = p1;
            ans_len++;
            p1++;
            p2++;
        } else if((int)(c1 - '0') > (int)(c2 - '0')) {
            p1++;
        } else {
            p2++;
        }
    }
    
    cout << "";
    return 0;
}
```

I see multiple critical errors in my implementation that would prevent any meaningful solution generation.

1. Both input reads reference `str` instead of distinct variables like `s1` and `s2`. This immediately corrupts the data since the second read overwrites the first string entirely.

2. I'm reassigning `strlen()`'s return value back to `str` itself - `str = strlen(something)` attempts to assign an integer length value to a character array, which creates a compilation error and loses the original string content.

3. The character-to-index conversion needs fixing: `s1[i] - 'A'` only works for uppercase letters. Input includes digits, so this breaks for numeric characters.

To fix these, I need separate string buffers for each input, proper length calculation without assignment, and a robust character mapping function that handles both uppercase letters and digits.
</think>

```cpp
#include <bits/stdc++.h>
using namespace std;

// Maximum string size: 10,000,000
const int MAX_LEN = 10000005;

// Position maps: store all occurrences of each character type
// Each character group gets its own linked list of positions
struct CharMap {
    vector<int> digit;   // positions of digits '0'-'9'
    vector<int> upper;   // positions of uppercase letters 'A'-'Z'
};

// Build position maps for all characters in a string
void buildMaps(const string& s, CharMap& cm) {
    for(char c : s) {
        cm.upper.push_back(0); // placeholder
    }
    
    // Collect all positions for each character type
    string sortedS = s;
    sort(sortedS.begin(), sortedS.end());
    
    int digitPos = 0;
    int upperPos = 0;
    
    // Count how many of each character type exist
    int digitCount = 0;
    int upperCount = 0;
    for(char c : s) {
        if(c >= '0' && c <= '9') digitCount++;
        else if(c >= 'A' && c <= 'Z') upperCount++;
    }
    
    cm.digit.resize(digitCount);
    cm.upper.resize(upperCount);
    
    // Fill with actual positions
    for(size_t i = 0; i < s.length(); i++) {
        if(s[i] >= '0' && s[i] <= '9') {
            cm.digit[digitPos++] = i;
        } else if(s[i] >= 'A' && s[i] <= 'Z') {
            cm.upper[upperPos++] = i;
        }
    }
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    string s1, s2;
    getline(cin, s1);
    getline(cin, s2);
    
    TrimWhitespace(s1);
    TrimWhitespace(s2);
    
    CharMap cm1, cm2;
    buildMaps(s1, cm1);
    buildMaps(s2, cm2);
    
    Result res1 = getResult(cm1, s1);
    Result res2 = getResult(cm2, s2);
    
    string result = "";
    string z = "", best_z = "";
    
    if(res1.score > res2.score || res2.is_subseq1) {
        z = res1.z;
        best_z = res1.z;
    } else {
        z = res2.z;
        best_z = res2.z;
    }
    
    cout << best_z << "\n";
    
    return 0;
}

// Trim leading/trailing whitespace from string
void TrimWhitespace(string& s) {
    auto it = s.find_first_not_of(" \t\n\r");
    if(it == string::npos) { s = ""; return; }
    it = s.find_last_not_of(" \t\n\r");
    s = s.substr(it, s.size() - it);
}

// Find longest common subsequence using dynamic programming with pruning
Result getResult(const CharMap& cm, const string& s) {
    Result res = {};
    string dp(MAX_LEN, '\0');
    bool used[MAX_LEN];
    
    for(auto ch : "abcdefghijklmnopqrstuvwxyz") {
        int idx = hash(ch)[s];
        if(idx != -1) {
            long long cnt1 = countOccurrences(s, ch);
            res.score = max(res.score, score(cnt1));
            
            dp[idx] = s[0];
            res.len = 1;
            res.z = s[0];
        }
    }
    
    for(size_t i = 1; i < dp.size(); i++) {
        dp[i] = max(dp[i], dp[i-1]);
        res.z += dp[i];
        res.len = min(res.len, (size_t)((int)res.z.size()));
        if((int)dp[i].length() > res.len) continue;
    }
    
    res.z = dp[0];
    res.len = dp[0].size();
    res.score = max(res.score, score(res.len));
    
    return res;
}

// Calculate score based on subsequence length
long long score(long long len) {
    return len * 10;
}

// Check if a character exists within given bounds
int exists(const CharMap& cm, int low, int high, char ch) {
    if(ch >= '0' && ch <= '9') {
        auto it = lower_bound(cm.digit.begin(), cm.digit.end(), low);
        return it != cm.digit.end() && *it <= high ? it - cm.digit.begin() : -1;
    } else {
        auto it = lower_bound(cm.upper.begin(), cm.upper.end(), low);
        return it != cm.upper.end() && *it <= high ? it - cm.upper.begin() : -1;
    }
}

// Find longest common subsequence between two strings
Result longestCommonSubsequence(const string& S1, const string& S2) {
    Result res = {};
    int N = S1.size();
    int M = S2.size();
    
    res.len = 0;
    res.z = "";
    
    for(size_t i = 0; i < S1.size(); i++) {
        res.z += S1[i];
        res.len = min(res.len, (int)res.z.size());
    }
    
    return res;
}

// Hash character for fast indexing
unordered_map<char, int> hash_func(char c) {
    unordered_map<char, int> h;
    h['a'] = 0;
    h['b'] = 1;
    h['c'] = 2;
    h['d'] = 3;
    h['e'] = 4;
    h['f'] = 5;
    h['g'] = 6;
    h['h'] = 7;
    h['i'] = 8;
    h['j'] = 9;
    h['k'] = 10;
    h['l'] = 11;
    h['m'] = 12;
    h['n'] = 13;
    h['o'] = 14;
    h['p'] = 15;
    h['q'] = 16;
    h['r'] = 17;
    h['s'] = 18;
    h['t'] = 19;
    h['u'] = 20;
    h['v'] = 21;
    h['w'] = 22;
    h['x'] = 23;
    h['y'] = 24;
    h['z'] = 25;
    return h;
}

// Get hash index for character in string
int hash_function(const char* s, int idx) {
    return hash_func(s[idx]) ^ idx;
}

// Count how many times a character appears in range [low, high]
long long countOccurrences(const string& s, char ch) {
    int left = lower_bound(s.begin(), s.end(), ch) - s.begin();
    int right = upper_bound(s.begin(), s.end(), ch) - s.begin();
    return max(0LL, right - left);
}
```
````
