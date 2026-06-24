# FrontierCS：Distinct Pairwise XOR Set（竞赛算法题） — Start（Qwen3.5-9B instruct，未训练对照）

> **本文件是完整、未截断的真实评测记录。** 模型类别：`start`。
> 评测：FrontierCS（thinking 口径，竞赛算法编程，官方评测器抽最长 ```cpp 块编译跑测试点给连续部分分）。
> 题目内部编号 `problem_idx=11` / 题库目录 `algorithmic/problems/111` / 第 `sample_idx=0` 个采样（共 5 采样/题）。
>
> **本采样得分（FCS score，0–100 连续部分分）：`43.654182399999996`**　|　生成 token 数：`22809`　|　error：`None`

---

## 1. 完整输入（喂给模型的真实 prompt）

模型实际收到的 user prompt = 一段固定的「C++ 竞赛求解」system 引导 + 下面这道题的 `statement.txt` 原文 + `Generate solution code:`。题面原文如下（逐字照抄题库）：

```text
Problem: Distinct Pairwise XOR Set

Time Limit: 1 second
Memory Limit: 512 MB

Description
Given an integer n, find a subset S ⊆ {1, 2, ..., n} such that:
1) For all pairs (a, b) with a, b ∈ S and a < b, the values (a XOR b) are all distinct (i.e., no two different unordered pairs produce the same XOR).
2) |S| ≥ floor(sqrt(n / 2)).

Input
A single integer n (1 ≤ n ≤ 10^7).

Output
- First line: an integer m — the size of the set S.
- Second line: m distinct integers in the range [1, n] — the elements of S, in any order.

Notes
- Any valid S is accepted. You do NOT need to maximize m; you only need m ≥ floor(sqrt(n/2)).
- The pairwise XOR distinctness means the set {a_i XOR a_j | 1 ≤ i < j ≤ m} has size m*(m-1)/2.
- Multiple correct outputs may exist for the same n.
- Print out the sequence with the longest length.

Sample
Input
49
Output
4
1 2 3 4
```

---

## 2. 完整输出（模型的真实生成，逐字照抄，未截断）

> 注：FCS 评测开启 thinking。下面是模型 `text` 字段的**全文**（包含推理与最终代码）。

````text
The problem asks us to find a subset S of {1, 2, ..., n} such that all pairwise XORs of elements in S are distinct, and |S| ≥ floor(sqrt(n/2)).

For the sample input n = 49, the output is size 4 with elements 1, 2, 3, 4.
Let's check:
Pairs from {1, 2, 3, 4}:
(1, 2) -> 3
(1, 3) -> 2
(1, 4) -> 5
(2, 3) -> 1
(2, 4) -> 6
(3, 4) -> 7
XORs: {3, 2, 5, 1, 6, 7}. All distinct. Size of S is 4.
sqrt(49/2) = sqrt(24.5) ≈ 4.94. floor is 4. So m=4 satisfies the condition.
Wait, for n=49, we need m >= 4. The output says 4.

Actually, there's a well-known construction for this type of problem involving Sidon sets or similar structures where differences (or sums, or XORs) are distinct.
However, finding a set where ALL pairwise XORs are distinct is related to "B_h-sets" in terms of XOR basis, but specifically here it means the XOR sum set has no collisions. This is exactly asking for a subset with the "distinct XOR property".
Specifically, if we pick elements such that their most significant bit positions are distinct or they form a structure like powers of 2, we can ensure distinctness.

Consider the binary representation. If we select numbers such that no two pairs have the same XOR sum, it's often easier to construct using properties of bits.
If we take numbers $2^0, 2^1, 2^2, \dots, 2^{k-1}$, then the XOR of any subset is unique. Wait, that's for subset XORs.
Here we need pairwise XORs $a_i \oplus a_j$ for $i < j$.
If we select numbers like $1, 2, 4, 8, \dots$, i.e., powers of 2.
Let $S = \{2^0, 2^1, \dots, 2^{m-1}\}$.
Then $a_i = 2^i$.
$a_i \oplus a_j = 2^i + 2^j$ (since $i \neq j$, no overlap).
Since the indices $i, j$ are unique, the sums $2^i + 2^j$ are distinct for each pair $\{i, j\}$.
Proof: The value is just the number formed by having bits $i$ and $j$ set. Since the set $\{i, j\}$ is different, the resulting integer is different.
So, any set consisting of powers of 2 guarantees distinct pairwise XORs.

How many powers of 2 fit in $n$?
We need $2^k \le n$. So $k \approx \log_2 n$.
For $n = 10^7$, $\log_2(10^7) \approx 23$.
So $m \approx 24$.
The constraint requires $m \ge \lfloor \sqrt{n/2} \rfloor$.
For $n = 10^7$, $\sqrt{n/2} \approx \sqrt{5 \times 10^6} \approx 2236$.
So simply taking powers of 2 is insufficient ($24 < 2236$). We need a larger set.

This looks like the "Sidon set" problem but with XOR operation instead of addition/difference. A set where pairwise XOR sums are distinct is equivalent to saying the set of elements forms a specific additive combinatorial structure over the field GF(2), but since we are restricted to integers up to n, we need an efficient construction.
There is a known construction called the "greedy construction" or simply finding a "maximal distinct XOR set", but we don't need maximal, just satisfying the lower bound.
However, usually, sets with distinct XORs have sizes bounded roughly by $O(\sqrt{n})$ or maybe $O(\sqrt[n]{n})$? No.
In general, for difference sets $a - b$, max size is around $\sqrt{n}$. For XOR sums, it might be similar.
Given $n \le 10^7$ and required size $m \ge \sqrt{n/2}$, we essentially need to output ANY set of size roughly $\sqrt{n}$ satisfying the property.
Since $\sqrt{10^7} \approx 3162$.
Is there a simple pattern?
The sample output suggests {1, 2, 3, 4} works.
Note that {1, 2, 3, 4} contains $\{1, 2, 3, 4\}$.
Wait, $1=01, 2=10, 3=11, 4=100$.
Pairs:
1^2 = 3 (011)
1^3 = 2 (010)
1^4 = 5 (101)
2^3 = 1 (001)
2^4 = 6 (110)
3^4 = 7 (111)
All distinct.
Notice 1, 2, 4 are powers of 2. 3 is 1+2.
Including non-powers of 2 allows increasing density.

There is a conjecture/construction for "Sets with distinct pairwise sums/XORs".
A common heuristic for these types of problems (finding large sets with distinct sums) is to use arithmetic progressions or specific modular arithmetic, but XOR behaves differently.
Actually, for XOR, there is a very standard result:
If we choose numbers such that their XOR sums are distinct, this is equivalent to finding a sequence such that $x_i \oplus x_j = x_u \oplus x_v \implies \{i, j\} = \{u, v\}$.
This is related to "Costas arrays" or similar, but on integers.
However, checking the time limit (1 sec) and $N=10^7$, we cannot run an exponential search. We need an $O(N)$ or $O(\sqrt{N})$ constructive solution.
We need to find $M \approx \sqrt{N/2}$ numbers.
Let's verify if we can just pick the first $K$ numbers or something close?
If we pick $\{1, 2, 3, 4, 5, 6\}$:
1^2 = 3
1^3 = 2
1^4 = 5
1^5 = 4
1^6 = 7
2^3 = 1
2^4 = 6
2^5 = 7  <-- Collision! 1^6 = 7 and 2^5 = 7.
So small ranges do not work directly.

Let's reconsider the requirement: $m \ge \lfloor \sqrt{n/2} \rfloor$.
For $n=10^7$, target $m \approx 2236$.
Can we output $1, 2, 3, \dots, k$ where $k \approx \sqrt{n/2}$? No, as seen above, collisions occur early.

What if we construct elements based on their index?
Let $S_k = \{ a_1, a_2, \dots, a_k \}$.
Maybe we can use a greedy approach to build such a set up to size $K$.
Greedy strategy: Iterate through $x \in 1 \dots n$. Include $x$ in $S$ if adding it doesn't create a duplicate XOR with any existing pair in $S$.
Checking all pairs for every candidate takes $O(|S| \cdot n)$ which is too slow if $|S| \approx 2000$. $2000^2$ checks per step? No, when considering $x$, we compute $x \oplus s$ for all $s \in S$. We need to check if any $(x \oplus s)$ has appeared before as a pair XOR.
To do this efficiently, we maintain a frequency map or boolean array `seen_xor` of size up to $2 \cdot 10^7$ (roughly $2N$ max XOR? No, max XOR of two numbers $< N$ can be slightly less than next power of 2). Max $N = 10^7$, so max XOR is $< 2^{24} \approx 1.6 \times 10^7$.
We can preallocate a `bool` array or `visited` vector.
Algorithm:
1. Initialize `vector<int> S`.
2. Initialize `vector<bool> used_xor` of sufficient size.
3. Start with `S = {1}` (empty set doesn't generate XORs, but singleton is fine). Actually start empty, try 1. Add 1.
   Wait, we need distinct *unordered* pairs.
   Actually, better to start `S` empty. Try adding candidates.
   However, the problem asks for $m \ge \sqrt{n/2}$.
   Is it guaranteed that a greedy set exists?
   Usually, random greedy works well for packing problems, but here we have strict constraints.
   
   Let's check the complexity of greedy:
   Loop $x$ from 1 to $n$.
     Check if adding $x$ violates distinctness.
     Violation happens if there exists $y \in S$ such that $x \oplus y$ is equal to some $z \oplus w$ where $z, w \in S, z < w$.
     Actually simpler: For each $y \in S$, calculate $val = x \oplus y$. If $val$ has already been seen as a pair XOR from previous steps, we cannot add $x$.
     To store seen XORs: Use a boolean array `xor_seen`. `xor_seen[v]` = true if value $v$ is the XOR sum of some pair in current $S$.
     When considering $x$:
       For each $y \in S$:
         val = $x \oplus y$.
         If `xor_seen[val]` is true: collision detected. Stop $x$.
         Else: mark `xor_seen[val]` = true.
       If no collision found for all $y \in S$:
         Add $x$ to $S$.
         Now, update `xor_seen` with new pairs generated by $x$ and existing $y$. But wait, the check `xor_seen[val]` handles old vs new.
         If we pass the check, we need to record the NEW pairs for future comparisons.
         The loop above marks pairs formed with $y \in S$ (where $S$ was the state BEFORE adding $x$). These represent pairs $\{x, y\}$ where one is new.
         Wait, does this handle the case where $val$ comes from TWO OLD pairs?
         Yes. Suppose we added $x$ earlier and $z$ earlier, and $x \oplus z$ was recorded.
         Now consider $w$. If $w \oplus u = x \oplus z$, and we process $w$, we check $w \oplus u$ against `xor_seen`.
         If $u \in S$, we see `xor_seen[w^u]` is true (set when $x, z$ were processed).
         Wait, my logic "For each $y \in S$" only checks if $x \oplus y$ was produced previously.
         It assumes the previous pairs were stored in `xor_seen`.
         When $x$ is accepted, we must add the pairs $(x, y)$ to `xor_seen` for future candidates.
         But also, we need to handle the condition that $a \oplus b$ is distinct from $c \oplus d$.
         The condition "For each $y \in S$, $x \oplus y$ not in `xor_seen`" ensures that no NEW pair $(x, y)$ collides with any OLD pair.
         Because `xor_seen` stores values of ALL old pairs.
         This seems correct.
         
   Complexity:
   We want size $K \approx \sqrt{n/2}$.
   For each element added, we iterate over existing elements.
   Total operations $\approx K^2$.
   With $n=10^7$, $K \approx 2236$. $K^2 \approx 5 \times 10^6$.
   This fits easily within 1 second.
   Wait, memory limit is 512 MB.
   We need an array of size approx $2^{24} \approx 1.6 \times 10^7$ booleans? Or `bitset`?
   Max possible XOR value with inputs up to $10^7$. $10^7 < 2^{24}$. Max XOR $< 2^{24}$.
   Array size $1.6 \times 10^7$ bytes $\approx 16$ MB. Fits.
   
   Is it fast enough to check $x$ up to $n$?
   Iterating $x$ from 1 to $n$ and performing the check.
   But we only need to find $K$ elements. Once we found $K$ elements, we can stop?
   The problem asks for ANY valid S. "Print out the sequence with the longest length" - wait, looking at notes.
   Notes say: "You do NOT need to maximize m; you only need m ≥ floor(sqrt(n / 2))."
   But note also says "Print out the sequence with the longest length." in the prompt text?
   Wait, the user prompt text says: "Output - First line: an integer m — the size of the set S."
   And under Notes: "Print out the sequence with the longest length."
   This implies maximizing $m$ IS preferred if multiple answers satisfy the constraint.
   However, given $K \approx \sqrt{n/2}$, finding a larger set is likely hard/unnecessary. The constraint is just a lower bound.
   Wait, "Print out the sequence with the longest length." is part of the problem description provided by user. Usually competitive programming problems ask for MAX size, but the Note explicitly says "You do NOT need to maximize m". This is contradictory.
   Re-reading: "Notes - Any valid S is accepted. You do NOT need to maximize m; you only need m ≥ floor(sqrt(n / 2))."
   Okay, so I should prioritize validity. If the greedy produces $m > \sqrt{n/2}$, that's good. If it stops earlier, I'm in trouble.
   However, it is known that greedy works reasonably well for this problem (often called the "Zarankiewicz problem" analog for XOR or simply "Distinct XOR Set").
   Given the tight time limit (1s), pure greedy with array lookup is likely the intended solution or a variation.
   
   Optimization:
   Instead of iterating $x$ from 1 to $n$, can we iterate candidates more intelligently?
   Since $S$ grows, $K$ is small (approx 2200).
   Checking if a candidate is valid takes $O(K)$.
   To find the optimal set, maybe random shuffle helps.
   However, deterministic greedy $x=1 \dots n$ works best.
   Actually, just picking the first $K$ that satisfy the condition is easiest.
   What if greedy fails to reach $K$?
   There is a theorem that for any set size $k$, there exists a set of size $k$ with distinct pairwise XORs in range $[1, n]$? Not necessarily true for arbitrary $k$.
   But given the constraint is $\sqrt{n/2}$, this is the typical threshold for such constructions.
   
   Wait, is there a faster way?
   The maximum XOR value could be close to $2^{\lceil \log_2 n \rceil}-1$.
   For $n=10^7$, max XOR is $2^{24}-1 = 16777215$.
   `std::vector<bool>` or `bitset` or raw `char` array.
   Using `vector<char>` to avoid bit manipulation overhead, though `vector<bool>` is specialized and might be slower/faster depending on access. `char` array is safer for speed (no bitwise ops). 16MB `char` array is negligible.
   
   Algorithm refined:
   1. Read $n$.
   2. Limit `max_xor = (1 << 25)`. Just enough to cover all possible XOR results of numbers in $[1, n]$. Since $n < 2^{24}$, $a, b < 2^{24} \implies a \oplus b < 2^{24}$. Safe upper bound.
   3. `used_xor` array of size $2 \cdot 10^7$ initialized to 0.
   4. `S` vector.
   5. Iterate $i$ from 1 to $n$:
      valid = true
      For each $s \in S$:
        val = $i \oplus s$
        If `used_xor[val]`: valid = false; break;
        // Mark the newly formed XOR sum for the existing set plus i
        // Wait, we need to mark pairs involving i AND existing s.
        // These pairs will be queried later.
        // BUT we haven't marked them yet!
        // If we don't mark them now, we won't detect collision with FUTURE pairs.
        // Correct logic:
        // 1. Check if adding $i$ conflicts with any existing pair $(x, y)$ from previous $S$.
        //    This corresponds to checking if $i \oplus s$ equals any previously seen XOR sum.
        //    So we query `used_xor[i ^ s]`.
        // 2. If valid, add $i$ to $S$.
        // 3. Then, for ALL $s \in S_{OLD}$, mark $i \oplus s$ in `used_xor`.
        //    Because new pairs $(i, s)$ exist and must not collide with FUTURE elements.
        
      If valid:
        Add $i$ to $S$.
        For each $s \in S$ (excluding $i$):
          used_xor[i ^ s] = true;
          
   Complexity again:
   Outer loop runs until we have collected enough elements.
   In worst case we scan all $n$.
   Inner loop runs $|S|$ times.
   If we succeed in finding $\approx 2200$ elements quickly, we stop.
   How quickly do we find them?
   Often the first few integers work.
   However, sometimes we might skip many numbers.
   Worst case scenario: we iterate $x$ from 1 to $n$, check, reject, accept.
   Number of checks: Sum of $|S|$. Since $|S| \le 2236$ (target).
   Even if we scan all $n=10^7$ numbers, the average cost is $2236/2$ ? No, cost depends on how many $|S|$ is.
   Initially $|S|=0$. Cost 0. Then 1. Then 2...
   Total cost roughly $\int_{0}^{K} x dx = K^2/2$ if we only count successful additions?
   Wait, NO. We check many numbers that FAIL.
   Does failure cost more?
   For a rejected number $x$, we still iterate all $s \in S$ to find a collision?
   Yes, because collision detection might be deep.
   If $x$ is rejected after checking $k$ pairs, cost is $k$.
   If almost all numbers are rejected, total cost could be high.
   Is it the case that many numbers are rejected?
   If we pick greedily small numbers, $|S|$ stays small for a long time?
   Actually, for $n=10^7$, if we start with 1, 2, 4, 8...
   Small numbers might form a dense set.
   Let's trace:
   $S=\{1\}$. xor\_used={0}.
   $x=2$. $2\oplus 1=3$. Used[3]=false -> OK. Used[3]=true.
   $S=\{1, 2\}$. used={0, 3}.
   $x=3$.
   $3\oplus 1=2$. Used[2]=false.
   $3\oplus 2=1$. Used[1]=false.
   OK. Used[2]=T, Used[1]=T.
   ...
   It turns out dense subsets like $\{1, \dots, k\}$ produce collisions very quickly.
   So we will likely skip a lot of numbers.
   However, the number of ACCEPTED elements is limited to $\sqrt{N}$.
   The number of REJECTED elements?
   Each time we add an element, we consume $\approx |S|$ slots in the XOR space.
   Total consumed slots $\approx K \cdot K \approx N/2$.
   The space of XORs is up to $N$.
   So we fill about half the space with valid XORs.
   The probability of finding a suitable $x$ should be reasonable.
   Is it guaranteed to finish within 1s?
   C++ execution for $10^7$ iterations with inner loop average 1000 iterations is $10^{10}$ ops. Too slow.
   We need to optimize.
   Do we really need to check all $x$ up to $N$?
   Or can we construct it differently?
   
   Maybe we don't need to check all numbers.
   Consider the problem statement constraint again.
   "Print out the sequence with the longest length."
   If I can implement a randomized greedy, it might be faster?
   Or maybe there's a mathematical construction.
   
   Known result: "Constructive algorithm for Distinct Pairwise XOR Subset".
   There is a paper "On the maximum size of a subset of $\{1, \dots, n\}$ with distinct XOR sums".
   It's known that max size is roughly $\frac{\log n}{\text{something}}$ or $\sqrt{n}$?
   Actually for "Sum-free" it's $\sqrt{n}$. For "Distinct Sums" it's $\approx \sqrt{2n}$.
   Here it is XOR sums.
   There is a construction using quadratic residues or similar?
   Actually, for $N \le 10^7$, the $\sqrt{N}$ requirement suggests $O(\sqrt{N})$ solution.
   Wait, I am iterating $O(N)$ times. That's $10^7$ ops outside, and inside is $O(K)$.
   If we optimize the search:
   Maybe we can pick numbers in a block?
   Or maybe we can skip the loop over $n$ and just assume certain properties?
   Wait, Sample $n=49$, Output $m=4$. $\sqrt{49/2} = 4.9 \to 4$.
   The output uses consecutive integers 1, 2, 3, 4.
   If I pick 1, 2, 3, 4... does it work?
   We saw 1..6 failed.
   However, maybe we only need to find $\approx 2236$ numbers.
   If we can just pick numbers $1, 2, \dots, m$? No.
   
   Let's look at constraints again. Time 1s. $N=10^7$.
   A naive check of every integer $1..N$ with an inner loop of size $K$ ($\sim 2200$) is definitely TLE ($10^7 \times 1000 = 10^{10}$).
   We MUST reduce the number of candidates we check.
   Most candidates will fail.
   Can we predict which ones fail?
   
   Alternative Idea:
   Instead of searching $1..n$, what if we construct elements to be sparse enough to guarantee distinctness?
   Powers of 2 is too sparse.
   What about numbers with specific bit patterns?
   Like, only using lower $d$ bits?
   If we restrict elements to range $[1, L]$, max XOR is $L$.
   If we select elements such that all pairwise XORs fall into disjoint ranges?
   Impossible.
   
   What if we reverse the problem?
   We need $K$ elements.
   Maybe use a randomized approach.
   Pick $K$ random numbers.
   Sort them. Check validity.
   Repeat until valid.
   Expected number of trials to find a valid set?
   If density of valid sets is low, this is bad.
   
   Wait, let's revisit the greedy strategy complexity.
   Why is it $O(N \cdot K)$?
   Because we try $x=1, 2, 3, \dots$.
   For each $x$, we check against current $S$.
   Can we skip $x$'s?
   Notice that if $x$ is too small, it might collide.
   If $x$ is too large?
   Actually, if we fix the size of $S$ to be around $K$, maybe we can find it without checking every $x$.
   
   But wait. Look at the constraints. $N \le 10^7$.
   We need $K \approx 2236$.
   The number of valid pairs in the entire space $[1, N]$ is roughly $N^2/2$? No, $m(m-1)/2 \approx N$.
   The number of available XOR values is $N$.
   By pigeonhole principle, we can't have more than $\sqrt{2N}$ elements if all pairwise XORs are distinct and lie in $[1, N]$?
   Wait. If we have $m$ elements, we generate $m(m-1)/2$ XOR values.
   These must be distinct and $\le 2N$ (since $a, b \le N \implies a \oplus b < 2^{\text{bits}}$).
   Specifically, if $N = 2^k$, XORs are in $[0, 2^k-1]$. Count $N$.
   We need $m(m-1)/2 \le N$.
   $m^2 \approx 2N \implies m \approx \sqrt{2N}$.
   So $m \approx \sqrt{2 \cdot 10^7} \approx 4472$.
   But the problem asks for $m \ge \sqrt{n/2} \approx \sqrt{0.5 \cdot 10^7} \approx 2236$.
   This matches the theoretical upper bound derived from $m^2 \approx 2N$ (if XOR values are confined to $N$).
   Wait, XOR values are in range $[1, 2^{\lceil \log_2(N+1) \rceil}-1]$.
   For $N=10^7$, $2^{24}-1 \approx 1.67 \cdot 10^7$.
   So we have room.
   
   Crucially, to achieve $m \approx \sqrt{n}$, the greedy strategy *should* work if we iterate smartly.
   But iterating $1..N$ is too slow.
   However, note that if we just pick $K$ numbers, we perform $K \cdot K$ updates.
   Is there a way to construct the set without trial and error?
   Maybe there's a pattern.
   The sequence 1, 2, 3, 4 worked for $N=49$.
   For $N=49$, $\sqrt{24.5} = 4.9 \to 4$.
   Elements 1, 2, 3, 4.
   Pairs:
   1^2=3
   1^3=2
   1^4=5
   2^3=1
   2^4=6
   3^4=7
   Values used: 1, 2, 3, 5, 6, 7. All distinct.
   
   Hypothesis: The set $\{1, 2, \dots, m\}$ works if $m$ is small enough.
   For $m=5$:
   1..5. 1^5 = 4. 2^4 = 6. 2^5=7. 3^4=7. Collision 7.
   So 1..5 fails.
   For $m=6$ fails.
   
   However, notice 1, 2, 3, 4.
   Bits: 001, 010, 011, 100.
   It seems like we can pick elements such that higher bits are 0 for a while?
   
   Another idea:
   Construct $S$ recursively or based on powers of 2?
   If $S = \{2^0, 2^1, \dots, 2^{p-1}\}$. Size $p$. Max $N \approx 2^p$.
   XORs are unique.
   If we take blocks?
   
   Wait, I recall a solution for this specific problem (it looks like a Codeforces or similar contest problem).
   Problem "Distinct Pairwise XOR Set".
   There's a known construction:
   Take numbers $x_i = 2i$? No.
   Take numbers $x_i$ such that the LSB is fixed?
   
   Actually, look at the constraint $m \ge \lfloor \sqrt{n/2} \rfloor$.
   This is quite low compared to $\sqrt{2n}$.
   Is it possible to construct this in $O(N)$ or $O(\sqrt{N} \log N)$?
   The bottleneck is filling the set.
   
   What if we iterate $x$ from 1 to $N$, but we keep track of used XOR sums.
   Wait, I said $O(N \cdot K)$ is too slow.
   But notice that we only ADD $x$ if it passes.
   For a fixed $x$, we check $x \oplus s$ for all $s \in S$.
   If we maintain a `valid_x[x]` flag.
   Also, observe that we only need to find $K$ numbers.
   If we can find a set of size $K$ relatively quickly.
   With $N=10^7$, if we check random numbers, expected time might be acceptable?
   Or if we check numbers with gaps.
   
   Let's reconsider the complexity.
   If we just output $S = \{1, 2, 4, 8, \dots\}$, size is ~24.
   We need ~2200.
   We can combine ideas.
   Actually, there is a specific construction:
   Use the fact that if we select numbers $x_i$ where each has a distinct most significant bit position, XORs might collide if bits cancel out?
   No.
   
   Wait! There is a construction for $K \approx \sqrt{N}$ using the "quadratic residue" idea or similar modulo arithmetic, but XOR is bitwise.
   
   Wait, I found a reference to a similar problem: "Maximum size of a subset of $\{1, \dots, n\}$ such that all pairwise sums are distinct" is $O(\sqrt{n})$. Construction involves $2x, 4x, \dots$ or similar? No, that's for sums.
   For XOR, it's harder to control cancellation.
   BUT, if we treat the bits independently?
   If we select $m$ numbers such that their lowest $b$ bits vary in a way that prevents collisions.
   
   Let's assume the greedy strategy is correct and the number of rejections is not too high?
   Wait, if we iterate $1..N$, we might spend too much time rejecting.
   However, notice that we don't need to check ALL numbers.
   If we can jump to likely candidates.
   Candidates with fewer bits set? Or specific bit patterns.
   
   Actually, the problem allows $m \ge \sqrt{n/2}$.
   Is it possible to find a solution that constructs $S$ such that elements are $x_i = i^2$?
   Let's test $x_i = i^2$ for small $i$.
   1, 4, 9, 16, 25.
   XORs:
   1^4 = 5
   1^9 = 8
   1^16 = 17
   1^25 = 24
   4^9 = 13
   4^16 = 20
   4^25 = 29
   9^16 = 25
   9^25 = 16
   16^25 = 9
   Values: {5, 8, 17, 24, 13, 20, 29, 25, 16, 9}. All distinct?
   Wait, 25 is in the set. 9^25 = 16 (which is in the set).
   But we need pairwise XORs to be distinct.
   The values themselves being in the set doesn't matter, unless a pair's XOR equals another pair's XOR.
   Are any XORs repeated?
   9^25=16. 16^? No 16^25 = 9.
   Check list: 5, 8, 17, 24, 13, 20, 29, 25, 16, 9.
   All distinct.
   It seems squares work well for sums?
   For XOR? Squares are sparse?
   $i^2 \approx 2236^2 \approx 5 \cdot 10^6 \le 10^7$.
   So we can pick $i^2$ for $i=1 \dots 2236$.
   Do they satisfy the property?
   We need $a^2 \oplus b^2 \neq c^2 \oplus d^2$.
   This is generally true for sufficiently sparse sequences.
   $i^2$ grows quadratically.
   Gaps between $i^2$ and $(i+1)^2$ are $2i$.
   As $i$ gets large, spacing is large.
   It's likely distinct.
   Is it always?
   Maybe not. Collisions could happen.
   Example: $16 = 4^2$. $25 = 5^2$. $9=3^2$.
   $9 \oplus 25 = 34$? No, $9=01001, 25=11001$. XOR=10000 = 16.
   Wait $9 \oplus 25 = 16$.
   And $16 \oplus \dots$?
   Is there a pair with XOR = 16?
   If we have $x, y \in S$ such that $x \oplus y = 16$.
   Here $9, 25 \to 16$.
   Is there $u, v \in S$ such that $u \oplus v = 16$?
   If $S$ is powers of 2 (sparse), maybe not.
   But for squares, maybe.
   However, if we check randomly selected numbers, it works well.
   
   Let's refine the Greedy Strategy.
   We want to avoid TLE.
   We can iterate $x$ from 1 to $n$, but use a pruning strategy or optimization.
   Actually, $10^7$ integers is not THAT big for modern CPUs.
   Inside the loop, we do:
   For $s \in S$: check `used[x ^ s]`.
   Average $|S|$ is $K/2$. $K \approx 2236$. $|S| \approx 1000$.
   Operations $\approx 10^7 \times 1000 = 10^{10}$. This is definitely too slow.
   So we cannot iterate all $x$.
   
   We must jump.
   Which $x$ are likely to be safe?
   Those that produce "large" XOR values? Or specific bit patterns?
   Actually, we can pick $x$ such that $x \oplus s$ covers a wide range of unused space.
   Wait, if $m$ is small, most space is unused.
   So almost any $x$ should work as long as $x$ itself isn't involved in previous collisions.
   But we don't know previous collisions without checking.
   
   Wait, is there a simpler construction?
   What about selecting $x$ based on binary representation?
   Maybe pick numbers where the LSB is 1?
   Or maybe just use a randomized greedy approach.
   Randomly permute $1..N$ and pick the first $K$ valid ones?
   This avoids the worst-case linear scan.
   We expect to find valid numbers frequently.
   For $N=10^7$, random order means we sample uniformly.
   Probability of $x$ being compatible with current $S$?
   Depends on how "dense" the `used_xor` array is filled.
   As $|S|$ grows, the fraction of occupied `used_xor` increases.
   However, we only occupy $K \cdot K \approx 5 \cdot 10^6$ entries in a space of $1.6 \cdot 10^7$.
   So density is $\approx 1/3$.
   So roughly $2/3$ of candidates should be valid immediately (first check).
   Wait, we check $x \oplus s$ for all $s \in S$.
   Probability that ANY of the $K$ queries hits a used slot?
   Occupancy fraction $\rho$. Prob of miss $\approx (1-\rho)^K$.
   $\rho \approx 0.5$ (worst case). $K \approx 2000$.
   $(0.5)^{2000} \approx 0$.
   So once density is high, it becomes hard to find valid $x$.
   But we stop once $m \approx 2236$.
   The problem is that $m(m-1)/2$ is the number of XORs.
   With $m \approx 2200$, we occupy $2.4 \times 10^6$ XOR values.
   Space size $\approx 1.6 \times 10^7$.
   Density $\approx 15\%$.
   $(0.85)^{2200}$ is tiny.
   So random selection will fail to find many elements.
   
   Wait, the constraint is $m \ge \sqrt{n/2}$.
   For $n=10^7$, $\sqrt{n/2} \approx 2236$.
   This requires us to find a set of size 2236.
   My calculation shows that the "random greedy" will fail because the density of available XOR sums is high.
   The theoretical max size is bounded by $m \approx \sqrt{2N}$.
   So we are pushing against the limit.
   Thus, a constructive method is needed, not random greedy.
   
   Is there a specific sequence?
   The "powers of 2" works but gives $O(\log n)$.
   We need polynomial growth in $n$.
   
   Let's check the sample $n=49$. $m=4$.
   Elements 1, 2, 3, 4.
   Notice $1, 2, 4$ are powers of 2. 3 is sum.
   Is it possible to take $\{1, 2, 4, \dots, 2^p\}$ and fill in gaps?
   
   Wait, I found the problem online. It is from a contest (e.g. Codeforces Round #... or similar).
   Problem name: "Distinct Pairwise XOR Set".
   Constraints: $N \le 10^7$.
   There is a simple solution:
   Construct a sequence of elements.
   One strategy is to select elements $a_i$ such that $a_i$ are powers of 2 times some factor?
   Actually, there is a construction using the property:
   $a_i \oplus a_j = \pm (a_u \oplus a_v)$? No, distinct.
   
   Let's assume the question implies a solution exists and is efficient.
   Maybe the constraint $\sqrt{n/2}$ is loose enough to allow a simple greedy if optimized?
   Or maybe we only need to check a subset of candidates.
   Wait! If we sort candidates by "how far they extend the XOR basis"?
   No.
   
   Let's rethink the density argument.
   The `used_xor` array size is determined by $N$.
   If we only use numbers up to $K^2$? No.
   If we pick numbers from range $[1, N]$.
   
   Wait, there is a known construction for this exact problem.
   It relies on the observation that if we pick $a_i$ carefully, the XOR sums are distinct.
   One such construction is to pick numbers $a_i$ such that the most significant bit (MSB) is distinct for each element?
   If MSBs are distinct, say $2^k$ for $a_i$.
   Then $a_i \oplus a_j$ has MSB $\max(k_i, k_j)$.
   If we have elements with MSBs at positions $k_1 < k_2 < \dots < k_m$.
   The pair $(i, j)$ with $i > j$ has MSB $k_i$.
   Since all $k$'s are distinct, each pair has a unique MSB position among the $k$'s?
   Wait. Pair $(i, j)$ with $i > j$. MSB is $k_i$.
   If we have another pair $(u, v)$ with $u > v$.
   If $k_u \neq k_i$, then MSBs are different. Distinct!
   If $k_u = k_i$, then MSBs match. We need the lower bits to differ.
   So, if we select exactly one element per MSB level, we have $m$ elements.
   We need distinct MSB levels $k_1, \dots, k_m$.
   For this to work, we need $2^k \le N$ for all $k$.
   Max $k$ is $\approx \log_2 N$. So $m \approx \log_2 N$.
   This gives only logarithmic size. We need square root size.
   So this MSB trick fails to provide enough elements.
   
   We need to group multiple elements per MSB level.
   If we have multiple elements with same MSB $k$.
   Say $2^k, 2^k+1, \dots$.
   Their XORs will be small (around $2^k$).
   
   Wait, actually, the solution might be:
   Select $a_i = i$ for $i=1 \dots m$ where $m$ is large enough?
   We checked 1..6 fails.
   But maybe we can select $1, 2, \dots, m$ skipping some?
   Or maybe $m$ is smaller than $\sqrt{n/2}$? No, we must satisfy it.
   
   Let's search for "Subset with distinct XOR sums size sqrt n".
   There is a result: For any set $S \subset \mathbb{Z}_n^+$, if all pairwise XOR sums are distinct, then $|S| \le O(\sqrt{n})$.
   Wait, the problem asks for existence of such set.
   It implies such sets ARE dense enough.
   
   Maybe the "Greedy" strategy works if we restart or shuffle?
   Wait, 1 second time limit is generous for $10^7$ simple ops.
   Maybe we don't need to check every $x$.
   We can iterate $x$ starting from a random offset.
   Or even better:
   Iterate $x$ such that $x$ is "far" from previous elements?
   
   Actually, looking at the problem source (likely CodeChef or similar), often the solution is simple.
   One simple solution for this problem (found in similar contexts):
   Take $a_i = 2^i$. This is too small.
   
   Wait! There is a specific algorithm:
   Initialize $S = \emptyset$.
   Maintain `used` set.
   For $i$ from 1 to $n$:
     Check if adding $i$ creates collision.
     If not, add $i$.
     If $|S| \ge \sqrt{n/2}$, break.
     
   Is this fast enough?
   The issue was checking too many $i$.
   BUT, if we check only $i$ where $i$ is a prime?
   Or $i$ where $i$ has few bits?
   Actually, if we just run this code, does it pass?
   Many solutions online for "distinct XOR pair" use simple greedy.
   Perhaps the constant factor is small enough or the "bad" cases are rare?
   Let's estimate.
   How many numbers do we need to skip?
   Suppose we need to find $K$ numbers.
   Total capacity of `used_xor` is $\approx N$.
   Each accepted number consumes $O(current\_size)$ slots.
   Total consumed $K^2/2$.
   If we iterate $x$ sequentially, the probability that $x$ is valid decreases as $|S|$ increases.
   However, the valid $x$ are distributed somewhat regularly?
   Actually, for this problem, it is known that the density of valid numbers is decent.
   Maybe $10^7$ checks is actually fine if the inner loop is short?
   But the inner loop is $|S| \approx 2000$.
   So $2000 \times 10^7 = 2 \cdot 10^{10}$ ops.
   In 1 second (usually $\approx 10^8$ ops/sec), this is 200 seconds.
   So simple greedy is TLE.
   
   We need a faster construction.
   What if we don't iterate $1..n$?
   We can iterate $i$ and construct $S$ such that elements are roughly $2i$?
   No.
   
   How about this:
   We need $m$ elements.
   If we pick elements $x_1, \dots, x_m$ such that they form a basis for a code?
   
   Wait, is there a formula for $m$-th element?
   Maybe $x_i$ is related to prime numbers?
   
   Let's try to adapt the solution logic from a known problem "D. XOR Set".
   Wait, there's a solution involving:
   $a_i = 2^{i}$? No.
   
   Actually, what if we use the property of the number 1?
   
   Let's consider the constraint $\sqrt{n/2}$ again.
   For $n=10^7$, target $\approx 2236$.
   Is it possible to construct $S$ by taking blocks of size $B$?
   
   Let's try to implement the Greedy with optimizations.
   Optimization: Bitset?
   `std::bitset<1<<24>`.
   Access is fast.
   
   Wait, can we use the fact that we only need *any* solution?
   If we find ONE solution of size $K$, we can stop.
   If we pick numbers in a way that maximizes separation.
   E.g. Randomly pick numbers?
   With $m \approx 2236$, we fill $\approx 2.5 \times 10^6$ entries in `1.7 \times 10^7` array.
   Free space $\approx 95\%$.
   So random selection has $95\%$ chance per check?
   No, for a single $x$, we check $K$ pairs.
   Prob(all pairs valid) $\approx 0.95^K \approx 0.95^{2200} \approx 0$.
   So random picking is impossible.
   
   This implies the elements cannot be arbitrary. They must be structured.
   Structure:
   Maybe elements with $k$ set bits?
   Maybe elements with $2^k + \text{small}$?
   
   Actually, look at the sample. 1, 2, 3, 4.
   These are numbers with bits $1, 10, 11, 100$.
   
   Key Insight:
   If we pick elements such that their binary representations do not interact in certain ways.
   What if we pick numbers such that the LSB is 1?
   Odd numbers.
   If all elements are odd.
   $a_i \oplus a_j$ is even? No, if LSB is 1, $1 \oplus 1 = 0$. LSB is 0.
   So all pairwise XORs are even.
   If we pick numbers in range $1..N$ that are congruent to 1 mod 4?
   
   Actually, I recall a solution for this problem that involves iterating and checking, but optimizing the loop by noticing that we don't need to check all previous elements?
   No.
   
   Let's go back to basics.
   We need $S$ of size $K$.
   $N \le 10^7$.
   Time 1s.
   This is extremely tight for $O(K^2)$ inside an $O(N)$ loop if we do $K$ checks.
   However, $K \approx 2200$.
   We can try to build $S$ incrementally.
   
   Wait, there is a known pattern for Sidon sets.
   Maybe we can use the sequence $1, 2, 4, 8, \dots$ but scaled?
   
   What if we use the following logic:
   Iterate $x$ from 1 to $N$.
   If $x$ is compatible, add to $S$.
   To speed up, we can skip many $x$.
   For example, check $x$ such that `used_xor` is empty? No.
   
   Is there a construction $S = \{ \lfloor n^{1/3} \cdot i \rfloor \dots \}$?
   
   Actually, looking at the code submission limits for similar problems (like Codeforces 1196E?), often $N$ up to $10^6$ allows $O(N \log N)$. Here $10^7$ is larger.
   Maybe we only need to output $S$ up to size $\sqrt{N/2}$.
   Is it possible to solve this by maintaining a queue of candidates?
   
   Wait! I found a relevant problem discussion.
   For distinct XOR sums, a randomized greedy with restarts can work?
   No, we established probability is low.
   
   However, there is a specific sequence of numbers that works.
   Consider the numbers $x_i$ such that $x_i$ has the $i$-th bit set? No.
   
   Let's think about the constraints on bits.
   If we restrict elements to have $k$ bits fixed to 0?
   
   Let's write a "smart greedy".
   Instead of $1..N$, pick candidates that are "most promising".
   Maybe candidates with large gaps?
   Or candidates that are powers of 2?
   
   Wait, if I pick $2^0, 2^1, 2^2, \dots, 2^{10}$.
   Max value $1024$.
   Then I can pick more numbers in between?
   
   What if I simply implement the Greedy with `std::vector` and `bool` array, but ONLY check up to $N$?
   Wait, maybe the number of rejects is not that high?
   Let's simulate mentally.
   1 (OK)
   2 (OK)
   3 (OK)
   4 (OK)
   5 (Check 5^1=4, 5^2=7, 5^3=6, 5^4=1). 1 is in used?
   Used: {1^2=3, 1^3=2, 1^4=5, 2^3=1, 2^4=6, 3^4=7}.
   Used = {1, 2, 3, 5, 6, 7}.
   Check 5:
   5^1=4 (Not in Used)
   5^2=7 (IN Used! 2^4=7). Collision.
   So 5 is rejected.
   6:
   6^1=7 (Used). Reject.
   7:
   7^1=6 (Used). Reject.
   ...
   Basically small numbers collide very fast.
   It will take a while to find the next number.
   But eventually we find one.
   Maybe around 100? 1000?
   If we have to skip 100 numbers for every acceptance, and we need 2200 numbers.
   Total scanned $\approx 2200 \times 1000 = 2.2 \times 10^6$.
   For each scan, inner loop $|S| \approx 1000$.
   Total ops $\approx 10^9$. Still high.
   But wait, inner loop runs over ALL existing elements.
   Average size is $1100$.
   So if we scan 2.2 million numbers, total checks = $2.2 \times 10^6 \times 1100 \approx 2.4 \times 10^9$.
   Might be TLE.
   
   Optimization:
   Use a fast check for `used_xor`.
   Use `unsigned char` array. `vector<unsigned char> used(MAX, 0)`.
   This is very fast.
   Maybe the "scan 2.2 million numbers" assumption is pessimistic.
   Actually, the density of usable numbers might be higher.
   Also, we can stop scanning once we have enough.
   If we pick random numbers, we save time.
   But random numbers collide too much.
   
   Is there a heuristic?
   Try to pick numbers $x$ where the lowest unset bit in the prefix is large?
   Or simply:
   Just implement the Greedy loop with a fast XOR check.
   And hope the constant factor is small enough or the rejection rate is low.
   Actually, if I submit this, I need to be sure.
   Wait, can I optimize the loop?
   If I only need to find $K$ elements.
   Maybe I can just check $x = 1, 2, 4, 8, \dots$ (Powers of 2)?
   That gives $O(\log N)$ elements.
   
   Wait, I missed something crucial.
   The problem statement: "Output - Print out the sequence with the longest length."
   "Any valid S is accepted. You do NOT need to maximize m".
   This suggests I just need to find ONE valid S.
   The sample output for 49 is 4 elements: 1 2 3 4.
   Wait, is it possible to just print $1, 2, 3, \dots, K$ where $K$ is small enough?
   No, we need $m \ge 2236$.
   
   Let's assume the Greedy solution works efficiently enough.
   To ensure it passes, I'll make it efficient.
   Use a `vector<char>` for `used_xor` (byte array) rather than `vector<bool>` (bit array) because byte access is faster than bitwise ops?
   Actually `vector<bool>` is specialized, might be slower due to bit extraction. `vector<char>` is contiguous memory.
   Given 512 MB memory, $1.7 \cdot 10^7$ chars is 17MB. Safe.
   
   Implementation details:
   - Use Fast I/O.
   - Max XOR value: $N=10^7$. Next power of 2 is $2^{24} = 16777216$.
   - Size of `used_xor` array: $2^{24} \approx 1.7 \times 10^7$.
   - `S` vector.
   - Loop $x$ from 1 to $n$.
   - Inside, check compatibility.
   
   Wait, there's a possibility that checking $x$ is too slow.
   Can we prune?
   If `count(S)` is small, checking is fast.
   If `count(S)` is large, checking is slow.
   But `count(S)` stops at 2236.
   The check is $O(|S|)$.
   If we find $x$ very frequently, total time is $N \times |S|$.
   If we find $x$ rarely (skip many), time is $(Skips \times |S|)$.
   If we skip $1000$ numbers for each accepted, and $|S|$ grows linearly.
   Total time $\sum_{k=1}^{K} (\text{skips}_k \cdot k)$.
   If skips $\approx 100$, $K=2236$. $\sum 100 \cdot k \approx 100 \cdot K^2 / 2 \approx 100 \cdot 2.5 \cdot 10^6 = 2.5 \cdot 10^8$.
   This is borderline but might pass in C++ (10^8 ops usually ok, 2.5*10^8 might be tight for 1s).
   But `char` array access is very fast (cache friendly if data fits in cache).
   Wait, `used_xor` is 17MB. L3 cache is usually 10-20MB on modern servers, but 1 second limit is generous.
   
   Also, we can optimize:
   Don't iterate $x$ from 1.
   Start checking from a random offset?
   No, deterministic is better.
   Maybe we can iterate $x$ but break early on collision?
   Yes, the check breaks on first collision.
   
   Wait, is there a better way?
   What if we simply generate the sequence $S = \{1, 2, 3, \dots, m\}$ where $m$ is calculated?
   No, collisions happen.
   
   Wait, I remember a solution for this problem (Codeforces 1085C is similar? No).
   Ah, for $N \le 10^5$, simple greedy works. For $N \le 10^7$, maybe we need a mathematical construction.
   
   Mathematical Construction Idea:
   $S = \{ a_1, \dots, a_k \}$.
   If we choose $a_i$ such that $a_i = i \cdot P$ for some $P$?
   Or $a_i$ in arithmetic progression?
   We know AP fails for XOR (collisions like $a \oplus b = c \oplus d \iff a-b=c-d$?).
   XOR doesn't relate to subtraction.
   
   What if we use the property that $x \oplus y$ is unique if we fix the bits.
   
   Actually, let's write the Greedy solution but with a twist.
   Use `vector<uint32_t>` for `S` and `vector<bool>` (bitset) for `used_xor`.
   If TLE, I'd switch to `std::vector<char>` (array).
   Actually, `vector<char>` is safest for performance here.
   
   One critical optimization:
   We only need to check `x` up to `n`.
   But do we need to check ALL `x`?
   We can pick `x` such that `x` has a high bit that hasn't been used?
   No.
   
   Okay, I will implement the greedy solution. It is the standard solution for this problem.
   With `10^7` constraint, `check(x)` is fast.
   
   Wait, I should check if $m \ge \lfloor \sqrt{n/2} \rfloor$.
   Calculated $K \approx 2236$.
   My loop finds $K$ elements.
   I will just exit when `S.size() >= K`.
   
   One more thing:
   To make it faster, I can shuffle the order of checks?
   If I iterate 1..N, I might get unlucky.
   If I iterate randomly?
   Random shuffling of candidates might be safer for performance distribution.
   Since `1..N` is static, checking 1 then 2 then 3...
   Maybe checking large numbers first works better?
   Large numbers have higher bits, potentially avoiding small XOR collisions?
   Let's try to iterate $N, N-1, \dots, 1$.
   Or pick multiples of a large prime?
   No, stick to 1..N or random.
   Actually, for this specific problem (known as "XOR basis" or similar), the sequence $1, 2, 3, \dots$ is problematic.
   But maybe iterating $N$ downwards works?
   Or simply random?
   Since I need to output *one* solution, I will implement random permutation of candidates $1..N$ to check.
   This spreads the difficulty and avoids the cluster of collisions in small numbers.
   Wait, does random help?
   If we pick random $x$, it has uniform distribution.
   It might land in the "collision free" spots.
   
   Wait, I need to output the sequence.
   Randomizing the order of candidates to pick helps finding a set faster.
   However, the actual values in S should be sorted? The problem says "in any order".
   So I just collect valid $x$'s.
   
   Wait, if I pick random numbers, I might pick duplicates? No, I skip if already in S.
   
   Wait, if I pick random numbers, I might waste time checking a number that is invalid.
   Is it worth it?
   Yes, it might help avoid the dense clusters of invalid small numbers.
   
   Final plan:
   1. Read $N$.
   2. Target $m = \lfloor \sqrt{N/2} \rfloor$.
   3. `used_xor` array (bool or char) size $2 \times 10^7$.
   4. `S` vector.
   5. Candidates generation: Generate a shuffled version of $1 \dots N$? No, generating a full shuffle of $10^7$ elements takes time and memory.
   Better: Iterate $i$ from 1 to $N$, but pick candidates using a stride or random.
   Actually, just iterating $1 \dots N$ is fine. The time limit is 1s.
   Wait, if I iterate $1 \dots N$ and it's too slow, I'm screwed.
   But let's trust the standard greedy is intended.
   Actually, the standard greedy usually iterates $1 \dots N$ but stops early.
   Wait, if it takes 2.5*10^8 ops, it should pass.
   Modern judges do $\approx 5 \cdot 10^8$ ops/sec.
   
   Code structure:
   - Fast IO.
   - Global array for `used_xor` to avoid allocation overhead (stack limit might be hit, so global or dynamic).
   - Vector `S`.
   - Loop $x=1 \dots n$.
     Check valid. If valid, push, update `used`.
     If $S.size() \ge m$, break.
   - Print size and elements.
   
   Memory: $10^7$ ints for S (if size N) -> 40MB.
   $1.7 \times 10^7$ bytes for used -> 17MB.
   Total well within 512MB.
   
   Double check: "Print out the sequence with the longest length."
   The problem says "You do NOT need to maximize m".
   This note contradicts the last bullet point.
   Usually "longest length" implies we should output the largest possible m found.
   But since we just need to satisfy the lower bound, finding a large set naturally satisfies it.
   Greedy tends to find a local optimum which is usually quite large.
   
   I'll wrap up the code.

```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;

// Maximum possible value for n is 10^7.
// XOR values can go up to 2^24 - 1 approx 1.67 * 10^7.
// We allocate a bit more to be safe.
const int MAX_VAL = 1 << 24; // 16,777,216
// We need slightly larger since n <= 10^7, max xor < 2^24.
// 10^7 < 2^24.
// a, b < 2^24 => a^b < 2^24.
// Indices 0 to MAX_VAL-1.

bool used_xor[MAX_VAL];
// Since we can't put bool array inside main stack, make it global/static.
// Actually, 16MB is fine in BSS.

int main() {
    // Optimize I/O
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int n;
    if (!(cin >> n)) return 0;

    // Target size
    // Problem requires m >= floor(sqrt(n/2))
    int min_m = (int)floor(sqrt(n / 2.0));
    
    // Although we need m >= min_m, let's try to find as many as possible
    // or just until we reach min_m. 
    // The problem asks for the longest length if possible.
    // Greedy builds a valid set.
    
    vector<int> S;
    // Reserve to avoid reallocations
    S.reserve(min_m + 100);

    // Iterate candidates
    // To optimize, we can iterate from 1 to n.
    // However, randomization might help to avoid clustering of rejections.
    // But full shuffle is expensive. Simple iteration is robust enough.
    // Wait, checking 1..N with inner loop O(|S|) is risky.
    // Let's try a randomized order of candidates generation just to be safe.
    // Generating random offsets?
    // Let's stick to 1..N. If it's too slow, then TLE.
    // But given the constraint, maybe 2.5*10^8 ops is acceptable.
    
    // Optimization: Start checking from 1.
    // Use 'used_xor' to track.
    
    // Resetting used_xor is not needed if global, but technically we run once.
    // Since it's global, it's zero-initialized.
    
    // To ensure we don't TLE, we can use a heuristic to skip some numbers?
    // Or just rely on efficiency.
    // Actually, checking 'x' requires checking 'x ^ s' for all 's' in S.
    // 'used_xor' is direct addressable.
    // Accessing 'used_xor' is very fast.
    
    // Let's refine the check to break early.
    
    int count = 0;
    
    // To improve chances, maybe we should not start from 1?
    // 1 is fine.
    
    for (int x = 1; x <= n; ++x) {
        bool possible = true;
        // Check all existing elements
        for (int s : S) {
            int val = x ^ s;
            if (val >= MAX_VAL) {
                // Should not happen if MAX_VAL > 2^24 and n < 2^24.
                // But safety check.
                possible = false; 
                break; 
            }
            if (used_xor[val]) {
                possible = false;
                break;
            }
        }
        
        if (possible) {
            // Update used_xor for new pairs
            // We need to add pairs (x, s) for all s in S
            // Mark them as used.
            // Note: x is not in S yet, so we don't need to check its self-pair.
            // We just need to ensure future numbers don't collide with (x, s).
            
            // Also, x^x = 0. Self XOR is 0.
            // Does 0 count? 
            // "pairs (a, b) with a, b in S and a < b"
            // So self pair doesn't exist.
            // But what if x^s = 0? => x=s. Since s in S and x not in S, x != s.
            // So val > 0.
            
            // We can reserve capacity for S
            // Optimization: check if S is full?
            // We don't need to store ALL pairs in S, just check used_xor.
            // However, we need to iterate S to mark new pairs.
            
            // To prevent reallocation issues, just reserve.
            S.push_back(x);
            
            // Mark the pairs
            // Note: S currently holds old elements. x is just added?
            // No, logic:
            // 1. Check if adding x creates conflict with existing S.
            // 2. If ok, add x to S.
            // 3. Now, the pairs (x, s) for s in S (old) are newly created pairs.
            // We must mark them so that future candidates y don't collide with (x, s).
            // So we iterate OLD S elements.
            
            // Let's rewrite slightly cleaner.
        }
        
        // Wait, logic correction:
        // If we add x to S, we need to mark x^s for ALL s in NEW S (including x? No, pairs).
        // Pairs are {x, s}. So we mark x^s.
        // BUT, we must check if x^s was ALREADY used by PREVIOUS pairs in S.
        // That is done in the check loop.
        // After adding, we update `used_xor` for the NEW pairs involving x.
        // So we iterate s in S (current state).
        // Wait, if we add x, we shouldn't iterate x itself in the marking phase?
        // Yes, we iterate s in S_OLD.
        // The vector S includes x only AFTER check.
        
        // Correct Logic:
        // S is initially empty.
        // Check x against S (all pairs in S are old pairs).
        // If valid:
        //   Add x to S.
        //   Mark x^s for all s in S (where s is the NEW element x, or old?).
        //   Wait, if we iterate S after adding x, S includes x.
        //   x^x = 0. Do we care?
        //   We don't have self-pairs.
        //   So we should mark x^s for all s in S where s != x?
        //   Or just s in S_old.
        //   It's easier to iterate S_old to mark.
        
        // Let's use the loop correctly.
    }

    // Wait, I wrote the logic inside the loop but didn't implement fully.
    // Let's fix the implementation inside main.
    
    return 0;
}
```

Wait, `MAX_VAL` needs to be larger if $n$ is $10^7$?
$10^7$ in hex is `0x27F5000` approx. $2^{23} \approx 8.4 \times 10^6$. $2^{24} \approx 1.6 \times 10^7$.
So $10^7$ fits in 24 bits ($2^{24}-1$).
Max XOR is $< 2^{24}$.
Size $2^{24}$ is 16,777,216 bytes $\approx 16$ MB.
Global array is fine.

Optimization for speed:
Since `S` grows, checking `used_xor` is fast (L3 cache).
But iterating `S` takes time.
I'll add a check `if (S.size() >= min_m) break;`?
The problem asks for "longest length". But note says "do NOT need to maximize m".
However, "Print out the sequence with the longest length" in the Notes section is strong.
It probably implies if multiple answers exist, pick the one with largest $m$.
My greedy approach naturally tries to find $m$. If it stops at $min\_m$, it satisfies the constraint.
If it continues past $min\_m$, it satisfies both.
So I will continue until I've scanned all $n$.
Or until $S$ cannot grow further.
Stopping condition: $S.size() == min\_m$?
No, I'll continue scanning until $x=n$.
This ensures I find a large set.
Wait, time limit might be exceeded if I scan all $n$.
If I scan all $n$, and $S$ is small, $S.size() \times N$ ops.
With $N=10^7$, $|S|=1000$, ops=$10^{10}$.
I SHOULD stop early.
Since I am allowed any $m \ge min\_m$.
Once I find $min\_m$, I can break?
The Note "Print out the sequence with the longest length" implies I should maximize $m$.
BUT "You do NOT need to maximize m" overrides it.
Usually, in CP, "Longest length" is for tie-breaking or encouragement. The hard constraint is $m \ge \dots$.
However, maximizing $m$ makes me risk TLE.
Is there a compromise?
Find $min\_m$ and stop?
Wait, if I stop at $min\_m$, I satisfy the requirement.
Does the checker check if $m$ is maximal? "Multiple correct outputs may exist". "You do NOT need to maximize m".
So I should stop at $min\_m$.
To be safe, I'll stop as soon as $S.size() \ge min\_m$.
But wait, if I stop early, do I ensure $m \ge min\_m$? Yes, by breaking when reached.
Is it possible that stopping early yields a smaller set?
No, I break *after* adding the element. So size is exactly or slightly above $min\_m$.
Wait, the requirement is $m \ge \dots$.
So breaking immediately after reaching $min\_m$ is valid.
To be safe, I'll run up to $min\_m$.
Actually, let's aim to find exactly $min\_m$.
Wait, if the greedy fails to find $min\_m$ within $n$ candidates?
Is it guaranteed?
Yes, theoretically.
Also, scanning *only* up to $n$ is required.
So:
Loop $x$ from 1 to $n$.
  Check.
  If valid, add, mark.
  If `S.size() >= min_m`, break?
  Actually, I should check if `S.size() >= min_m`.
  If yes, break.
  This minimizes time spent.
  But wait, the "print longest length" might suggest I should try to go further?
  Given the instruction "You do NOT need to maximize m", I will prioritize correctness and efficiency.
  Breaking at $min\_m$ is safe.
  
  Wait, I should double check if `min_m` is small.
  For $N=10^7$, `min_m` ~ 2236.
  Scanning $x$ up to $N$ is necessary to find these 2236 elements?
  Or do they appear early?
  If they appear early, we find them quickly.
  If we scan 1..N, and find them quickly, we stop.
  The risk is if we scan $1..N$ and find nothing.
  But we stop when found.
  The risk is if we find few and scan many.
  If we stop at $min\_m$, we stop as soon as found.
  Is it possible we have to scan almost all $N$?
  Unlikely.
  I'll assume the density is okay.

One detail: `used_xor` array size.
$n=10^7$.
Max element $10^7$.
Max XOR $a \oplus b < 2^{24} = 16777216$.
So array size $2^{24}$ is sufficient.
Indices up to $16777215$.
So `MAX_VAL = 1 << 24`.

Final check on logic:
If $S = \{1, 2, 3\}$. Used: $1\oplus 2=3$, $1\oplus 3=2$, $2\oplus 3=1$.
$x=4$.
$4\oplus 1 = 5$. OK.
$4\oplus 2 = 6$. OK.
$4\oplus 3 = 7$. OK.
Add 4.
Mark 5, 6, 7.
Correct.

Double check time complexity.
Assume we need $K$ elements.
We scan $X$ numbers.
Complexity $\approx \sum_{i=1}^{K} ( \text{candidates}_i \cdot i )$.
Where $\text{candidates}_i$ is average gap between accepted numbers.
Total time $\approx \frac{1}{2} X \cdot K$.
Wait, cost is $\sum (\text{checks for element } i)$.
Checks for element $i$ is $|S_{i-1}|$.
So $\sum_{i=1}^{K} |S_i| \approx K^2/2$.
This is IF we check every candidate and every candidate is accepted?
No.
If we accept $K$ items, and check $X$ items.
Number of successful checks: $K$.
Number of failed checks: $X - K$.
Cost = $\sum_{acc=1}^{K} |S_{acc}| + \sum_{rej} \text{cost}(rej)$.
Cost(rej) is proportional to how far we check before collision.
In worst case, we check all $|S|$.
So total cost $\approx (X \cdot K)$.
Wait, if we reject most items, cost is $X \cdot K$.
If $X=10^7, K=2236$, cost $\approx 2.2 \cdot 10^{10}$.
This is the danger.
Can we optimize?
If we find a solution quickly, $X$ will be small?
For $N=49$, we found $1, 2, 3, 4$ (size 4). $X=4$.
If we need size 2236 for $N=10^7$.
Will we find them within small $X$?
Probably not. They will be scattered.
However, maybe $X$ is smaller.
Maybe we can check numbers $1, 2, \dots, K^2$?
Because $m \approx \sqrt{N/2} \implies m^2 \approx N/2$.
So checking $1 \dots N$ is consistent with finding them.
Maybe we can restrict range to $1 \dots N$ but skip multiples?
Actually, if $X \approx N$, then it's $O(N \cdot \sqrt{N})$.
Which is $10^{10}$.
Is there a faster construction?
Maybe $S = \{ \lfloor n \cdot \frac{i}{K} \rfloor \}$?
Arithmetic progression?
$1, 2, 3, \dots$.
If $a_i = i$.
$a_i \oplus a_j = i \oplus j$.
We need distinct $i \oplus j$.
This is not true for consecutive integers.
But maybe $S = \{ \text{primes} \}$?
Primes are sparse.
Density $\ln N$.
Number of primes in $N$ is $N/\ln N \approx 6 \cdot 10^5$.
We need 2236.
Primes might work?
Prob of collision for primes is low?
Primes are random-ish.
If we pick primes, $X \approx 600,000$.
Inner loop 2200.
Ops $\approx 600000 \times 2200 \approx 1.3 \cdot 10^9$.
Still high.

Wait! Is there a constraint that $S \subseteq \{1, \dots, n\}$? Yes.
Is it possible to output just the first `min_m` integers that satisfy the property?
Wait, if I iterate $x$ and check.
I can optimize the inner loop.
Instead of iterating $s \in S$, iterate only those $s$ where $x \oplus s$ is NOT in `used_xor`.
But that's the same logic.
I can't skip checks.

Wait, maybe the solution uses the observation that $m$ is small enough.
Wait, $10^7$ is the LIMIT.
Time limit 1s.
This is tight.
However, in C++, `10^9` simple operations might pass if operations are cheap (like `char` array access).
And `min_m` is not worst case.
If the elements are packed densely, `used_xor` fills up fast.
If `used_xor` fills up fast, collisions happen, we skip.
Wait, if we skip, cost is high.
If we accept, cost is low (we mark, but marking is fast? No, marking is iterating).
Wait, marking is iterating $|S|$ elements.
Accepting $S_i$ costs $|S_i|$.
Rejecting costs $\le |S_i|$.
Total cost $\approx \sum_{all\_x} |S_{at\_x}|$.
Since $|S_{at\_x}|$ grows up to $K$.
Average $|S| \approx K/2$.
Total cost $\approx X \cdot K/2$.
If $X$ is close to $K$, then $O(K^2)$.
If $X$ is close to $N$, then $O(N \cdot K)$.
If elements are clustered (e.g. first $K$ numbers work), then $X \approx K$.
Then cost is $O(K^2) \approx 2236^2 \approx 5 \cdot 10^6$. Very fast.
Are the elements clustered?
We saw 1, 2, 3, 4 work for $N=49$.
Does 1, 2, 3, ..., 2236 work for $N=10^7$?
We saw 1..5 fails.
But maybe 1..2236 is not fully valid, but we can pick a subsequence.
If elements are spread out, $X$ is large.
BUT, if we can prove that we can find $K$ elements within a small range?
Or that density is high?
Actually, random greedy suggests $K^2/N$ probability?
No.
Given the constraints, maybe the test cases are weak or the density is higher.
I'll stick to the greedy.
I'll just add `#pragma GCC optimize("O3")` to boost.
I will assume that the elements are somewhat clustered or easy to find.
The worst case (scattered valid numbers) is unlikely for XOR distinct sets constructed greedily.
Usually, the valid numbers come early.

One final optimization:
Use a pointer to `used_xor` array base.
Avoid function calls.
Use `std::vector<int>` for S.
`reserve` S.

Memory limit 512MB.
`used_xor` (16MB) + `S` (4KB). Fine.

One more check: `n` can be 1.
If $n=1$, $m \ge 0$. Empty set?
Problem says $S \subseteq \{1, \dots, n\}$. $|S| \ge 0$.
Sample $n=49 \to m=4$.
Empty set valid for $n=1$?
Input $n=1$. $\lfloor \sqrt{0.5} \rfloor = 0$.
Code should handle $n=1$.
Loop 1 to 1.
If valid add.
S empty -> no checks. Valid. Add 1.
Mark pairs (none).
Break if size >= 0.
Wait, break condition `size >= min_m`.
If `min_m` is 0, we might break immediately without adding?
Better to break after adding.
Condition: `if (S.size() >= min_m && min_m > 0) break;`
Actually, if $n=1$, we output 0 or 1?
Constraint says $m \ge 0$.
But sample output says $m=4$ for $n=49$.
Usually we want non-empty if possible.
But for $n=1$, we can output 1.
The loop will add 1.
Break check `S.size() >= min_m`.
If `min_m`=0, it breaks immediately.
I should change break to: `if (S.size() >= min_m)` -> add element -> check size?
Wait, logic:
Loop $x$.
Check.
Add.
Update.
Check `if (S.size() >= min_m) return/print`.
This ensures we stop exactly when satisfied.

Wait, what if `min_m` is 0?
Then we output $m=0$?
Problem: $n=1$. $\sqrt{1/2}=0$.
Is $m=0$ allowed? $S \subseteq \{1\}$. Empty set is valid.
But sample says "Print out the sequence...".
Usually $m \ge 1$ implied by context?
Constraints $1 \le n \le 10^7$.
If $n=1$, $m=0$ is mathematically correct.
But maybe better to output 1?
Problem: $|S| \ge \lfloor \sqrt{n/2} \rfloor$.
0 >= 0. Valid.
I'll stick to the logic. If it finds any valid set, good.
If $min\_m=0$, it might output empty.
But usually test cases expect something.
Maybe check if $min\_m == 0$, force add 1?
No, just follow spec.

One catch: `S.size()` returns `size_t` (unsigned). Comparison with `int` is safe.
Use `int` for loop counters.

Final check on header includes. `vector`, `iostream`, `cmath`.

```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;

// Using a flattened array for XOR usage tracking
// Max N = 10^7. Max XOR < 2^24 = 16777216.
// We need size 2^24.
const int MAX_N = 1 << 24; // 16777216
// Using char array to be cache-friendly and avoid vector<bool> slowness
static unsigned char used_xor[MAX_N]; 

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int n;
    if (!(cin >> n)) return 0;

    int min_m = (int)floor(sqrt(n / 2.0));
    
    // Edge case: if min_m is 0, we still need to output a valid set.
    // We can output an empty set or {1}. 
    // Since 1^1 is undefined (not a pair), and pair needs distinct elements.
    // {1} has size 1, pairs 0. 0 distinct.
    // Let's try to find at least min_m.
    
    vector<int> S;
    if (min_m > 0) {
        S.reserve(min_m + 100);
    } else {
        // If constraint is 0, we can just output 0 or empty.
        // But usually m>=1 is preferred for non-trivial.
        // Let's just run the loop.
    }

    int count = 0;
    for (int x = 1; x <= n; ++x) {
        // Optimization: if we have enough elements, we might still want to find longer ones?
        // But to be safe on time, stop at min_m.
        // The problem statement says "do NOT need to maximize m".
        // So we stop.
        if (S.size() >= min_m) break; 
        
        bool possible = true;
        // Check against all existing elements
        for (int s : S) {
            // Calculate XOR
            // Ensure indices within bounds
            if ((x ^ s) >= MAX_N) {
                 // This case is theoretically impossible if n < MAX_N
                 // But for safety:
                 possible = false; 
                 break;
            }
            if (used_xor[x ^ s]) {
                possible = false;
                break;
            }
        }

        if (possible) {
            // Add x to S
            S.push_back(x);
            // Mark new XOR sums involving x
            for (int s : S) {
                // s iterates over all elements including x?
                // Wait, logic: we need to mark pairs (x, prev_s).
                // If we iterate S containing x, then (x, x) -> 0.
                // 0 is never a valid XOR for distinct positive integers?
                // Wait, if 0 is marked, does it affect anything?
                // x ^ x = 0.
                // Future y: y ^ s = 0 => y = s.
                // Since s in S (positive) and y in candidate (positive).
                // If we mark 0, and later we find y such that y ^ s = 0, we reject y.
                // y=s. But s is already in S. We check distinct elements.
                // So we should not reject y=s.
                // However, we only check y if y is NOT in S?
                // The loop is for y=x (candidate).
                // If x is in S, we won't process it again.
                // So marking 0 is harmless?
                // Yes.
                // But strictly, we should iterate only previous elements.
                // Or just iterate all, ignore 0.
                // But 0 is not in `used_xor` valid range?
                // It's fine.
                
                // Optimization: if we iterate S, it grows.
                // S has size up to 2300.
                // This is fast.
                
                int val = x ^ s;
                if (val < MAX_N) {
                    used_xor[val] = 1;
                }
            }
        }
    }

    // If min_m was 0, we might output empty. 
    // Let's ensure at least 1 if possible.
    // Actually, if n >= 1, {1} is valid.
    // But loop runs x=1. S is empty. Loop s: none. Possible=true.
    // Adds 1. Marks used[0].
    // If min_m=0, we break.
    // Output 0? No, output size m and elements.
    // If S is empty, output 0.
    // Is empty set valid? Yes, condition |S| >= 0.
    // But maybe judge expects non-empty.
    // If min_m=0, we can just output 0 elements.
    // Wait, problem: "Print out the sequence with the longest length."
    // Empty set is valid.
    // But for n=1, {1} is better.
    // I will add a check to ensure we don't output empty if not needed.
    // Actually, I'll just run the loop until n or size >= min_m.
    // If size >= min_m, we break.
    // If min_m=0, we break immediately after loop?
    // Loop x=1. Adds 1. Checks size >= 0. Breaks.
    // So S will contain {1}.
    // Correct.
    
    cout << S.size() << "\n";
    for (size_t i = 0; i < S.size(); ++i) {
        cout << S[i] << (i == S.size() - 1 ? "" : " ");
    }
    cout << "\n";

    return 0;
}
```
Wait, one logic fix:
Inside marking loop `for (int s : S)`, if I include `x` (since I pushed it to S before loop), `x^x=0`.
Then `used_xor[0] = 1`.
Later, if I consider `y` and find `y^x = 0` (meaning `y=x`), I would reject `y`.
But since `y` iterates from 1 to `n` distinct, and `x` is in `S`, `y` will never be `x` (loop increments).
So `y=x` never happens.
Wait, `y` is candidate. `x` is in `S`.
We want to check if `y` is compatible with `x`.
Compatibility check: `used[y^x]`.
If `used[0]` is set, then `y^x=0` triggers rejection.
`y=x`.
Since `y` loop starts 1, if `x=1` is in `S`.
When loop reaches `x=2`. Check `2^1`.
If `used[0]` is set, we don't care.
Wait, does `y` ever become `x`? No.
Does `y` ever cause `y^x=0`? Only if `y=x`.
But `y` goes through $1..n$. `x` is one specific number.
When processing `y`, if `y=x`, we wouldn't process it?
Wait, loop `y` from 1 to `n`. `x` is current candidate.
If `y` reaches value of some previous `S` element?
We should check if `y` is already in `S`?
The problem doesn't require `S` to contain unique candidates?
"m distinct integers".
So `S` must be distinct.
My code iterates `x` and adds to `S`.
Since `x` increases, `x` is never added twice.
So uniqueness is satisfied.
So `y^x=0` implies `y=x`. Since `x` was already in `S` (if it was a previous element) or we are at `x`.
Wait, if `x` was added in previous iteration, it is in `S`.
Current candidate `y` (variable `x` in outer loop) can be that `x`? No, `x` loops.
Wait, let's trace variables.
Outer loop `candidate`.
Inner loop `s`.
After adding `candidate`, `used_xor[0]` set.
Later, loop reaches `candidate`? No, loop continues to next integer.
So we never check `candidate` against itself.
So `used[0]` is irrelevant.
Okay, logic holds.

Wait, the `break` condition `S.size() >= min_m`.
If `min_m` is 0, `S` can be empty initially.
Loop `x=1`. Check `possible` (True). `S` becomes `{1}`.
Break `if (S.size() >= min_m)`.
Output 1.
Seems correct.

Complexity:
Time: $O(N \cdot m)$.
Worst case $10^7 \cdot 2200$.
With bitset/array access and fast compilation, it might pass.
Also, `min_m` is small enough.
Most importantly, I'll rely on the fact that valid numbers are found early.
If not, TLE is unavoidable without better math.
But standard greedy is the intended solution for this constraint level usually.

One last optimization:
`S.reserve` is good.
`ios_base` sync off.
`MAX_N` const.

I'll provide the code.
</think>

```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;

// Maximum N is 10^7.
// Max possible XOR value for inputs in [1, 10^7] is < 2^24.
// 2^24 = 16,777,216.
// We allocate a byte array of this size for O(1) lookups.
const int MAX_XOR = 1 << 24; 
static unsigned char used_xor[MAX_XOR];

int main() {
    // Optimize standard I/O operations for speed
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int n;
    if (!(cin >> n)) return 0;

    // Calculate minimum required size of the set
    // Constraint: |S| >= floor(sqrt(n / 2))
    int min_m = (int)floor(sqrt(n / 2.0));

    // Store the elements of the set
    vector<int> S;
    // Pre-allocate memory to avoid reallocations
    if (min_m > 0) {
        S.reserve(min_m + 50);
    }

    // Iterate through candidates from 1 to n
    // We attempt to add 'x' to S if it maintains the distinct XOR property.
    for (int x = 1; x <= n; ++x) {
        // Optimization: Stop if we meet the minimum requirement
        // We need at least min_m elements. Finding more is optional but allowed.
        // However, to respect the time limit, stopping early is safer.
        if (S.size() >= min_m) break;

        bool possible = true;
        // Check if adding x creates any duplicate XOR sum with existing pairs in S
        for (int s : S) {
            int val = x ^ s;
            // Safety check, though val < MAX_XOR should hold for n <= 10^7
            if (val >= MAX_XOR || used_xor[val]) {
                possible = false;
                break;
            }
        }

        if (possible) {
            // Add x to S
            S.push_back(x);

            // Mark the XOR sums formed by x with every element in S (now updated)
            // We need to ensure future candidates don't collide with new pairs (x, s)
            // Note: x is the newest element, iterating S covers all (x, s) pairs including x itself (val=0)
            // used_xor[0] is set but harmless as distinct positive integers won't yield 0 XOR sum for distinct pairs.
            for (size_t i = 0; i < S.size(); ++i) {
                int s_val = S[i];
                int val = x ^ s_val;
                if (val < MAX_XOR) {
                    used_xor[val] = 1;
                }
            }
        }
    }

    // Output the result
    cout << S.size() << "\n";
    for (size_t i = 0; i < S.size(); ++i) {
        cout << S[i] << (i == S.size() - 1 ? "" : " ");
    }
    cout << "\n";

    return 0;
}
```
````
