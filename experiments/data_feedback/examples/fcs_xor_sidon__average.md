# FrontierCS：Distinct Pairwise XOR Set（竞赛算法题） — Average（q35 a100 method model-soup soup10）

> **本文件是完整、未截断的真实评测记录。** 模型类别：`average`。
> 评测：FrontierCS（thinking 口径，竞赛算法编程，官方评测器抽最长 ```cpp 块编译跑测试点给连续部分分）。
> 题目内部编号 `problem_idx=11` / 题库目录 `algorithmic/problems/111` / 第 `sample_idx=0` 个采样（共 5 采样/题）。
>
> **本采样得分（FCS score，0–100 连续部分分）：`37.3876537`**　|　生成 token 数：`22294`　|　error：`None`

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
The problem asks us to find a subset S of {1, 2, ..., n} such that all pairwise XORs of elements in S are distinct, and the size of S is at least floor(sqrt(n/2)). We need to output the size of S and the elements themselves.

Let's analyze the condition: all pairwise XORs must be distinct. This is related to Sidon sets but with XOR instead of addition. These sets are sometimes called "Golomb rulers" for XOR or similar structures. However, the constraint here is relatively weak: we only need |S| >= sqrt(n/2). The sample shows n=49, output size 4, which satisfies 4 >= floor(sqrt(49/2)) = floor(sqrt(24.5)) = 4. Wait, actually the sample output says size 4, and elements 1, 2, 3, 4. Let's check their pairwise XORs.
1^2 = 3
1^3 = 2
1^4 = 5
2^3 = 1
2^4 = 6
3^4 = 7
All XORs: {3, 2, 5, 1, 6, 7}. Are they distinct? Yes. So {1, 2, 3, 4} works for n=49.
Actually, for small n, maybe {1, 2, ..., m} works if m is small enough?
Let's check if the sequence 1, 2, 4, 8... (powers of 2) works.
If elements are powers of 2, say $2^k$, then the XOR sum is just the bitwise OR since bits don't overlap. So $a_i \oplus a_j$ would have exactly two bits set. Thus all pairwise XORs are distinct because each corresponds to a unique pair $(i, j)$ mapping to a bitmask with bits at positions $p_i$ and $p_j$.
Wait, if I take $\{2^0, 2^1, 2^2, \dots, 2^{m-1}\}$, then any pair XOR has exactly two bits set. The value determines which two powers of 2 were used. So yes, powers of 2 give distinct pairwise XORs.
How many powers of 2 fit in $n$? Powers of 2 are $1, 2, 4, 8, 16, \dots, 2^k \le n$. So $k+1$ elements. Since $2^k \le n$, roughly $\log_2 n$ elements. But we need $\sqrt{n/2}$. For large $n$, $\log n \ll \sqrt{n}$. So powers of 2 are not sufficient for large $n$.

We need a larger set. We know that there exist sets with distinct pairwise XOR sums of size much larger than $\sqrt{n}$, possibly $O(n^{1/2})$. In fact, the problem statement says "You do NOT need to maximize m; you only need m ≥ floor(sqrt(n/2))". And also says "Print out the sequence with the longest length" in the Notes. Wait, let me re-read.
"Notes ... Print out the sequence with the longest length."
This contradicts "you do NOT need to maximize m; you only need m ≥ floor(sqrt(n/2))". Usually, if it doesn't need to maximize, printing any valid one is fine. But the note explicitly says print the one with the longest length.
Given the limit $n \le 10^7$, we cannot iterate through all subsets or complex combinatorial optimization for maximum size. We need a constructive solution that guarantees size around $\sqrt{n/2}$.

Let's consider specific constructions.
One simple construction for distinct pairwise XORs is to use numbers whose binary representations have disjoint sets of bits. If every number in S has only bits from a specific partition of bit positions, say all numbers use only even bits and all others use odd bits... no, that restricts the values too much.

Actually, there is a known property about sets where pairwise XOR sums are distinct. If we construct numbers such that each number $x \in S$ has a 'prefix' or high-order bit structure that ensures uniqueness.
Consider numbers of the form $2k$ and $2k+1$.
Another idea: The Walsh-Hadamard transform or Reed-Solomon codes properties suggest bounds like $O(\sqrt{n})$.
Specifically, a classic construction for distinct subset sums (additive Sidon sets) involves polynomials. Here it's XOR.
XOR corresponds to addition in GF(2)^k without carry.
A set $S \subseteq \{0, \dots, n\}$ with distinct pairwise XOR sums is sometimes called an XOR-Sidon set.
It is known that the maximum size of such a set in $\{0, \dots, 2^L - 1\}$ is bounded, but lower bounds are better constructed.
Wait, for additive Sidon sets, the bound is roughly $\sqrt{n}$. For XOR-Sidon sets, similar results apply.
Let's look for a simple pattern.
Sample output for 49 is 1, 2, 3, 4. Size 4. $\lfloor\sqrt{49/2}\rfloor = 4$. It matches exactly.
Let's try $S = \{1, 2, 3, \dots, k\}$. When does this fail?
We need $a \oplus b = c \oplus d$ implies $\{a,b\} = \{c,d\}$.
For $\{1, 2, 3, 4\}$:
$1 \oplus 2 = 3$
$1 \oplus 3 = 2$
$1 \oplus 4 = 5$
$2 \oplus 3 = 1$
$2 \oplus 4 = 6$
$3 \oplus 4 = 7$
Values: 1, 2, 3, 5, 6, 7. Distinct.
Try $\{1, 2, 3, 4, 5\}$.
$1 \oplus 5 = 4$
$2 \oplus 4 = 6$
$3 \oplus ? \dots$
Is $1 \oplus 4 = 5$ present as another pair?
We found $1 \oplus 4 = 5$. Do we have another pair giving 5? No, 5 is in S, so $x \oplus y = 5$ means $x \oplus y \notin S$ (since XOR result is outside S usually, wait, no, XOR result can be anything).
Wait, $1 \oplus 4 = 5$. Is there any other pair with XOR 5?
$2 \oplus x = 5 \implies x = 7$. Not in S.
$3 \oplus x = 5 \implies x = 6$. Not in S.
So 5 is produced by $\{1, 4\}$. Is 5 in the set of XORs?
Pairs from $\{1, 2, 3, 4\}$ gave $\{1, 2, 3, 5, 6, 7\}$.
Now add 5 to S. Pairs involving 5:
$1 \oplus 5 = 4$ (4 was in S as element, now XOR value 4)
$2 \oplus 5 = 7$ (7 already in XOR set)
$3 \oplus 5 = 6$ (6 already in XOR set)
$4 \oplus 5 = 1$ (1 already in XOR set)
Check conflicts:
Old XORs: $\{1, 2, 3, 5, 6, 7\}$.
New XORs with 5: $\{4, 7, 6, 1\}$.
Intersection: $\{1, 6, 7\}$.
Collision: $2 \oplus 3 = 1$ and $4 \oplus 5 = 1$. Two different pairs map to 1.
So $\{1, 2, 3, 4, 5\}$ fails.
Generally, taking first $k$ integers doesn't scale well.
However, we only need size $\approx \sqrt{n/2}$.
For $n=10^7$, $\sqrt{5 \times 10^6} \approx 2236$.
So we need to find ~2000 numbers in range up to 10 million.
We can use a randomized approach or a greedy approach to build the set. Given 1 second time limit and 512 MB memory, we can afford $O(N \cdot M)$ or $O(M^2 \log N)$ where $M \approx \sqrt{N}$.
Wait, checking distinctness of pairwise XORs takes $O(M^2)$. If $M \approx 2000$, $M^2 = 4 \times 10^6$, which fits comfortably in 1 second.
Can we greedily select elements?
Algorithm:
Initialize $S = \emptyset$.
Iterate $x$ from 1 to $n$.
  Calculate set of current pairwise XORs for $S \cup \{x\}$.
  Check if any new XOR collides with existing XORs or internal collisions.
  Specifically, for each $y \in S$:
    val = $x \oplus y$.
    If val is in $X$ (the set of pairwise XORs generated so far), reject $x$.
    Else insert val into $X$.
  Insert $x$ into $S$ and its own internal XORs (none) into $X$.
Stop when size reaches desired target or $x > n$.
But we need $|S| \ge \lfloor \sqrt{n/2} \rfloor$. Actually, maximizing is preferred according to notes ("Print out the sequence with the longest length").
With $M \approx 2000$, inserting into a hash set or vector is fast.
Memory limit is 512MB. Storing $X$ of size $\binom{M}{2} \approx 2 \times 10^6$ integers. Each int 4 bytes -> 8 MB. Very safe.
Checking collisions: using a boolean array or hash set.
Since XOR results are up to approx $2 \times 10^7$ (max possible xor is slightly less than next power of 2 covering $n$, i.e., $\approx 1.2 \times 10^7$ max?), actually max value in range $[1, n]$ XORing two values $\le n$ is $< 2^{24} \approx 1.6 \times 10^7$.
So we can use a `std::vector<bool>` or `bitset` of size $\approx 16 \times 10^6$. That's 2MB. Very safe.
Or `unordered_set`. Vector/Array is faster.
Max value of $n$ is $10^7$. Max XOR of two numbers in $1..n$ is bounded by $2^{\lceil \log_2 n + 1 \rceil} - 1$.
$n=10^7 < 2^{24}$. So max XOR < $2^{24} = 16,777,216$.
Array size $16,777,216$ bits is ~2MB. `std::vector<uint64_t>` or `bool` or `char`.
Given 512MB, `vector<char>` is fine.

Greedy strategy:
Iterate candidates $x = 1, 2, \dots, n$.
  Try adding $x$ to $S$.
  Check all previously selected $y \in S$:
    If $x \oplus y$ exists in $UsedXor$ set, then $x$ is invalid.
  If valid, add $x$ to $S$ and mark $x \oplus y$ for all $y \in S$ in $UsedXor$.
  If $|S|$ exceeds some reasonable threshold, maybe stop? Or continue until $n$ runs out?
  Wait, does the greedy selection guarantee reaching size $\ge \sqrt{n/2}$?
  Intuitively yes, density is sufficient.
  However, note says "Print out the sequence with the longest length." This suggests if multiple valid sets exist, pick the largest one. Greedy usually finds a good size but might get stuck if order matters. With random shuffle, maybe better? Or sorting by something?
  Given constraints and typical CP problem types for "constructive", usually there's a deterministic construction. But given the "print longest" hint, maybe we just want to run the greedy algorithm to maximize size within limits?
  Actually, for $N=10^7$, iterating up to $10^7$ checking $M$ times each ($M$ grows to $\approx 2000$) gives complexity $\sum_{i=1}^M (n-i) \approx O(n \cdot \sqrt{n})$.
  $10^7 \cdot 2200 \approx 2 \cdot 10^{10}$ operations. Too slow for 1 second.
  Wait, complexity of greedy is $O(n \cdot m)$ where $m$ is final size. Since $n \approx 10^7$ and we can't afford scanning all numbers.
  Wait, do we need to scan ALL numbers $1 \dots n$?
  Maybe we only need to generate a smaller pool of candidates?
  Or maybe there's a mathematical construction.
  
  Let's reconsider the required size: $\sqrt{n/2}$.
  If $n = 10^7$, $\sqrt{5 \times 10^6} \approx 2236$.
  If we can construct such a set quickly without iterating all $n$.
  
  Known Construction:
  A set of the form $\{ 2^k \}$ works, but size $\log n$.
  We need $\sqrt{n}$.
  There is a construction based on finite fields or coding theory, but implementing them from scratch is hard.
  
  Another option:
  What if we split the bits?
  Consider numbers $v$ such that $v = i \cdot 2^k + j$? No.
  
  Let's check the constraint again. Time Limit 1s, $n \le 10^7$.
  Maybe the intended solution is simpler.
  Maybe the set size requirement is loose enough for a randomized greedy approach that selects candidates from a restricted pool?
  What if we select numbers $1 \dots K$ where $K \approx \text{something}$?
  Wait, looking at similar problems (e.g. Codeforces "XOR-MST" etc), often randomization helps to avoid specific bad cases.
  
  Alternative Construction Idea:
  Divide the integers into blocks or layers.
  However, the simplest efficient way to find *any* large set with property X is random shuffle + greedy.
  Complexity: If we try to pick $m$ elements, and we maintain $UsedXor$.
  If we pick random candidate from a pre-calculated list, say shuffle $1 \dots n$? No, iterating $1 \dots n$ is slow.
  Wait, if $M$ needs to be around 2200, and we have 1 sec.
  Checking collision takes $O(M)$. Total steps to find $M$ elements:
  If density of available numbers is high, we will accept almost every checked number.
  But we don't know which ones work without checking.
  If we check $1, 2, 3 \dots$ sequentially, for $x$, we check against $S$ (size $m$).
  Operations $\approx \sum_{i=1}^m i = m^2/2$ is if we only check candidates that are accepted immediately? No.
  To find $m$ elements, suppose acceptance probability is $P$. We check $E[\text{candidates}]$ elements.
  Then total ops $\approx E[\text{candidates}] \cdot (\text{current } m)$.
  If we reach $m=2200$, cost per step is 2200.
  Number of candidates needed: if probability of conflict is low, we check a few.
  Conflict happens if $x \oplus y = z \oplus w$ for $y,w,z \in S$.
  This is rare for small sets.
  Wait, checking "distinctness of pairwise XORs".
  The condition is: all pairwise XORs of $\{a_1, \dots, a_k\}$ are distinct.
  When adding $x$, we need to ensure $x \oplus y$ for all $y \in S$ hasn't appeared before.
  Previously, pairwise XORs of $S$ were stored in `UsedXor`.
  So checking is $x \oplus y \in UsedXor$?
  Yes.
  If $S$ has size $k$, we compute $k$ XORs.
  We need to check if they collide with existing $k(k-1)/2$ XORs.
  So lookup in `UsedXor` takes $O(1)$ (with array).
  Total time to build $S$ up to size $m$:
  $\sum_{i=1}^m (\text{trials for step } i) \cdot i$.
  If trials is constant (say 10), then $\sum i \approx m^2 \approx 4 \cdot 10^6$.
  This is very fast! $4 \cdot 10^6$ ops is well within 1 sec.
  The only catch is generating candidates efficiently.
  If we iterate $x$ from 1 upwards, how many do we need to skip?
  We want to maximize $m$.
  Also we need to stay within $n$.
  So the strategy:
  Initialize $S=\emptyset$.
  Precompute `UsedXor` array (boolean/bitset).
  Iterate $x$ from 1 to $n$:
    check if valid:
      bool ok = true;
      for each y in S:
        if UsedXor[x ^ y] == true:
          ok = false; break;
      else:
         for each y in S:
           UsedXor[x ^ y] = true;
         S.push_back(x);
    
    // Stop early if |S| is large enough?
    // Note says "Print out the sequence with the longest length".
    // This implies we should keep going until $n$ is reached or we hit a local max?
    // Actually, if we just fill greedily, we get one maximal set under greedy order.
    // Does "longest length" imply we need global optimum?
    // Global optimum for distinct XOR set is likely related to finding a maximal independent set or similar, which is NP-hard.
    // So "longest length" in context of CP usually means "as long as your method allows" or simply "don't minimize size unnecessarily".
    // However, the note "Any valid S is accepted" followed by "Print out the sequence with the longest length" suggests the grader checks validity and prefers longer sets. Since exact max is hard, we just aim for max greedy.
  
  Wait, does $m \approx \sqrt{n}$ hold for this greedy approach starting from 1?
  Sample 49 -> 1, 2, 3, 4.
  Let's trace manually:
  Start S={}.
  x=1: S={1}. Used={0 (but we don't care about 0)}. Actually pairs are non-empty. $1 \oplus y$. $S=\{1\}$, no pairs. Used empty. OK.
  x=2: $2 \oplus 1 = 3$. Used={3}. S={1, 2}. OK.
  x=3: $3 \oplus 1 = 2$, $3 \oplus 2 = 1$. Used={3, 2, 1}. S={1, 2, 3}. OK.
  x=4: $4 \oplus 1 = 5$, $4 \oplus 2 = 6$, $4 \oplus 3 = 7$. Used={3, 2, 1, 5, 6, 7}. S={1, 2, 3, 4}. OK.
  x=5: $5 \oplus 1 = 4$. Check Used. 4 is NOT in Used (Used has 1,2,3,5,6,7). Wait, earlier I said 4 is in XOR set for $\{1,2,3,5\}$?
  My manual trace earlier:
  Pairs from $\{1,2,3,4\}$:
  1^2=3, 1^3=2, 1^4=5, 2^3=1, 2^4=6, 3^4=7.
  Set U = {1, 2, 3, 5, 6, 7}.
  Now try x=5.
  5^1=4. In U? No.
  5^2=7. In U? YES.
  So x=5 fails because 5^2 collides with 3^4=1? No, 3^4=7. 5^2=7.
  Collision is on 7.
  Wait, earlier I said collision on 1.
  5^2 = 7. 3^4 = 7. Yes.
  So 5 is rejected.
  x=6: 6^1=7(U), fail.
  x=7: 7^1=6(U), fail.
  x=8: 8^1=9, 8^2=10, 8^3=11, 8^4=12.
  Check U={1,2,3,5,6,7}.
  None match. Accept 8.
  S={1,2,3,4,8}. Size 5.
  Condition: size $\ge \sqrt{49/2} = 4$.
  Greedy on $\{1, \dots, n\}$ worked well.
  
  How many can we get for $n=10^7$?
  With $U$ growing as $O(m^2)$, collisions increase.
  But still, expected size should be quite good.
  However, iterating $x$ from 1 to $n$ (10 million) might be slow if we spend time inside loop.
  But if we only succeed a few times?
  Wait, we need to execute loop 1 to $n$.
  Inside loop: check $S$ (size $m$).
  Initially $m$ small. Later $m \approx 2000$.
  Even if we iterate all $n=10^7$, and for each check $m=2000$, total operations $2 \cdot 10^{10}$ which is TLE.
  We must not iterate all $n$ with $O(m)$ check.
  
  Optimization:
  We need to skip candidates that are definitely invalid.
  Or simply: we only need to find *some* subset.
  We can construct the subset more directly.
  There is a theorem/construction: 
  For distinct subset sums (Sidon sets), greedy works. For XOR Sidon, maybe similar.
  Actually, since we can choose ANY subset, we can construct it from bits.
  Consider numbers of form $k \cdot C + r$? No.
  
  Better idea: Use a randomized greedy approach but restrict search space.
  Or better, notice that for $N=10^7$, $\sqrt{N} \approx 3162$.
  If we can find 3000 numbers such that no pairwise XORs clash.
  We can pick numbers $2^k, 2^{k+1} \dots$? No.
  
  Let's revisit the constraint on execution time. 1 sec is tight for $O(n \cdot m)$.
  We need to reduce the number of checks.
  Notice that for any $x$ considered, if $x$ is already in $S$, obviously reject (though $x$ increases).
  If $x \oplus y \in U$ for some $y$, reject.
  Most numbers will collide.
  Is there a subset of candidates we can check quickly?
  
  Wait, what if we construct the set using the property that elements are powers of 2 shifted by offsets?
  Or simply: We need size $\approx \sqrt{n}$.
  Maybe we can iterate only odd numbers? No.
  
  Actually, look at the constraint: "Time Limit: 1 second".
  Input $N$ up to $10^7$.
  Output size $\approx 2236$.
  If we implement the greedy algorithm but ONLY test random candidates?
  Probability of success decreases as set gets larger.
  But initially it's high.
  Actually, if we start with a random permutation of numbers?
  No, just iterating 1..n is ordered.
  The bottleneck is $O(m)$ per iteration.
  Can we speed up collision check?
  $U$ is a set of size $m^2/2$.
  Lookup is $O(1)$.
  So check is $O(m)$.
  We iterate $x$.
  If we skip many $x$ because they collide, we save nothing unless we can predict collisions.
  However, note that $U$ contains XOR sums.
  If $x$ is large, $x \oplus y$ tends to be larger.
  We only reject if $x \oplus y \in U$.
  
  Maybe we can assume that if $S$ is dense in low bits, higher numbers won't collide easily?
  
  Wait, I recall a standard trick for this specific problem type (Distinct Pairwise XOR Set / XOR Sidon set):
  One construction involves taking elements that have a zero at the most significant bit, recursively?
  Or using Gray code ideas.
  However, the easiest valid construction that guarantees large size is actually randomizing the greedy process or just trying to fill the set with available numbers.
  The issue is the $O(n \cdot m)$ complexity.
  Wait, if $n=10^7$, $m \approx 3000$.
  $10^7 \times 3000 = 3 \times 10^{10}$. Too big.
  We need $O(m^2)$ or $O(m^2 \log n)$ total.
  This means we only visit $O(m^2)$ candidates or optimize the check.
  
  Wait, if we only need to output *one* valid set, maybe we can just select specific numbers?
  Consider the set $S = \{1, 2, 4, 8, \dots \} \cup \{ \dots \}$.
  Actually, for competitive programming with such constraints, typically there's a specific property.
  Let's consider the problem source or type.
  Problem: Distinct Pairwise XOR Set.
  Common trick: Pick numbers with disjoint set bits? No, that gives $\log n$.
  
  Wait, what if we pick $S$ such that all elements have distinct bits? No.
  
  Correct approach for large $N$:
  It turns out that a random set of size $\approx \sqrt{2n}$ often works?
  If we randomly pick numbers, prob of collision is roughly proportional to $M^2$.
  By birthday paradox, if we pick $M$ items, there are $M^2$ pairs.
  We need these $M^2$ values to fall in range $[0, 2^n-1]$ uniquely.
  Wait, we are choosing from $N$ items.
  If we pick $M$ items, we form $M(M-1)/2$ XOR sums.
  We require them to be unique.
  With random choices, this is guaranteed if $N$ is large enough relative to $M^2$.
  Specifically, if we pick $M$ elements uniformly at random from $1 \dots N$.
  Expected number of collisions?
  Total pairs $P \approx M^2/2$. Each sum uniform in $2 \dots 2N$ (roughly).
  Prob of specific collision is $1/N_{space}$.
  But sums are not uniformly distributed.
  However, heuristics suggest that for distinct XOR sums, size $M \approx \sqrt{N}$ is achievable.
  Actually, it is proven that there exists a subset of size $\Omega(\sqrt{N})$ with distinct XOR sums.
  Randomized greedy:
  Shuffle $1 \dots n$? No, shuffle candidates?
  We want to pick a subset of size $K \approx \sqrt{N}$.
  Just picking $K$ random numbers might fail (collisions likely).
  Better: Pick numbers greedily but prioritize "good" candidates.
  Which candidates are good? Those that don't create collisions.
  If we check random candidates $x$:
  Check if $x$ compatible with $S$.
  If yes, add. Repeat until $|S| \approx \sqrt{N/2}$.
  How many trials?
  Since we just need $\sqrt{N}$, and density of valid numbers is not zero (it's roughly $N/M^2$? No), we should find them.
  Actually, the density argument suggests that we can find such a set by just picking candidates.
  Algorithm:
  Shuffle $1 \dots n$ (using `mt19937`).
  Iterate through shuffled list.
  For each $x$:
     if $x$ compatible with $S$: add to $S$.
     if $|S| \ge \text{target}$: stop.
  Target $m = \lfloor \sqrt{n/2} \rfloor$.
  Wait, is it guaranteed that a random shuffle allows us to find this size within reasonable time?
  Probably yes. The number of compatible elements remaining is high.
  If $S$ grows to size $K$, we have checked $C$ candidates.
  If $C$ is not too large, we pass.
  Wait, we don't need to check all $N$. We just need to find enough.
  If we check 100,000 random candidates, and each takes $O(K)$ checks.
  $K \approx 2000$. Ops $\approx 10^5 \times 2000 = 2 \cdot 10^8$. Feasible in 1 sec!
  So randomized selection is the way.
  But we must ensure we don't overshoot time.
  Also we need to output *the* sequence with the longest length.
  Since greedy with random shuffle gives a valid set, is it the longest?
  "Print out the sequence with the longest length." implies we should maximize $m$.
  However, maximizing subset size for XOR Sidon set is hard.
  "You do NOT need to maximize m; you only need m ≥ floor(sqrt(n / 2))."
  This is a contradiction or clarification? "Notes ... Print out the sequence with the longest length."
  Usually, if there's a tie or multiple solutions, any is accepted. But the "longest" instruction overrides "NOT need to maximize" in practice.
  However, finding the GLOBAL longest is impossible.
  Likely interpretation: Find the largest valid subset using a reasonable method, which will satisfy the lower bound.
  So the randomized greedy stopping when size $\approx$ target is okay, OR run until exhaustion of candidates.
  Since $N$ is large, running until exhaustion of $1 \dots N$ is TLE.
  Running until time limit is better?
  Or just rely on the fact that small subset is always sufficient to satisfy lower bound.
  Given $N=10^7$, target is 2236.
  If we just perform greedy on $1, 2, 3 \dots$ (deterministic), does it fail?
  Earlier manual trace showed for 49 it works perfectly.
  For larger $N$, the sequence $1, \dots, M$ (first $M$ ints) works for $M$ small.
  But for $M \approx \sqrt{N}$, collisions appear earlier.
  However, with greedy we can skip bad numbers.
  Deterministic greedy $1 \dots N$ with skipping takes $O(N \cdot M)$ which is TLE.
  But wait! $M$ is small (2200). $N$ is 10,000,000.
  $N \times M$ is huge.
  BUT, if the set $S$ remains small (which it is), $M$ stays $\approx 2000$.
  We need to avoid iterating $N$.
  
  Strategy:
  Just generate candidates using a heuristic.
  Maybe multiples of some number?
  Or simply: iterate $x$ from 1 to $n$. But skip ranges?
  Actually, if we just need *any* set satisfying the condition, maybe a subset of $1 \dots N$ with stride $k$?
  Take $x = k, 2k, 3k \dots$.
  Then $a \oplus b = (u k) \oplus (v k) = (u \oplus v) k$.
  Distinctness depends on $u \oplus v$.
  This reduces to same problem scaled down. Not helpful.
  
  Let's go back to Randomized Greedy.
  Since we have plenty of memory (512MB), we can store `UsedXor` array.
  We need to pick $K = \lfloor \sqrt{n/2} \rfloor$ elements.
  Let's just run the loop $1 \dots N$ but optimized.
  Actually, $O(N)$ is too slow?
  Wait, is $O(N \cdot \frac{1}{2})$ acceptable?
  Checking collision takes $O(M)$.
  Sum of $M$ over iterations.
  If we only accept $K$ elements, we must have checked $\ge K$ numbers.
  If we accept almost every number, $M$ grows fast.
  If we accept few, we iterate more.
  If we iterate $10^7$ times, doing check $M=2000$, total $2 \cdot 10^{10}$. Definitely TLE.
  We need to drastically reduce the number of candidates checked, or optimize the check.
  
  What if we construct the set using the structure of XOR?
  We need $a \oplus b$ distinct.
  Consider elements with only certain bits set?
  If we take elements from $1 \dots N$, we can group them.
  Is there a construction that picks elements in arithmetic progression? No.
  How about powers of 2 plus some offset?
  
  Actually, there's a known solution for this problem (it looks like a variation of a Codeforces problem).
  The problem is asking for a set with distinct XOR sums.
  This is exactly finding a "XOR-Sidon Set".
  One efficient way is to pick numbers with distinct "high bits".
  But we established $\log N$ is too small.
  
  Wait, if $N \le 10^7$.
  We can use the fact that $M \approx 3000$.
  $3000^2 \approx 9 \times 10^6$.
  If we can verify existence of valid set quickly.
  Actually, let's write code that uses randomized greedy with a time/speed cap.
  But better: Just iterate $1 \dots N$ is bad.
  How about we pick a random permutation of indices?
  No, indices matter for values.
  
  Refined Plan:
  1. Target size $M_{req} = \lfloor \sqrt{n/2} \rfloor$.
  2. Use a set `S` to store chosen numbers.
  3. Use a boolean array/bitset `used_xor` to store pairwise XOR sums.
  4. Instead of $1 \dots n$, we can use a randomized shuffle of $1 \dots n$ limited to some pool size, say $200 \times M_{req}$?
     No, shuffling $10^7$ takes time/memory overhead.
     Maybe just `std::random_device` to generate candidates.
     We need to avoid cycles/repetition.
     Keep track of tried numbers? Or just allow repeats and ignore duplicates (since we check if valid).
     Generating random numbers up to $n$.
     How many tries do we need?
     If we try $T$ numbers.
     Cost $T \times (\text{avg } m)$.
     If $m \approx 2000$, and $T \approx 10^5$, cost $2 \cdot 10^8$. OK.
     Will we find enough elements?
     Heuristically yes.
     But what if $N$ is small? Handled by `min(T, N)` logic.
     Wait, if $N$ is small, we can just brute force.
     If $N$ is large, we use random.
     Threshold $N=10^5$. Below, linear scan. Above, random.
     Wait, time limit is strict. 1 sec for $N=10^7$ is the challenge.
     
  Wait, there is a deterministic construction for size $O(\sqrt{n})$ using "finite geometry" or similar?
  Actually, constructing a set with distinct XOR sums is equivalent to finding a code with minimum distance 2? No, distinct sums means distance > 1?
  If $a \neq b$, then $a \oplus b \neq 0$. (Already satisfied for non-zero).
  Condition: $a \oplus b \neq c \oplus d$ for $\{a,b\} \neq \{c,d\}$.
  Equivalent to: $a \oplus b \oplus c \oplus d \neq 0$ for distinct sets? No.
  Distinct pairwise XORs.
  
  Let's reconsider the "print longest" note.
  If I output a random valid set of size $K \ge \text{floor}(\sqrt{n/2})$, it might be rejected if grader expects strictly largest.
  But finding largest is hard.
  Usually "print longest" means "try to make it as large as possible with your method".
  
  Code structure:
  Read n.
  If $n$ small, simple iteration.
  If $n$ large:
    Initialize `S`.
    `used_xor` array (vector<char> size $16 \times 10^6$ or similar).
    Loop while $|S| < \text{limit}$ and attempts < count.
       pick random candidate `val` in $[1, n]$.
       check compatibility with `S`.
       if ok:
         add to `S`.
         update `used_xor`.
    Print.
  
  Wait, if we use random, we need to seed properly.
  And handle case where we don't find enough elements (unlikely).
  Also, check memory for `used_xor`.
  Max possible XOR value: $2^{\lceil \log_2(10^7 + 1) \rceil} - 1 = 2^{24} - 1 = 16,777,215$.
  Array size ~16.8M chars = 16.8MB. Perfectly fine.
  `used_xor` access is fast.
  
  One detail:
  We need to maximize size.
  Random greedy might get stuck early due to bad luck.
  Better: Iterate $1 \dots n$ but skip heavily?
  No, simply: Iterating $1 \dots n$ is too slow due to $M$ factor.
  However, we can optimize the check.
  Notice that `used_xor` stores the XOR sums.
  When checking `x`, we iterate `y` in `S`.
  If `used_xor[x ^ y]` is true, `x` is bad.
  This is the bottleneck.
  
  Wait, if we just want to satisfy the lower bound, maybe just $1, \dots, K$ works?
  For $N=10^7$, $\sqrt{N/2} \approx 2236$.
  Does $\{1, \dots, 2236\}$ work?
  Let's estimate probability of collision.
  $P(\text{collision}) \approx \binom{M}{2} / N_{max\_xor}$.
  $N_{max\_xor} \approx 1.6 \times 10^7$.
  $M = 2236$. $\binom{M}{2} \approx 2.5 \times 10^6$.
  Ratio $\approx 0.15$.
  Collision probability is high! So $\{1, \dots, K\}$ will likely fail.
  So we really need to search.
  
  Since 1 second is generous for $2 \cdot 10^8$ ops.
  If we run greedy with random candidates, we need to pick candidates such that they are not too likely to collide.
  How about picking numbers with a specific bit pattern?
  e.g. numbers where lowest set bit is at position $p$?
  Or simply, use the random candidate pool.
  To ensure finding enough, maybe mix strategies?
  Try $1 \dots N$ but with a step? No.
  Try $1 \dots N$ but with a block processing?
  
  Let's trust the randomized greedy with a sufficiently large pool.
  We need to avoid TLE.
  We can limit the number of checks.
  However, if we stop at target, it's faster.
  Is it safe to stop at target? Yes, "Any valid S is accepted" provided $m \ge \text{floor}(n/2)$.
  Wait, if the random set finds $m$ elements, it's valid.
  We should maximize $m$ as per notes.
  So we should keep going until we can't find more or time runs out?
  Or just run until we hit $N$ (too slow).
  Maybe we can prune the search space.
  Only pick $x$ such that $x$ doesn't collide.
  If we shuffle $1 \dots n$ and iterate:
  First pass: collect valid $x$.
  Since checking takes $O(M)$, and $M$ increases, we stop quickly if we are lucky.
  The first few candidates are likely valid.
  Wait, actually, if we pick random candidates, collisions might be frequent as $M$ grows.
  But we stop once $M$ is large enough.
  So the average complexity is dominated by early $M$.
  Average check cost is low.
  Total time $T \times \frac{M_{final}}{2}$.
  If $T \approx 10^5$ and $M_{final} \approx 2000$, ops $\approx 10^8$.
  It should pass.
  
  Wait, one critical thing:
  If we just pick $x=1$, it's valid.
  $x=2$ valid.
  ...
  The first $\sqrt{N}$ elements?
  We calculated collision probability for $M \approx 2236$ as 0.15.
  This is per pair.
  The probability that *no* collision occurs for the whole set is not $(1-P)^{\binom{M}{2}}$.
  Roughly, if density is low, we are good.
  But for $\{1 \dots 2236\}$, collisions are likely.
  So we definitely cannot just take $1 \dots K$.
  But greedy on $1 \dots n$ with skipping might work better than random?
  Numbers near $1$ are denser in XOR sums?
  Maybe large numbers are sparser?
  If we pick large numbers (near $n$), their XOR sums are spread out.
  So maybe iterating backwards?
  Or simply: Iterate $1 \dots n$ is too slow.
  But notice: checking `used_xor` is fast.
  Maybe we can limit the number of candidates.
  We need to find at least $M_{req}$.
  We can run for `time_limit`.
  Given the strictness, perhaps we should optimize further.
  
  Optimization:
  We can try to find elements in batches.
  But random approach seems most robust and fastest to implement.
  
  One edge case:
  Memory limit 512 MB.
  `vector<bool>` is specialized to use 1 bit per entry.
  `vector<char>` uses 1 byte.
  $16.8 \times 10^6$ bytes $\approx 16$ MB.
  `vector<bool>` is smaller (2 MB).
  Access `vector<bool>` is slightly slower (bit manipulation) but negligible.
  Let's use `vector<char>` for speed.
  
  Implementation details:
  - Include headers.
  - Fast IO.
  - Seed RNG.
  - Function to check.
  - Main loop.
  - Output formatting.
  
  Wait, "Print out the sequence with the longest length."
  This implies we should try to extend the set as much as possible.
  If we find a valid set of size $m$, can we try to extend it?
  Yes, continue searching.
  Just stop after some reasonable number of checks or when $n$ exhausted?
  Given time limit, we can't exhaust.
  But we can prioritize candidates that are more likely to fit?
  Or just accept whatever greedy finds.
  Since "longest" is likely a hint towards maximizing, random shuffle order affects the result.
  Shuffling the order of candidates maximizes the chance of finding a larger set (like simulated annealing but greedy).
  Actually, sorting candidates by some metric?
  Candidates $x$ such that $x$ has fewer collisions?
  Calculating this metric requires $O(M)$ anyway.
  Just random shuffle is best trade-off.
  
  What is the worst case?
  $n=10^7$. $m \approx 2236$.
  We need to find 2236 numbers.
  Probability of picking a valid number $p$.
  With $M$ chosen, we have blocked $\binom{M}{2}$ XOR values.
  Total XOR space $R \approx 1.6 \times 10^7$.
  Blocked fraction $\beta \approx \frac{M^2}{2R}$.
  For $M=2236$, $\beta \approx 0.15$.
  So 85% of random candidates should be valid?
  Wait, this assumes uniform distribution of $x \oplus y$.
  $x \oplus y$ distribution is somewhat uniform for large $x, y$.
  So yes, acceptance rate is decent.
  So with random greedy, we shouldn't struggle much.
  We will accept many numbers.
  So finding $M_{req}$ is fast.
  Maximizing $m$ beyond that?
  It will grow to maybe $3000-4000$ before space fills up.
  This is optimal.
  
  Final Algorithm:
  1. Read $n$.
  2. Limit $M_{req} = \lfloor \sqrt{n/2} \rfloor$.
  3. If $n$ is small, maybe just brute force $1..n$.
  4. Define max array size $MAX\_VAL = 2^{24}$ (approx).
  5. Initialize `S` and `blocked`.
  6. Create a vector of candidates $1 \dots n$. Shuffle it.
  7. Iterate candidates.
     For each $x$:
       check if $x \oplus s \in blocked$ for all $s \in S$.
       If valid:
         add $x$ to $S$.
         update blocked.
       If $|S| \ge 2 \cdot M_{req}$ or time spent too long?
       Actually, just run for fixed number of checks, say 1,000,000 checks?
       Or until $n$ exhausted.
       With $N=10^7$, iterating $1 \dots N$ with shuffle is $10^7$.
       Cost $10^7 \times M_{avg} \approx 10^7 \times 2000 = 2 \cdot 10^{10}$ ops -> TLE.
       Wait, my cost calculation was wrong?
       $M_{avg}$ starts small, grows to 2000. Average $M \approx 1000$.
       $10^7 \times 1000 = 10^{10}$. Still TLE.
       
       We MUST not iterate all candidates.
       We only need to find *enough*.
       How many do we need to check to find $k$ elements?
       If acceptance rate is 50%, we check 2 candidates to find 1.
       To find $k=2236$, we check $4500$ candidates.
       Cost $4500 \times M_{avg}$.
       This is tiny!
       So we just need to pick candidates until we have enough.
       Why did I think we need to check $1 \dots n$?
       Because we assumed we need to find *the* largest or something.
       But finding *a* valid set is easy with low acceptance rate.
       Wait, as $S$ grows, blocking increases.
       If we fill up the "valid space", we stop finding.
       But we only need $M \approx \sqrt{N}$.
       Space size $N \approx 10^7$.
       Blocking is only $O(M^2) \approx 10^7$.
       So we consume roughly 0.6 of space.
       Acceptance rate drops to 40%.
       So finding 2236 elements is easy.
       Finding MAXIMUM?
       If we run out of valid numbers?
       The set of all XOR sums occupies space.
       We can stop when we can't find more.
       But time limit is key.
       So we just loop until we have collected enough, or check limit.
       If we just collect 2236 elements, cost is negligible.
       Is that enough?
       "Print out the sequence with the longest length."
       This implies we should try to push it further.
       If we run for a few minutes? No, 1 sec.
       We can try 100,000 candidates max.
       Or until time expires.
       Actually, simpler: Just try candidates $1, 2, \dots$ but SKIP.
       No, random is better.
       Wait, checking 100,000 random candidates.
       Suppose we do $10^6$ checks.
       $M_{avg} \approx 1000$ to $3000$.
       Total ops $\approx 3 \cdot 10^9$ is risky.
       We need to limit the number of checks.
       Or realize that $M$ stays small for a long time.
       Wait, we need $M \approx 2236$.
       If we just need to reach that.
       Suppose acceptance rate $\rho$. Checks $= M_{req}/\rho$.
       If $\rho \approx 0.5$, Checks $\approx 4500$.
       Cost $\sum_{i=1}^{4500} i \approx 10^7$ ops. Very fast.
       So finding the MINIMUM required is trivial.
       Finding MORE?
       If we continue, $M$ goes up to say 3000.
       More checks needed.
       But we are limited by time.
       Maybe simply: Run random greedy with a timeout or check count limit.
       Since we need *longest*, we should maximize checks within time.
       How many checks per second?
       Maybe $5 \cdot 10^7$ basic ops.
       Each check involves iterating $S$.
       If $|S|=2000$, loop takes 2000 ops.
       We can afford $\approx 25000$ checks.
       If acceptance is 0.5, we get 12500 elements.
       Wait, target is 2236. We will reach it instantly.
       So we have slack.
       We can continue searching until we can't find more?
       But eventually $S$ saturates the XOR space or candidates run out.
       The XOR space is filled.
       So we stop naturally?
       With random sampling, we might cycle.
       Keep track of tried numbers? `seen` array size $10^7$ bits (1.25MB).
       If `seen` used, generate new random.
       We should use `seen` array to prevent infinite loops or repeated useless checks.
       If `seen[x]` is true, skip.
       This makes finding new elements slow if $n$ is exhausted.
       But we only need to find some number of elements.
       So `seen` check is important to avoid wasting time on duplicates.
       Wait, duplicate numbers are obvious failures ($x \oplus x = 0$, but we check against distinct $y \in S$).
       If $x \in S$, $x \oplus x = 0$. If $0 \in Used$?
       We never put 0 in Used.
       But $x$ is in $S$. If we try to add $x$ again, we check $x \oplus y$ for $y \in S$.
       Includes $x \in S$ (if we consider $x$ already in $S$).
       Actually we check distinct $y$.
       If $x \in S$, it's already added.
       We should not pick already picked numbers.
       
       Revised Logic:
       Use `vector<char> used_elements(n + 1, 0)` to track chosen numbers.
       Use `vector<char> used_xor(MAX_XOR_VAL, 0)` to track forbidden values.
       Loop:
         Generate random `x` in [1, n].
         If `used_elements[x]`: continue.
         Check valid:
            flag = true
            for s in S:
               if used_xor[s ^ x]:
                  flag = false; break;
         If valid:
            S.push(x)
            update xor marks
            used_elements[x] = 1
         Check if done?
            If we have checked a lot and didn't add?
            Or just rely on finding enough.
      
      Actually, we want to print the LONGEST sequence.
      Maybe we should sort the output?
      No, "in any order".
      
      Is it safer to pick $x=1, 2, \dots$ and use a `visited` array?
      Linear scan with `visited` is $O(N \cdot M)$.
      But we can skip if `used_elements[x]` is true?
      Wait, linear scan $1..N$:
      For $x=1..N$:
         If `used_elements[x]` is false:
             Check valid with $S$.
             If valid: add to $S$.
      This is the greedy approach we discussed earlier.
      It takes $O(N \cdot M_{final})$.
      $N=10^7, M=2236$.
      $2 \cdot 10^{10}$ ops.
      Too slow.
      
      So we MUST limit candidates.
      Random candidates allow us to find solution without visiting everyone.
      We just need to find $M_{req}$.
      To find $M_{req}$, we need $\approx 2 \times M_{req}$ successful tries.
      If acceptance is 0.5, we try $4000$ numbers.
      Checking $4000$ numbers takes $\sum_{i=1}^{4000} i \approx 8 \cdot 10^6$ ops.
      Very fast.
      So we can find the minimal set extremely fast.
      To find longer set?
      Just keep looping.
      How many loops?
      Say we do $100,000$ iterations total?
      If we do $10^5$ checks, and $M$ grows to 2000.
      Ops $\approx 10^5 \times 1000 = 10^8$. Fits in 1 sec.
      So limiting iterations to 100k-200k is safe.
      And since we want longest, maybe we can increase limit for smaller $n$?
      For $N=10^7$, limit is 200k.
      Wait, for small $N$, we iterate $1..N$.
      If $N < 5000$, iterate all.
      
      So the plan:
      - Determine target $m_{target} = \lfloor \sqrt{n/2} \rfloor$.
      - Use randomized greedy.
      - Maintain a counter of `checks`.
      - Break if `checks` reaches limit.
      - Limit should depend on $n$ and time budget.
      - Maximize size, so don't break prematurely.
      - Just break if $m_{target}$ reached AND we feel confident?
      - No, we should try to get bigger.
      - But time is strict.
      - Just loop with `checks` limit $2 \times 10^5$ or $10^6$.
      - If we stop before filling space, output what we have.
      - But wait, does "longest length" imply we should find the absolute maximum?
      - "Notes: Any valid S is accepted. You do NOT need to maximize m..." vs "Print out the sequence with the longest length."
      - It seems contradictory. "Longest" usually means "maximal".
      - However, if we interpret "You do NOT need to maximize m" as the primary instruction for validity, then meeting the threshold is enough.
      - The "longest length" note might be a suggestion to provide a better solution than just satisfying threshold.
      - Random greedy will find a valid solution. It will likely find close to max size given time.
      
      Let's refine the limit.
      If $n=10^7$, we can afford $\approx 2000$ checks per $100ms$. Total $2 \cdot 10^5$ checks.
      $2 \cdot 10^5 \times M_{avg}$. $M_{avg}$ reaches 2000.
      Worst case ops $\approx 2 \cdot 10^5 \times 2000 = 4 \cdot 10^8$.
      Might be tight for 1 sec.
      Optimization: Bitset is faster. `std::vector<bool>` or `bitset`.
      With bitset, check $x \oplus y$ is $O(1)$ (check index).
      Wait, we iterate $y \in S$.
      So complexity is still $O(|S|)$.
      Can we optimize check?
      We need to check if $\exists y \in S$ such that $(x \oplus y) \in UsedXor$.
      This is essentially: intersection of $\{x \oplus y \mid y \in S\}$ with $UsedXor$ is empty.
      Since $UsedXor$ is a boolean array, it's fast.
      But we still loop $y$.
      However, we only check if valid. If invalid, we discard.
      If we accept, we grow.
      Since we need to output valid S, maybe we only need to verify.
      
      Let's stick to $N \le 10^5$ brute force, $N > 10^5$ randomized with limit.
      Limit: 200,000 trials.
      Seed RNG.
      Ensure distinct candidates (track seen).
      Wait, `seen` array for candidates is useful?
      If we try $x$ twice, we waste a trial.
      With random generation, collision in input space is $1/n$. For $n=10^7$, rare.
      So `seen` not needed for input space if we generate fresh.
      But `used_xor` prevents reuse of values? No, `used_xor` stores sums.
      Wait, `used_xor` prevents picking $x$ that creates collision.
      We don't need to track `seen` input numbers if we generate random.
      Unless we want to deterministically cover space?
      Just random is fine.
      
      Wait, if we just output what we found, and size is $\ge$ target.
      The code should be clean.
      
      Corner cases: $n=1$. $\sqrt{0.5} = 0$. Output 0?
      Wait, $n=1$. $\lfloor \sqrt{0.5} \rfloor = 0$.
      Can we output size 0?
      Problem says "subset S ⊆ {1, ..., n}".
      Empty set satisfies pairwise distinctness vacuously.
      But usually sets are non-empty.
      If $n=1$, output 0 or 1?
      Constraints $1 \le n$.
      If we output 0, valid. But maybe 1 is better?
      $\{1\}$ has 0 pairs. Condition holds. Size 1 $\ge$ 0.
      Sample $n=49$, output size 4.
      If $n=1$, output size 1, element 1.
      
      Wait, I missed "You do NOT need to maximize m; you only need m ≥ floor(sqrt(n / 2))".
      And "Print out the sequence with the longest length."
      Okay, so I should maximize.
      
      Wait, the memory limit is 512 MB.
      I can allocate a larger array for `used_xor`?
      Max value $1.6 \times 10^7$.
      `vector<bool>` takes 2MB.
      `vector<char>` takes 16MB.
      Both fit.
      
      One small issue: `x ^ y` could exceed bounds?
      Max $x, y \le n$.
      XOR sum $< 2 \times 2^{\lceil \log_2 n \rceil} \approx 2 \cdot 2 \cdot n$.
      No, XOR sum $\le 2^{\text{next power of 2}} - 1$.
      Next power of 2 for $10^7$ is $2^{24} = 16777216$.
      So max value is $16777215$.
      Size 16777216 is safe.
      Use `1<<24`.
      
```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <iomanip>

using namespace std;

// Fast I/O
void fast_io() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
}

int main() {
    fast_io();

    int n;
    if (!(cin >> n)) return 0;

    // Target size based on problem description
    // Note: "Print out the sequence with the longest length" implies we should maximize it as much as possible.
    // But "You do NOT need to maximize m" suggests any valid >= threshold is fine.
    // We will try to maximize with a limit to avoid TLE.
    
    int min_size = (int)floor(sqrt(n / 2.0));
    
    // Maximum possible XOR sum
    // n <= 10^7 < 2^24. Max XOR is 2^24 - 1.
    int max_val = 1 << 24; 
    
    // Using vector<char> for used_xors to allow faster access than vector<bool> (bit manipulation)
    // 16MB memory usage is well within 512MB.
    static vector<char> used_xor(max_val, 0);
    // Reset or static? Static initializes to 0.
    // Since it's single run, no reset needed.
    // Wait, for safety, we should check memory usage. 16MB is fine.
    // However, static inside function allocates in BSS/data segment.
    
    vector<int> s;
    vector<bool> used_element(n + 1, false); // Track chosen elements
    
    // Random generator
    mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
    uniform_int_distribution<int> dist(1, n);

    auto last_time = chrono::steady_clock::now();
    chrono::duration<double> elapsed;
    
    // We want to find the largest possible set.
    // Given 1s time limit, we can afford roughly 10^7 - 10^8 operations.
    // Checking a candidate takes O(s.size()).
    // Let's limit the number of trials.
    // If we find valid elements, we add them.
    // If s.size() reaches a reasonable upper bound (e.g. 5000), we can stop or relax?
    // Let's just run a fixed number of attempts or until time expires.
    
    // Actually, for n=10^7, we can probably find ~2000-3000 elements easily.
    // We'll set a safety break.
    int trials = 0;
    const int MAX_TRIALS = 200000; // Adjust based on performance tests
    
    while (true) {
        // Check timer?
        elapsed = chrono::steady_clock::now() - last_time;
        if (elapsed.count() > 0.95) break; 
        
        int x = dist(rng);
        
        if (used_element[x]) continue;
        
        bool possible = true;
        // Check collisions
        for (int y : s) {
            if (used_xor[x ^ y]) {
                possible = false;
                break;
            }
        }
        
        if (possible) {
            s.push_back(x);
            for (int y : s) {
                if (x != y) { // Should not happen if we add unique
                    used_xor[x ^ y] = 1;
                }
            }
            // Actually, if we add x, new pairs are (x, s[i]).
            // We iterate s BEFORE adding x.
            // Correct logic:
            // Add x. Then mark x^s[i] for all s[i] in old S.
            // Wait, I wrote above `for (int y : s)` which iterates current s (including x)?
            // No, we need to store pairs carefully.
            // If x is added, pairs are (x, prev_y).
            // Mark x ^ prev_y.
        }
        
        // If we updated 's', we should update 'last_time'?
        // Wait, if I just iterate until time or trials limit.
        // Better strategy: Just keep checking random numbers until we have "enough"?
        // No, we want longest.
        // But we also have "Any valid S is accepted" if length >= min_size.
        // So if we reach min_size, we are safe, but trying longer is better.
        
        // Optimization: If s.size() becomes large, checking takes longer.
        // So we balance.
    }
    
    // Output
    cout << s.size() << "\n";
    for (int i = 0; i < (int)s.size(); ++i) {
        cout << s[i] << (i == (int)s.size() - 1 ? "" : " ");
    }
    cout << "\n";

    return 0;
}
```
Wait, one correction in the logic inside loop:
When `possible` is true, we add `x` to `s`.
Then we must update `used_xor`.
The pairs are formed between `x` and all currently in `s` (before adding `x`?).
No, pairs are between `x` and `y` where `y` is in the new set `s` (excluding `x` itself).
So correct update:
```cpp
        if (possible) {
            s.push_back(x);
            for (int y : s) { // s now includes x. Pairs are (y, x). But we double count?
                 // Pairs (y, x) and (x, y).
                 // We only care about uniqueness of value x^y.
                 // We can iterate up to previous size.
                 // Wait, if we just iterate all y != x, it covers all new pairs.
            }
        }
```
Better logic:
Store `prev_size`.
Loop `j` from `0` to `prev_size - 1`.
`used_xor[s[j] ^ x] = 1;`
Also `used_xor[x ^ s[j]]` is same.
Wait, do we need to mark `used_xor` for `s`?
No, `used_xor` tracks values generated by pairs.
Initial `used_xor` empty.
First element: no pairs.
Second element: pair (1st, 2nd) -> value marked.
Third element: (1st, 3rd), (2nd, 3rd) -> marked.
So when adding `x`, we iterate existing elements `y` in `S` (before adding) and mark `x^y`.
My code snippet had `for (int y : s)` which would include `x` if pushed first.
Need to ensure we don't iterate `x`.
`used_xor` initialized to 0.
Note: `x^y` is symmetric.
Also `x` must be in `[1, n]`.
If `s` is empty, loop doesn't run. `x` added.
Next, `y` is `s[0]`. Mark `x^s[0]`.
Correct.

Optimization:
Inside the loop, if we haven't met any requirement (min_size) AND trials are exhausted, we stop.
Actually, we should output even if size is small.
Wait, `used_xor` allocation.
Static vector is allocated at load time.
If memory limit exceeded?
16MB is very safe.
However, allocating `vector<char>` on heap inside `main` is safer or `static` to BSS.
Using `static` inside function puts it in data segment, avoiding stack overflow.
Good.

Double check time complexity.
Loop runs up to `MAX_TRIALS`.
Inner loop size `s.size()`.
Sum of `s.size()` over trials $\approx \sum_{k=0}^m k \approx m^2/2$.
If we find $m$ elements successfully.
But trials include failed checks.
If acceptance rate is $\approx 0.5$, then for each element we try $\approx 2$ numbers.
Total trials $\approx 2m$.
Cost $\approx \sum_{k=1}^{m} 2k \approx m^2$.
For $m=2000$, $m^2 = 4 \times 10^6$.
This is VERY fast.
So we don't need `MAX_TRIALS` limit if we rely on accepting elements quickly.
We should just keep running until we run out of time or find valid candidates run out?
Finding valid candidates won't run out easily because $m \ll n$.
Wait, if we run for full time.
If we just run until time limit (e.g. 0.95s).
We can afford $\approx 10^8$ ops.
$m \approx 2000$.
$m^2 \approx 4 \cdot 10^6$.
Even if we do 100 such passes, we are fine.
So basically, with random greedy, we converge very fast.
We can just output the result when loop finishes naturally?
Wait, loop finishes when `used_trial_count` reached? No.
We should just run indefinitely until time limit?
Or until we can't find more?
The only reason to stop is time.
So:
```cpp
while (chrono::steady_clock::now() - last_time < duration_cast<duration<double>>(0.95));
```
But calculating time repeatedly?
Just one loop.
Inside, generate `x`.
If `x` valid, add.
Else skip.
After loop (when timeout), print.

Wait, is it possible that random greedy gets stuck?
With $n=10^7$, we have lots of space.
We should find elements.
What if $n$ is small?
Code handles small $n$ naturally (dist range shrinks).
Is it possible we never find a valid element after initial ones?
Only if we filled all XOR combinations?
But $m \approx 2000$ consumes $\approx 2 \cdot 10^6$ XOR values.
Space $1.6 \cdot 10^7$. Plenty left.
So we will keep finding elements until `used_element` space is exhausted or we run out of time.
Wait, if we pick random elements, we might pick same `x` multiple times.
`used_element[x]` prevents this.
Also `used_xor` prevents conflicting `x`.
So the state space of valid `x` shrinks.
Eventually we might run out of valid `x`.
This happens when we saturated the system.
This is desirable (we found maximal set for the greedy strategy).

Final check on logic.
Problem says $S \subseteq \{1, \dots, n\}$.
Output size $m$.
Output elements.
Order any.
Constraints $n \le 10^7$.
Memory 512MB.
Time 1s.
My solution uses $\approx 16MB$ memory.
Operations $\approx O(\text{valid\_elements\_found} \times m + \text{failed\_checks} \times m)$.
If we find $K$ elements, and failure rate $F$, operations $\approx (K+F)K$.
Since failure rate likely decreases as $m$ increases.
Assuming $K \approx 3000$, $F \approx 10000$?
Ops $\approx 4 \cdot 10^7$. Very safe.

One minor point: `static vector<char> used_xor(max_val, 0);`
Does this initialize correctly on every run?
In a program execution, static is initialized once.
If multiple test cases in one file (not specified here), it should be reset.
Here "Input: A single integer n". Implies single test case per run.
So static is fine.
But to be safe/clean, move to `main` scope and let RAII handle it (allocates memory).
However, `vector` default constructor zeros out memory.
Allocating 16MB on heap is fine.
I will use local variable or static. Static avoids heap fragmentation or stack issues.
But 16MB on heap is totally fine.
I'll use `static` inside `main` or global. Global is safer.
Or just declare `vector<char> used_xor` locally.
Default init takes time. $16MB$ zeroing is fast (memset).
Local is fine.

One edge case: `min_size`.
$n=1$, `min_size` = 0.
My code outputs size 1 (element 1). Valid.
Is empty set allowed?
Subset $S \subseteq \{1\}$. $S=\emptyset$.
Pairs: none. Condition 1 holds.
Size $\ge 0$ holds.
But usually CP judges prefer non-trivial sets.
My code produces non-empty.
If $n=1$, it picks 1.
Condition: $a, b \in S, a < b$. If $|S| < 2$, condition vacuously true.
So $S=\{1\}$ is valid.

```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>

using namespace std;

void solve() {
    int n;
    if (!(cin >> n)) return;

    // Minimum required size
    int min_req = (int)(sqrt(n / 2.0));

    // Constraints
    // n <= 10^7.
    // Max possible XOR sum < 2^24.
    int MAX_XOR = 1 << 24; // 16,777,216

    // We use a flat array for XOR tracking. 
    // vector<char> uses 1 byte per entry -> ~16MB memory.
    // Safe within 512MB.
    // Initializing to 0.
    // Using dynamic allocation to ensure clean zeroing.
    vector<char> used_xor(MAX_XOR, 0);
    vector<int> S;
    vector<bool> used_element(n + 1, false);

    // Seeding RNG
    mt19937 rng(time(0)); // Time based seed for variety
    uniform_int_distribution<int> dist(1, n);

    auto start = chrono::steady_clock::now();
    
    // Timeout: 0.9 seconds
    auto end = start + chrono::seconds(900000000); // Approx 0.9s in steady clock (differs by platform)
    // Better: use duration comparison.
    // Actually, simple loop break condition.
    
    // Heuristic limit on trials to prevent TLE in worst case scenarios (e.g. if unlucky)
    // Though with n=10^7 we shouldn't get unlucky much.
    int trials = 0;
    const int MAX_TRIALS_LIMIT = 500000; 

    while (true) {
        // Check time
        if (chrono::steady_clock::now() >= start + chrono::milliseconds(900)) break;

        trials++;
        if (trials > MAX_TRIALS_LIMIT) break;

        int x = dist(rng);
        
        // If already selected, skip
        if (used_element[x]) continue;

        // Check if adding x creates duplicate XOR sums
        bool possible = true;
        for (int y : S) {
            if (used_xor[x ^ y]) {
                possible = false;
                break;
            }
        }

        if (possible) {
            S.push_back(x);
            used_element[x] = true;
            // Mark the new XOR sums
            for (int y : S) { // S now includes x. Loop goes over all. 
                // Wait, if we iterate S including x, x^x = 0.
                // Did we mark 0? No.
                // But we need to mark x^y for all y != x.
                // If we iterate S after push_back, we encounter x.
                // x^x = 0. used_xor[0] is 0. OK.
                // But we will mark x^x twice? No, logic is:
                // pairs are {a, b} with a!=b.
                // We iterate y in S.
                // If y==x, x^y=0.
                // Other pairs y!=x produce positive XOR.
                // Since XOR is commutative, we mark all.
                // But we marked them twice if we check both (x,y) and (y,x).
                // Actually, iterating all y in S (after adding) checks (x,y) for all y in S_old, AND (x,x).
                // We should only mark for y in S_old (which are y < x in our set construction order? No, random order).
                // Wait, if y comes before x in S, we mark x^y.
                // If we iterate all S, we encounter y and later x.
                // We mark x^y.
                // Later when we encounter x, we mark x^x=0.
                // So effectively we mark all pairs involving x and other elements.
                // Wait, x^x=0. Is 0 relevant?
                // Problem says a<b. 0 is not in range of valid XORs?
                // Wait, a!=b => a^b > 0 (since a!=b).
                // So 0 doesn't affect correctness if we don't treat it specially.
                // Just ensure we don't falsely detect collision on 0?
                // If we mark 0, future x=z, y=w s.t. x^y=0 (i.e. x=y).
                // Since we ensure x != y (by used_element check), x^y will never be 0 for valid checks.
                // So marking 0 is harmless but useless.
                // It might cause used_xor[0]=1.
                // But x^y can never be 0 for distinct x,y.
                // So it doesn't matter.
                
                // Wait, I need to mark x^y for y in S_old.
                // My code loops over S (which now includes x).
                // If I use `y` in loop, it includes x.
                // I can iterate `int i=0; i<S.size(); ++i;`? No `y=S[i]`.
                // Just iterate all.
                // However, `used_xor` should track valid XOR sums (non-zero).
                // If I mark 0, it's fine as long as I don't check it.
                // Checking `used_xor[x^y]` where x,y distinct implies x^y != 0.
                // So checking 0 is safe.
                
                // Optimization: We can iterate `const int& y : S`
                // But to avoid checking `x^x`, loop `int y : S; y != x`?
                // `x` is the newly added.
                // But `S` is a container.
                // Simpler:
                // Before push_back, loop. Mark. Then push_back.
                // This is cleaner.
            }
        }
    }

    cout << S.size() << "\n";
    for (int i = 0; i < (int)S.size(); ++i) {
        cout << S[i] << (i == (int)S.size() - 1 ? "" : " ");
    }
    cout << "\n";
}

int main() {
    fast_io(); // Need to define this if not included globally
    // Re-declare fast_io if necessary or inline
    solve();
    return 0;
}
```
Wait, I realized I need to correct the update part.
If I push `x` then iterate `S` (containing `x`), I do `used_xor[x^x] = 1` (value 0).
And for `y` in `S_old`, I mark `x^y`.
Is this symmetric?
Yes.
Wait, if I use `x` as the newly added element, and iterate `S` after adding, I will see `x` itself.
So I must skip `y == x`.
Actually, better to mark before adding.
```cpp
        if (possible) {
            // Mark pairs
            for (int y : S) {
                used_xor[x ^ y] = 1;
            }
            // Then add
            S.push_back(x);
            used_element[x] = true;
        }
```
Wait, if I mark first, `S` does not contain `x` yet.
So loop `y` iterates over previous elements.
We mark `x^y` for all old `y`.
Then we add `x`.
This is correct and avoids self-loop.
Also `used_xor` marks are set.
Wait, `used_xor` persists across checks?
Yes. We accumulate pairs.
Wait, if I find a valid `x`, I add it.
Then next time, I check `x_new` against `S` (which contains `x`).
Pairs are marked as `(x_new, y)` for all `y` in `S` (old and `x`).
So we accumulate all pairs in `used_xor`.
Wait, do we need to check against ALL pairs ever generated?
Yes, because we need ALL pairwise XORs of the set `S` to be distinct.
So if `S` grows, the set of pairs grows.
All these values must be distinct.
So `used_xor` must persist.
Correct.

Wait, one logic error in my check.
If I iterate `y` in `S` (which contains `y`) and check `used_xor[x^y]`.
If `x^y` was already set (from a previous pair `u, v` where `x^y == u^v`), then `x` is invalid.
This is correct.
But we also need to ensure `x` doesn't create a pair equal to another pair involving `x`.
Wait, if `x` is new, its pairs are `{x^y}` for `y \in S_old`.
We check if any of these `x^y` values exist in `used_xor`.
`used_xor` contains all pairwise XORs of `S_old`.
So if any `x^y` is in `used_xor`, it means a collision occurred.
This is the correct check.

Wait, do I need to ensure distinctness of pairs within `S` itself?
Initially `used_xor` is empty.
Add 1st element. No pairs. `used_xor` empty. OK.
Add 2nd. Pairs `{s[0]^x}`. Check against `used_xor` (empty). OK.
Add `used_xor[s[0]^x] = 1`.
Add 3rd. Pairs `{s[0]^x_new, s[1]^x_new}`.
Check if these are in `used_xor`.
`used_xor` contains `{s[0]^s[1]}`.
So if `s[0]^x_new == s[0]^s[1]` -> `x_new == s[1]`. Impossible since distinct.
But if `s[0]^x_new == s[0]^s[1]` is checked.
Wait, if `x_new` creates collision, it's detected.
Also we must ensure that `x_new` doesn't form collision with *other* pairs.
E.g. `a^b = c^x_new`.
Here `a,b,c` are in `S_old`.
If `a^b` is in `used_xor`. And we check `c^x_new`?
We check `x_new` vs all `y`.
Wait, I said we check `x^y` for `y \in S_old`.
Does this cover all potential collisions?
The condition is: all $a \oplus b$ ($a,b \in S, a<b$) are distinct.
Let $S_{new} = S_{old} \cup \{x\}$.
Pairs in $S_{new}$ are:
1. Pairs entirely within $S_{old}$. Their XORs are already in `used_xor` (and must be distinct).
2. Pairs involving $x$: $\{x \oplus y \mid y \in S_{old}\}$.
   We need these to be distinct among themselves (guaranteed since $x$ is fixed and $y$ distinct).
   And we need these to be distinct from values in category 1.
   So we check if $x \oplus y \in used\_xor$ for any $y \in S_{old}$.
   This is exactly what the loop `for (int y : S) if (used_xor[x ^ y])` does.
   
   Wait, I assumed we mark pairs as we go.
   `used_xor` stores $\{u \oplus v \mid u,v \in S_{old}, u<v\}$.
   So yes, logic holds.

One small bug potential:
If I add `x` to `S` and update `used_xor`.
Next candidate `z`.
We check `z^y` for `y \in S`.
Does `z^y` need to be distinct from `x^w`?
Yes, `used_xor` contains all values generated so far.
`z^y` is checked against `used_xor`.
If `z^y == x^w` (where `x,w` in `S_old`?), wait.
`x` is in `S_old` for `z`'s check.
`x^w` is in `used_xor`.
So yes, checking `used_xor[z^y]` detects collision with any previous pair.
Logic is sound.

Wait, `used_element` initialization size.
If `n` is $10^7$, `vector<bool>` is $10^7$ bits $\approx 1.25$ MB.
OK.

Double check headers. `<vector>`, `<cmath>`, `<iostream>`, `<chrono>`, `<random>`, `<algorithm>`.
Everything standard.
`mt19937` requires `<random>`.
`uniform_int_distribution` requires `<random>`.
`vector` requires `<vector>`.
`sqrt` in `<cmath>`.

Final code assembly.
One detail: `fast_io` function call.
Put it in main or global.
I will put it in main to avoid scope issues.
Also `return 0`.

The prompt asks for ONLY the code.

```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>

using namespace std;

void fast_io() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
}

int main() {
    fast_io();

    int n;
    if (!(cin >> n)) return 0;

    int min_size = (int)floor(sqrt(n / 2.0));

    int MAX_XOR = 1 << 24; // 2^24 covers up to 1.67e7, sufficient for n=10^7
    // n <= 10^7, so max value < 2^24.
    // vector<char> is used for speed (byte access) vs bitset.
    // Size is ~16MB, within 512MB limit.
    vector<char> used_xor(MAX_XOR, 0);
    vector<int> S;
    vector<bool> used_element(n + 1, false);

    // Seeding RNG with current time
    mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
    uniform_int_distribution<int> dist(1, n);

    auto start = chrono::steady_clock::now();

    // While loop until we find a set of sufficient size or timeout
    // We aim for a valid set. Since we want longest, we keep running.
    // But to prevent TLE, we enforce time limit.
    
    // Safety break to prevent hanging
    int trials = 0;
    const int MAX_TRIALS = 500000; // Upper bound on random checks

    while (true) {
        // Timeout check: 0.9 seconds
        if (chrono::steady_clock::now() - start > chrono::milliseconds(900)) break;
        if (trials > MAX_TRIALS) break;

        trials++;
        int x = dist(rng);
        
        if (used_element[x]) continue;

        bool possible = true;
        for (int y : S) {
            if (used_xor[x ^ y]) {
                possible = false;
                break;
            }
        }

        if (possible) {
            S.push_back(x);
            used_element[x] = true;
            for (int y : S) { 
                 // Note: After push_back, S contains x.
                 // We only care about x^y for y in OLD_S.
                 // But iterating all is easier?
                 // If we iterate all S, we encounter x. x^x = 0.
                 // used_xor[0] set to 1. 
                 // Since x != y for distinct pair, 0 is never checked for collision 
                 // (because collision requires x^y == some old XOR, which is >0).
                 // So checking used_xor[0] is safe.
                 // Wait, if used_xor[0] is set, and later we check some x, y such that x^y=0?
                 // That means x=y. But we check x against unique elements.
                 // So used_xor[0] being 1 is harmless.
                 // But to be pedantic, we should only mark x^y for y in S BEFORE pushing.
                 // Or simply iterate and exclude x.
                 // Let's modify loop to exclude x to be cleaner.
                 // But we can't easily access 'old S' without saving size.
                 // Save size.
            }
            // Let's rewrite the loop for clarity in head:
        }
        // Actually, rewriting loop:
        // if (possible) {
        //    for (int y : S) used_xor[x ^ y] = 1; // This marks all pairs (x, y) where y is in current S
        //    // Wait, S includes x. So loop includes x.
        //    // x^x = 0. Marked.
        //    // Others marked.
        //    // Correct.
        // }
        // BUT WAIT.
        // If I mark `x^y` where `y` is in `S` (which includes `x`), I am marking `x^y` for pairs formed with `x`.
        // This adds `x` as an endpoint to existing pairs.
        // BUT, I haven't added `x` to `S` yet?
        // I need to mark AFTER adding `x`? No, I need to add `x` and update state.
        // State is `S` and `used_xor`.
        // If I mark first, then add `x`, I use `S` (old).
        // If I add `x` first, then mark, I use `S` (new).
        // In `used_xor` update, I need to record all pairs involving `x` and any existing element.
        // Existing elements are those in `S` before push.
        // So loop should be over `S` (old).
    }

    // Correct implementation with loop logic:
    // Check loop is over `S` (before push).
    // Update loop is over `S` (before push).
    // So `S` needs to be tracked.
    
    // Let's restructure to be sure.
    
    // Re-run logic mentally:
    // Possible check: for (int y : S) if (used_xor[x^y]) -> fail.
    // If success:
    //   for (int y : S) used_xor[x^y] = 1;
    //   S.push_back(x);
    //   used_element[x] = true;
    
    // This works perfectly. `S` inside the loop refers to old set.
    // So we don't need separate tracking.
    // I will adopt this structure.
```

Final Code Structure.
</think>

```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>

using namespace std;

void fast_io() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
}

int main() {
    fast_io();

    int n;
    if (!(cin >> n)) return 0;

    // Minimum required size
    int min_size = (int)floor(sqrt(n / 2.0));

    // n <= 10^7.
    // Max possible XOR sum < 2^24 (since 2^24 = 16,777,216 > 10^7).
    // We use an array/vector to track which XOR values have been used.
    // vector<char> provides fast byte access and consumes ~16MB memory, well within 512MB.
    int MAX_XOR = 1 << 24;
    vector<char> used_xor(MAX_XOR, 0);
    
    vector<int> S;
    vector<bool> used_element(n + 1, false);

    // RNG setup
    mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
    uniform_int_distribution<int> dist(1, n);

    auto start = chrono::steady_clock::now();

    int trials = 0;
    // Limit trials to prevent TLE, though random greedy converges quickly.
    const int MAX_TRIALS = 500000; 

    while (true) {
        // Timeout check: approximately 0.9s
        if (chrono::steady_clock::now() - start > chrono::milliseconds(900)) break;
        if (trials++ > MAX_TRIALS) break;

        int x = dist(rng);
        
        // Skip if already selected
        if (used_element[x]) continue;

        // Check validity: ensure x^y is not in used_xor for any y in S
        bool possible = true;
        for (int y : S) {
            if (used_xor[x ^ y]) {
                possible = false;
                break;
            }
        }

        if (possible) {
            // Mark new pairs before adding to S
            for (int y : S) {
                used_xor[x ^ y] = 1;
            }
            S.push_back(x);
            used_element[x] = true;
        }
    }

    // Output result
    cout << S.size() << "\n";
    for (int i = 0; i < (int)S.size(); ++i) {
        cout << S[i] << (i == (int)S.size() - 1 ? "" : " ");
    }
    cout << "\n";

    return 0;
}
```
````
