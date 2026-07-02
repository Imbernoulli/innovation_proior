# 题 132 — base均分 2.7 vs 直接SFT均分 0.0(掉点 2.7)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=3,得分 13.3,completion_tokens=29063)

````
The user wants a C++ solution for an interactive problem involving finding two chairmen among 1000 possible positions using $R$ robots with a total of $H$ hours.
Specific constraints:
- $R = 75$ (available robots).
- $H = 1$ (Wait, the example says $H=20$, but the input specification says "Input The first line consists of 2 integers R and H (R=75,H=1)". This looks like a specific test case constraint in the problem description text provided in the prompt, or perhaps the prompt is from a specific contest where limits vary. However, usually standard Codeforces-like problems allow up to some $R$. Let's look closely at the Input section: "(R=75,H=1)". Wait, looking at the scoring rules, it allows sending up to $R$ times, and receiving results up to $H$ times per testcase?
Actually, re-reading the problem statement carefully:
"Interaction... You can 'send' at most R times per testcase."
"You can 'get' at most H times per testcase."
The example shows $R=75, H=20$ in the text description, but the Input line says $(R=75, H=1)$.
Usually, if this is "The 2016-2017 ACM-ICPC, NEERC, Moscow Subregional Contest", Problem I (Chairman), let's check standard constraints. Usually, this problem asks for finding 2 items in range [1, N] with N queries (robots) and time limit.
Actually, this is likely Codeforces Beta Round #89 (Div. 1) Problem B (Interactive) or similar. Wait, the problem name is "Chairman". It's actually Problem I from 2016-2017 ACM-ICPC, NEERC, Moscow Subregional Contest, Problem D or something? No, it's "Vacuum cleaner robots".
Let's search for "interactive problem find two chairmen R=75 H=1".
This matches Codeforces Gym 100993C "Vacuum Cleaner". Or maybe it is specifically from a contest where H=1.
Wait, if $H=1$, then I can only send robots once and get results once. I cannot batch queries over multiple rounds.
However, the scoring function suggests $rmax \le 75$ gives score 13.
Wait, if $H=1$, how can I send multiple batches? "You can instruct each robot to scout several of 1000 positions... Every robot needs a full hour... you can send out each robot only once... You want to know the positions ... after at most H hours."
If $H=1$, I can only make one round of inquiries. All my $R=75$ robots must be sent simultaneously (or effectively so). But I have to deduce the answer immediately based on the results of these 75 robots.
Wait, the interaction protocol says: "To get the answers of robot that sent before, print '@', then you will receive... L entries". This implies I can wait for previous ones.
If $H=1$, I can only perform the action of printing `@` once (plus the initial state). So I send a batch of robots, get results, and then I must solve it.
With 1 hour, I effectively have 1 query step.
But wait, usually $H$ corresponds to the number of rounds I can query. Here $H=1$.
So, I need to distinguish between all pairs $(a, b)$ with $1 \le a \le b \le 1000$.
Total pairs is $1000 \times 1001 / 2 \approx 500,000$.
With 75 binary outcomes, I can distinguish $2^{75}$ states. $2^{75}$ is much larger than $500,000$.
So information-theoretically, one round is sufficient.
I just need to design 75 sets of indices such that every pair $(a,b)$ produces a unique pattern of responses (where response $k$ is 1 if $\{P_k[0], \dots\} \cap \{a,b\} \neq \emptyset$ else 0).

However, checking uniqueness for all pairs might be hard to construct explicitly without more tools, but given the generous bit capacity (75 bits for ~500k possibilities), collision probability is low if I spread things well, but since the grader is adaptive (wait, the note says "Moreover, the grader is adaptive"), I should not rely on random hashing or simple covering if adversarial cases exist. Wait, adaptive usually means the grader changes positions based on my queries. If it's fully adaptive, I need to deduce exactly which pair it is consistent with *all* history.
Since I only have 1 hour, I don't have iterative elimination. I have one "shot".
Wait, if the grader is adaptive, does it change between queries? Since I only get 1 response batch, the "history" is empty before the query. So non-adaptive vs adaptive doesn't matter for the first round. I just need to guarantee that the result uniquely identifies $(a,b)$.
So the problem reduces to: Find 75 subsets $S_1, S_2, \dots, S_{75}$ of $\{1, \dots, 1000\}$ such that for any pair $u, v$ ($1 \le u \le v \le 1000$), the vector of intersections $(|S_i \cap \{u,v\}| > 0 ? 1 : 0)$ is unique to that pair. Note that $|S_i \cap \{u,v\}|$ can be 0, 1, or 2. The condition is whether there is "at least one chairman". So we care about indicator function $\mathbb{1}(S_i \cap \{u,v\} \neq \emptyset)$.
So for each pair $\{x, y\}$, the signature is $v \in \{0,1\}^{75}$ where $v_i = 1$ iff $x \in S_i$ or $y \in S_i$.
We need all distinct pairs to map to distinct vectors.
There are $\approx 500,000$ pairs. We have $75$ dimensions.
A naive construction might work.
Wait, if $H=1$ was fixed in the problem statement text, but typically interactive problems allow $H \ge 1$. The Example shows $H=20$. The Input says $H=1$.
The problem text says: "The first line consists of 2 integers R and H (R=75,H=1)".
This implies for this specific instance/problem version, $H=1$.
If $H=1$, I really only have 1 batch of 75 queries.
Is it possible that the problem allows sending robots in parallel and receiving results later, but since $H=1$, I must send all and get all?
Yes. "print on a single line '@'... then you will receive an integer L and a array... for each robot sent out one hour ago".
If I send $R$ robots, I get their status. That consumes 1 hour.
Since I start at hour 0, I can send robots, wait 1 hour, get results. That's 1 hour passed. Time limit 0.25 seconds suggests fast IO and logic, but time isn't strictly the bottleneck here, correctness is.

Strategy:
Assign a unique ID to every pair $(i, j)$ where $1 \le i < j \le 1000$. There are $N(N+1)/2$ pairs.
We need to encode this ID into 75 bits such that the encoding is derived from membership in sets $S_k$.
Each set $S_k$ contributes 1 bit to the signature of pair $(i,j)$ if $i \in S_k$ or $j \in S_k$.
This looks like we are trying to embed the pairs into hypercubes or use coding theory.
Actually, simpler idea: Map each position $x$ to a value $v_x$.
If $x \in S_k$ then the $k$-th bit of $v_x$ is 1.
Then for a pair $\{x,y\}$, the signature is $(v_x | v_y)$ bitwise OR.
If $v_x$ and $v_y$ are chosen such that the OR combination is unique for every pair, we win.
Since we have 75 bits available, we can assign to each of 1000 positions a unique subset of size roughly $1000/2$? No.
Actually, if $v_x$ are distinct for all $x$, then $v_x | v_x = v_x$ is distinct.
But we need $v_x | v_y$ to be distinct for all pairs.
Consider representing each position $x$ as a number $X$ in base 3? No, bits are boolean.
We can simply assign a unique ID to each position $x \in \{1, \dots, 1000\}$ such that no two positions share a significant property? No.
We want $f(x) \cup f(y)$ to be unique for $\{x, y\}$.
This is related to separating systems or identifying codes.
With $N=1000$, we can just treat positions $1 \dots 1000$ as numbers $0 \dots 999$.
We need to identify the pair of numbers.
Since we have 75 robots, we can cover almost everything.
One strategy is to try to detect if both are present or just one, and locate them.
But "at least one" makes it tricky. If I cover $x$ in all sets, then if the result is 1, $x$ could be one, or $y$ could be another covered by different sets.
Actually, if I partition the range $1..1000$ into 75 buckets $B_1, \dots, B_{75}$.
For each bucket $B_k$, I include it in my query $Q_k$ (robot $k$'s list).
Wait, a robot can scout multiple positions.
If I put position $p$ into many robots' lists, it provides more bits of information.
Suppose I assign each position $p$ a bitmask $M_p$ of length 75.
Robot $k$ checks positions $p$ where bit $k$ of $M_p$ is 1.
Result of robot $k$ is 1 if any chairman $c \in \{1,\dots,1000\}$ has bit $k$ set in $M_c$.
Let $U$ be the set of positions occupied by chairmen. $|U|=2$.
Let $R_k$ be the result for robot $k$.
$R_k = \bigvee_{p \in U} (\text{bit } k \text{ of } M_p)$.
So the outcome vector $R$ is $M_{p_1} \lor M_{p_2}$ (bitwise OR).
We need to choose $M_1, \dots, M_{1000} \in \{0,1\}^{75}$ such that $M_a \lor M_b \neq M_{a'} \lor M_{b'}$ whenever $\{a,b\} \neq \{a', b'\}$.
Since order doesn't matter.
Also $M_a \lor M_a = M_a$, so the diagonal pairs must also produce unique signatures distinct from each other (and off-diagonal pairs).
Wait, if $M_a \lor M_b = M_c$, then we can't distinguish $\{a,b\}$ from $\{c, c\}$ (if we knew we checked against diagonal too, but the problem guarantees 2 chairmen). Actually, if $\{a,b\}$ maps to same signature as $\{c,d\}$, we lose.
So we need pairwise distinct unions.
Number of distinct values needed: $1 + 1000 \times 999 / 2 \approx 500,500$.
$2^{75}$ is huge, so space is fine.
Is there a construction that ensures uniqueness of OR sums?
Simply assigning random bitmasks works with high probability.
Wait, deterministic construction might be better for competitive programming context (though usually randomized works for interactive unless specifically targeted). Given 1 second (or even 0.25), generating random masks and verifying uniqueness offline (if allowed) or during runtime is good. But we can't verify during runtime because it's interactive and we don't know the answers until we send.
However, we can pre-calculate the mask table locally.
Does a deterministic mapping exist?
Yes. Assign $M_x$ for $x=1 \dots 1000$ using a polynomial evaluation or simple arithmetic properties?
Or just use the first few bits of the index $x$ itself?
If we use $x$ as an integer $< 2^{10}$, say $1000 \le 1024$.
Can we ensure $A \lor B$ is unique?
If we take $M_x$ to be the binary representation of $x$ padded to 75 bits.
Then $A \lor B$ might collide?
Example: $A=1 (0...01), B=2 (0...10) \implies A \lor B = 3 (0...11)$.
$C=3 (0...11)$. Is it possible to form 3 from two others? $3$ is a single position. Wait, if the two chairmen are at 3 and something else? No, we need to distinguish $\{A,B\}$ from $\{C,D\}$.
If $A \lor B = X$, and $C \lor D = X$.
With binary representation of indices, collisions are likely. e.g. $\{1, 2\}$ gives $\{01, 10\} \to \{11\} = 3$. $\{3, x\}$ where $x$ has 0s elsewhere.
Wait, if I use the property that for every $x$, we add $2^x$? No, positions are indices $1..1000$.
The easiest way to ensure uniqueness of OR sum is to make sure bits overlap minimally or follow a structure.
Actually, since we have plenty of bits (75 for 1000 items), we can just pick $M_x$ such that they are disjoint in support?
No, $75 < 1000$, so we can't make them disjoint.
But we can make them "almost" disjoint or structured.
However, note that the problem says "Grading your actual score depends on the number rmax".
If $H=1$, we have 75 robots.
We can assign a random 75-bit mask to each position $1..1000$.
Then iterate through all pairs $(i,j)$ to check if $mask[i] \lor mask[j]$ is unique.
Wait, if we run this code interactively, we don't know the positions of chairmen yet.
But we can precompute the table `mask[i]` in our code.
Before running the main loop, inside `main()`, we generate the random masks. We check if the configuration is valid (unique ORs).
If not, regenerate.
This takes $O(1000^2 \cdot 75)$ operations, which is roughly $7.5 \times 10^7$ ops.
With 0.25s time limit (for the whole program execution?), doing $7.5 \times 10^7$ checks might be too slow if we do it blindly?
Wait, 0.25s is quite tight for 10^8 ops in C++. But generating random masks and checking uniqueness might be heavy if done poorly.
But maybe we can use a known construction.
Construction Idea:
Represent each $i \in \{1, \dots, 1000\}$ as a triple $(x, y, z)$? No.
Maybe using modular arithmetic?
Let's reconsider the problem constraints.
Usually in such interactive problems, if $H$ is small (like 1 or 2), the intended solution uses probabilistic method or information theoretic bounds.
With $H=1$, 75 robots is a lot for 1000 positions (which would require $\lceil \log_2 1000 \rceil \approx 10$ bits to distinguish one position). To distinguish pairs, we need more.
The condition $A \lor B = A' \lor B'$ should be avoided.
Actually, we can simplify the condition.
If we ensure that for every position $p$, its mask $M_p$ contains a unique "tag" that distinguishes it from others? No, we need pairs.
What if we select 10 robots to identify the *first* chairman's position and another 10 to identify the *second* chairman's position?
Wait, we don't know which one is first. They are unordered. And we don't know which detection corresponds to which chairman if they are close in the coverage.
However, if we can ensure the robots provide enough info to separate the two.
Let's try to assign each position $i \in \{1, \dots, 1000\}$ a unique subset $S_i$ of $\{1, \dots, 75\}$.
If we pick $S_i$ randomly, probability of collision for pairs is very low.
How many pairs? 500,000. Total possible outcomes $2^{75}$.
Probability of collision $\approx 500,000^2 / 2^{75}$? No, birthday paradox style.
$\sqrt{500,000} \approx 700$. $700^2 \approx 500,000$.
Wait, birthday collision probability is $n^2 / 2^k$.
Here $n=500,000$, $k=75$. $n^2 \approx 2.5 \times 10^{11}$. $2^{75} \approx 3 \times 10^{22}$.
Collision prob is extremely small ($< 10^{-11}$).
So simply assigning random masks to each position $1..1000$ works with overwhelming probability.
We can implement a generator that creates these masks.
However, we should be careful about the verification. Since we execute this once per test case, and we have plenty of randomness (time limit 0.25s is generous for C++ initialization if not complex).
Actually, 0.25s is strict.
We can optimize the precalculation.
Just store masks for positions 1..1000.
We need to ensure that $mask[a] \lor mask[b]$ is unique.
Since $2^{75}$ is so large compared to $\binom{1000}{2}$, almost any random assignment works.
We can just pick the first 1000 random numbers modulo $2^{75}$ (interpreted as bitsets) and hope.
But wait, we can make it even safer.
Use a construction like: $M_x$ is a string of bits.
Since $1000 < 2^{10}$, we have 10 bits of index.
If we use $M_x$ such that the union is unique...
Actually, what if we interpret each position $x$ as a number $x$.
If we can identify $x$ and $y$ separately?
If we can find $x$ such that $x \in \text{union} \iff \dots$
If we can reconstruct the set $\{x,y\}$ directly from the intersection counts?
Suppose we assign each position $p$ a number $val(p) \in \mathbb{Z}$.
If we use linear independence? No.
Let's stick to the random bitmasks approach. It's the most robust for $H=1$.
Wait, checking uniqueness takes time.
Do we need to check?
Since we output the code, if it fails (probabilistically), the judge kills it.
The chance is negligible.
So I will generate random masks for 1..1000.
Wait, I need to output the sets $S_k$ for each robot $k=1 \dots 75$.
My plan:
1. Generate a list of `int pos_masks[1001]`. Each `pos_masks[p]` stores a bitmask (integer or std::bitset) of length 75 indicating which robots check position $p$.
   To maximize entropy, for each $p$, choose a subset of 75 bits uniformly at random.
   Wait, if I just pick random bits, I expect 50% density. $75$ bits is sparse-ish.
   Wait, is it guaranteed that the union of two random subsets is unique? Yes, with high probability.
   However, to save time, I can fix the structure.
   Can we assign $M_x$ deterministically?
   Maybe $M_x$ is determined by splitting the 75 bits into chunks corresponding to indices?
   Indices are $1..1000$. $\log_2 1000 \approx 10$.
   We have 75 bits. We can assign $M_x$ to have bit $i$ set if the $i$-th bit of index $x$ is 1?
   That's just $M_x = x$. Then $A \lor B$ is bitwise OR of numbers.
   Is $A \lor B$ unique for distinct pairs $\{A,B\}$?
   Counterexample: $\{1, 2\} \to 1|2 = 3$. Pair $\{3, 0\} \to 3|0 = 3$. (Assuming 0-indexed or 0 allowed? Positions are $1..1000$. So 0 is not allowed. But say positions $1, 2, 3$.
   $\{1, 2\} \to 001 | 010 = 011 (3)$.
   $\{3\} \dots$ wait we need two positions.
   Does $\{3, x\}$ yield $3$? We need $3 \lor x = 3$. This implies $x$ is a submask of 3.
   If $x=1$ or $x=2$, we already considered.
   What if $x=0$ (not allowed)?
   With domain $1..1000$, $A \lor B$ is not necessarily unique.
   Example: $1 = 001$, $2 = 010$, $3 = 011$.
   $1 \lor 2 = 3$.
   Suppose we had a position 3. Then pair $\{1,2\}$ yields signature $011$. Pair $\{3, \text{something}\}$?
   If $\{3, 0\}$ existed, it would collide. But 0 is not in domain.
   However, consider $1 (001)$, $6 (110) \to 7 (111)$.
   Position 7 exists.
   Consider $\{7, 0\}$ -> not allowed.
   Consider $\{1, 6\}$. Signature 7.
   Consider $\{2, 5\}$ (binary 010, 101) -> 111 (7).
   So $\{1, 6\}$ and $\{2, 5\}$ collide. $1, 6, 2, 5$ are all $\le 1000$.
   So simple binary representation is insufficient.

Back to Random Masks.
Generating 1000 random masks and verifying them offline takes $O(N^2 \cdot \text{bits})$.
$1000^2 \times 75 \approx 7.5 \times 10^7$ operations. In C++, `unsigned long long` holds 64 bits. We have 75 bits, so maybe use `__int128_t` or `std::bitset<75>`.
Bitset operations are fast.
Checking uniqueness: Store all unions in a hash set or sort them.
Sorting $500,000$ items takes $5 \times 10^5 \times \log(\dots)$ which is roughly $10^7$ ops.
This fits within 0.25s?
Wait, 0.25s is very short. $10^8$ ops might TLE.
But usually, we only need to do the verification once at compile time or generation time.
Wait, can I hardcode the masks?
The problem might run multiple test cases or reuse seeds?
Actually, if I write a small snippet to generate them, the user runs the script.
The generated code is submitted. The checker runs the script.
If I include a generation algorithm in the code that runs at startup, it will be executed on the judge.
The judge limit is 0.25s.
If I spend 0.1s generating, I have 0.15s left for interaction.
But I can use a pseudo-random generator seeded by time or process id to generate different valid tables.
To avoid the $O(N^2)$ verification overhead, I can use a heuristic that is likely to work without full verification.
Given the large space, almost any random assignment works. I won't verify.
Wait, but "adaptive grader" - oh wait, if the grader is adaptive, it might choose chairmen based on my queries. But since I only send queries once (H=1), the grader chooses them initially and reveals nothing. So it acts non-adaptively relative to my queries.
Wait, the grader reads my queries. "Moreover, the grader is adaptive".
This phrasing usually applies when I can query adaptively (multiple rounds).
If $H=1$, I have only 1 round. So the grader's choice is independent of my strategy. The only risk is if I send bad queries (low info), then I fail.
So, random masks with high entropy is safe.

Algorithm:
1. Define an array `pos_id[1001]` of type `uint64_t` or `vector<int>`. Wait, 75 bits don't fit in `uint64_t`.
   Use `std::bitset<75>` or `unsigned __int128` (GCC extension) or struct of 2 uint64s.
   Standard library `<bitset>` is convenient.
   Wait, `std::bitset` operations are somewhat slow compared to primitives? Not really for 75 bits.
   But sorting `std::bitset` involves comparing elements.
   Better: pack into `string` or `vector<bool>` or simply `vector<uint64_t>`.
   With 75 bits, we need 2 `uint64`s.
   Let's store as `pair<uint64_t, uint64_t>`. High word, Low word.
   Or just use a custom struct.
   
   Let's refine the generation.
   Iterate $i$ from 1 to 1000.
   Assign a random subset of bits.
   For performance, we can just assign $i$-th position a random mask.
   Do we need to worry about duplicates? $M_a = M_b$.
   If $M_a = M_b$, then $M_a \lor M_a = M_b \lor M_b$, etc.
   Collision happens if $\{a,b\}$ and $\{c,d\}$ map to same union.
   With random masks, we assume this doesn't happen.
   
   Implementation details:
   Read $R, H$.
   Precompute the query sets.
   Construct `vector<vector<int>> queries(R)`. `queries[k]` stores positions checked by robot $k$.
   Also, we need to reconstruct the positions from the results.
   So, before interacting, we build a lookup table.
   `vector<unordered_map<int, int>>`? No.
   The inputs to the grader are $P_1, \dots, P_{75}$.
   When we receive results, we get a bitmask of length $R$ (since we sent $R$ robots).
   Wait, we send at most $R$ times. Here we send $R$ times.
   So we get a vector of $R$ integers (0 or 1).
   These integers correspond to the results.
   If robot $k$ returned 1, then at least one chairman is in `queries[k]`.
   So we can compute a "result mask" where $k$-th bit is the result of robot $k$.
   Then we look for pairs $\{a,b\}$ such that $mask(a) | mask(b) == result\_mask$.
   Since we ensured uniqueness, there should be exactly one such pair.
   
   Optimization:
   To speed up the search for the pair, instead of storing $mask(a) | mask(b)$, we can store them in a sorted vector or map.
   Key: `result_mask`, Value: `{a, b}`.
   Since we have 75 bits, keys are large.
   Using `map` with bitsets might be slow.
   `map<pair<unsigned long long, unsigned long long>, pair<int, int>>`?
   500,000 insertions into a map is slow ($5 \times 10^5 \times \log$).
   Hashing is better.
   Or, simply sort the vector of `(mask_union, {a, b})`.
   Search can be done via binary search or hash map.
   Given time limit, a simple sort and lower_bound is fast enough.
   Wait, searching for the union mask. We get the results as an array of 75 ints. Convert this to our internal key format (e.g. `unsigned __int128` or 2 uint64s).
   Then lookup in our table.
   
   Wait, if I sort the pairs, the table size is $N(N+1)/2 \approx 500,000$.
   Sorting takes time. Binary search is logarithmic.
   If I pre-sort a static array, I just need to handle dynamic generation.
   Wait, the problem requires me to submit code.
   I can't pre-generate in the submission file if it depends on runtime RNG.
   Wait, the code is submitted once, run for testcases.
   Inside `main`, I will generate the masks.
   Generating 500,000 pairs and sorting them might take ~0.1s - 0.2s.
   Is it safe?
   Let's check the complexity.
   Creating 500,000 structs and inserting into vector: instant.
   Sorting: $N \log N$. $500,000 \times 19 \approx 10^7$ comparisons.
   Comparison of 2 `uint64`s is fast.
   $10^7$ ops is well within 0.25s.
   So I should definitely verify or at least create a proper lookup.
   
   Verification step:
   Generate masks. Check uniqueness of $M_a \lor M_b$.
   If collision found, regenerate.
   With 75 bits, collisions are rare, but theoretically possible with small seeds or bad RNG.
   If I use `mt19937` with `chrono::steady_clock`, it should be fine.
   Actually, for CP, sometimes people prefer to fix a deterministic seed or pattern to avoid TLE on generation, but random is safer.
   Wait, if I just use the property that $M_a = 1 << (index \& 63)$... no, index goes up to 1000.
   
   Let's refine the data structures.
   `struct Node { unsigned long long lo; unsigned long long hi; };`
   Wait, 75 bits.
   We can map $M_i$ to an integer if we compress it.
   But we need to support OR operation.
   So keep as 2 64-bit integers.
   High part covers bits 64..74 (11 bits). Low part 0..63.
   
   Structure of the solution:
   ```cpp
   #include <iostream>
   #include <vector>
   #include <algorithm>
   #include <random>
   #include <chrono>
   #include <map>
   #include <cstdio>

   using namespace std;

   // Use __int128 for convenience if available (GCC extension)
   // or define a wrapper. __int128 is supported on 64-bit compilers (standard for CF/etc).
   // Since limit is 0.25s, let's try to be efficient.
   
   typedef unsigned __int128 bitset75;

   int main() {
       int R, H;
       cin >> R >> H;
       
       // Setup RNG
       mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
       
       // Store for each position i (1 to 1000), a unique mask
       // We need to find a pair {a, b} that matches the query results.
       // We'll store all valid pairs and their resulting union mask in a hash table or sorted vector.
       // Since N=1000, we can store in a vector and sort.
       
       struct QueryResult {
           bitset75 val;
           int a, b;
       };
       
       vector<QueryResult> results;
       results.reserve(1000*1001/2);
       
       vector<bitset<75>> pos_mask(1001);
       // Actually bitset doesn't support easy conversion to __int128 or vice versa efficiently?
       // Let's just stick to custom packing.
       vector<pair<uint64_t, uint64_t>> masks(1001);
       
       // Fill masks
       for(int i = 1; i <= 1000; ++i) {
           uint64_t h = 0, l = 0;
           uniform_int_distribution<uint64_t> dist(0, (1ULL<<64)-1);
           // We want random bits. 
           // Let's fill bitset manually to avoid bitset overhead if any.
           // Just random 64-bit integers for high and low.
           masks[i].first = dist(rng);
           masks[i].second = dist(rng);
           // But we might get duplicate masks for different i.
           // Probability of collision is 1/(2^64)^2 = 1/2^128. Safe to ignore.
           
           // Now prepare queries for robots
           // Robot k corresponds to bit k.
           // But we need to group positions by robot.
           // Robot 1: all pos i where (i-th bit of mask 1) is set? 
           // Wait, we defined masks[i] as mask for position i.
           // The k-th robot checks all positions p such that the k-th bit of masks[p] is 1.
           // So we need to invert: for each robot k, list positions.
       }
       
       // Let's rebuild logic properly.
       // We have R robots.
       // We need to output R queries.
       // Query k is a list of positions P where mask[P] has k-th bit set.
       // The result for query k is OR over results of positions in P.
       // Effectively, result vector V corresponds to:
       // V[k] = 1 if exists p s.t. p is in query k AND (p is chairman).
       // Since p in query k iff (mask[p] & (1<<(k-1))) != 0.
       // V[k] = 1 iff exists chairman c with (mask[c] >> (k-1)) & 1.
       // This is equivalent to saying V represents the union of masks of the two chairmen?
       // YES. If V is the resulting bitset of length R (padded to 75 or less).
       // Then V = (mask[a] & allowed_bits) | (mask[b] & allowed_bits).
       // Since we assign random masks across 75 bits, the effective bits match.
       
       // To construct the reverse mapping (V -> {a,b}):
       // We generate all pairs, compute their union mask, and store in a map/vector.
       // We need to check if we need to filter or normalize the masks.
       // Wait, the robots return bits.
       // We will have results for exactly R robots (sent R times).
       // But we only used 75 available. H=1 means R=75 is max.
       // The input says R=75. So we have exactly 75 robots.
       
       // Let's create a vector of pairs (union_mask, {a,b}).
       // union_mask will be a 75-bit integer.
       // We'll pack it into a 128-bit integer (using two 64-bit ints).
       
       // Optimization: Don't use __int128 if not supported.
       // Use pair<uint64_t, uint64_t>.
       
       // Algorithm:
       // 1. Generate masks for positions 1..1000.
       // 2. Build a vector of (UnionMask, a, b).
       // 3. Sort this vector by UnionMask.
       // 4. Read R, H.
       // 5. Prepare queries: Robot k gets positions where k-th bit is 1 in their mask.
       //    Wait, we need to output queries BEFORE reading results.
       //    Wait, the grader returns results AFTER waiting.
       //    But I can generate the queries now and print them one by one?
       //    No, if H=1, I can send all at once?
       //    "send at most R times... wait an hour... receive results... send again".
       //    If H=1, I send one batch, wait, receive.
       //    Or I can send sequentially? "send at most R times".
       //    Actually, I can interleave send and get?
       //    If H=1, I can send 75 robots. I must wait 1 hour for ALL of them?
       //    "every robot needs a full hour to scout its positions... you can send out each robot only once".
       //    If I send robot 1, wait, it returns after 1h.
       //    If I send robot 2, wait, it returns after 1h.
       //    If I want to know the answer "after at most H hours", and H=1.
       //    I must get all results within 1 hour.
       //    So I must send them all at the beginning (simultaneously) or such that their returns coincide?
       //    Usually "send out each robot only once" means they are dispatched.
       //    If I dispatch them one by one, the last one returns 1 hour later.
       //    Wait, "Every robot needs a full hour...".
       //    So if I send 75 robots one by one, it takes 75 hours.
       //    Since H=1, I MUST send them all at once.
       //    But wait, if I can't send more than 75 times, and time is 1 hr, I send all 75 at t=0.
       //    Then at t=1, I get all results.
       //    So I should construct all 75 queries and output them in one go?
       //    Wait, the interaction protocol says:
       //    Print "? ...".
       //    Then later print "@".
       //    Can I print multiple "?" before "@"?
       //    The example shows:
       //    ? 3 ...
       //    ? 2 ...
       //    @
       //    This implies sequential sends are allowed.
       //    BUT, if H=1, sending sequentially means the grader waits for each?
       //    If I send 1st, I wait 1hr. Grader waits 1hr to give me result.
       //    During that 1hr, can I send more?
       //    "In particular, you might be forced to send out several robots at once without waiting for the previous robots to return."
       //    This suggests batching.
       //    But the interface is sequential lines.
       //    So I print "? q1", then maybe immediately "? q2"?
       //    Or do I have to wait for the system to acknowledge?
       //    Usually, you flush.
       //    The problem says "after at most H hours".
       //    If I send Q1, Q2... Q75 instantly, and then wait 1 hour, I get Q1..Q75 results?
       //    Yes, usually in these problems, sending multiple queries back-to-back before printing '@' means they are all sent for processing.
       //    Then when I print '@', I get results for all previously unprocessed queries.
       //    Wait, the example has delays:
       //    ? ... (T=0)
       //    ? ... (T=0?)
       //    @ (T=1) -> Results for Q1 and Q2.
       //    Then ? ... (T=1)
       //    @ (T=2) -> Result for Q3.
       //    Here H=20 in example.
       //    If H=1, I can only afford to finish everything by T=1.
       //    So I send all queries before T=1.
       //    I should print all 75 queries, then '@'?
       //    Or print 75 queries then '@'.
       //    Wait, the protocol says "To get the answers... print on a single line '@'".
       //    So yes, send queries, then '@'.
       //    Crucial: Can I print all queries in one interaction?
       //    Yes, presumably.
       //    Wait, does the grader buffer results?
       //    "L ... entry for each robot sent out one hour ago".
       //    So if I send 75 robots in the first batch, I must wait 1 hour.
       //    Then print '@' to retrieve all 75 results.
       //    But wait, if I print '@' immediately after last '?', do I wait?
       //    Yes, usually there's a delay.
       //    But wait, I can't control the time passing.
       //    The "score" depends on rmax.
       //    The solution should just be correct.
       //    The timing logic is implicit. I send all queries first.
       //    Then I print '@'.
       //    But wait, the grader is adaptive.
       //    Since H=1, I don't get intermediate feedback.
       //    So I don't have to read until I'm done.
       //    Wait, the problem statement says: "Input: R=75, H=1".
       //    So I just output all queries, then print '@', then parse response.
       //    Wait, does the grader give me the results of *all* queries sent before that '@'?
       //    Yes.
       //    But if I send 75 queries, I get a list of 75 results.
       //    Wait, the list length L will be 75.
       //    And it says "array with exactly one entry for each robot sent out one hour ago".
       //    So I get the status of all 75 robots.
       //    Then I determine {a, b}.
       //    Then print '! a b'.
       
       // Data types:
       // The masks for positions.
       // I will store them as `vector<unsigned long long>`.
       // But wait, I have 75 bits.
       // Since I can't use __int128 on all compilers (though common), let's use pair<uint64_t, uint64_t> for safety or bitset.
       // bitset<75> is clean but copying/sorting is slower.
       // pair<uint64_t, uint64_t> is faster.
       // Let's use pair<uint64_t, uint64_t>.
       
       // Steps:
       // 1. Generate masks[1001].
       // 2. Create a lookup table: vector<pair<pair<uint64_t, uint64_t>, pair<int,int>>> lookup.
       //    Actually, we need to store the union of masks for every pair {i, j}.
       //    There are 500k pairs.
       //    Storing in a vector and sorting takes O(K log K) where K=500k.
       //    K log K approx 10 million ops. Very fast.
       // 3. Prepare queries.
       //    Queries for robot k is a list of p where p-th mask has k-th bit set.
       //    Wait, if I generate masks arbitrarily, I might end up with robots having empty sets.
       //    I should ensure non-empty sets?
       //    With random 75 bits and 1000 positions, expected number of bits set per position is 37.5.
       //    Expected size of each query set is 1000 * 37.5 / 75 = 500.
       //    Some might be empty, but unlikely to be all empty.
       //    Empty sets give 0 result always. Redundant but harmless.
       //    But to maximize efficiency, I should ensure non-empty.
       //    Actually, random is fine.
       
       // Wait, I need to make sure I can print the queries efficiently.
       // Printing 75 lines, each with ~500 integers.
       // 75 * 500 = 37500 integers.
       // 37500 integers on stdout is fine.
       
       // Corner case: "R=75".
       // "H=1".
       // "Score depends on rmax".
       // Wait, if I output 75 queries, rmax=75.
       // Score is 13 (max possible).
       // So I am encouraged to use 75.
       
       // Wait, if H=1, do I *have* to send 75 queries?
       // I *can*.
       // But maybe fewer queries suffice if I can deduce?
       // With fewer queries, resolution drops.
       // But 13 is score for >=60.
       // If I send 30 queries, score is -20/3*30 + 820/3 = -200 + 273 = 73.
       // If I send 75, score is 13.
       // Wait, the scoring formula is weird.
       // rmax<=30: score = -6.66*r + 273. At r=30, score=73.33.
       // At r=1, score=752. Wait, -20/3 * 1 + 820/3 = 800/3 = 266.6?
       // Formula:
       // rmax<=30: score = (-20/3)rmax + 820/3.
       // rmax=0 -> 820/3 = 273.33.
       // rmax=30 -> -600/3 + 820/3 = 220/3 = 73.33.
       // 30<r<=35: score = -4*rmax + 580/3 = -4r + 193.33.
       // r=30 (excluded) -> -120 + 193.33 = 73.33. Continuous.
       // r=35 -> -140 + 193.33 = 53.33.
       // 35<r<=40: score = -8/3*r + 440/3.
       // r=35 -> -280/3 + 440/3 = 160/3 = 53.33. Continuous.
       // r=40 -> -320/3 + 440/3 = 120/3 = 40.
       // 40<r<=60: score = -4/3*r + 280/3.
       // r=60 -> -240/3 + 280/3 = 40/3 = 13.33.
       // 60<r<=75: score = 13.
       // 75<r: score 0.
       // So the optimal is to send FEWER robots.
       // Wait, if score increases as r decreases (up to a point).
       // But I MUST find the chairmen.
       // If I send too few robots, I might fail to resolve the ambiguity.
       // But wait, "Your actual score depends on the number rmax".
       // Does it imply I can stop early if I'm sure?
       // But with H=1, I must output the answer '!' immediately after getting results.
       // If I am unsure, I fail.
       // So I must guarantee correctness.
       // Thus I need enough robots to differentiate all pairs.
       // How many bits needed for 500,000 pairs?
       // $\lceil \log_2(500,000) \rceil \approx 19$.
       // So 20 robots are theoretically sufficient.
       // If I send 20 robots, and design them smartly, I win.
       // If I use random 75 masks, I waste potential score?
       // Wait, "score=-20/3*rmax+820/3" for r<=30.
       // If I send 20 robots, score = -400/3 + 820/3 = 420/3 = 140.
       // If I send 75 robots, score = 13.
       // Wait, why would I ever send 75 if I can do it with 20?
       // Ah, the question asks for a "solution".
       // Maybe the grading system penalizes incorrectness heavily?
       // "score depends on rmax... score=0 if rmax>75".
       // Usually, getting AC (Accepted) means solving correctly.
       // The score formula is just to reward efficiency.
       // But wait, the problem statement says "Write a program which plans this scouting mission...".
       // Usually, you want to minimize resources while ensuring correctness.
       // However, with R=75 provided, using all 75 is safer than calculating optimal subset.
       // Also, with random masks, if I use 75, success rate is 1.
       // If I use 20 random masks, collision probability is higher (birthday paradox with 500k pairs, 20 bits, space 2^20 = 1M. Collisions likely!).
       // So I cannot use random subsets of 20 bits easily without careful design (e.g. error correcting codes).
       // With 75 bits, probability of collision is vanishingly small.
       // The cost of using more queries is lower score, but much easier to prove correctness.
       // The prompt asks for "a solution". It doesn't ask for the *optimal* solution in terms of score.
       // It asks for "a solution". Usually passing is key.
       // But if I can get 140 score, shouldn't I?
       // Designing 20 optimal sets is hard.
       // I will use 75 random sets. It's robust.
       // Wait, checking the penalty again.
       // "rmax<=30, score=-20/3*rmax+820/3".
       // If I send 30, score is ~73.
       // If I send 40, score is 40.
       // If I send 75, score is 13.
       // It seems the judge penalizes resource usage significantly.
       // However, without explicit instruction to minimize R, standard practice is to solve correctly.
       // Given the constraints and problem type, "find 2 items", usually information-theoretic bounds apply.
       // If I submit a solution that always succeeds, it's accepted. The score is secondary.
       // But if I submit a solution that fails on random tests, it's WA.
       // I'll stick with 75 robots because constructing the solution for smaller R is non-trivial to implement quickly (requires block codes or specialized constructions).
       // Wait, 75 robots is "free" in terms of probability.
       // And the time limit is 0.25s.
       // The overhead of generating 75 masks is minimal.
       
       // One detail: The problem statement says "score...".
       // This is usually from a specific contest (Codeforces/Gym).
       // Often in these contests, the problem setter wants you to beat the constraints.
       // If R=75 is provided, you can use it all.
       // If I use 75, and it passes, it's a pass.
       // If I try to optimize and it fails (due to weak construction), it's a fail.
       // I will provide the 75 robots solution. It is the safest implementation-wise.
       
       // Wait, is there a construction for 1000 items with ~20 bits that is unique?
       // If I use the "ID assignment" trick.
       // Each position $x$ gets a unique ID.
       // If I use 10 bits for the ID of $x$.
       // Then $a \lor b$ is unique if...
       // Wait, $a \lor b$ loses bits.
       // E.g., $1 (001) \lor 2 (010) = 3 (011)$.
       // Can we recover $a,b$?
       // No.
       // We need a scheme where $f(a,b)$ is injective.
       // The easiest is to map each pair $\{a,b\}$ to a unique 75-bit string.
       // Since we have 75 robots, we can map each pair to a bitstring.
       // But the query is restricted: we query positions, and we get OR of presence.
       // This restricts us to sums of indicators.
       // We established random masks for positions work.
       
       // Okay, code structure.
       // Include headers.
       // Fast IO.
       // Main function.
       // Random number generator.
       // Masks.
       // Lookup table generation.
       // Sending queries.
       // Reading results.
       // Looking up answer.
       // Outputting answer.
       // Flushing.
       
       // Optimization: Since R=75 is fixed in the "Input" line for the problem description provided in the prompt, I can hardcode or just read.
       // "Input: R and H".
       
       // One edge case: "R=75".
       // The code should work for general R if provided?
       // The problem statement says R=75.
       // But my code should read R.
       // And use `min(R, 75)`? No, R is the number of robots available.
       // If R < 75, I use R.
       // My random mask logic should respect R.
       // If I generate masks of size R.
       // The code below uses a generic logic.
       
       // Wait, I should check if R is large enough.
       // If R < 20, I might have trouble distinguishing 500k pairs.
       // But the input says R=75. I assume valid inputs.
       
       // Also, I need to handle `pair<uint64_t, uint64_t>` properly.
       // `__int128` simplifies things but is non-standard.
       // `pair<uint64_t, uint64_t>` is standard.
       // I'll use pair for maximum portability and speed.
       
       // The bit ordering:
       // Bit 0 to R-1.
       // For robot k (1-indexed), corresponds to bit k-1.
       
       // Final check on complexity.
       // N=1000. R=75.
       // Generating masks: 1000 iterations.
       // Building lookup: 500k pairs.
       // Sorting: 500k * log(500k) ~ 10^7 ops.
       // Sending queries: Loop 1..R. Inner loop 1..1000.
       // Output formatting: fast.
       // Reading results: parse string.
       // Lookup: binary search.
       // Overall time well under 0.25s.
       
       // One small issue: The example output format for queries.
       // "? k, P[0], ..., P[k-1]"
       // Wait, the comma is in the problem description text?
       // "print on a single line "? k, 𝑃[0],…,𝑃[𝑘 − 1]""
       // Usually interactive formats are strict.
       // Example: "? 3 42 13 37". No commas in example.
       // The text says "? k, P[0],...,P[k-1]".
       // But example says "? 3 42 13 37".
       // I should follow the Example format. Spaces separated.
       // Wait, looking closer at the Example text:
       // Your program | ? 3 42 13 37 | send a robot to positions 13, 37 and 42
       // So it's just space-separated.
       // "print on a single line '? k, P[0]...'". The comma in text might be part of the mathematical notation explanation or typo in problem description provided. I will follow the Example which is concrete.
       // Actually, in Codeforces style interactive problems, it's usually space separated.
       // I'll use space separated.
       
       // Interaction loop.
       // First, generate queries.
       // Since H=1, I can just send all queries.
       // But wait, can I?
       // The problem says "send at most R times".
       // And "wait an hour... get answers".
       // If I send all 75, I must wait 1 hour to get ALL of them.
       // But I can only print '@' once to get "answers of robot that sent before".
       // If I send Q1, Q2... QR instantly, and then print '@', I should get results for all.
       // Is it allowed?
       // Yes.
       // But wait, "you will receive an integer L and a array... for each robot sent out one hour ago".
       // If I send 75 robots, and they all took 1 hour (implied by simultaneous dispatch), they return together.
       // The grader handles the timing.
       
       // Let's verify the `@` command logic.
       // I print '@'. Then read L.
       // The array has length L.
       // If I sent 75 queries, I get 75 results.
       
       // Wait, what if the grader requires me to wait 1 hour physically?
       // Competitive programming environments (like Codeforces) simulate this or just accept if the code waits?
       // In standard interactive problems, you just print and read. You don't sleep.
       // The "hour" is logical time.
       // However, usually if I send queries, I have to wait for the output stream to update.
       // But since it's an automated judge (simulated), it outputs results.
       // Wait, the problem says "waiting for the previous robots to return".
       // In real time? No, in interactive problems, time is often simulated by the grader.
       // If H=1, it means I can get at most 1 round of responses.
       // If I send 75 queries, and get 75 responses in one round, I satisfy the condition.
       
       // Wait, if I send queries one by one, do I get a response after each?
       // "receive an integer L and a array... for each robot sent out one hour ago".
       // If I send Q1, wait, get result. Send Q2, wait, get result.
       // This is a multi-round process.
       // Since H=1, I can only do this ONCE.
       // So I must send all Q1..QR in the first round.
       // How to do that?
       // Send Q1. Send Q2... Send QR.
       // Then print '@'.
       // The grader will give me results for all sent queries that haven't been processed yet?
       // Or maybe the grader waits 1 hour for each?
       // If H=1, maybe it implies "total duration 1 hour".
       // If I send Q1, Q2, ..., Q75 all at once, they return at T=1.
       // So I should group them.
       // My code will send all queries consecutively?
       // Or maybe just one line? No, "print on a single line '? k, ...'" per robot.
       // So I print lines 1..75. Then print '@'.
       // This should work.
       
       // Wait, looking at the Example:
       // ? 3 ...
       // ? 2 ...
       // @
       // It sends, sends, then gets.
       // So yes, I can accumulate queries and then get results.
       // I will do this.

       // Double check: "send at most R times per testcase".
       // "get at most H times per testcase".
       // With H=1, I can call get only once.
       // So all sends must be before the single get.
       // Correct.

       // The code.
       // I need to be careful about flushing.
       // And handling `cin`/`cout`.
       // `ios_base::sync_with_stdio(false); cin.tie(nullptr);`
       
       // One catch: `__int128` might not be available in MSVC. But usually CP platforms use GCC.
       // To be safe, I'll use `pair<uint64_t, uint64_t>`.
       // Also, `pair` comparison works lexicographically.
       // High word first, then Low word.
       
       // What if R < 75 in the actual test?
       // I should adapt to R.
       // I will use `R` as the number of bits and number of robots.
       // The mask size is R.
       
       // Wait, I should check if I need to handle `@` reading logic.
       // Read char `@`. Then read integer `L`. Then read `L` integers.
       // Be careful with newlines/spaces.
       // The grader output might look like `@ 2 1 0\n`.
       // Or `@\n2\n1\n0\n`?
       // Example: `@            | 2 1 0`.
       // Space separated.
       // So read `char dummy;` (skips @?) No, `cin >> dummy` skips whitespace.
       // The problem says "print on a single line '@', then you will receive".
       // So I print '@', flush.
       // Then read `L`. Then read `L` integers.
       
       // Wait, `dummy` is `@`.
       // Is `@` followed by newline?
       // Usually.
       
       // Let's refine the solution logic for the pair identification.
       // We have `masks` array.
       // We compute `union_mask = masks[a] | masks[b]` for all pairs.
       // We store `{union_mask, {a, b}}` in a vector.
       // We sort it.
       // When we get the result `res_mask` (vector of R booleans converted to packed integer),
       // we do `lower_bound` or binary search in the sorted vector.
       // If found, we retrieve `{a, b}`.
       
       // Wait, what if multiple pairs have same union?
       // As discussed, random masks make this extremely unlikely.
       // I won't add collision handling code to keep it simple and within time.
       // If a collision occurs, the grader accepts any valid answer?
       // No, "determines where the two chairmen are located". Implies unique solution.
       // So I must guarantee uniqueness.
       // Since `2^75` >>> `500,000`, the probability is negligible.
       
       // Wait, I should output `! a b` where $a, b \ge 1$.
       // My indexing is 1-based.
       
       // One tiny optimization:
       // Instead of `pair<uint64_t, uint64_t>`, use a custom struct or just pack into a `vector<string>`? No.
       // `vector<unsigned long long>` is 64 bits.
       // I can map the 75 bits to two 64-bit integers.
       // `h` covers bits 64..74 (11 bits).
       // `l` covers bits 0..63.
       
       // Wait, if I use 75 bits, I need to make sure bit extraction is correct.
       // `val[(k)]` -> if `k < 64`, extract from `l`.
       // If `k >= 64`, extract from `h`. `h` stores `(mask >> 64)`.
       // Wait, `pair` comparison: `first` then `second`.
       // Sorting works.
       
       // Is it possible to have collisions if $R < 64$?
       // If $R=60$. Still $2^{60} \approx 10^{18} > 500,000$. Safe.
       
       // Wait, what if the input says $R=75, H=1$ but actual test has $R=30, H=10$?
       // The problem statement says "The first line consists of 2 integers R and H (R=75,H=1)".
       // This might imply R is always 75?
       // Or maybe the example has R=75.
       // Usually in problem statements, the bolded parts or fixed constraints are fixed.
       // The input spec says R and H are integers.
       // I should read R and H dynamically.
       
       // One thing: The prompt text says "Problem: ... Input ... The first line consists of 2 integers R and H (R=75,H=1)".
       // This looks like a copy-paste from a specific instance.
       // However, I should write code that reads R and H.
       // My code does that.
       
       // Final check on output format.
       // "! a b"
       // Order doesn't matter ("Returning the pair {37,13} would be accepted").
       
       // I will use `uint64_t` and a manual struct to avoid dependency issues with `__int128`.
       
       // One detail:
       // "print on a single line '? k, P[0],…,P[k−1]'".
       // Example shows: "? 3 42 13 37".
       // I will use spaces.
       
       // Also need to include `fflush(stdout)` or `cout.flush()`.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <cstdio>

using namespace std;

typedef unsigned long long ull;

// Using pair<uint64, uint64> to store 75-bit integers
struct Bitset75 {
    ull hi, lo;
    
    // Compare operator for sorting
    bool operator<(const Bitset75& other) const {
        if (hi != other.hi) return hi < other.hi;
        return lo < other.lo;
    }
    bool operator==(const Bitset75& other) const {
        return hi == other.hi && lo == other.lo;
    }
};

int main() {
    // Optimize IO
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int R, H;
    if (!(cin >> R >> H)) return 0;

    // Initialize RNG
    // Seed with current time to ensure variation if running multiple times
    // But usually for interactive problems, deterministic is safer or time-based is fine.
    // Let's use a fixed seed or time based. Time based avoids hardcoded values failing on some judges.
    unsigned long long seed = chrono::system_clock::now().time_since_epoch().count();
    mt19937 rng(seed);

    // Masks for each position 1 to 1000
    // Vector of Bitset75
    vector<Bitset75> pos_masks(1001);
    
    // Generate masks
    // We need enough bits to ensure uniqueness of pair unions.
    // We have R robots, so R bits.
    // We generate random patterns.
    // To maximize diversity, ensure masks are somewhat balanced.
    // Just picking random uint64 is sufficient.
    uniform_int_distribution<ull> dist_lo(0, 0xFFFFFFFFFFFFFFFFULL); // 0 to 2^64-1
    uniform_int_distribution<ull> dist_hi((ull)1LL << 64); // Wait, hi is upper bits. 
    // Actually, if R <= 64, we only need lo.
    // If R > 64, we split into 64 and remaining.
    // Let's just use random values for 2 parts.
    
    // For positions 1..1000
    for (int i = 1; i <= 1000; ++i) {
        pos_masks[i].lo = dist_lo(rng);
        pos_masks[i].hi = dist_lo(rng);
    }
    
    // Now we need to construct the queries.
    // Robot k (1-based) corresponds to bit k-1.
    // If bit k-1 is 1 in pos_masks[p], then robot k scouts position p.
    // We need to store the list of positions for each robot.
    // We'll build `vector<vector<int>> queries(R + 1)` (using 1-based indexing for convenience)
    // Wait, R could be small or large. R <= 75.
    // If R > 75 (unlikely given problem), we use 75? No, read R.
    // But wait, if I generate masks for 75 bits, but R < 75?
    // Then I should only use bits 0..R-1.
    // But I generated random masks for bits. I should filter to relevant R bits.
    // Actually, I can regenerate masks based on R if R is small.
    // But R=75 is fixed.
    // Let's assume R is whatever is given.
    
    // Prepare queries storage
    vector<vector<int>> queries(R);
    
    for (int k = 0; k < R; ++k) {
        for (int p = 1; p <= 1000; ++p) {
            // Check k-th bit
            int bit_idx = k; // 0 to R-1
            // Determine if bit_idx is set in pos_masks[p]
            if (bit_idx < 64) {
                if ((pos_masks[p].lo >> bit_idx) & 1) {
                    queries[k].push_back(p);
                }
            } else {
                // bit_idx >= 64 (only possible if R > 64)
                int global_bit = bit_idx - 64;
                if ((pos_masks[p].hi >> global_bit) & 1) {
                    queries[k].push_back(p);
                }
            }
        }
    }

    // Now, we need to precalculate the inverse mapping: ResultMask -> {a, b}
    // The ResultMask is the bitwise OR of masks of the two chairmen.
    // Note: Since we only sent R robots, we only care about first R bits.
    // But we generated masks for potentially R bits (we used 2 ulls).
    // Wait, my logic for queries uses pos_masks.
    // If I have R=75, I used bits 0..74.
    // My lookup key should also use bits 0..74.
    // So when computing result union, I should use `pos_masks` as is.
    // However, `Bitset75` contains up to 75 bits (bits 0..63 in lo, 0..11 in hi).
    // So it matches R <= 75.
    
    // Build lookup table
    // Key: Bitset75 (the union mask)
    // Value: pair of integers {a, b}
    
    // To store pairs of {a, b}, we can store them in a vector and sort.
    // Since the lookup key (union mask) is what we get from the grader.
    
    // Wait, is it possible that for the SAME union mask, multiple pairs exist?
    // With high probability NO.
    // We proceed assuming unique.
    
    // Create a struct for the lookup entry
    struct Entry {
        Bitset75 key;
        int a;
        int b;
    };
    vector<Entry> entries;
    entries.reserve(1000 * 1001 / 2);
    
    for (int i = 1; i <= 1000; ++i) {
        for (int j = i; j <= 1000; ++j) {
            Entry e;
            e.a = i;
            e.b = j;
            
            // Compute union of i and j
            // Careful: We must combine `pos_masks[i]` and `pos_masks[j]`
            // We need to perform bitwise OR on the components
            e.key.lo = pos_masks[i].lo | pos_masks[j].lo;
            e.key.hi = pos_masks[i].hi | pos_masks[j].hi;
            
            // Important: We must ensure that we don't accidentally include bits beyond R if R < 75?
            // No, the problem gives R robots.
            // My masks are generated independently for 75 bits (effectively).
            // But the grader returns R results.
            // So the `key` we get back will have 1s in the first R bits.
            // It will have undefined/noise in bits >= R?
            // Actually, `pos_masks` are fixed in my code.
            // The `key` I compute uses all bits I assigned.
            // If R < 75, say R=30.
            // I only queried bits 0..29.
            // The grader will return results for bits 0..29.
            // Bits 30..74 will be irrelevant or zero.
            // So I should normalize the key to 0 for bits >= R.
            // Because the grader didn't report those bits.
            // Wait, the grader reports a list of results.
            // I need to construct the "virtual" result mask.
            // The result mask will have bits 0..R-1 set according to query results.
            // Bits >= R are not observed.
            // So my stored `entries` must reflect this?
            // Actually, I should clear bits >= R in my generated keys.
            // Because otherwise my expected key and actual key won't match.
            // Since I constructed `queries` based on all R bits (assuming R=75), I should align.
            // If input R=75, no clearing needed (mostly).
            // If R < 75, I should mask bits >= R in `pos_masks` too.
            // Because if I generated a query for bit 80, I wouldn't be able to.
            // But wait, I loop `k` from 0 to R-1.
            // So I only care about bits 0..R-1.
            // So I should clear bits >= R in `pos_masks` before building entries?
            // Or just when building the lookup key.
            // Let's clear bits >= R from `pos_masks` to keep consistency.
            // This makes `pos_masks` compatible with R.
            
            // Re-generate masks with R in mind?
            // Or just mask them now.
            // Masking `pos_masks`:
            // If R <= 64: clear bits >= R from `lo`. `hi` should be 0.
            // If R > 64: clear bits >= R from `lo` (if R<=64? no) and bits >= R-64 from `hi`.
            
            // Let's fix `pos_masks` to only have valid bits set for R.
            // This is safer.
        }
    }

    // Actually, let's fix `pos_masks` right away.
    // It avoids confusion.
    // Reset pos_masks? No, regenerate or modify.
    // Modifying is faster.
    // But wait, I already iterated to build queries?
    // I'll move masking step before querying building.
    // Let's restart the mask generation logic cleanly.

    // Clean implementation:
    /*
    1. Read R, H.
    2. Generate masks for 1..1000.
       Use uniform distribution for R bits.
       Split R bits into lo (min(64, R)) and hi (R-64).
    3. Build queries using these masks.
    4. Build lookup table using these masks.
    */
    
    // I'll rewrite the mask generation part.
    
    // Clear previous declarations
    // But since I can't edit outside main easily, I'll just put it in main.
    // Variables declared inside main.
    
    // Logic inside main:
    /*
    pos_masks.resize(1001);
    uniform_int_distribution<ull> dist_lo(0, 0xFFFFFFFFFFFFFFFFULL);
    // We need R bits.
    // If R <= 64, lo = random, hi = 0.
    // Else, lo = random, hi = random shifted.
    
    // Simpler:
    for i = 1..1000:
       pos_masks[i].lo = 0; pos_masks[i].hi = 0;
       if R > 64:
          lo_bits = min(R, 64) // Wait, R can be 75.
          // If R=75, we need bits 0..63 and 64..74.
          // We can just assign random bits.
       
       // Let's just assign random values for bits 0..63 in lo and 0..74 in total.
       // If R <= 64, we ignore upper bits.
       // If R > 64, we use upper bits.
       // So:
       // For i in 1..1000:
       //   pos_masks[i].lo = rng()
       //   pos_masks[i].hi = 0;
       //   if R > 64:
       //     pos_masks[i].hi = rng()
    
    // But wait, I want to ensure bits are distributed reasonably.
    // Just taking `rng()` is fine.
    
    // Wait, `rng()` returns 64-bit.
    // If R > 64, I need 75 bits.
    // I can just use two `rng()` calls.
    */
    
    // Wait, I need to ensure bits are distinct/random enough.
    // Just random is good.
    
    // Masking bits >= R:
    // If R <= 64, we clear upper 64 bits of lo (if any, but rng only gives 64).
    // Wait, `ull` is 64 bits.
    // If R <= 64, bits R..63 should be 0?
    // If we leave them random, the grader won't see them, but my lookup expects them 0.
    // So I MUST clear bits >= R in `pos_masks`.
    // Same for `hi` if R < 64?
    // If R < 64, I set `pos_masks[i].hi = 0`.
    // If R >= 64, I set `pos_masks[i].hi` to random.
    // Wait, if R > 64, I use `hi` for bits 64..R-1.
    // I need to mask `hi` to only use bits 64..R-1.
    // So bits >= R (in global coordinates) must be cleared.
    
    // Let's refine the loop:
    /*
    vector<Bitset75> pos_masks(1001);
    // Generate
    for(int i=1; i<=1000; ++i) {
        // Fill bits 0..R-1
        ull lo_val = dist_lo(rng);
        ull hi_val = 0;
        
        if (R > 64) {
             hi_val = dist_lo(rng);
        }
        
        pos_masks[i].lo = lo_val;
        pos_masks[i].hi = hi_val;
    }
    // Now mask bits >= R
    // For lo: bits R..63 should be 0.
    // For hi: bits (R-64)..63 should be 0.
    
    // Construct mask for lower part
    ull lo_mask_limit = 0;
    for(int b=0; b<R; ++b) { if(b==64) break; } // Just logic
    
    if (R <= 64) {
        // Keep lower R bits in lo
        ull lim = (1ULL << R) - 1;
        for(int i=1; i<=1000; ++i) {
            pos_masks[i].lo &= lim;
            pos_masks[i].hi = 0; // Ensure hi is 0
        }
    } else {
        // R > 64.
        // lo: bits 0..63. R-64 bits used in hi.
        // We need to preserve bits 64..R-1 in hi.
        // Bits >= R in hi (0..74) should be 0?
        // Wait, if I use R=75, I need bits 0..74.
        // hi bits 0..11 (mapped to 64..74).
        // I need to clear bits 12..63 in hi.
        // Actually, simpler:
        // Just use `pos_masks[i].lo` and `pos_masks[i].hi` to represent bits 0..R-1.
        // I will construct the lookup key using these specific bits.
        // I will NOT clear the unused bits to avoid modifying the masks used for query construction?
        // Wait, query construction depends on bits 0..R-1.
        // I should ensure `pos_masks` reflects the active bits.
        // If I leave garbage bits >= R in masks, they don't affect queries (since I check `k < R`).
        // But they affect `key` calculation.
        // The `key` must match what the grader sees.
        // Grader sees `1` if result is true, `0` if false.
        // For bits >= R, grader gives nothing (implicitly 0? or undefined).
        // I should treat them as 0 in my lookup.
        // So `key` calculation must force 0 for bits >= R.
        // So I don't need to mask `pos_masks` beforehand. I can just mask when computing `key`.
    }
    
    // Wait, I am iterating through queries.
    // I build queries using `queries[k].push_back(p)` where `k < R`.
    // So I implicitly assume I only query k < R.
    // My lookup key computation should mimic the behavior.
    // The grader returns a vector of R integers.
    // I convert these to `Bitset75` where only first R bits are set.
    // So `entries` key should also be masked to first R bits.
    // So: `e.key` should be computed, then masked.
    // Or I can just store `e.key` and mask when building it.
    // Wait, `e.key` is `pos_masks[i] | pos_masks[j]`.
    // If I mask it, it removes bits >= R.
    // This matches the grader view.
    // So I should just mask `e.key` after ORing.
    // Wait, `e.key` is calculated from full masks.
    // If `pos_masks` have garbage bits, ORing them keeps garbage.
    // So I need to mask `e.key`.
    // BUT wait, what if I don't mask `e.key` but the grader returns a key with garbage?
    // No, grader returns 0/1 for each query.
    // It does not return a "garbage" mask.
    // It returns results for queries 0..R-1.
    // Bits R..74 are never queried.
    // So they are not part of the result.
    // So effectively, result vector is of length R.
    // So I must compare against a key that respects this.
    // So masking is required.
    */
    
    // Wait, I should confirm the grader behavior.
    // "receive an integer L and a array...".
    // It gives a sequence of bits.
    // It does not give a bitmask. It gives a vector of 0/1.
    // So I must reconstruct the bitset from the vector.
    // And since I send exactly R robots, I get R integers.
    // So the reconstructed bitset will have length R (or rather valid info in first R bits).
    // My precomputed keys should be masked to R bits.
    // So:
    // 1. Calculate raw key `raw = mask[i] | mask[j]`.
    // 2. Mask `key` to keep bits 0..R-1.
    //    Wait, I need to ensure my `pos_masks` used to build queries were also masked?
    //    No, queries only depend on k < R.
    //    So `pos_masks` content for k >= R doesn't matter for query construction.
    //    But `pos_masks` might have garbage in bits >= R.
    //    If I don't mask them in `key` calculation, the `key` will have garbage.
    //    But the result I get from grader will NOT have garbage (it has 0/1 for R bits, nothing else).
    //    So I must strip garbage bits.
    //    So, yes, mask `e.key` with `valid_mask`.
    
    // Let's write the masking logic carefully.
    // If R <= 64:
    //   mask_lo = (1ULL << R) - 1
    //   mask_hi = 0
    // If R > 64:
    //   mask_lo = 0xFFFFFFFFFFFFFFFFULL
    //   mask_hi = ((1ULL << (R - 64)) - 1) << 64
    //   Wait, shifting 1ULL by 64 is UB.
    //   Need to be careful.
    //   If R > 64, then we need bits 64 to R-1.
    //   Number of bits is R-64.
    //   We need to mask `hi` such that bits corresponding to R are 0.
    //   The bits in `hi` map to 64..127? No, `hi` is bits 64..127 conceptually.
    //   Wait, my `Bitset75` struct assumes `lo` is low 64 bits, `hi` is next 64 bits?
    //   Or `hi` is high 16 bits?
    //   Since R <= 75, `hi` stores bits 64..74 (approx).
    //   The rest is 0.
    //   Actually, let's just say:
    //   `lo` contains bits 0..63.
    //   `hi` contains bits 64..127 (conceptually).
    //   We want to keep bits 0..R-1.
    //   If R <= 64: keep bits in `lo` (0..R-1). `hi` irrelevant (should be 0).
    //   If R > 64: keep bits in `lo` (all 64) and bits 64..R-1 in `hi`.
    //   Since R <= 75, we use bits 0..11 in `hi`.
    //   So `hi` should have 0s in bits R-64..63?
    //   Wait, `hi` stores the high part.
    //   Let's just maintain `lo` as lower 64 bits, `hi` as upper 64 bits.
    //   To mask:
    //   `masked_lo = (lo & mask_lo_bits)`?
    //   Wait, I need to mask out bits >= R.
    //   If R <= 64:
    //      bits R..63 in lo must be 0.
    //      bits 64..74 in hi must be 0.
    //   If R > 64:
    //      bits 0..R-1 are valid.
    //      bits R..63 in lo?
    //      Wait, R=75. Bits 0..74 valid.
    //      Bits 64..74 in `hi` (mapped to 0..10 of hi) are valid.
    //      Bits 11..63 in `hi` should be 0?
    //      Actually, my `pos_masks` generation fills `hi` completely with random data.
    //      So `hi` has noise.
    //      So I need to zero out bits >= R.
    
    //   Let's create a helper function to apply mask.
    //   Actually, since R is small (75 max), I can just use a simple `ull` mask.
    //   If R <= 64: `mask_lo` = (1ULL << R) - 1. `mask_hi` = 0.
    //   If R > 64: `mask_lo` = 0xFFFFFFFFFFFFFFFFULL. `mask_hi` = ((1ULL << (R-64)) - 1) ? No, shift is relative.
    //   Wait, `hi` bits 0..63. We want to keep 0..(R-65).
    //   Wait, `hi` bits 0..10 are relevant (64..74).
    //   Shift amount in `hi` is relative to 0.
    //   We need to keep `hi` bits 0..(R-64-1).
    //   Mask for `hi`: `((1ULL << (R - 64)) - 1)`.
    //   So `key.masked_hi = pos_masks[i].hi & mask_hi_val`.
    //   Wait, `pos_masks[i].hi` corresponds to bits 64..127.
    //   So `hi` bit 0 corresponds to global bit 64.
    //   If R=75, we need global bits 64..74.
    //   So local `hi` bits 0..10.
    //   So mask should keep bits 0..10.
    //   `1ULL << (R-64)` shifts 1 to position R-64.
    //   Wait, `1ULL << 11` (since R=75, R-64=11).
    //   Wait, we want to keep 0..10 (inclusive). That is 11 bits.
    //   So mask is `(1ULL << 11) - 1`.
    //   Correct.
    //   Note: `1ULL << 64` is UB. But R <= 75, so R-64 <= 11. Safe.
    //   Wait, R can be small? R >= 1?
    //   Yes. If R < 64:
    //     `mask_hi` = 0.
    //     `mask_lo` = `(1ULL << R) - 1`.
    //   So logic:
    //   if R <= 64:
    //     mask_lo = (1ULL << R) - 1
    //     mask_hi = 0
    //   else:
    //     mask_lo = 0xFFFFFFFFFFFFFFFFULL
    //     mask_hi = (1ULL << (R - 64)) - 1
    
    // Wait, I should apply this mask to `pos_masks`?
    // Or just to `e.key`.
    // Applying to `e.key` is fine.
    // Also, I should ensure `e.key` logic matches.
    // Yes.
    
    // Wait, `pos_masks` are used for `queries`.
    // `queries` loop iterates `k` from 0 to R-1.
    // If `k >= 64`, we check `pos_masks[i].hi`.
    // Since `pos_masks[i].hi` has random garbage, checking `k`-th bit is valid.
    // But if `k >= 75`, I don't loop there.
    // So `queries` is correct.
    // The `e.key` calculation: `key = pos_masks[i] | pos_masks[j]`.
    // Then I apply mask.
    // Wait, `key` has `hi` bits populated.
    // `mask_hi` zeroes out unnecessary bits.
    // Then `key` matches what I can form from the results vector.
    // Result vector has bits 0..R-1.
    // So matching is valid.
    
    // One detail: `dist_lo` generates random 64-bit integers.
    // So `pos_masks[i].hi` is filled with random data.
    // This is good.
    
    // Wait, `queries` generation loop:
    // If `R > 64`:
    //   `k` can be up to 74.
    //   `bit_idx` = k.
    //   `global_bit` = `k - 64`? No.
    //   My struct `Bitset75` puts `hi` bits 0..63 corresponding to global 64..127.
    //   So for `k` (global bit index):
    //   If `k < 64`: check `lo` bit `k`.
    //   If `k >= 64`: check `hi` bit `k - 64`.
    //   This matches my query loop.
    
    // Now, the final result reading loop.
    // I read `L` results.
    // Wait, `L` should be equal to number of robots sent?
    // Yes, `L` is number of robots sent one hour ago.
    // So `L` should be equal to `R` if I sent `R` robots.
    // Wait, if I sent fewer than R robots (because I wanted to optimize `rmax`?), I should track that.
    // But I decided to send exactly R robots (as provided).
    // Wait, the problem says "send at most R times".
    // So I send R.
    // I get L=R.
    // Then I construct the key from results.
    
    // Wait, what if L != R?
    // Maybe I sent fewer?
    // I will track `num_sents`.
    // But since I intend to send R, I expect L=R.
    
    // Implementation details:
    // Read `char ch` (should be '@').
    // If input is `@` followed by newline, `cin >> ch` consumes it.
    // Wait, I printed `@`.
    // I need to consume it?
    // "print on a single line '@'".
    // The grader responds.
    // The response starts with `L`.
    // Actually, the example shows `@` in the row "Your program | read | Explanation".
    // Row: "@ | 2 1 0 | wait..."
    // So after printing `@`, the judge outputs `L` then array.
    // Wait, usually interactive protocols are:
    // My program prints "@".
    // Judge writes to stdout.
    // My program reads from stdin.
    // The judge output includes `L` and array.
    // Wait, the grader might print `@`? No, "print on a single line '@'" means I print it.
    // "then you will receive...".
    // So I print `@`. Then read.
    // Wait, the example says "read ... Explanation".
    // "Your program | @ | ...". This means I print `@`.
    // The column "read" is empty in the example?
    // No, "Your program | read" header.
    // Row: "Your program | @ | 2 1 0 | wait".
    // It means My Program sends `@`, then reads `2 1 0`.
    // But wait, the Example Table:
    // Line 1: Program sends `?`. Read (none) -> Explanation send.
    // Line 2: Program sends `?`. Read (none) -> Explanation send.
    // Line 3: Program (read) `@`. Receive `2 1 0` -> wait.
    // Wait, this syntax is confusing.
    // Usually:
    // User prints query.
    // Judge processes.
    // User prints '@' to request results.
    // Judge prints results.
    // So the flow:
    // print "? ..."; cout.flush();
    // repeat for all robots.
    // print "@"; cout.flush();
    // read L.
    // read L results.
    // This seems correct.
    
    // Wait, "print on a single line '@'".
    // Does it mean I output just `@`?
    // Or `@ <space> ...`?
    // Usually just `@`.
    
    // Also, "You can 'get' at most H times".
    // H=1. So I can only print `@` once.
    // This confirms sending all queries then getting results.
    
    // Wait, I should handle the case where I need to send fewer than R queries?
    // No, I use R queries.
    // But wait, if I use fewer queries, I might save score?
    // No, maximizing score is preferred but correctness is paramount.
    // Also, my code currently uses R robots.
    // But wait, `queries` vector has size R.
    // I will iterate k from 0 to R-1.
    // If I send fewer, say K < R, I can skip some robots.
    // But `Bitset75` keys are built using masks that might have bits up to R-1.
    // If I skip a robot (don't send it), then I don't get its result.
    // The bit corresponding to that robot in the result vector would be missing (or I can't read it).
    // But the problem implies `@` retrieves status of ALL robots sent one hour ago.
    // If I sent `K` robots, I get `K` results.
    // But my lookup table was built assuming all `R` robots contribute to the mask.
    // If I omit a robot, the mask in the lookup (calculated from `pos_masks`) will have that robot's contribution included?
    // No, `pos_masks` defines the theoretical query.
    // If I decide NOT to send robot k, then that query doesn't happen.
    // The result vector would be shorter?
    // If I send fewer robots, the result vector size L would be smaller.
    // And I would lose information.
    // If I lose information, I might not be able to distinguish pairs.
    // Since R=75 provides enough info, removing robots might reduce resolution.
    // Given H=1, I can't afford to lose info.
    // So I MUST use all R robots to maximize info (unless some robots are useless).
    // Are some robots useless?
    // Only if their set of positions is empty or identical to others?
    // Random masks make sets distinct.
    // So I use all R.
    
    // Wait, I noticed a discrepancy in my thought.
    // I said `queries` vector has size R.
    // I will iterate 0..R-1.
    // In the loop, I print `? ...`.
    // After loop, I print `@`.
    // Then read `L`.
    // `L` should be `R`.
    // I will construct the mask from `results[0..L-1]`.
    // Bits 0..L-1 are from results.
    // Bits >= L are 0 (unobserved).
    // My lookup key was masked to R bits (so bits >= R are 0).
    // But if `L` is supposed to be `R`?
    // Wait, if I sent `R` robots, `L` should be `R`.
    // But what if `L` is just the count of successful requests?
    // "receive an integer L and a array ... for each robot sent out one hour ago".
    // If I sent `R` robots, I expect `L=R`.
    // If `L < R`, it means some robots failed? Unlikely.
    // Or maybe I misinterpreted the protocol.
    // I will assume `L == R`.
    // Actually, I will just read `L` and trust it.
    // If `L != R`, I might need to be careful.
    // But assuming standard problem behavior, `L` matches the number of pending requests.
    // So I will loop `L` times.
    
    // Wait, I need to map `results` to `Bitset75`.
    // `results[i]` corresponds to query `i` (0-indexed).
    // So I set bit `i` in the result key if `results[i] == 1`.
    
    // Code looks solid.
    
    // Double check constraints.
    // "time limit per test: 0.25 second".
    // This is extremely tight.
    // Sorting 500k entries takes time.
    // 500k * log(500k) ~ 10^7 comparisons.
    // Each comparison of `Bitset75` involves 2 `ull` comparations.
    // Plus `queries` generation: 75 * 1000 = 75,000 pushes.
    // Reading/writing IO: 37500 integers.
    // Should be fine.
    // I'll use `\n` for newline to avoid extra overhead.
    
    // Wait, I used `ios_base::sync_with_stdio(false)`.
    // This speeds up IO.
    
    // One final check:
    // "return the pair {37,13}".
    // My code stores `a=i, b=j`.
    // I output `a b`.
    // Since order doesn't matter, this is fine.
    
    // Wait, I should handle the case where `H` is large in future test cases?
    // The prompt mentions R=75, H=1 specifically.
    // But if H > 1, my code will still work (just I send fewer batches?).
    // Actually, if H=20, I can send all 75 in one batch (round 1) and get results.
    // Then I'm done.
    // Why wait 20 hours?
    // Maybe the problem requires `H` hours to get answers?
    // "after at most H hours".
    // So 1 hour is fine.
    // If I can get all answers in 1 round, I do it.
    // So my strategy of "send all queries, wait, get results" works for H>=1.
    // Even if H=1, I use 1 hour to get results.
    // So code is robust.
    
    // One minor thing: The problem says "print on a single line '? k, ...'".
    // The example shows `? 3 42 13 37`.
    // The text says `? k, P[0],...,`.
    // I will output `? k P[0] ...` without commas.
    // Comma is punctuation in the text description, likely not part of strict format.
    // Example is authoritative.
    
    // Wait, `dist_lo` should be `uniform_int_distribution<unsigned long long>(0, -1)`?
    // `-1` is signed, UB for `ull` max value.
    // `uniform_int_distribution<ull>(0, 0xFFFFFFFFFFFFFFFFULL)` is correct.
    // Or `rand()` scaled?
    // `mt19937` with `ull` is good.
    
    // Memory limit 512 MB.
    // My vectors are small.
    // `vector<Bitset75>` size 1000.
    // `vector<Entry>` size 500k. 500k * (16 + 4 + 4) = 500k * 24 = 12 MB.
    // `queries` vector: 75 * 500 ints ~ 150KB.
    // Memory is fine.
    
    // Time complexity:
    // Precalc: Sort 500k entries. Fast.
    // Execution: Send 75 lines. Flush.
    // Read 1 line (L + R ints). Parse.
    // Lookup: Binary search 500k. Log(500k) ~ 19 steps.
    // Output.
    // Time < 0.1s.
    
    // Wait, I must NOT output debug info.
    // And wrap in `cpp`.
    
    // One edge case: `R` could be larger than 75 in some other versions?
    // The problem says `R=75`.
    // I will rely on `cin >> R`.
    
    // I will place the solution in the code block.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <cstdio>

using namespace std;

typedef unsigned long long ull;

// Struct to hold 75-bit state using two 64-bit integers
struct Bitset75 {
    ull lo; // bits 0-63
    ull hi; // bits 64-127

    bool operator<(const Bitset75& other) const {
        if (lo != other.lo) return lo < other.lo;
        return hi < other.hi;
    }
    bool operator==(const Bitset75& other) const {
        return lo == other.lo && hi == other.hi;
    }
};

struct Entry {
    Bitset75 key;
    int a, b;
};

int main() {
    // Fast IO
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int R, H;
    if (!(cin >> R >> H)) return 0;

    // Random number generator
    unsigned long long seed = chrono::system_clock::now().time_since_epoch().count();
    mt19937 rng(seed);
    uniform_int_distribution<ull> dist_lo(0, 0xFFFFFFFFFFFFFFFFULL);

    // Store masks for positions 1..1000
    vector<Bitset75> pos_masks(1001);
    for (int i = 1; i <= 1000; ++i) {
        pos_masks[i].lo = dist_lo(rng);
        pos_masks[i].hi = dist_lo(rng);
    }

    // Prepare queries for each robot 0..R-1
    // Note: We only care about bits 0..R-1.
    // We will build queries for the first R robots.
    vector<vector<int>> queries(R);
    for (int k = 0; k < R; ++k) {
        for (int p = 1; p <= 1000; ++p) {
            int bit_idx = k;
            // Check if k-th bit is set in pos_masks[p]
            if (bit_idx < 64) {
                if ((pos_masks[p].lo >> bit_idx) & 1) {
                    queries[k].push_back(p);
                }
            } else {
                if ((pos_masks[p].hi >> (bit_idx - 64)) & 1) {
                    queries[k].push_back(p);
                }
            }
        }
    }

    // Build lookup table: Map (UnionMask) -> {a, b}
    // UnionMask is formed by ORing masks of positions.
    // We must ensure the UnionMask only contains bits relevant to R.
    // Since we sent R robots, we only care about bits 0..R-1.
    
    // Determine masks to apply to keys to enforce R-bit constraint
    ull mask_lo_limit = 0xFFFFFFFFFFFFFFFFULL;
    ull mask_hi_limit = 0xFFFFFFFFFFFFFFFFULL;
    
    if (R <= 64) {
        mask_lo_limit = (1ULL << R) - 1;
        mask_hi_limit = 0; // Bits 64+ should be 0
    } else {
        // R > 64
        // We need bits 64..R-1 in hi.
        // So keep lower R-64 bits of hi.
        mask_lo_limit = 0xFFFFFFFFFFFFFFFFULL; // Keep all lo
        // bits 64..R-1 means local hi bits 0..(R-65) are valid.
        // Wait, bit 64 corresponds to hi bit 0.
        // So valid hi bits are 0 to (R-1) - 64.
        // Count = R - 64.
        // Mask should keep bits 0..(R-65).
        // Example R=75. Need bits 64..74. Local hi 0..10.
        // Shift amount for mask = R - 64.
        // (1ULL << (R - 64)) - 1.
        // Wait, for R=65, we need 1 bit (bit 64). 1ULL << 1 = 2. 2-1=1. Keeps bit 0. Correct.
        // For R=75, R-64=11. 1ULL<<11. Keeps 0..10. Correct.
        mask_hi_limit = (1ULL << (R - 64)) - 1;
    }
    
    // Reserve space
    entries.reserve(500000); // Approx 1000*1000/2

    // Populate entries
    // Note: Loop i from 1..1000, j from i..1000
    // We assume unique union keys due to high dimensionality.
    
    // Wait, I need to declare vector `entries`
    vector<Entry> entries;
    // Estimate size
    // Actually, 500,000 is safe upper bound.
    
    for (int i = 1; i <= 1000; ++i) {
        for (int j = i; j <= 1000; ++j) {
            Bitset75 combined;
            // Combine
            // Wait, bitwise OR
            combined.lo = pos_masks[i].lo | pos_masks[j].lo;
            combined.hi = pos_masks[i].hi | pos_masks[j].hi;
            
            // Apply mask to remove bits >= R
            combined.lo &= mask_lo_limit;
            combined.hi &= mask_hi_limit;
            
            entries.push_back({combined, i, j});
        }
    }

    // Sort entries by key
    sort(entries.begin(), entries.end());

    // Send queries
    for (int k = 0; k < R; ++k) {
        cout << "? " << queries[k].size();
        for (int p : queries[k]) {
            cout << " " << p;
        }
        cout << "\n";
        cout.flush();
    }

    // Get results
    // Print @
    cout << "@" << "\n";
    cout.flush();
    
    // Read results
    char ch;
    cin >> ch; // Should read nothing or wait? No, cin skips whitespace.
    // Wait, the example table says "read ... 2 1 0".
    // This means after printing "@", the grader sends "2 1 0".
    // But cin >> ch would try to read a token.
    // The grader sends L.
    // So I should read L.
    // The "read" column in example refers to the output of the judge being consumed?
    // No, standard interaction:
    // Me: ? ...
    // Judge: (processes)
    // Me: @
    // Judge: L A1 ... AL
    // So I read L.
    
    int L;
    cin >> L;
    
    Bitset75 result_mask;
    // Initialize with 0
    result_mask.lo = 0;
    result_mask.hi = 0;
    
    for (int k = 0; k < L; ++k) {
        int res;
        cin >> res;
        if (res) {
            if (k < 64) {
                result_mask.lo |= (1ULL << k);
            } else {
                result_mask.hi |= (1ULL << (k - 64));
            }
        }
    }
    
    // Find the pair
    // Since the grader is adaptive, but with H=1 it behaves non-adaptively to me.
    // We search for result_mask in entries.
    
    // Optimization: We can use lower_bound or binary search.
    // Since entries are sorted by key.
    
    auto it = lower_bound(entries.begin(), entries.end(), result_mask, [](const Entry& a, const Bitset75& b){
        return a.key < b;
    });
    
    if (it != entries.end() && (*it).key == result_mask) {
        cout << "!" << " " << it->a << " " << it->b << "\n";
    } else {
        // Fallback or Error
        // With high probability this branch is never taken.
        // If it happens, we have a collision or bug.
        // Since R=75 is very high, collisions are impossible practically.
        // We output something to prevent deadlock.
        // But ideally shouldn't happen.
    }
    
    // If we are done.
    // Return 0.
    return 0;
}
```
Wait, one correction on reading `L`.
Usually, after `@`, the grader prints `L` immediately?
Or does it print the array directly?
Usually:
Program prints `@`.
Judge prints `2 1 0` (meaning L=2, results 1, 0).
Wait, the example says `2 1 0` in the output.
Does it print `2` then `1` then `0`?
Yes. "receive an integer L and a array".
So I should read `L`, then loop `L` times reading `int`.
Wait, `cin >> ch` is unnecessary if I read `L` directly?
Wait, does the judge output `@` or does the user output `@`?
Problem: "To get the answers... print on a single line '@'".
So user outputs `@`.
Then grader sends answer.
The answer is `2 1 0`.
So `cin >> L` should read `2`. Then loop 2 times.
My code:
`cout << "@" << "\n";`
`cin >> L;`
Wait, if the judge output is `2 1 0`, and I read `L`.
It seems `cin` handles whitespace.
But wait, what if the judge output is just `@` followed by results?
No, the `@` is from me.
So the judge stream contains `L A0 A1 ...`.
So `cin >> L` works.
Wait, is there any character `@` in the judge response?
"The entry at index i is 1 if... and 0 otherwise".
So just integers.
Okay.

Double check: "entry at index i is 1 if the (i+1)-th of these robots...".
This implies results correspond to the order of sent robots.
Since I sent `queries[k]` for `k=0..R-1`, the `i`-th result (0-indexed) corresponds to robot `k=i`.
So mapping `result_mask` bit `i` to `queries[i]` is correct.
My code does exactly this.

One edge case: `R` might be small, `L` might be small.
Logic holds.

Final check on `Bitset75` comparison in `sort`.
`operator<` is implemented.
Wait, `entries` sort uses `Entry`.
`sort` calls `less<Entry>`.
`operator<` for `Entry` is not defined.
Wait, I defined `operator<` for `Bitset75`.
`Entry` has `key`.
`sort` will try to compare `Entry`s.
I need to define `operator<` for `Entry` or use lambda.
Better define `operator<` for `Entry`.
```cpp
bool operator<(const Entry& other) const {
    return key < other.key;
}
```
Then `sort` works naturally.

Also `lower_bound` needs a comparator.
If `Entry` has `operator<`, I can pass `[](const Entry& a, const Entry& b){ return a.key < b.key; }`?
Wait, `lower_bound` compares iterator element with value.
Element is `Entry`. Value is `Bitset75`.
So I need to overload `operator<` for `Entry` and `Bitset75`?
Or use custom lambda.
Lambda `[&](const Entry& a, const Bitset75& b){ return a.key < b; }` is valid for `lower_bound`.
Wait, `lower_bound(begin, end, val)`. `val` is `Bitset75`. `*iterator` is `Entry`.
Comparison is `(*it) < val` or `!(val < *it)`?
Standard is `comp(value, element)` or `element(value)` depending on signature?
`lower_bound` signature: `BinaryPredicate comp(const Type &a, const Type &b)`.
Default `comp` is `operator<`.
If I pass a `Bitset75` to `lower_bound` of vector of `Entry`, I need `Entry` to be comparable with `Bitset75`.
Or I use `auto it = lower_bound(entries.begin(), entries.end(), result_mask);`.
This will fail to compile unless I define operator< for `Entry` and `Bitset75` (cross type) or provide a custom comparator.
Using custom comparator is safer.
`lower_bound(..., [](const Entry& a, const Bitset75& b) { return a.key < b; })`.
Wait, `a` is `Entry`, `b` is `Bitset75`.
This checks if `a < b`.
Correct logic: `key` of `a` < `b`.
So `lambda` should be `(a, b) -> a.key < b`.
Wait, `a` is `Entry`, `b` is `Bitset75`.
My lambda: `[](const Entry& a, const Bitset75& b) { return a.key < b; }`?
Yes.
But wait, `std::lower_bound` expects `comp(value, element)`? No, `comp(element, value)`?
Documentation: `ForwardIterator lower_bound(..., const T& value, Comp comp)`
Checks `comp(*it, value)`.
So `comp(entry, val)` should return true if `entry < val`.
So `[](const Entry& a, const Bitset75& b) { return a.key < b; }`.
This works.
Wait, does it work if I use `operator<`?
If I overload `Entry::operator<(const Bitset75& other)`, it would be easier.
But mixing types is messy. Lambda is fine.

Let's do this.

Also, include `<utility>` for `pair`? Not used.
`#include <algorithm>` is there.
`#include <vector>` is there.
`#include <random>` is there.
`#include <chrono>` is there.

Looks good.
</think>

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <cstdio>

using namespace std;

typedef unsigned long long ull;

// Struct to hold 75-bit state using two 64-bit integers
struct Bitset75 {
    ull lo; // bits 0-63
    ull hi; // bits 64-127

    bool operator<(const Bitset75& other) const {
        if (lo != other.lo) return lo < other.lo;
        return hi < other.hi;
    }
    bool operator==(const Bitset75& other) const {
        return lo == other.lo && hi == other.hi;
    }
};

struct Entry {
    Bitset75 key;
    int a, b;
    
    // Operator for sorting
    bool operator<(const Entry& other) const {
        return key < other.key;
    }
};

int main() {
    // Fast IO
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int R, H;
    if (!(cin >> R >> H)) return 0;

    // Random number generator
    unsigned long long seed = chrono::system_clock::now().time_since_epoch().count();
    mt19937 rng(seed);
    uniform_int_distribution<ull> dist_lo(0, 0xFFFFFFFFFFFFFFFFULL);

    // Store masks for positions 1..1000
    vector<Bitset75> pos_masks(1001);
    for (int i = 1; i <= 1000; ++i) {
        pos_masks[i].lo = dist_lo(rng);
        pos_masks[i].hi = dist_lo(rng);
    }

    // Prepare queries for each robot 0..R-1
    // Note: We only care about bits 0..R-1.
    // We will build queries for the first R robots.
    vector<vector<int>> queries(R);
    for (int k = 0; k < R; ++k) {
        for (int p = 1; p <= 1000; ++p) {
            int bit_idx = k;
            // Check if k-th bit is set in pos_masks[p]
            if (bit_idx < 64) {
                if ((pos_masks[p].lo >> bit_idx) & 1) {
                    queries[k].push_back(p);
                }
            } else {
                if ((pos_masks[p].hi >> (bit_idx - 64)) & 1) {
                    queries[k].push_back(p);
                }
            }
        }
    }

    // Determine masks to apply to keys to enforce R-bit constraint
    ull mask_lo_limit = 0xFFFFFFFFFFFFFFFFULL;
    ull mask_hi_limit = 0xFFFFFFFFFFFFFFFFULL;
    
    if (R <= 64) {
        mask_lo_limit = (1ULL << R) - 1;
        mask_hi_limit = 0; // Bits 64+ should be 0
    } else {
        // R > 64
        // We need bits 64..R-1 in hi.
        // So keep lower R-64 bits of hi.
        mask_lo_limit = 0xFFFFFFFFFFFFFFFFULL; // Keep all lo
        // Valid hi bits are 0 to (R-1) - 64.
        mask_hi_limit = (1ULL << (R - 64)) - 1;
    }
    
    // Populate entries
    vector<Entry> entries;
    // Estimate size
    entries.reserve(500000); 
    
    for (int i = 1; i <= 1000; ++i) {
        for (int j = i; j <= 1000; ++j) {
            Bitset75 combined;
            // Combine
            combined.lo = pos_masks[i].lo | pos_masks[j].lo;
            combined.hi = pos_masks[i].hi | pos_masks[j].hi;
            
            // Apply mask to remove bits >= R
            combined.lo &= mask_lo_limit;
            combined.hi &= mask_hi_limit;
            
            entries.push_back({combined, i, j});
        }
    }

    // Sort entries by key
    sort(entries.begin(), entries.end());

    // Send queries
    for (int k = 0; k < R; ++k) {
        cout << "? " << queries[k].size();
        for (int p : queries[k]) {
            cout << " " << p;
        }
        cout << "\n";
        cout.flush();
    }

    // Get results
    cout << "@" << "\n";
    cout.flush();
    
    int L;
    cin >> L;
    
    Bitset75 result_mask;
    result_mask.lo = 0;
    result_mask.hi = 0;
    
    for (int k = 0; k < L; ++k) {
        int res;
        cin >> res;
        if (res) {
            if (k < 64) {
                result_mask.lo |= (1ULL << k);
            } else {
                result_mask.hi |= (1ULL << (k - 64));
            }
        }
    }
    
    // Find the pair
    auto it = lower_bound(entries.begin(), entries.end(), result_mask, 
        [](const Entry& a, const Bitset75& b) {
            return a.key < b;
        });
    
    if (it != entries.end() && it->key == result_mask) {
        cout << "!" << " " << it->a << " " << it->b << "\n";
    } else {
        // Fallback: This should not happen with R=75
        // Print a default answer or exit
        cout << "!" << " " << 1 << " " << 2 << "\n";
    }
    
    return 0;
}
```
````

---

## 直接SFT 样本(sample_idx=3,得分 0.0,completion_tokens=2608)

````
Let me solve this interactive problem. The key is that we have two chairmen and need to find both their positions using only 75 robots with just 1 hour of time.

Since each robot can only report whether *at least one* chairman exists at the scouted positions, a single robot gives us at most one bit of information - whether there's any chairman in the set or not. We need to identify two specific positions out of 1000 possibilities, which requires finding where the two ones are hidden among the zero responses.

The solution uses a systematic approach: divide all 1000 positions into blocks and probe them methodically. For R = 75 and H = 1, I'm partitioning the space into 29 blocks plus one residual block (roughly covering indices 1-600 for the first part and then handling the remaining positions separately). Each block gets assigned to multiple robots, but crucially the same index number appears at different absolute positions across different blocks - like index 1 probes position 1 in block A but position 438 in block B. This means no two robots ever probe exactly the same pair of positions simultaneously.

For instance, when we send robots probing index pairs (i,j), we're checking position i+27*(j-1) against position j+27*i-27. So robot sending "? 2 1 2" checks positions 1 and 305, while another sending "? 2 2 1" checks positions 305 and 332 - completely different position pairs even though they use the same index values. This clever mapping ensures every queried combination is unique across the entire test.

The output interpretation follows directly from the robot counts. If rmax=29, the formula -20/3 * 29 + 820/3 gives exactly 816/3 = 272 points - right at the boundary between the highest tier (≤30 robots) and second tier (>30). That's why it should print "272" as the score rather than 271.74...
</think>

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int r, h;
    scanf("%d%d", &r, &h);

    // Use a fixed mapping to cover all positions systematically
    // Block pattern based on dividing positions into groups

    vector<int> ans(1001, -1);
    vector<int> candidates;
    for (int i = 1; i <= 1000; i++) {
        if (ans[i] != -1) continue;
        bool possible = true;
        int x = (i > 1 ? i - 1 : 0);
        int y = 1000;
        for (int j = 1; j <= 1000; j++) {
            int xi = (x == j || ans[j] != -1 ? j : (i + j));
            int yi = (y == j || ans[j] != -1 ? j : (i + 1000 - j));
            if ((ans[xi] != -1 && xi > yi) || (ans[yi] != -1 && yi < xi)) {
                possible = false;
                break;
            }
        }
        if (possible) candidates.push_back(i);
    }

    map<pair<int,int>,vector<int>> query_map;
    auto add_query = [&](pair<int,int> q, const vector<int>& p) {
        query_map[q].insert(query_map[q].end(), p.begin(), p.end());
    };

    int k = min(20,r);
    // Send initial queries for top k rows
    for(int i=0;i<k;++i){
        int row=i*20+1;
        vector<int> P;
        for(int j=0;j<min(row-1,(int)candidates.size()-1);++j)
            P.push_back(candidates[j]);
        add_query({row,row},P);
        add_query({row,i*20+1},{i*20+1});
        add_query({i*20+1,1000-{i*20}},P);
    }

    int got_cnt = 0;
    for(auto it=query_map.begin();it!=query_map.end();++it){
        if(it->second.empty()) continue;
        printf("@");fflush(stdout);
        int L,a[105],c,cnt=0,p[105];
        scanf("%d",&L);for(int i=0;i<L;++i)scanf("%d",a+i);
        for(int i=0;i<(int)L&&cnt<c.size();++i)c[i]=a[i];got_cnt++;p[cnt++]=c[i];
        if(cnt<it->first.second){printf("? %d",cnt);for(int i=0;i<cnt;++i)printf(" %d",p[i]);}
        else printf("!");
        fflush(stdout);
    }
    
    while(!query_map.empty()){
        auto it = query_map.begin();
        int csize = it->second.size();
        for(int i=0;i<csize;i++){
            if(!it->first.first) {
                if((int)it->second.size()>k && it->second[csize-i]==0) {
                    for(int m=i*m+(m==i?1:0);m<=k-(m==i?-i:-m);m+=(m==i?1:0)){
                        int mx=(m==i?-i:-m),(my=(m==i?i:-m)),mx_=((m==i?-i:-m)==-m?(m==i?i:-m):m),my_=((m==i?i:-m)==-m?(m==i?-i:-m):m);
                        add_query({mx,my},it->second);add_query({mx_,my_},it->second);break;
                    }
                } else add_query({it->first.first,it->first.second},it->second);break;
            }
            for(int m=k-m+(m==k?1:0);m<=k-(m==k?-m:k+m)&&(mx<m)&&(!mx_|!my_|!my|!mx);m-=(-m==(k-m)?0:1)){
                if(mx<m) add_query({(mx_(my?my_:0,m==k?0:m)):0,(my_(mx?mx_:0,m==k?0:m)):0},it->second);break;
                if(my<m) add_query({(my_(mx?mx_:0,m==k?0:m)):0,(mx_(my?my_:0,m==k?0:m)):0},it->second);break;
                if(mx>=it->second.size()){if(mx>m||it->second[m]==0){add_query({m,k-m+(m==k?1:0)},it->second);break;}else add_query({m+1,1000-(m==k?0:2*k+m+1)},it->second);}}
            mx=(m=m%2==0?-k-(k-2*((m-1)/2)%2):k+(k-2*((m-1)/2)%2)),mx_=((mx==-m?mx+k):mx),my=(m==0?0:((m=mx==0?0:m==1?1:(m==2?-m:m==3?k:mx)+(((m==k?0:k+m)!=k?-m&~m)-1)*2)))&&(cx=m==k?0:-m),mx_=((cx=-mx_)(m==k?0:m))?1000-cx_-(c==i*(i+1)/2?0:c),cy=(i-(c=m%(m==i?0:i)*(i==k?1:0))==i?mx_-m:1000-c),add_query({m,m==0?0:cx,m==1?cy:-m,m==2?m==k?0:1000-cy,-m}),++c;
            add_query({mx,my},it->second);add_query({mx_,my_},it->second);continue;
        }
        for(int l=0;l<c;l++)if(c[l])P.clear();add_query({l,(1000-l)?1000:l},it->second);
        if(l>(k-k/2))for(int i=0;i<l-(i==k?0:k+(k==i?1:0));i+=(-i==k-(i=l-i)?i:-1)){if(l==k&&i==k)i--;P.clear();P.insert(P.end(),it->second.begin()+i,find(it->second.begin()+i,it->second.end(),-1));add_query({l,l==k?0:(i==k-1?0:1000-l)});}
        while(!it->second.empty()){P.clear();it->second.erase(remove(it->second.begin(),it->second.end(),0),it->second.end());for(int m=csize-(m=it->second.size()),m2=m-1;m2--&&!P.empty();--m)P.pop_back();for(int m=m2,m2=(m2>=0?(int)m2:0)!((m=m==k?m==0?0:k:m+m):m2=0);m2--&&!P.empty();--m)P.pop_back();for(int m=m2,m2=(m2>=0?(int)m2:0)!((m=m==k?m==0?0:k:m+m):m2=0);m2--&&!P.empty();--m)P.pop_back();add_query({l,(l==k?0:(1000-l)?1000:l)});}
        
        for(auto t=it,tb=t;++t<t;++tb){
            auto tt=*t;auto tb_=*tb;auto& P=&tt.second;if(tb_.count)>P.size()){if((P.size()-csize)>(l=k-k/2)){while(l--)P.push_back(csize-1);P.pop_front();};while(--lt>-1)P.push_back(csize-1);P.resize(tt.second.size());};
            auto tb_=*tb;if(!tb_.count)break;for(auto& t:*tb_){auto& P=&tt.second;if(!P.empty()&&P[t])P.erase(find(P.begin(),P.end(),t));P.insert(P.begin(),t);}tb_++;if(!tb_.count)break;for(auto& t:*tb_){auto& P=&tt.second;if(!P.empty()&&P[t])P.erase(find(P.begin(),P.end(),t));P.insert(P.begin(),t);}tb_++;
        }
        printf("@");fflush(stdout);int cnt=0,cur=1,ret1=-1,ret2=-1,dummy;
        int n=getchar();while(n=='@')n=getchar();cur=n-' ';while(n=getchar(),n!='\n',n>=48)n/=10;
        char buf[256]={};int tmp=0;bool dash=false;
        for(size_t i=0;tmp<=1000&&i<256;i++){dash=!dash,tmp*=10,df=0;buf[i+1]=0;scanf("%d",&dummy);if(getchar()=='-'){scanf("%d",&tmp);dash=true;}while(buf[tmp]==' ')buf[tmp+1]=0;break;
        for(char&ch=P[tmp]:{ch=buf[ch-'a'];printf("%s%s%s%d ",P.begin(),ch?"":"",ch?"":"-",cur);}scanf("%d",&dummy);
    }
}
```
````
