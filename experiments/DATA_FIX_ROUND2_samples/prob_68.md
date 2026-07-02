# 题 68 — base均分 0.0 vs 直接SFT均分 8.9(掉点 -8.9)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=4,得分 0.0,completion_tokens=11142)

````
The problem is about finding two pens with a total of at least $n$ units of ink, given that we don't know the exact amounts. We only know that each pen has a unique amount of ink from $0$ to $n-1$.

Let's think about this. Since the amounts form a permutation of $0, 1, \dots, n-1$, if we have access to all pens and their initial amounts, then the sum would be $\frac{n(n-1)}{2}$. However, we can't query directly - we can only try writing on them.

If we write on a pen, it consumes 1 unit of ink if it had ink. If it was empty (0 units), writing does nothing and returns 0.

The key insight here is that we want to find two pens whose remaining ink sums to at least $n$. Since the original amounts are distinct integers from $0$ to $n-1$, by the Pigeonhole Principle or similar reasoning:

If we select any two pens that initially had more than $(n-3)/2$ units each (approximately), they might work. But wait, let me think differently.

Consider the largest values in the permutation. The sum of the two largest values would be $(n-1) + (n-2) = 2n - 3$, which is definitely at least $n$ for $n \geq 10$.

But we don't know which pens have how much ink. However, we can try to "discover" the pens with more ink.

One strategy could be:
1. Try writing on each pen multiple times to determine how much ink it initially has.
2. Once we identify enough pens and their ink levels, pick the two with most ink.

But trying multiple times per pen might use too many queries. Let's check the constraints:
- $t \leq 1001$ test cases
- $n \leq 25$

Wait, actually I need to re-read the interaction protocol. The judge gives us feedback when we try to write:

If output: `0 p` (where p is a pen number)
Read back: x where x=1 means the pen still had ink (consumed 1), x=0 means it was already empty.

Important: When x=1, it means we consumed 1 unit but there might be more left. When x=0, it was already empty before we tried.

Actually, I realize we should NOT consume ink unnecessarily since we're checking final states. But wait, to find how much ink each pen has, we do need to keep writing until it becomes empty.

A better approach might be:
1. Try writing once on each pen to see which ones still have ink.
2. For pens that respond with 1, continue writing until they respond with 0.
3. Count how many writes were needed for each pen - this tells us the initial ink level.
4. Pick the two pens with highest ink levels. Their sum would be at least n.

Let me verify: If two pens have initial amounts $a$ and $b$ (distinct integers from $0$ to $n-1$), what's the minimum possible value of $a+b$ where $a+b \geq n$?

The smallest two values that sum to $\geq n$: 
- If $a = \lfloor n/2 \rfloor$ and $b = \lceil n/2 \rceil$, then $a+b = n$ ✓

Since we can only make sure the sum is at least $n$, we should aim for pens that we believe have more ink.

Given $n \geq 10$:
The average ink is $(n-1)/2$. Two pens above average would likely sum to $> n$.

Algorithm:
For each pen from $0$ to $n-1$:
1. Keep writing on it (0 p) until it says 0 (empty).
2. Record the number of successful writes (count of 1s before getting a 0).
3. This count is the initial ink amount.
4. After testing all pens, select the two with highest counts.
5. Output those two pens (format: 1 p1 p2).

Time complexity per test case: At most $2n$ iterations of queries (each pen tested up to $n$ times worst case). Total queries ≤ $n^2$. With $n \leq 25$, that's at most 625 queries per test case, which is fine.

Actually, wait - I'm consuming ink during my testing phase. That's fine because we're learning about the initial state, and we need to report which pens to select based on what remains AFTER our operations... 

Hmm, but the problem says: "when you finally select them, their total remaining ink must be at least n."

This means after we write on pens during discovery, we shouldn't have consumed too much! Otherwise, our selected pens might not meet the requirement anymore.

Oh, but reading carefully again: "You cannot directly query the amount of ink in any pen. Instead, you can only 'try to write' with a pen — then you will learn whether it can still write or not. If it can write, it will consume 1 unit of ink."

So every time I try to write, I consume ink. If I figure out that a pen started with 5 units, I've now consumed say 3 of them through my exploration. Those won't be counted toward the final check.

That changes things significantly! I need to either:
1. Find two pens without consuming their ink during discovery, OR  
2. Make sure my consumption doesn't reduce the final sum below n.

Since each test consumes ink we can't get back, option 2 seems impossible - we'd be wasting our own ink. So I should look for option 1.

How to determine ink level WITHOUT consuming it?

Actually, thinking more carefully: When a pen responds with 1, we learned that "the pen still had ink before consuming 1 unit" - meaning it went from having ≥1 to having one less. So yes, we consumed ink.

Wait, maybe the answer isn't about discovering ALL pens, but strategically picking two that we're confident will have enough combined ink even after some consumption.

Here's a different idea: What if I just pick any two pens that I've tried to write on first, and they responded with 1 both times? That proves they had ink. But how do I know their remaining ink?

Maybe I need fewer queries and be smarter:

Key observation: Since the original amounts are a permutation of 0..n-1, at least two pens must have at least floor((n+1)/2) ink initially (by pigeonhole principle, if we sort the amounts). Actually, let me think again...

If we have numbers $0, 1, ..., n-1$, the two largest sum to $(n-1)+(n-2) = 2n-3$.

But I'm worried about losing ink during investigation...

NEW APPROACH:
Instead of testing individual pens completely, maybe I should:
1. Select a candidate pair tentatively
2. Use other pens to verify if they have lots of ink
3. Be careful not to drain candidates too much

OR EVEN BETTER:

What if I never fully test any pen for its full capacity, but instead just use a randomized approach: keep asking various pens to check if they have ink (they respond 1 or 0). After some checks, make an educated guess?

But we need deterministic correctness...

Let me reconsider. In competitive programming interactive problems, sometimes we CAN allow some consumption if we're strategic about it. The guarantee is that we need to find two pens whose INITIAL values sum to at least n, OR whose FINAL values after our consumption sum to at least n?

Re-reading: "Note that the requirement refers to the remaining ink in the two chosen pens: when you finally select them, their total remaining ink must be at least n."

So it's the REMAINING ink that matters, not the initial ink. That's good - it means whatever ink remains after my tests will count toward the requirement.

OK so now I DO get to consume ink during discovery, as long as the two pens I eventually select end up with total ≥ n remaining ink.

Strategy refined:
1. For each pen, partially test to estimate its current remaining ink (without necessarily reducing it to 0).
2. Select the two pens with the highest estimated remaining ink.
3. Verify and output them.

But how to estimate remaining ink efficiently? I could write on each pen once - those returning 1 definitely have at least 1 ink. Writing again reduces further.

Let me think about worst-case consumption:
If I use k writes on pen i before selecting it, I reduce its value from p_i to max(0, p_i-k).

Actually here's a smart approach: For each pen, I'll write ONCE. Count how many return 1 (meaning non-empty before that write). This takes exactly n queries.

For each pen i:
- Write with pen i once → if response is 1, increment counter for this pen.
- After all n pens, I have some indication of which pens are currently "strong".

The pens that returned 1 definitely have ≥1 ink remaining. Pens that returned 0 have 0.

But this doesn't help me distinguish between pens with different ink levels...

OK let's try yet another approach. Since the amounts are a permutation:
- At most one pen has 0
- At most one pen has 1
- ...
- At most one pen has n-1

So if I write on n-1 different pens and ALL respond 1, that means none of them was empty, so one pen MUST be empty (the one I didn't try).

Now here's the critical insight: If I write on EVERY pen exactly once, I'll learn which pens became empty (those that said 0). All others said 1, meaning they originally had at least 1, and now each lost 1.

Then the remaining ink for each pen i is max(0, p_i - 1), where p_i was original.

But I need TWO pens such that remaining ink sums to ≥ n...

Let me think about what maximum sum I can guarantee.

Suppose I write on every pen once. Then:
- Any pen that originally had 0 stays at 0 (writes 0, returns 0).
- Any pen that originally had k≥1 becomes k-1 (writes 1, returns 1).

If I know the original amounts, I can compute current = original - 1 (if original > 0, else 0).

I want two pens i and j where (p_i - 1) + (p_j - 1) ≥ n if both p_i, p_j ≥ 1.
Or (0) + (something) if one is empty.

Best case: Pick the two that originally had largest amounts. Say n-1 and n-2.
After one write each: (n-1-1) + (n-2-1) = n-2 + n-3 = 2n-5.

Is 2n-5 ≥ n? Yes for n ≥ 5.

Second best: If one pen was originally 0 and stays 0, I'd pick another large one like n-1 (now n-2). Sum = n-2. This doesn't satisfy ≥ n.

So I should avoid empty pens entirely.

Better approach:
Don't write on all pens. Instead, selectively choose which to explore.

What if I designate certain pens as "candidates" and only write on the OTHERS to figure out something?

Wait, actually here's a simpler guaranteed approach:

Since amounts are a permutation of 0 to n-1:
- There's at least one pen with ≥ 1 ink (actually at least n-1 of them)
- There's at least one pen with ≥ 2 ink (actually at least n-2 of them)
- ...

More specifically, if I randomly pick two pens, what's the probability they satisfy the condition? Hard to reason probabilistically here.

OK NEW INSIGHT:

What if I never try to FULLY drain ANY pen, but instead just write ONCE on each pen? After that write:
- Some pens respond 1 → they had ≥1 initially, now have ≥0 (but we know they had ink)
- Some pens respond 0 → they were 0 initially, stay 0

Now suppose I have identified which pens have ink and which are empty. Among the pens that returned 1 (so had ink), I should try to find two that together have enough.

BUT HERE'S THE KEY: After writing once on all pens, if I now SELECT any two pens that BOTH responded 1, I'm saying "these two remain with enough combined ink". BUT I don't know how much they have individually anymore, just that each ≥0.

Hmm, but I wrote on each pen ONCE. So if pen i responded 1, it means it went from p_i → p_i-1, and I observed it went successfully. So current remaining ink is p_i-1.

Wait no, let me be very clear:
- Query: write with pen i
- Response 1: Pen had ink before, consumed 1. Remaining: p_i - 1
- Response 0: Pen had 0 before, stays 0. Remaining: 0

So if I write ONCE on every pen, and record responses, I can calculate current remaining for each pen i as:
- response[i] == 1 ? p_i - 1 : 0
Where p_i is unknown initial amount... 

But I don't know p_i, so I can't compute p_i - 1.

UNLESS:
For pen i, if I write again immediately and get response 1 again, then I know p_i - 1 ≥ 1, so p_i ≥ 2, current remaining = p_i - 2 after second write.

Actually, let me track cumulative writes per pen:
- Write twice on pen i:
  - First response 1: remaining becomes p_i - 1
  - Second response:
    - If 1: remaining is p_i - 2, pen wasn't empty
    - If 0: remaining is 0, pen became empty during second write, so p_i - 1 = 1, p_i = 2

So by writing multiple times on EACH pen, I can discover p_i exactly!

Write m times on pen i:
- If last m-th response is 1 and all previous were 1: p_i ≥ m
- If m-th response is 0: then p_i = m-1 (since after m-1 writes it had 1 left, after m-th write it's 0)

Actually:
Start with m attempts:
- Attempt 1: gets 1 → p_i >= 1, now has p_i-1
- Attempt 2: gets 1 → p_i >= 2, now has p_i-2
- ...
- Attempt k: gets 0 → p_i = k-1, now has 0

Yes! So if I exhaustively write on every pen until it goes empty, I learn exactly p_i for all i.

Complexity: In worst case, sum of p_i = n(n-1)/2, so ~1275 writes for n=25. That's too many! Each test case can take up to O(n²) queries, times t≤1001... that's potentially 3 million+ queries across all test cases, which might timeout.

But wait, we DON'T need to know p_i exactly. We just need to find two pens where after SOME consumption, their remaining sum ≥ n.

CRITICAL REALIZATION:
If I write on pen i exactly p_i times, I drain it completely. But if I write fewer times, I leave it with some remaining ink.

So my strategy can be:
1. Write ONCE on all pens to get initial info.
2. For each pen, note whether it returned 1 or 0.
3. If it returned 1, it's a candidate (has ≥1 ink, now has p_i-1 remaining).
4. From among those that returned 1, pick the ones I suspect have MORE remaining ink.

But I don't know p_i for those candidates...

UNLESS: Among pens that returned 1 after ONE write, if I write AGAIN on the same pen, and it returns 1, then I know it had ≥2 initially, so it now has ≥1 remaining. If it returns 0, it had exactly 1 initially, so now has 0 remaining.

AH HA! Here's the optimal strategy:

**Strategy:**
1. Write ONCE on each pen 0..n-1. Record responses r[0..n-1].
2. Collect indices of pens where r[i] == 1 into list L. These are pens with ≥1 ink remaining.
3. For each pen i in L, write ONCE MORE on it. Let new responses be s[i].
   - If s[i] == 1: this pen had ≥2 initially, current remaining = (original_p_i - 2) ≥ 1
   - If s[i] == 0: this pen had exactly 1 initially, current remaining = 0
4. Now, pens where s[i] == 1 are "super-candidates" (they have ≥1 remaining after two writes, so original ≥2).
5. Pick two super-candidate pens. Why? Because if a pen responded 1 after TWO writes, its remaining ink is original - 2 ≥ 1.
   Wait, that's not right. After two writes that both succeed (return 1), current remaining is original_p_i - 2. If original_p_i = 2, then remaining = 0. So I can't be sure it has remaining.

Let me recalculate:
- Original: p_i
- Write #1: if p_i >= 1, returns 1, remaining = p_i - 1
- Write #2: if p_i - 1 >= 1 (i.e., p_i >= 2), returns 1, remaining = p_i - 2
- Write #k: if p_i - (k-1) >= 1, returns 1, remaining = p_i - k

So if pen returned 1 after k consecutive writes (1st, 2nd, ..., k-th write all return 1), then:
- Current remaining = p_i - k ≥ 1 (since it would be 0 otherwise)
- Thus p_i ≥ k+1
- And current remaining could be anything from 1 onward

Now here's the brilliant part: If I write k times on each pen and the LAST write returns 1 (not 0), then:
- pen i's remaining = p_i - k
- I don't know p_i, but I do know p_i ≥ k (otherwise write k wouldn't succeed)

Actually, if the k-th write returns 1:
- Before k-th write: remaining was p_i - (k-1) ≥ 1
- After k-th write: remaining is p_i - k ≥ 0
- But since k-th write returned 1, it means p_i - (k-1) ≥ 1, so p_i ≥ k
- And remaining after k writes = p_i - k ≥ 0

If k-th write returns 0:
- Before k-th write: remaining was p_i - (k-1) = 1 (exactly)
- After k-th write: remaining is 0
- So p_i = k

Great! So:
- Write m times on pen i:
  - If last (m-th) write returns 1: pen has p_i ≥ m, current remaining = p_i - m ≥ 0
  - If last (m-th) write returns 0: pen has p_i = m-1, current remaining = 0

OK so if I write exactly m times on pen i and the m-th write returns 0, then p_i = m-1 and remaining = 0.

If I write m times and the m-th write returns 1, then p_i ≥ m and remaining = p_i - m.

Now, strategy:
1. For each pen i, start writing repeatedly until:
   - Either it goes empty (returns 0 on some attempt)
   - Or I hit my budget limit

But budget... let me think about what's safe.

For safety, I should ensure that whichever two pens I pick, their SUM of remaining ink is ≥ n.

If I pick pen i where I wrote k_i times (last returned 1) and pen j where I wrote k_j times (last returned 1):
- Remaining_i = p_i - k_i
- Remaining_j = p_j - k_j
- Need: (p_i - k_i) + (p_j - k_j) ≥ n
- I know p_i ≥ k_i and p_j ≥ k_j
- Best case: p_i is large, p_j is large

But I also spent k_i + k_j writes draining these pens!

Alternative thought: What if I ONLY write ONCE on every pen initially? Then for each pen i:
- If response = 1: remaining = p_i - 1 ≥ 0, and since original was a permutation, p_i ∈ {1,2,...,n-1}, so remaining ∈ {0,1,...,n-2}
- If response = 0: remaining = 0

Among those with response=1, I have pens with remaining ≥ 0 (at least one of them, possibly zero). But I don't know which ones are good.

However, I know the ORIGINAL amounts are 0,1,...,n-1. The sum of all originals = n(n-1)/2.

After writing once on all pens:
- Total remaining ink overall = sum(p_i) - (#non-empty)
- #non-empty = n minus (#that were originally 0)
- Originally, EXACTLY ONE pen has 0, n-1 pens have ≥1
- After one write on all:
  - One pen was 0, stays 0
  - n-1 pens were ≥1, each reduced by 1
  - Total remaining = 0 + sum_{all except 0}(p_i - 1) = sum(p_i) - (n-1) - 0 = n(n-1)/2 - n + 1 = (n²-n-2n+2)/2 = (n²-3n+2)/2 = (n-1)(n-2)/2

For n=25, total remaining = 24*23/2 = 276.

I need to pick two pens summing to ≥ n=25. On average, each of the two would need 12.5. Total available is 276, so plenty.

But I need to FIND two such pens. Since I don't know which pens have what, I need more information.

Let me try this concrete approach:

**Final Strategy:**
1. **Initial pass**: Write ONCE on every pen i=0..n-1. Save responses in array r[].
   - Pens with r[i]==1 have current remaining = p_i - 1 ≥ 0. (Original p_i ∈ {1..n-1})
   
2. **Secondary pass**: For each pen with r[i]==1, write ONCE MORE on it. Save responses in array s[][].
   - For pen i where r[i]==1:
     - If s[i]==1: this pen originally had p_i ≥ 2, current remaining after 2 writes = p_i - 2 ≥ 0. More importantly, I can write again to check further.
     - If s[i]==0: this pen originally had p_i = 1, current remaining after 2 writes = 0. Should discard.
   
3. **Selection**: Among pens where s[i]==1 (meaning they survived at least 2 writes), I need two such pens.
   - Wait, survival of 2 writes means p_i ≥ 2, current after 2 writes = p_i - 2 ≥ 0.
   - Best candidates are those that SURVIVED MANY WRITES.

Actually, here's the cleanest approach given the constraints:

Since n ≤ 25, I can afford to write roughly 3-5 times per pen.

**Revised Algorithm:**
1. Create arrays cnt[0..n-1] initialized to 0.
2. For each pen i from 0 to n-1:
   - While true:
     - Output: `0 i`
     - Read response x
     - If x == 1:
       - cnt[i]++ (this means pen survived this write, current ink decreases by 1)
     - Else (x == 0):
       - Break loop (pen is now empty)
   
3. After this phase, for each pen i:
   - If cnt[i] < n, current remaining ink is: I don't know p_i, but I know cnt[i] writes succeeded, and the (cnt[i]+1)-th write failed (or pen exhausted).
   - If the (cnt[i]+1)-th write would fail, then p_i = cnt[i], current remaining = 0.
   - If the (cnt[i]+1)-th write succeeds, then p_i ≥ cnt[i]+1, current remaining = p_i - (cnt[i]+1).
   
   Wait, I broke when x==0. So after breaking, if last write returned 0:
   - Previous state had 1 ink
   - So p_i = cnt[i] (number of successful writes before failure)
   - Current remaining = 0
   
   Perfect! After my while loops:
   - For pen i, cnt[i] = number of successful writes (responses of 1)
   - If I read 0, pen is empty, remaining = 0
   - If I haven't reached 0, remaining = ??? 

No, wait. In my loop, I break WHEN I get 0. So after the loop completes:
- cnt[i] = number of times I got 1 before hitting 0
- The next (implied) write would have been on empty pen
- So p_i = cnt[i] (because after cnt[i] successful writes, the pen had 1 left, then we'd get 0)
- Current remaining = 0

Hmm, that drains everything! Not good.

RETHINK:

The requirement is about REMAINING INK when I FINALLY REPORT the two pens. So if I drain pens during discovery, I can't report them.

Solution: Don't drain the selected pens! Leave them alone until the final selection.

So:
1. Discover ink levels for all pens BY NOT reporting any selected pen during discovery.
2. During discovery, I can probe pens, but mark them as "probed".
3. When ready to select two, choose two unprobed pens that I predict will satisfy the condition.
4. Report them. They won't have been touched, so remaining = original.

Perfect! Now the question is: how do I know which unprobed pens to pick without probing them?

I can probe SOME pens to gather statistics, then predict.

Since the original amounts are a permutation of 0..n-1:
- The average is (n-1)/2 ≈ 12.25 for n=25
- By pigeonhole, at least one pen has ≥ ceil((n-1)/2) + 1 = 13 (for n=25, average is 12)
- Actually, more precisely: if sorted amounts are q_0 < q_1 < ... < q_n-1, then q_i ≥ i

The best sum comes from the two largest: (n-1) + (n-2) = 2n-3.

Any two pens that survive probing will have remaining equal to their original (if untouched). So if I pick two random pens from the remaining set, and they happen to be the two largest, great!

But I need to GUARANTEE success...

OK, here's a winning strategy based on the structure:

**Guaranteed Approach:**
Since there's exactly one pen with amount 0, and one with 1, etc.:

1. Probe pens sequentially. For each pen i:
   - Output `0 i`
   - Read response x
   - If x == 1: store that pen i is "alive" (has ≥1 remaining)
   - If x == 0: mark pen i as empty (has 0)

2. After probing all n pens exactly once:
   - I have n alive pens and 1 empty pen (wait, no - I probed each, so if a pen had 0 initially, I got 0 and it stayed 0)
   - Alive pens (returned 1) have current remaining = p_i - 1 ∈ {0, 1, ..., n-2}
   - Empty pen (returned 0) has current remaining = 0

3. Among the alive pens, I need two whose CURRENT remaining sum ≥ n.
   - Current remaining_i = p_i - 1 for alive pens
   - Need (p_i - 1) + (p_j - 1) ≥ n ⇒ p_i + p_j ≥ n + 2
   - Since original amounts are a permutation, I can bound this...

Actually, the BEST two original amounts are n-1 and n-2. Their current after one write each is n-2 and n-3. Sum = 2n - 5.

For n=25, 2n-5 = 45 ≥ 25. ✓

For the WORST case among alive pens:
- If I picked the two smallest alive pens originally, they might be small...

Wait, but I can control which alive pens I pick! I can pick the ones I believe have high original amounts.

But after one probe, all alive pens look identical to me (all returned 1). No differentiation.

OH WAIT! I can probe TWICE on certain pens to differentiate them!

**Enhanced Strategy:**
1. First, probe EVERY PEN ONCE. Mark responses:
   - r[i] = 1 means pen i has ≥1 original ink, now has p_i - 1 remaining
   - r[i] = 0 means pen i has 0 original, now has 0 remaining
   
2. For pens where r[i] == 1, probe AGAIN (second write):
   - If second returns 1: pen i has p_i ≥ 2, current remaining after 2 writes = p_i - 2 ≥ 0
   - If second returns 0: pen i has p_i = 1, current remaining after 2 writes = 0
   
3. Categorize:
   - Level 0 (after 2 writes): pen became empty during second write → p_i = 1, remaining = 0 (DON'T PICK)
   - Level 1+: pen survived 2 writes → p_i ≥ 2, remaining = p_i - 2 ≥ 0

4. Among Level 1+ pens, pick ANY TWO. Their original amounts are ≥ 2. After 2 writes each, remaining are ≥ 0 each.

But I drained them by 2 writes each! If I picked a pen with original p_i = 2, after 2 writes it has 0 remaining. BAD.

I need to LEAVE the selected pens UNTouched.

FINAL CORRECT STRATEGY:
- Use a subset of pens for PROBE DISCOVERY, don't touch the ones I intend to select.
- Select two pens that I HAVEN'T PROBED.
- For these unprobed pens, their remaining ink equals their original p_i.
- I need p_i + p_j ≥ n.

Can I guarantee this?

Probe n-2 pens to learn about them. Leave 2 probes untouched. Among the n-2 probed ones, I can identify which ones are likely to be low ink. The unprobed ones might be high ink.

Specifically:
1. Choose to leave pens with HIGH indices unprobed (e.g., pen n-2 and pen n-1).
2. Probe all OTHER pens (0 to n-3).
3. Among probed ones, if I find a pen with p_i ≤ some threshold, eliminate it.
4. If remaining candidates include n-2 and n-1, select those!

But the permutation is arbitrary, so n-2 and n-1 might have low amounts.

OK, the REAL answer must involve randomness or a clever property.

WAIT! Let me re-examine the problem statement more carefully...

"You cannot directly query the amount of ink in any pen."

But maybe I'm overcomplicating. Let me consider a simple randomized strategy:

Randomized Strategy:
- Randomly shuffle pens 0 to n-1
- For each pen in shuffled order:
  - Try writing once
  - Track responses
- Eventually select two pens that I'm most confident about

But competitive programming solutions generally need to be deterministic unless specified otherwise.

ACTUALLY, let me reconsider the core constraint more carefully:

"The sum of all original ink levels = 0 + 1 + ... + (n-1) = n(n-1)/2"

"After we write on some pens, their ink decreases. But we need two pens with remaining ink sum ≥ n."

Here's the mathematical guarantee:

If I write ONCE on EVERY pen, total remaining ink = n(n-1)/2 - (number of non-empty pens) = n(n-1)/2 - (n-1) = (n-1)(n-2)/2.

As computed earlier, for n=25, this is 276.

I need two pens summing to 25. By averaging, I need each to average 12.5. With 276 total spread across up to 25 pens (excluding the empty one), most pens have substantial ink.

By picking the TWO pens with HIGHEST CURRENT remaining ink (which I can discover through probing), I should be able to exceed n.

But to discover current remaining, I need to probe... but probing drains. So I can't do both.

UNLESS: I only partially probe each pen - stop when I have enough info to identify top candidates, but leave them mostly intact.

**Partial Probing Strategy:**
1. For each pen i, write exactly ONCE. Record response r[i].
   - r[i] == 1: pen has p_i ≥ 1, current remaining = p_i - 1
   - r[i] == 0: pen has p_i == 0, current remaining = 0
   
2. Among pens with r[i] == 1, pick TWO of them arbitrarily.
   - Their current remaining = p_i - 1 and p_j - 1
   - Since p_i and p_j are distinct integers from 1 to n-1:
   - Best case: n-1 and n-2, current = n-2 and n-3, sum = 2n-5 ≥ n for n ≥ 5
   - Worst case among r[i]==1: 1 and 2, current = 0 and 1, sum = 1 (bad!)

Problem: The two smallest non-zero values sum to 1 << n.

But wait, I only have n-1 pens with r[i]==1 (since exactly one pen has p_i=0). The values are {1, 2, ..., n-1}. If I pick the two smallest (1 and 2), their current remaining after probe = 0 and 1. BAD.

So random selection from r[i]==1 is risky!

**Smart Selection from Partial Probes:**

The issue is distinguishing which probed pens had higher original values.

After one probe, all pens with r[i]=1 look identical (current remaining = p_i - 1). To distinguish, I need to probe MORE.

But each additional probe on a candidate reduces its remaining ink!

KEY INSIGHT: What if I designate certain pens as "victims" (to be heavily probed to learn their levels), and others as "candidates" (never probed)?

Victims:
- Probe them thoroughly to discover their original levels.
- Discard them from selection.

Candidates:
- Never probe. Their current ink = original.
- Select from these.

If I designate n-2 victims and 2 candidates:
- Probe all victims thoroughly to learn their levels
- Candidates have unknown levels, but I know one victim is empty (amount 0), one victim is 1, etc.
- After knowing victim levels, I can infer candidates' levels

Example: If I probe all pens 0..n-3, and find that some have levels 0, 1, 2, 3, ..., then I know the two unprobed pens (n-2, n-1) must have the largest remaining values in the permutation!

PERFECT! This works!

**Final Algorithm:**
1. Designate two pens as CANDIDATES (e.g., pens 0 and 1).
2. Designate the remaining n-2 pens (2 to n-1) as VICTIMS.
3. Thoroughly probe all VICTIMS to discover their original ink levels:
   - For each victim pen i:
     - Continuously write (0 i) until response is 0
     - Count how many writes were successful (responses of 1)
     - The count equals the original p_i (since after p_i writes, it empties)
4. From the discovered victim levels, identify which original amount is MISSING (the two candidates must have).
   - Victims have levels v_2, v_3, ..., v_n-1
   - Original levels are {0, 1, ..., n-1}
   - Missing levels go to candidates c_0 and c_1
   - Let these missing values be m_0 and m_1
   - Current remaining of candidates = m_0 and m_1 (never touched!)
5. Check if m_0 + m_1 ≥ n
   - If yes, select them! Done.
   - If no, swap roles! Designate different pens as candidates.

To ensure robustness, I can:
- Try multiple pairs as candidates
- For each try, probe the other n-2 pens
- Select the candidate pair with maximum predicted sum
- But predicting requires knowledge...

Actually, simpler: Just designate pens n-2 and n-1 as candidates (indices n-2, n-1). Probe 0 to n-3 thoroughly. Identify which levels are missing from {0, 1, ..., n-1}. Those missing levels belong to n-2 and n-1. Their sum is p_{n-2} + p_{n-1}. If this sum ≥ n, report them. Otherwise, try a different pair...

But wait, can the sum of ANY two distinct values from 0 to n-1 be < n?
- Smallest sum of two distinct: 0 + 1 = 1
- Largest sum: (n-1) + (n-2) = 2n-3

For n=25, smallest sum 0+1=1 ≪ 25. So it's possible to have a bad pair!

My job is to find ANY pair with sum ≥ n, not necessarily the largest pair.

So if I fix candidates as n-2 and n-1, they might be unlucky (e.g., both 0 and 1). I need to try multiple configurations or a smarter selection.

**Optimal Robust Strategy:**

Try all possible candidate pairs systematically:
1. Iterate through all combinations of 2 pens as potential candidates.
2. For each pair, probe all OTHER pens to determine their levels.
3. Calculate the missing levels (which must belong to candidates).
4. If missing_levels_sum ≥ n, SELECT AND REPORT THIS PAIR. STOP.
5. Continue to next combination if condition fails.

Worst case: I might need to probe O(n * n²) pens? Too slow.

Optimization: Start with promising candidates first. Pick candidates that are most likely to have high values - maybe the last ones added in the permutation? But I don't know the permutation order.

Practical approach:
1. Arbitrarily pick pair (0, 1) as candidates initially.
2. Probe all other pens (2 to n-1).
3. Calculate their discovered levels. Missing levels = candidates' original values.
4. If missing_sum ≥ n, done!
5. Otherwise, try pair (0, 2), then (0, 3), etc., until find working pair.
6. To speed up, memoize partial results.

But given t=1001 test cases and n≤25, I need efficiency.

**Efficient Strategy:**

Since I need ANY pair with sum ≥ n:
- The "average" pair (middle values) around n/2 should work well
- (⌊(n-1)/2⌋, ⌈(n-1)/2⌉) sums to n!

So I should focus on finding pens with middle-range values.

Let me code a solution that:
1. Tries to designate candidate pairs starting from balanced values
2. Efficiently discovers levels for non-candidate pens
3. Reports the first pair found with sufficient sum

```cpp
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    int t;
    cin >> t;
    
    while (t--) {
        int n;
        cin >> n;
        
        // Track ink levels of probed pens
        vector<int> levels(n, -1); // -1 means unknown
        
        // Function to probe a pen until empty, return its original amount
        auto probe_pen = [&](int pen_idx) -> int {
            int count = 0;
            while (true) {
                cout << "0 " << pen_idx << "\n";
                fflush(stdout);
                int response;
                cin >> response;
                if (response == 1) {
                    count++;
                } else {
                    break;
                }
            }
            return count; // Original amount
        };
        
        // Try to find two pens with total >= n
        bool found = false;
        
        // Strategy: Try different candidate pairs, probe others
        // Prioritize pairs that would have balanced/high values
        
        for (int i = 0; i < n && !found; i++) {
            for (int j = i + 1; j < n && !found; j++) {
                
                // Check if i or j already probed
                bool i_known = (levels[i] != -1);
                bool j_known = (levels[j] != -1);
                
                if (!i_known || !j_known) {
                    // Need to probe others first, or skip if known low
                    
                    // Probe all other pens
                    for (int k = 0; k < n; k++) {
                        if (k == i || k == j) continue;
                        
                        // Skip if already probed
                        if (levels[k] != -1) continue;
                        
                        levels[k] = probe_pen(k);
                    }
                    
                    // Now calculate levels for i and j by elimination
                    // Collected levels of other pens
                    vector<int> collected;
                    for (int k = 0; k < n; k++) {
                        if (k == i || k == j) continue;
                        collected.push_back(levels[k]);
                    }
                    
                    // Fill in levels[i] and levels[j] from permutation logic
                    int missing_low = 0;
                    int expected_count = 0;
                    for (int lvl : collected) {
                        if (lvl < n) {
                            int count_occ = 0;
                            for (int k : collected) if (k == lvl) count_occ++;
                            
                            // Just collect all and find duplicates aren't needed
                            // Original has 0 to n-1 exactly once
                        }
                    }
                    
                    // Better approach: create set of used levels
                    vector<bool> used(n, false);
                    for (int val : collected) {
                        used[val] = true;
                    }
                    
                    // Find first unused (level for i)
                    int level_i = -1;
                    for (int v = 0; v < n; v++) {
                        if (!used[v]) {
                            level_i = v;
                            break;
                        }
                    }
                    
                    // Find second unused (level for j)
                    int level_j = -1;
                    for (int v = 0; v < n && level_j == -1; v++) {
                        if (!used[v]) {
                            // Should only find two unused
                            level_j = v;
                        }
                    }
                    
                    // Assign levels
                    int original_i = (i < j) ? level_i : level_j;
                    int original_j = (i < j) ? level_j : level_i;
                    
                    levels[i] = original_i;
                    levels[j] = original_j;
                    
                    // Check if sum >= n
                    if (original_i + original_j >= n) {
                        cout << "1 " << i << " " << j << "\n";
                        fflush(stdout);
                        found = true;
                    } else {
                        // Reset i and j levels so we can try different candidate pair
                        levels[i] = -1;
                        levels[j] = -1;
                    }
                } else {
                    // Both already known, just check
                    if (levels[i] + levels[j] >= n) {
                        cout << "1 " << i << " " << j << "\n";
                        fflush(stdout);
                        found = true;
                    }
                }
            }
        }
        
        if (!found) {
            // Fallback
            // Output any two pens - though mathematically one should exist
            cout << "1 0 1\n";
            fflush(stdout);
        }
    }
    
    return 0;
}
```

I realize the nested loops might be inefficient. Let me simplify the algorithm to be more direct. Given the constraints and problem nature, I should focus on a straightforward probing strategy that guarantees finding valid pen pairs.

Actually, I need to be very careful about the interaction protocol. Every time I output a query line, I must flush and read a response before continuing. My solution should follow this strictly.

Let me write a cleaner version:
</think>

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int t;
    cin >> t;

    while (t--) {
        int n;
        cin >> n;

        vector<int> levels(n, -1); // stores original ink amounts, -1 if unknown

        // Probe a pen until empty and count its original amount
        auto probe_until_empty = [&](int pen) {
            int count = 0;
            while (true) {
                cout << "0 " << pen << "\n";
                fflush(stdout);
                int response;
                cin >> response;
                if (response == 1) {
                    count++;
                } else {
                    break;
                }
            }
            return count;
        };

        bool found = false;

        // Greedy approach: try pairs that are likely to have sufficient ink
        // Focus on larger index pens first since they tend to have higher values
        for (int i = n - 2; i >= 0; i--) {
            for (int j = n - 1; j > i; j--) {
                
                if (levels[i] != -1 && levels[j] != -1) {
                    // Both already known
                    if (levels[i] + levels[j] >= n) {
                        cout << "1 " << i << " " << j << "\n";
                        fflush(stdout);
                        found = true;
                        goto next_test_case;
                    }
                    continue;
                }

                // Identify pens we need to probe (not i, not j)
                vector<int> probed_values;
                vector<bool> known_used(n, false);

                for (int k = 0; k < n; k++) {
                    if (k == i || k == j) continue;
                    if (levels[k] != -1) {
                        known_used[k] = true;
                        probed_values.push_back(levels[k]);
                    } else {
                        // Need to probe this pen
                        levels[k] = probe_until_empty(k);
                        probed_values.push_back(levels[k]);
                        known_used[k] = true;
                    }
                }

                // Determine which original values belong to i and j by elimination
                vector<bool> used(n, false);
                for (int v : probed_values) {
                    used[v] = true;
                }

                int val_i = -1;
                int val_j = -1;
                int search = 0;
                for (int v = 0; v < n && search < 2; v++) {
                    if (!used[v]) {
                        if (search == 0) {
                            val_i = v;
                            search++;
                        } else {
                            val_j = v;
                        }
                    }
                }

                if (val_i == -1 || val_j == -1) {
                    // Something wrong, reset and try next pair
                    levels[i] = -1;
                    levels[j] = -1;
                    continue;
                }

                // Store the inferred levels (we assume they are correct based on permutation property)
                levels[i] = val_i;
                levels[j] = val_j;

                if (val_i + val_j >= n) {
                    cout << "1 " << i << " " << j << "\n";
                    fflush(stdout);
                    found = true;
                    goto next_test_case;
                } else {
                    // Backtrack: remove levels so we can try another pair
                    levels[i] = -1;
                    levels[j] = -1;
                }
            }
        }

next_test_case:
        ;
    }

    return 0;
}
```
````

---

## 直接SFT 样本(sample_idx=1,得分 0.0,completion_tokens=965)

````
I can solve this problem by using a constructive algorithm based on the Pigeonhole Principle and an interactive "try to write" operation. Here's my approach:

## Key Insight
The pens contain ink amounts that form a permutation of 0, 1, ..., n-1, so the total amount of ink across all pens equals:  
Total Ink = 0 + 1 + 2 + ... + (n-1) = n(n-1)/2

Since we need two pens whose combined remaining ink is at least n, and the total ink available is n(n-1)/2, I should identify pens that definitely have ink left (value ≥ 1) rather than just consuming ink from every pen. This avoids exhausting pens early when trying to find a valid pair.

## Strategy
1. **Track Used Pens**: Keep a set of pens already tried with to avoid redundant queries.
2. **Find Pens With Ink Left**: For each unused pen, query if it has ink (x = 1 means it has ink).
3. **Store Results**: When a pen confirms having ink, store its number in `inkLeft`.
4. **Check If Enough Pens Have Ink**: Maintain `cnt` (count of pens found with ink). When `cnt >= n`, we've likely satisfied the condition since there are only n pens. Stop early to optimize.
5. **Select Valid Pair**: Once enough pens with ink are identified, output any two as the answer. Since the total ink is n(n-1)/2 ≥ n for n ≥ 10, a valid pair exists among pens with non-zero values.
6. **Output Answer**: Print type 1 followed by two distinct pen numbers separated by space.

## Correctness Reasoning
The problem requires finding two pens where remaining ink ≥ n. The key insight is avoiding over-consumption before checking validity. By first identifying which pens have ink left (query return x=1), we select pens with non-zero residual ink—guaranteeing their sum meets or exceeds the threshold due to the high total ink value n(n-1)/2. This satisfies the "at least n units remaining" requirement without wasting queries testing empty pens, ensuring efficient solution within the n ≤ 25 limit and 60-query constraint.

```cpp
#include <iostream>
#include <vector>
#include <unordered_set>
#include <algorithm>

using namespace std;

void solve(int n) {
    unordered_set<int> used;
    vector<int> inkLeft;
    
    for (int i = 0; i < n; i++) {
        cout << "0 " << i << endl;
        int x; cin >> x;
        
        if (used.find(i) == used.end()) {
            if (x == 1) {
                inkLeft.push_back(i);
            }
            used.insert(i);
            
            // Optimization: if enough pens found with ink left, stop early
            if ((int)inkLeft.size() >= n) break;
        }
    }
    
    // Select any two pens with ink left
    cout << "1 " << inkLeft[0] << " " << inkLeft[1] << endl;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    int t;
    cin >> t;
    while (t--) {
        int n;
        cin >> n;
        solve(n);
    }
    return 0;
}
```
</think>

```cpp
#include <iostream>
#include <unordered_set>

using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    int t;
    cin >> t;
    while (t--) {
        int n;
        cin >> n;
        unordered_set<int> used;
        vector<int> ans;
        for (int i = 0; i < n; i++) {
            if ((int)ans.size() >= 2) break;
            cout << "0 " << i << endl;
            int res;
            cin >> res;
            if (!used.count(i)) {
                if (res == 1) ans.push_back(i);
                used.insert(i);
            }
        }
        cout << "1 " << ans[0] << " " << ans[1] << endl;
    }
    return 0;
}
```
````
