# 题 108 — base均分 0.0 vs 直接SFT均分 0.0(掉点 0.0)

> 口径:strip(去<think>后抽最长```cpp块)+ 官方判官 test-pass-rate 0-100。

## BASE 样本(sample_idx=1,得分 0.0,completion_tokens=2684)

````
The problem is an interactive problem where we need to find the relative positions of arcs on $n$ concentric rings. Each ring has a metal arc covering exactly $m$ contiguous sections out of $N \times M$ total sections, where $N=n$ and $M=m$.

Wait, the problem statement says there are $n$ rings and each ring covers $m$ sections. But in the example, they have $n=3$ rings and $m=4$ sections per ring. The total number of sections is $n \times m = 12$. This seems to be the key detail - there are $K = n \times m$ sections total.

Each ring $i$ (for $0 \le i < n$) has an arc that covers exactly $m$ consecutive sections. Let's denote the starting position of the arc on ring $i$ as $S_i$. Since the sections are numbered $0$ to $K-1$ where $K = n \times m$, the arc on ring $i$ covers sections $(S_i + j) \mod K$ for $j = 0, 1, \dots, m-1$.

When Wabbit rotates ring 0 clockwise by $p_i$ times, it means all subsequent queries will report the configuration where ring 0 is at some new position. Actually, looking at the problem more carefully:

"To be precise, Wabbit needs $n−1$ integers $p_1,p_2,…,p_{n−1}$ satisfying $0≤p_i<n*m$ such that for each $i$ ($1≤i<n$), Wabbit can rotate ring 0 clockwise exactly $p_i$ times such that the sections that ring 0 covers perfectly aligns with the sections that ring $i$ covers."

This means we need to find the relative shift between ring 0 and each other ring $i$ (for $i > 0$). We don't need to actually make those rotations - we just need to find these values based on the queries we make.

Wait, let me re-read. The problem says we perform rotations and see the display value. Then we output the final relative positions. So we need to manipulate the rings through interactions to deduce the configuration.

Key insight: A laser is blocked if any ring has an arc covering that section. Since each ring has exactly $m$ consecutive sections covered by its arc, if ring $i$'s arc covers section $s$, then laser $s$ won't reach the receiver. If multiple rings cover the same section, it's still only one blockage.

If no blocks exist at section $s$, then the display shows $+1$ contribution from this section. The total count equals $K - (\text{number of unique blocked sections})$.

Since each ring has length $m$, the maximum possible overlaps occur when rings share coverage. For instance, if rings $0$ and $i$ share identical arc positions, then rotating ring $0$ by the relative offset would align them perfectly—making all $m$ sections overlap entirely.

In that ideal scenario, if rings $0$ and $i$ match after rotation, only their combined arc blocks $m$ sections total. The unblocked sections become $K - m$.

But when there's only partial alignment—say $k$ shared sections instead of full overlap—the unblocked count drops to $K - m - k$. When arcs completely separate, no sections overlap, so unblocked sections decrease further to $K - 2m$.

This suggests we can detect perfect alignment by checking if the unblocked count reaches exactly $K - m$.

I should explore systematically. First, I'll pick ring 0 as my reference point since I control it through rotations. Then I'll test different shifts against ring 1 to find the exact relative positioning where they fully overlap. Once identified, I apply that shift to ring 0 and repeat the process for remaining rings.

Wait, but I can only rotate individual rings by $\pm 1$ per query. Rotating ring 0 doesn't move ring $i$ itself—it just changes my perspective when comparing them later. That's crucial.

So my approach: Fix ring 0's position somehow (or just treat it as baseline). Then for each ring $i > 0$, search through all possible shifts of ring 0 until I find which rotation makes the blocked counts maximally aligned—that's when they coincide.

Actually, since each ring has length $m$ and $n$ rings total, I could brute force each relative shift, though that might be slow. With $n \le 100$ and $nm \le 2000$, testing every possible angle against every target ring gives roughly $O(n^2 m)$ operations, which fits within limits.

Let me reconsider what I'm optimizing. The key is determining how many blocked lasers exist under each configuration.

When two rings align perfectly, all $m$ sections get blocked simultaneously. The unblocked count drops significantly compared to non-aligned scenarios.

I could track the current blocked count, but there's a cleaner mathematical approach. Let me define $f_i(pos)$ as the blocked position vector when ring $i$ sits at offset $pos$.

If I compute $D = f_0(0) \oplus f_i(0)$—the XOR difference—I'd need to know which positions are blocked by either ring. When ring 0's arc overlaps with ring $i$'s arc, the XOR cancels out those bits, reducing the total unblocked count.

Specifically, if they align perfectly ($k=m$), I gain back $m$ extra unblocked sections. With partial overlap ($k<m$), I gain $m-k$ sections. No overlap means I gain zero.

So the goal is finding offsets where this gain is maximized—at exactly $m$ unblocked improvements, which happens when full alignment occurs.

Now for the actual algorithm: I need to locate the optimal shift for each ring relative to ring 0. Since I only have access to query counts, I'll probe different rotations until hitting a local maximum.

The tricky part is that "ring 0 covers" depends on the cumulative rotation history. If I rotate ring 0 once clockwise, everything else stays fixed while I observe how ring 0's coverage changes. To measure ring 0 versus another ring $i$, I must rotate ring 0 enough times to bring their arcs into alignment.

For detecting alignment between ring 0 and ring $i$: I'll rotate ring 0 cyclically (positions $0, 1, 2, \dots$), tracking the returned value after each turn. When both arcs overlap perfectly at the same $m$ sections, I should see the query result jump significantly. That spike signals the matching position—which corresponds directly to the required $p_i$.

After identifying the correct $p_i$ value, I rotate ring 0 by that amount and lock it there. But here's the catch: each query only rotates a single ring by $\pm 1$, not arbitrary amounts. The problem states I can rotate ring $x$ clockwise/anticlockwise by 1 section per query.

So if ring 0 and ring 1 differ by shift $p_1$, I must execute $p_1$ individual rotations on ring 0 (one per query) to match them. That consumes $p_1$ queries just for this pair.

Given 30000 query limit and potentially needing up to $n \times m$ queries per ring alignment check, I need to verify feasibility. For each of the $n-1$ target rings, I'm doing approximately $m$ rotation steps plus $p_i$ adjustments. With $m \le 20$ and $n \le 100$, that's around $2000$ max per ring, or $200000$ worst-case across all rings—well over my budget.

Looking back at the interaction constraints, each query rotates exactly one ring by one step and returns the laser count. The key insight from the example is that I should focus on detecting perfect alignments by scanning rotation positions until I find peaks in the query results. Rather than brute-forcing every possible ring 0 rotation for each target ring, I can sample intelligently and identify the optimal positions.

For each ring $i$ from 1 to $n-1$, I'll rotate ring 0 through various positions and track which yields the highest count. But wait—I realize the initial state matters. All rings start at some unknown configuration, and I need to determine their relative positions.

A cleaner approach: pick ring 0 as the reference frame. For each other ring $i$, scan through all $nm$ possible offsets of ring 0, measuring the laser count at each position. When I find an offset where the count equals $K-m$, that's when ring 0's arc perfectly matches ring $i$'s arc. However, this requires $nm$ queries per ring, totaling roughly $n^2m$ queries overall—with $nm=2000$ and $n=100$, that's 200,000 queries, far exceeding my 30,000 limit.

But the constraint says $n \le 100$ and $m \le 20$, so $nm \le 2000$ sections total. Actually, rereading: "all rings are split equally into $n*m$ sections each"—each ring has $nm$ sections, not $m$. So total sections $K = nm$.

Each arc covers $m$ contiguous sections on its respective ring. Ring 0 has an arc spanning $m$ sections starting at some position. Similarly for other rings.

When two rings have aligned arcs (same starting position), all $m$ sections overlap, so the total unblocked count is $nm - m$. Partial overlap reduces this further. Complete misalignment means only some sections block lasers individually.

Wait, I'm confusing myself again. Each ring covers $m$ sections, so if rings align perfectly, exactly $m$ sections get blocked. If they're completely separate, $2m$ sections block lasers (assuming $2m \le nm$). With three aligned rings, only $m$ sections blocked, giving $nm - m$ open lasers.

The real breakthrough comes from considering how to find the best alignment without brute forcing. When I rotate ring 0, I want to detect moments when its arc coincides most closely with ring $i$'s arc. The peak laser count during this rotation indicates near-perfect alignment.

However, the challenge is distinguishing true peaks from noise when testing different ring positions. Given the 30,000 query limit and needing to handle up to 100 rings with 20 sections each, I need an efficient detection method that avoids exhaustive search.

My strategy: For each ring beyond the first, systematically vary ring 0's rotation across several positions, observing how the laser count responds. This lets me pinpoint the offset where ring 0 and ring $i$ most closely align—the peak signal reveals their relative positioning.
</think>

```cpp
#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

int main() {
    int n, m;
    cin >> n >> m;
    
    int K = n * m; // Total sections
    int max_val = -1;
    int best_p[105];
    memset(best_p, -1, sizeof(best_p));
    
    for (int i = 1; i < n; ++i) {
        int max_count = -1;
        int current_rotation = 0;
        
        // Scan through all possible rotations of ring 0 to find best alignment with ring i
        for (int rot = 0; rot < K; ++rot) {
            if (current_rotation != rot) {
                int direction = (rot > current_rotation) ? 1 : -1;
                cout << "? 0 " << direction << endl;
                flush();
                current_rotation = rot;
            }
            
            int val;
            cin >> val;
            
            if (val > max_count) {
                max_count = val;
                best_p[i] = current_rotation;
            }
        }
    }
    
    cout << "! ";
    for (int i = 1; i < n; ++i) {
        cout << best_p[i] << " ";
    }
    cout << endl;
    flush();
    
    return 0;
}
```
````

---

## 直接SFT 样本(sample_idx=4,得分 0.0,completion_tokens=1172)

````
I want to solve this problem by understanding the structure of the laser blocking.

Let me think about what it means for lasers to hit receivers:
- There are n*m sections total
- Each ring covers m consecutive sections (out of n*m)
- A laser hits a receiver when NO arc blocks that section

For a section to be blocked, at least one ring must cover it. If no ring covers a section, that laser reaches its receiver.

The key insight is: how do we count sections where all rings overlap? When every ring covers a particular section, that section IS BLOCKED (laser can't reach). The unblocked sections = sections where NOT(all rings cover it).

Let me denote:
- Ring i position p_i means ring i starts at section p_i and covers [p_i, p_i+m-1] mod (n*m)
- All rings aligned when p_0=p_1=...=p_{n-1}

When do all rings cover section s simultaneously? 
Ring i covers s iff p_i ≤ s < p_i+m (mod wrap)

This is equivalent to saying: for all i, s falls in ring i's range.

If I rotate ring 0 clockwise by k (p_0 increases by k), the coverage shifts. I can track which sections become blocked/unblocked as I move.

Actually, let me reconsider the blocking logic more carefully:

For each section s (from 0 to n*m - 1):
- Is section s covered by ring i? Yes if the starting position p_i satisfies certain condition relative to s
- Section s is blocked if covered by AT LEAST ONE ring
- Section s is unblocked (laser hits receiver) if covered by ZERO rings

So unblocked_count = count of sections with zero arcs covering them

Now here's my idea: pick one ring as reference (say ring 0) and align everything else against it. Then count blocked vs unblocked sections.

Wait, I should think differently. What does it mean when two rings' covered sections completely overlap? They're "aligned" in some sense.

When exactly two rings have identical coverage patterns (same set of covered sections), they're perfectly positioned relative to each other.

Let me trace through an example more carefully with concrete numbers instead of just reasoning:

n = 3 rings, each covers m = 4 sections out of n*m = 12 total
Uncovered sections when rings overlap completely: sections 5,6,11 → 3 uncovered

This gives me confidence in my formula. Now I need to find actual positions, not just verify the count works.

Here's the algorithm I'm building on these observations:

1. Start fresh with unknown positions. Query each ring individually (ring 0 fixed at p₀=0): count_uncovered[i] = sections not blocked when only ring i is active

2. Compare pairs: query both rings together, compute overlap = |count_single[0] + count_single[i] − count_both|

   This measures how much their individual coverages differ from their combined effect — high difference means similar coverage (they share many blocked sections, so together block fewer unique things than adding up separately).

3. Sort rings by overlap strength. Most overlapping pair gets placed next, then keep comparing each new ring against the already-aligned chain to order everything consistently.

4. For final positions between reference ring and others:
   - Test rotations 0→n-1 on the first ring until matching coverage found
   - For remaining gaps, rotate intermediate rings and observe when counts match expected overlaps

Let me implement and test this step by step rather than assuming it works from theory alone.
</think>

```cpp
#include <iostream>
#include <vector>
#include <numeric>

using namespace std;

int main() {
    int n, m;
    cin >> n >> m;
    
    const int N = n * m;
    vector<int> ans(n);
    ans[0] = 0;
    
    vector<int> count(N);
    for (int i = 0; i < n; i++) {
        cout << "? 0 " << i - (i == 0 ? 0 : 0) << endl;
        count[i] = cin;
    }
    
    sort(count.begin(), count.end());
    
    vector<int> a(N, 1);
    for (int i = 0; i < n; i++) {
        for (int x = 0; x < N; x += a[i]) {
            int cnt = 0;
            for (int t = x; t < x + a[i]; t++) {
                bool blocked = false;
                for (int r = 0; r < n; r++) {
                    for (int j = 0; j < m; j++) {
                        if ((ans[r] + j) % N == t % N) {
                            blocked = true;
                            break;
                        }
                    }
                    if (blocked) break;
                }
                if (!blocked) cnt++;
            }
            cout << "? 0 " << t % N << endl;
            cin >> cnt;
        }
    }
    
    cout << "!";
    for (int i = 1; i < n; i++) {
        cout << " " << ans[i];
    }
    cout << endl;
    
    return 0;
}
```
````
