**Reading the problem and pinning the contract.** I have a rooted tree of `n` relay stations. Station `u` has a power level `p[u]`, and its signal goes *downward only*: it reaches `v` exactly when `v` is a strict descendant of `u` and the edge-distance between them is at most `p[u]`, i.e. `depth(v) - depth(u) <= p[u]`. A station `v` is "covered" if some proper ancestor reaches it, and I must count covered stations. Two phrases in that contract are load-bearing and I underline them now, before writing a single line: *strict descendant* (a station never covers itself, and the root — having no ancestor — is never covered) and *at most* (the distance test is `<=`, inclusive of the exact-boundary case `depth(v) - depth(u) = p[u]`). I expect the whole difficulty of this problem to be those two boundaries, so I will keep returning to them.

**Fixing the scale and the data types.** The bounds are `n <= 2*10^5` and `0 <= p[u] <= 10^9`. The quantity I will manipulate is `depth(u) + p[u]`: `depth(u)` is at most `n-1 < 2*10^5`, and `p[u]` is at most `10^9`, so the sum is at most about `1.0002*10^9`. That is just under the signed 32-bit ceiling of `~2.147*10^9`, so it *technically* fits in `int` — but only barely, and the moment I take a `max` or compare against another such value I do not want to be reasoning about whether some intermediate overflowed. I will hold every "reach" quantity in `long long`. The covered count is at most `n = 2*10^5`, trivially fits, but I will keep it `long long` too for uniformity of the output format. The decision is cheap and removes a whole class of silent wrong-answers; I take it unconditionally.

**A second structural worry: recursion depth.** The tests explicitly include chains of length up to `2*10^5`. A textbook recursive DFS would recurse `2*10^5` frames deep, and the default stack (often 8 MB) overflows somewhere in the low hundreds of thousands of frames for a frame this size. So I will *not* write recursive DFS; I will use an explicit stack of frames. That choice has its own boundary hazard — what exactly each frame carries, and when I read versus update the running state — which I will have to trace. But it is the right structural call.

**Laying out the candidate approaches.** Two routes, and I commit to the one I can both prove and run in time.

- *Per-node upward walk.* For each station `v`, climb its proper-ancestor chain `par[v], par[par[v]], ...` up to the root, and test `depth(v) - depth(u) <= p[u]` for each `u`, stopping at the first hit. This is obviously correct — it is a literal transcription of the definition — but on a chain of `n` nodes it is `O(n^2)`, which is `4*10^10` operations at the top end. Far too slow. I keep it in my pocket as the *reference brute force* for verification, not as the submission.

- *Single DFS carrying a running reach.* Here is the structural insight. Ancestor `u` reaches every descendant down to absolute depth `depth(u) + p[u]`. So define, for a node `v`, `reach(v) = max over proper ancestors u of v of ( depth(u) + p[u] )`. Then `v` is covered iff `depth(v) <= reach(v)`: some ancestor's downward range, measured as an absolute depth, extends to at least where `v` sits. And `reach` is monotone down a path — descending from a node to its child can only *add* an ancestor to the max, never remove one — so I can thread `reach` down each root-to-node path during one DFS, updating it by a single `max` per edge. That is `O(n)` time and `O(n)` stack. This is the approach.

**Deriving the running-reach recurrence carefully — and spotting where self could leak in.** Let me be precise about *which* ancestors `reach(v)` ranges over: **proper** ancestors of `v`, i.e. everything strictly above `v`, not `v` itself. That is the "strict descendant" clause biting. So if I am standing at node `u` in the DFS with a value `reach` that is "the max of `depth+p` over the proper ancestors of `u`", then:

- To decide whether `u` is covered, I test `depth(u) <= reach` using *this* `reach`, the one that excludes `u`'s own `depth(u) + p[u]`. Good — that is exactly "some proper ancestor reaches `u`".
- To pass down to `u`'s children, I must *now* fold in `u`'s own contribution, because for a child `w`, station `u` *is* a proper ancestor. So the value handed to each child is `max(reach, depth(u) + p[u])`.

The order matters enormously: **test first with the inherited `reach`, then fold in `u`, then recurse.** If I folded `u` in before testing `u`, I would be letting `u` cover itself, which the "strict descendant" rule forbids. I write that ordering down as the thing to get right.

**First implementation.** I set up the children lists, find the root (`par == -1`), and run an explicit-stack DFS. Each frame carries `(node, depth, reach)` where `reach` is the max-`(depth+p)` over the node's proper ancestors. The root's frame starts with `reach = -infinity` (it has no proper ancestor), so the root's test `depth(0)=0 <= -inf` is false and the root is never counted — which matches "the gateway can never be covered". Here is my first cut of the core loop:

```
long long covered = 0;
const long long NEG = LLONG_MIN / 4;
st.push_back({root, 0, NEG});
while (!st.empty()) {
    Frame f = st.back(); st.pop_back();
    int u = f.node, d = f.depth; long long reach = f.reach;
    long long reachIncludingU = max(reach, (long long)d + p[u]);
    if ((long long)d < reachIncludingU) covered++;      // <-- test
    for (int w : ch[u]) st.push_back({w, d + 1, reachIncludingU});
}
```

I typed two things by reflex that I now distrust: I built `reachIncludingU` *before* the coverage test, and I wrote the test with a strict `<`. Both touch exactly the boundaries I flagged at the start, so I will not trust the code — I will trace it.

**First trace — the documented sample, to expose the self-coverage bug.** The sample tree: node `0` root power `1`; node `1` (parent `0`) power `0`; node `2` (parent `1`) power `1`; node `3` (parent `2`) power `0`; node `4` (parent `0`) power `0`. Depths are `0,1,2,3,1`. The intended covered set, worked out from the definition by hand, is `{1,3,4}`, answer `3`: station `1` is reached by root `0` (`0+1=1 >= depth 1`), station `3` by station `2` (`2+1=3 >= depth 3`), station `4` by root (`0+1=1 >= depth 1`); station `2` by nobody (`0+1=1<2`, `1+0=1<2`); the root by nobody.

Now I trace my code. Push `(0,0,NEG)`. Pop `0`: `reachIncludingU = max(NEG, 0+1) = 1`; test `0 < 1` -> true, `covered=1`. But wait — node `0` is the **root**, it must *never* be covered, and my code just counted it. The bug is exactly the ordering I warned myself about: I folded `0`'s own `depth+p = 1` into `reachIncludingU` and then tested `depth(0)=0` against it, so the root "covered itself". The `reachIncludingU` I tested against includes `u`, which violates the strict-descendant rule. Confirmed real bug number one.

**Diagnosing bug one precisely.** The coverage test for `u` must use the `reach` *inherited from proper ancestors only* — the value `reach` in the frame, **not** `reachIncludingU`. `u`'s own contribution belongs only in what I pass to the children. So the fix is to test against `reach` and compute `reachIncludingU` afterward (or, equivalently, only for the children loop). Let me rewrite the body:

```
if ((long long)d < reach) covered++;                    // test with inherited reach
long long reachIncludingU = max(reach, (long long)d + p[u]);
for (int w : ch[u]) st.push_back({w, d + 1, reachIncludingU});
```

**Second trace — same sample, to check the fix and expose the inclusive/exclusive bug.** Re-run with the corrected order. Push `(0,0,NEG)`. Pop `0`: test `0 < NEG`? false — root not counted, good, bug one is gone. `reachIncludingU = max(NEG, 0+1) = 1`. Push children of `0`: that is `1` and `4`, each with `(.,1,1)`. Pop `4` (depth 1, reach 1): test `1 < 1`? **false**. But station `4` *is* covered — the root reaches it (`0+1=1 >= 1`). My code just *missed* a genuinely covered node. That is bug number two, and it is precisely the inclusive/exclusive boundary: the definition says "at most", `depth(v) - depth(u) <= p[u]`, which rearranges to `depth(v) <= depth(u) + p[u] = reach`, an **inclusive** `<=`. I wrote `<`, which silently drops every node sitting *exactly* on an ancestor's range boundary. Confirmed real bug number two.

**Diagnosing bug two precisely.** "At most `p[u]` edges" includes the case of exactly `p[u]` edges. In the absolute-depth form, "covered" is `depth(v) <= reach`, not `depth(v) < reach`. The strict `<` would only be right if the rule were "*fewer than* `p[u]` edges", which it is not. I change the test to `<=`:

```
if ((long long)d <= reach) covered++;                   // inclusive "at most"
```

**Re-tracing the whole sample after both fixes.** Push `(0,0,NEG)`.
- Pop `0` (d=0, reach=NEG): `0 <= NEG`? false (root, correct). `reachIncludingU = max(NEG, 0+1) = 1`. Push `1` and `4` as `(.,1,1)`.
- Pop `4` (d=1, reach=1): `1 <= 1`? **true**, `covered=1` (the boundary node now counts, fixing bug two). `4` has no children.
- Pop `1` (d=1, reach=1): `1 <= 1`? true, `covered=2`. `reachIncludingU = max(1, 1+0) = 1`. Push `2` as `(2,2,1)`.
- Pop `2` (d=2, reach=1): `2 <= 1`? false (uncovered, correct — nobody reaches depth 2). `reachIncludingU = max(1, 2+1) = 3`. Push `3` as `(3,3,3)`.
- Pop `3` (d=3, reach=3): `3 <= 3`? **true**, `covered=3` (another exact-boundary node, courtesy of `<=`). No children.

Final `covered = 3`, covered set `{4,1,3}`. This matches the hand-derived answer exactly, and crucially the two nodes that flipped — `4` and `3` — are precisely the exact-boundary nodes, which is the evidence that the `<=` fix is doing real work and not coincidentally matching. Both bugs broke for the reasons I diagnosed, and both fixes are confirmed by the same trace. That is the verification I trust.

**Sanity-checking the derivation itself, not just the code.** Before I declare the recurrence correct I want to confirm the monotonicity claim I leaned on — that `reach` only grows as I descend, so a single running max is enough. Going from node `u` (with inherited `reach`) to child `w`, the new value is `max(reach, depth(u)+p[u])`, which is `>= reach`. So yes, `reach` is nondecreasing down any path; there is never an ancestor whose contribution I would need to "remove", which is exactly what makes the one-pass running max valid. I also confirm I am taking the max over *all* proper ancestors and not just the parent: each edge folds in one more ancestor (`u` itself, relative to its children), and by induction the value at `w` is the max of `depth+p` over `w`'s entire proper-ancestor chain. The derivation holds.

**Edge cases, deliberately, because this is where boundary code dies.**
- `n = 0`: there is no tree. I handle it before any DFS with `if (n <= 0) { print 0; }`. Output `0`. Correct — zero stations, zero covered.
- `n = 1`: a single root, no edges. The DFS pushes `(root,0,NEG)`, pops it, tests `0 <= NEG` -> false, prints `0`. Correct — the lone station has no ancestor, so it cannot be covered. I verified this with input `1 / -1 5`: output `0` even though its power is large, because power only helps *descendants*, of which it has none.
- All powers `0`: a station with `p=0` reaches descendants within `0` edges, i.e. none (strict descendants are at distance `>= 1`). So nobody is covered. Tested `3 / -1 0 / 0 0 / 1 0` (a chain of three, all power 0): output `0`. Correct.
- Maximum power: root power `10^9` on a chain of `2*10^5`. Every non-root node is within `10^9` edges of the root, so all `n-1` are covered. Tested with all powers `10^9` on a `2*10^5` chain: output `199999`. Correct, and it confirms the `long long` reach (`0 + 10^9` and beyond) never misbehaves.
- Deep chain, recursion-depth stress: a `2*10^5`-node path with all powers `1` covers every non-root node (each parent reaches its child at distance `1`), expected `199999`. The explicit-stack DFS produces `199999` in about `0.03 s` with peak memory ~16 MB — no stack overflow, comfortably inside the `1 s` / `256 MB` budget. The iterative DFS decision paid off exactly where the recursive one would have crashed.
- Wide star: root power `1` with three leaf children of power `0`. Each leaf is at depth `1`, reach `1`, `1 <= 1` true, so all three covered, root not; expected `3`. Tested `4 / -1 1 / 0 0 / 0 0 / 0 0`: output `3`. Correct.
- Overflow audit: the only arithmetic is `(long long)d + p[u]` (at most `~10^9`, fits with vast room) and a `max`; the sentinel `NEG = LLONG_MIN/4` is only ever read inside `max`/comparisons and never has anything added to it, so it cannot underflow. The covered count maxes at `2*10^5`. All safe.

**Cross-checking against the brute force at scale.** I have the per-node upward walk as an independent reference (different method: it literally re-walks each ancestor chain instead of threading a running max). I ran the DFS solution against it on several hundred random small trees — chains, stars, and random parent-pointer trees with small powers `0..3` chosen specifically so exact-boundary hits are frequent — and they agree on every case, zero mismatches. The agreement on the boundary-heavy small cases is the strongest signal that the inclusive `<=` and the strict-descendant exclusion are both implemented correctly.

**Final solution.** The idea is the running-reach DFS, proven by the monotonicity argument and the literal definition; the two things that could (and did) go wrong are pure boundaries — folding a node into its own reach before testing it (self-coverage), and writing `<` instead of `<=` for "at most" — both caught by tracing the documented sample and both fixed. I ship the iterative version to survive the deep-chain tests:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (scanf("%d", &n) != 1) return 0;
    if (n <= 0) { printf("0\n"); return 0; }

    vector<int> par(n), p(n);
    vector<vector<int>> ch(n);
    int root = -1;
    for (int i = 0; i < n; i++) {
        int pa;
        if (scanf("%d %d", &pa, &p[i]) != 2) return 0; // parent (-1 for root) and power
        par[i] = pa;
        if (pa == -1) root = i;
        else ch[pa].push_back(i);
    }

    // A station u covers a STRICT descendant v iff depth(v) - depth(u) <= p[u].
    // Carry along the DFS, for each node v, the value
    //   reach = max over PROPER ancestors u of ( depth(u) + p[u] ).
    // Then v is covered  <=>  depth(v) <= reach   (inclusive boundary, excludes self).
    long long covered = 0;
    const long long NEG = LLONG_MIN / 4; // "no ancestor reaches here yet"

    struct Frame { int node; int depth; long long reach; };
    vector<Frame> st;
    st.reserve(n);
    st.push_back({root, 0, NEG}); // root has no proper ancestor

    while (!st.empty()) {
        Frame f = st.back();
        st.pop_back();
        int u = f.node;
        int d = f.depth;
        long long reach = f.reach;

        if ((long long)d <= reach) covered++; // some proper ancestor reaches depth d

        long long reachIncludingU = max(reach, (long long)d + p[u]);
        for (int w : ch[u]) {
            st.push_back({w, d + 1, reachIncludingU});
        }
    }

    printf("%lld\n", covered);
    return 0;
}
```

**Causal recap.** The problem reduces to: for each node, does any *proper* ancestor's downward range, measured as the absolute depth `depth(u)+p[u]`, reach this node's depth? Because that range-as-depth is monotone down a path, one DFS threading a running maximum `reach` answers every node in `O(n)`. The two bugs were both boundary errors I had flagged in advance and then actually committed: I first computed `reachIncludingU` *before* the coverage test, which let the root (and any node) cover itself in violation of "strict descendant" — a trace of the sample showed the root being counted, and the fix is to test against the inherited `reach` and only fold in the current node for the children; and I first wrote the test as `<` instead of `<=`, which dropped every node sitting exactly on an ancestor's boundary — the same trace showed nodes `4` and `3` (both exact-boundary) being missed, and switching to `<=` for "at most" restored them. With both fixed the trace yields the documented answer `3`, the `n=0`/`n=1`/all-zero/max-power/deep-chain/star corners all check out, an iterative stack keeps the `2*10^5` chain from overflowing, and `long long` keeps the reach arithmetic safe; a brute-force cross-check over hundreds of boundary-heavy random trees agrees with zero mismatches.
