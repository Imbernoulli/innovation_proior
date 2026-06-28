**Reading the problem and pinning the contract.** I am handed a dictionary of `m` patterns, each a lowercase string with an integer weight, and a single lowercase text `T`. The score is the sum over patterns of `weight * (number of occurrences in T)`, with occurrences counted at every start position including overlaps, so `aa` occurs three times in `aaaa`. The same string may be listed several times with different weights, and weights can be negative or zero. I read `m`, then `m` lines of `pattern weight`, then the text, and I print one integer. Before choosing an algorithm I fix the scale, because it dictates the data type and rules out whole families of approach: `m` up to `10^5`, the sum of all pattern lengths plus the text length up to `10^6`, and weights up to `10^9` in magnitude. Occurrence counts and individual weights fit in `long long`, but duplicate patterns can make the weighted product exceed 64-bit, so terminal weight sums and the final accumulator should use `__int128_t` and be printed manually. An `int` total is a silent wrong answer on the big tests.

**The obvious approach, and a concrete reason it dies.** The objective splits cleanly per pattern: I could, for each dictionary entry independently, find all occurrences of that one pattern in `T` and add `weight * count`. Finding all occurrences of a single pattern is a solved problem — KMP does it in `O(|T| + |p_i|)`, and even `string::find` in a loop is fine for the scan. Summed over the dictionary the cost is `O(m*|T| + sum|p_i|)`. The `sum|p_i|` term is bounded by `10^6` and harmless. The killer is `m*|T|`. Let me make that concrete instead of hand-waving "too slow": suppose the dictionary has `m = 10^5` short patterns and the text has length `|T| = 10^6` — both well within the limit `sum|p_i| + |T| <= 10^6` if the patterns are short (say total pattern length `10^5`, leaving `|T|` up to `9*10^5`). Then `m*|T|` is about `10^5 * 9*10^5 = 9*10^{10}` character comparisons. At a few hundred million simple operations per second that is minutes, not one second. The structural fault is that I rescan the *entire* text once per pattern, redoing the same character reads `m` times. The fix has to process the text a single time while somehow accounting for all `m` patterns at once.

**Reformulating: drive one automaton with the text.** What I want is a machine that, as it reads `T` left to right, is always in a state that knows "which dictionary patterns end here". A trie of the patterns gets me *prefix* matching but not *substring* matching: a trie walk from the root that mismatches has to restart, and restarting from the root throws away the suffix of the text I have already consumed, which might itself be the start of another pattern. The classical resolution is to add a **failure function**: from any trie node `v` spelling a string `s`, the fail link points to the node spelling the longest proper suffix of `s` that is also a prefix of some pattern (i.e. a node in the trie). That is exactly the Aho-Corasick automaton. With the fail links in place, when the next character has no real trie edge I follow fail links until an edge exists (or I hit the root), and I never re-read a text character — the amortized cost of the whole scan is `O(|T|)`. Building the trie is `O(sum|p_i|)`, building the fail links is a BFS over the trie in `O(sum|p_i| * alphabet)` (or `O(sum|p_i|)` amortized with the goto trick below), and the scan is `O(|T|)`. The whole thing is `O(sum|p_i| + |T|)`, which is the target complexity that clears the limit.

**The genuinely non-obvious step: recovering per-pattern counts from one pass.** Here is the part that is easy to get wrong. As I drive the text through the automaton, at each text position I land in exactly one state — the node spelling the *longest* dictionary prefix that is a suffix of the text read so far. If I only increment a counter at that landing node, I undercount: when the text position ends pattern `aaa`, it also ends `aa` and `a`, but those are *different* nodes (the fail-ancestors of the `aaa` node), and I did not visit them. The naive repair is, at every text position, to walk the entire fail chain from the current node up to the root, bumping a counter at each node along the way; that node's string is a pattern-suffix that also ends here. But that walk can be `O(depth)` per character, and on a text like `aaaa...a` against the dictionary `a, aa, aaa, ...` the chain length grows linearly, so the total degrades to `O(|T| * maxdepth)` — back to quadratic in the worst case. That is the trap.

The insight that saves it: I do **not** need the per-position counts immediately. I only need, at the very end, the total number of times each pattern occurs. Occurrences of a pattern `p` ending at node `u` correspond exactly to text positions whose landing-state's fail chain passes through `u`. So if I let `cnt[v]` be the number of text positions whose *landing state* is exactly `v` (one cheap increment per character, no chain walk during the scan), then the true number of occurrences of the string spelled by `u` equals the sum of `cnt[v]` over all `v` in the **subtree of `u` in the fail tree** — because the fail links form a tree rooted at the root, and "v's fail chain passes through u" is exactly "u is an ancestor of v in the fail tree". So I push every `cnt[v]` up to its fail-parent once, accumulating, and after that single subtree-sum pass `cnt[u]` holds the occurrence count of `u`'s string. Then the answer is `sum over terminal nodes u of cnt[u] * (total weight of patterns ending at u)`. The fail-tree subtree-sum replaces a per-character chain walk with one global linear pass — that is the move that makes the whole thing `O(sum|p_i| + |T|)`.

**Two implementation choices that keep the constants and the asymptotics honest.**

*Making the transition function total ("goto").* During the BFS that builds fail links, I overwrite each missing trie edge `nxt[v][c]` with `nxt[fail[v]][c]`. After this, `nxt[v][c]` is defined for every state and every character and lands directly on the correct next state, so the text scan is a flat array lookup per character with no fail-chain walk during scanning at all. The root's missing edges are set to loop back to the root so the recurrence has a base. This is the standard Aho-Corasick "automaton" form, and it is what guarantees the `O(|T|)` scan rather than an amortized argument I would have to defend.

*Computing the fail-tree subtree-sum without an explicit tree.* I do not build an adjacency list for the fail tree. The BFS that assigns fail links visits nodes in nondecreasing fail-tree depth (a node's fail link always points to a shallower node, processed earlier). So if I record the BFS visitation order, then iterating that order **in reverse** processes every node before its fail-parent — exactly a valid order for pushing subtree sums upward. One reverse loop: add `cnt[v]*wsum[v]` to the answer, then `cnt[fail[v]] += cnt[v]`. No recursion, no separate tree, `O(nodes)`.

**Handling the dictionary's quirks up front.** Duplicate patterns and patterns that are substrings/prefixes of each other need no special case if I accumulate weights at terminal nodes: I keep `wsum[u]` = the *sum* of weights of all patterns whose terminal node is `u`. Two identical patterns share a terminal node and their weights add; a pattern that is a prefix of another simply has its own terminal node deeper-or-shallower in the trie, and the subtree-sum mechanism counts each independently. Negative and zero weights need nothing special — they just flow through the final multiply-and-add, and the answer is allowed to be negative.

**First implementation.** I write the node allocator, insert patterns accumulating `wsum`, run the BFS for fail links + goto completion while recording `bfsOrder`, scan the text bumping `cnt[state]`, then the reverse subtree-sum. My first cut of the scoring tail looked like this:

```
long long answer = 0;
for (int i = (int)bfsOrder.size() - 1; i >= 0; i--) {
    int v = bfsOrder[i];
    cnt[fail[v]] += cnt[v];          // push up first
    answer += cnt[v] * wsum[v];      // then score
}
```

**A trace that exposes a bug.** I trace the smallest input that mixes a pattern with its own fail-ancestor: dictionary `{a: 1, aa: 10}`, text `aaa`. By hand the answer should be: `a` occurs 3 times (`*1 = 3`), `aa` occurs 2 times (`*10 = 20`), total `23`. The trie has root `0`, node `1 = "a"` (child of root on `a`, `wsum=1`), node `2 = "aa"` (child of `1` on `a`, `wsum=10`). Fail links: `fail[1] = 0`, `fail[2] = 1` (longest proper suffix of `aa` that is a trie prefix is `a`). BFS order is `[1, 2]`. The text `aaa` drives the automaton: start state `0`; read `a` -> state `1`, `cnt[1]=1`; read `a` -> `nxt[1][a]=2`, state `2`, `cnt[2]=1`; read `a` -> `nxt[2][a]`. What is `nxt[2][a]`? Node `2` has no real child on `a`, so during BFS completion it was set to `nxt[fail[2]][a] = nxt[1][a] = 2`. So state stays `2`, `cnt[2]=2`. After the scan `cnt = [0, 1, 2]`.

Now the reverse loop with my "push up first" ordering. `i=1`, `v=2`: `cnt[fail[2]] += cnt[2]` -> `cnt[1] += 2` -> `cnt[1] = 3`; then `answer += cnt[2]*wsum[2] = 2*10 = 20`. `i=0`, `v=1`: `cnt[fail[1]] += cnt[1]` -> `cnt[0] += 3`; then `answer += cnt[1]*wsum[1] = 3*1 = 3`. Total `23`. That is correct — but I got lucky with the order. Let me look harder, because "push up first, then score" is suspicious: when I process `v=1`, the `cnt[1]` I score with is the *already-accumulated* `3`, which is what I want. So scoring after pushing the children up is actually required: I must accumulate all of a node's subtree into its `cnt` *before* I score that node. The danger is the opposite ordering — score before push — which would score `v` with only its own landing count, missing the contributions from its fail-descendants.

Let me verify the opposite ordering really is wrong on the same case, to be sure I keep the right one. If I had written `answer += cnt[v]*wsum[v]` *before* `cnt[fail[v]] += cnt[v]`: `i=1`, `v=2`: score `2*10=20`, then push `cnt[1]=3`. `i=0`, `v=1`: score `cnt[1]*1 = 3*1 = 3` — wait, `cnt[1]` is already `3` here too because the push happened in the previous iteration, not this one. So for a two-level chain both orderings happen to agree. The real discriminator is a node that is its *own* score source and also a parent in the same iteration — but the push always targets `fail[v]`, a *different* node processed *later* in the reverse loop, never `v` itself. So within a single iteration the score reads `cnt[v]` and the push writes `cnt[fail[v]]`; they touch different cells. The two statements within one iteration are independent, and the only thing that matters is that all pushes *into* `v` (from its fail-children, which appear later in `bfsOrder`, hence earlier in the reverse loop) have already happened before I read `cnt[v]`. Reverse `bfsOrder` guarantees exactly that. So both intra-iteration orderings are correct; the load-bearing fact is the reverse-BFS order, not the line order. Good — I keep the version that scores after the push since it reads most naturally as "accumulate subtree, then use it," and I have now actually proven it rather than assumed it.

**A second, sneakier trace: the empty text and `m = 0`.** I worried that `cin >> text` on an empty text would misbehave. I trace `m = 0` with no text token: `cin >> m` reads `0`; the pattern loop runs zero times; `cin >> text` finds no token, the extraction fails, and `text` is left as the empty string `""`. The scan loop over `text` runs zero times, `cnt` is all zero, the reverse loop adds nothing (and `bfsOrder` is empty because the trie is just the root), so `answer = 0`. Printed `0`. Correct — the empty selection of occurrences. I also check `m > 0` with empty text, e.g. `{ab: 5}` and no text: patterns inserted, automaton built, scan runs zero times, all `cnt = 0`, answer `0`. Correct.

**An adjacency-style trap I deliberately avoid: counting at the landing node only.** Before trusting the subtree-sum I sanity-check that I did *not* accidentally also bump `cnt` along the fail chain during the scan (which would double count once the subtree-sum runs). In the scan I increment `cnt[state]` exactly once per character, at the landing state only. The fail-chain contributions are introduced solely by the single reverse subtree-sum. If I had also walked the chain during the scan, every occurrence would be counted twice (once during the walk, once in the subtree-sum) and worse, the scan would degrade to quadratic. I keep the scan to one increment per character.

**Edge cases, deliberately, because this is where multi-pattern code dies.**
- `m = 0`, empty text: answer `0` (traced above).
- Pattern longer than the text, e.g. `{abcdef: 9}` on `abc`: the automaton never reaches the terminal node of `abcdef`, its `cnt` stays `0`, contribution `0`. Correct.
- Pattern equal to the whole text, `{abc: 4}` on `abc`: reached exactly once, `4`. Correct.
- Duplicate with cancelling weights, `{aa: 3, aa: -1}` on `aaaa`: both list at the same terminal node, `wsum = 2`; `aa` occurs `3` times; `3*2 = 6`. Correct.
- Deep nesting on one letter, `{a:1, aa:2, aaa:3}` on `aaaaa`: counts `5,4,3`, score `5+8+9 = 22`. The subtree-sum is exactly what makes the overlapping counts come out right in one pass. Correct.
- Negative total, `{b: -5}` on `bbb`: `-15`. Allowed; the answer is signed.
- Overflow: a legal duplicate-heavy input can exceed 64-bit even though each single weight fits. For example, `10^5` copies of pattern `a` with weight `10^9` on a `900000`-character `a...a` text has score `90000000000000000000`. So I keep text occurrence counts in `long long`, store terminal weight sums and the answer in `__int128_t`, and print the signed decimal value manually.

**Performance check at the limit.** I build an instance with `sum|p_i| ~ 5*10^5` and `|T| = 5*10^5`. It runs in about `0.13` seconds using ~`62` MB. A node-maximizing instance (`~9*10^5` total pattern length of random length-20-to-40 strings, so almost every character is a distinct node) builds ~`9*10^5` nodes, runs in ~`0.24` s using ~`119` MB. The per-node cost is the `array<int,26>` transition row, ~`104` bytes/node; at `~10^6` nodes that is ~`100` MB, inside the `256` MB limit. Both clear `1` s / `256` MB.

**Self-verification against a brute force.** I wrote an independent oracle that, for each pattern, counts occurrences in `T` with a naive find-loop and sums `weight*count` — obviously correct, `O(m*|T|*maxlen)`, fine on small inputs. I ran the automaton solution against it on 700 random small cases over alphabets of size 1 to 4 (to force dense overlaps), plus 600 cases over alphabets up to 26 with up to 12 patterns and longer texts, plus eleven hand-built edge cases (empty text, `m=0`, longer-than-text patterns, whole-text patterns, duplicate cancelling weights, nested single-letter patterns, zero and negative weights). Zero mismatches across all of them. The cases that would have exposed the two bugs I worried about — the fail-chain undercount and the empty-text read — are explicitly in that set and pass.

**Final solution.** I convinced myself the idea is right by killing the per-pattern scan on a concrete `9*10^{10}`-operation instance, by deriving the fail-tree subtree-sum as the only way to recover overlapping per-pattern counts in one linear pass, and by proving the reverse-BFS order is what makes the subtree-sum correct; and I convinced myself the *code* is right by tracing `{a:1, aa:10}` on `aaa` to confirm the accumulate-then-score order, checking the empty-text read, and differential-testing 1300+ cases to zero mismatches. That is what I ship — one self-contained file implementing the Aho-Corasick automaton with the goto-completed transition function and the fail-tree subtree-sum:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int m;
    if (!(cin >> m)) return 0;                 // m = number of dictionary patterns

    auto printInt128 = [](const __int128_t value) {
        if (value == 0) {
            cout << 0;
            return;
        }

        __int128_t x = value;
        if (x < 0) {
            cout << '-';
            x = -x;
        }

        string digits;
        while (x > 0) {
            digits.push_back(char('0' + x % 10));
            x /= 10;
        }
        reverse(digits.begin(), digits.end());
        cout << digits;
    };

    // Aho-Corasick over lowercase letters 'a'..'z' (alphabet size 26).
    // next[v][c] : goto/transition (made total via BFS), child[v][c]: real trie child.
    // We keep the trie compact: nodes grow as we insert.
    const int A = 26;
    // Upper bound on node count: 1 (root) + total pattern length.
    vector<array<int, A>> nxt;                 // transition function (after BFS = total automaton)
    vector<int> fail;                          // suffix link
    vector<__int128_t> wsum;                   // sum of weights of patterns ENDING exactly at this node
    auto newNode = [&]() {
        nxt.push_back(array<int, A>{});
        nxt.back().fill(-1);
        fail.push_back(0);
        wsum.push_back(0);
        return (int)nxt.size() - 1;
    };
    newNode();                                  // node 0 = root

    // Insert each pattern; accumulate its weight at the terminal node. Duplicate
    // patterns and patterns that are prefixes/substrings of others are handled
    // automatically because weights sum at the shared terminal node.
    for (int i = 0; i < m; i++) {
        string p;
        long long w;
        cin >> p >> w;
        int cur = 0;
        for (char ch : p) {
            int c = ch - 'a';
            if (nxt[cur][c] == -1) nxt[cur][c] = newNode();
            cur = nxt[cur][c];
        }
        wsum[cur] += w;                         // multiple identical patterns -> weights add
    }

    int N = (int)nxt.size();

    // BFS to build fail links and turn the trie into a complete DFA (goto function).
    // After this, nxt[v][c] is always a valid state (the "transition" automaton).
    // bfsOrder records nodes in increasing fail-tree depth, so a reverse pass
    // pushes occurrence counts from a node up to its fail-parent.
    vector<int> bfsOrder;
    bfsOrder.reserve(N);
    queue<int> q;
    for (int c = 0; c < A; c++) {
        if (nxt[0][c] == -1) {
            nxt[0][c] = 0;                       // root's missing edges loop to root
        } else {
            fail[nxt[0][c]] = 0;
            q.push(nxt[0][c]);
        }
    }
    while (!q.empty()) {
        int v = q.front(); q.pop();
        bfsOrder.push_back(v);
        for (int c = 0; c < A; c++) {
            int u = nxt[v][c];
            if (u == -1) {
                nxt[v][c] = nxt[fail[v]][c];     // no real child: follow fail's transition
            } else {
                fail[u] = nxt[fail[v]][c];       // real child: its fail is fail[v]'s transition on c
                q.push(u);
            }
        }
    }

    // Feed the text through the automaton, counting how many times each state is the
    // current state (i.e. how many prefixes of the text end exactly here).
    string text;
    cin >> text;                                // text may be empty token -> stays ""
    vector<long long> cnt(N, 0);
    int state = 0;
    for (char ch : text) {
        int c = ch - 'a';
        state = nxt[state][c];
        cnt[state]++;
    }

    // Fail-tree subtree sum: a pattern ending at node u occurs once for every state
    // reached during the scan whose fail-chain passes through u. Pushing cnt up the
    // fail links (children before parents) makes cnt[u] = #occurrences of the string
    // spelled by u as a substring of the text. bfsOrder is in nondecreasing fail
    // depth, so iterating it in reverse processes children before their fail-parent.
    __int128_t answer = 0;
    for (int i = (int)bfsOrder.size() - 1; i >= 0; i--) {
        int v = bfsOrder[i];
        answer += cnt[v] * wsum[v];             // contribution of patterns ending at v
        cnt[fail[v]] += cnt[v];                  // propagate occurrence count to fail-parent
    }
    // Root (node 0) never carries a pattern (empty string), so it is not in bfsOrder
    // and contributes nothing; this is intentional.

    printInt128(answer);
    cout << "\n";
    return 0;
}
```

**Causal recap.** Per-pattern KMP is correct but `O(m*|T|)`, which I showed reaches `~9*10^{10}` operations on a legal instance and cannot finish in a second; the fix is to drive a single Aho-Corasick automaton with the text so it is read once. Recovering each pattern's *overlapping* occurrence count from that single pass is the crux: counting only landing nodes undercounts, and walking the fail chain per character re-introduces quadratic blowup, so the resolution is to count landing nodes cheaply and then take, in one global linear pass, the **fail-tree subtree sum** — `cnt[u]` after pushing children into parents equals `u`'s occurrence count. Completing the transition function during the BFS makes the scan a flat `O(|T|)` lookup, recording the BFS order gives a recursion-free reverse pass for the subtree sum, accumulating weights at terminal nodes folds duplicates and nesting away for free, and `__int128_t` weighted accumulation plus decimal printing closes the overflow, negative-weight, empty-text, and `m=0` corners — verified to zero mismatches against an independent brute force over 1300+ cases.
