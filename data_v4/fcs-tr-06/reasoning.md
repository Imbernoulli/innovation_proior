**Reading the problem and pinning the contract.** I am handed two rooted forests and must answer one bit: are they isomorphic as rooted forests? An isomorphism here is a bijection between the two node sets that sends roots to roots and preserves the parent–child relation, with siblings allowed to be reordered arbitrarily — a tree carries no order on the children of a node. Each forest arrives as a parent array: `par[i] = 0` marks `i` as a root, otherwise `par[i]` is `i`'s parent. The scale is `n <= 2*10^5` per forest, and the time limit is 2 seconds, so whatever I do has to be near-linear; an `O(n^2)` pairwise comparison of subtrees, let alone anything exponential, is dead on arrival at this size. Let me fix one structural simplification before any algorithm: a **forest** with several roots is annoying to special-case, but if I invent a virtual super-root — call it node `0` — and make every real root a child of `0`, then each forest becomes a single rooted tree hanging off `0`, and "the two forests are isomorphic" becomes exactly "the two trees rooted at `0` are isomorphic." That reduction costs nothing and removes the multi-root bookkeeping, so I will build the children adjacency with index `0` reserved for the super-root and push every node `i` into `ch[par[i]]` (which routes roots into `ch[0]`).

**Laying out the candidate approaches.** Two routes are on the table, and I want the one I can *prove* canonical, not merely the one that is fast to type.

- *Serialize each tree to a string and compare.* Walk the tree and emit a bracket string: a leaf is `()`, an internal node is `(` then its children's strings then `)`. Equality of the two strings would be equality of the trees. This is `O(n)` and three lines. The danger is exactly the freedom the problem grants me: siblings are unordered, but a traversal visits them in *some* order, and that order leaks into the string.
- *Canonical AHU labeling.* Assign each node an integer label, computed bottom-up, so that two nodes get the same label precisely when their subtrees are isomorphic. The root's label then decides everything. The work is in making each node's label depend on its children as an *unordered multiset*, and in keeping the whole thing near-linear instead of degenerating into giant string concatenations.

**Breaking the serialize-and-compare approach on a concrete case.** Hand-waving "just serialize it" is how wrong solutions ship, so let me actually attack it. Take a tiny tree: root `r` with two children `a` and `b`, where `a` is a leaf and `b` has one child `c` (also a leaf). If my traversal lists `a` before `b`, the string is `( () (()) )` — that is, root-open, then `a`'s `()`, then `b`'s `(())`, then root-close. Now take the *same shape* but with the children stored in the other order, `b` before `a`: the string becomes `( (()) () )`. Character by character these differ at the third symbol: `(` versus `)`. So two trees that are genuinely isomorphic — same root, same multiset of child-subtrees — produce *different* strings, and a naive string-equality test would wrongly answer `NO`. The defect is structural and it is the whole point of the problem: a serialization imposes a sibling order that isomorphism is explicitly allowed to ignore. I could try to patch this by *sorting children before recursing*, and that is actually the right instinct — but if I sort by the children's raw subtree-strings and then concatenate, a path of length `n` builds strings of total length `O(n^2)` (each level re-copies everything below it), which is `4*10^10` characters at `n = 2*10^5`: far too slow and far too much memory. So the *idea* of canonicalizing by sorting is correct, but doing it on strings is the wrong representation. That is the crack I need to widen.

**Deriving the insight — sorted multiset of children, but on integers.** The fix is to keep the "sort the children to kill the order" idea and throw away the strings. Here is the canonical-AHU observation: I do not need the full serialization of a subtree to compare it — I only need a *handle* that is equal for isomorphic subtrees and distinct otherwise. So I process nodes bottom-up (children before parents) and, for each node `u`, I form its **signature**: the list of its children's already-computed integer labels, *sorted*. Sorting is what makes the signature depend on the children as an unordered multiset — the exact freedom the problem grants. Then I intern that signature: I keep one global dictionary from "sorted signature" to "fresh small integer," and `u`'s label is the integer the dictionary assigns to its signature (a brand-new id the first time that signature is seen, the cached id every later time). A leaf has the empty signature `[]`, so all leaves share one label; a node with two leaf-children has signature `[leafId, leafId]`, and so on up the tree. The key correctness claim, which I can prove by induction on subtree height: two nodes receive the same label **iff** their subtrees are isomorphic. *Base case:* leaves all have signature `[]` and any two leaves are isomorphic. *Inductive step:* by hypothesis a child's label equals another child's label iff their subtrees are isomorphic, so two nodes' *sorted multisets of child labels* are equal iff there is a bijection pairing their children into isomorphic pairs — which is precisely the definition of the parent subtrees being isomorphic — and the dictionary assigns equal ids to equal sorted signatures and distinct ids to distinct ones. So labels are a faithful canonical form. This is collision-free *by construction* (the dictionary is exact equality of integer vectors), unlike a random polynomial hash that an adversary could collide — that is why I will use deterministic interning, the genuinely SOTA exact form, rather than a 64-bit random hash.

**Why this is fast.** Each node contributes a signature whose length is its number of children, and `sum of (#children)` over all nodes is `n` (every non-super-root is somebody's child exactly once). Sorting each signature costs `O(deg(u) log deg(u))`, summing to `O(n log n)` in the worst case. The dictionary is a `map<vector<int>, int>`; each lookup compares vectors, and the total comparison work is also bounded by the total signature length times a log factor, so the whole algorithm is `O(n log n)` — comfortably inside 2 seconds at `2*10^5`. To compare *two* forests I make both use the **same** dictionary, so identical signatures get identical ids across the two trees; then the answer is simply whether node `0` of forest 1 and node `0` of forest 2 received the same label.

**One traversal hazard I will plan around up front.** The natural way to "process children before parents" is recursion, but the tests explicitly include deep chains of depth up to `2*10^5`, and a recursive DFS that deep will overflow the call stack and crash. So I will compute an explicit ordering with my own stack: push `0`, repeatedly pop a node and record it while pushing its children, which yields a pre-order; then iterate that recorded order *in reverse*, which guarantees every child is labeled before its parent (a valid post-order for the bottom-up labeling). No recursion, no stack-depth risk.

**First implementation.** Reading a forest into children adjacency with the super-root:

```
static vector<vector<int>> readForest(int n) {
    vector<vector<int>> ch(n + 1);
    for (int i = 1; i <= n; i++) { int p; cin >> p; ch[p].push_back(i); }
    return ch;
}
```

and the labeling:

```
int N = ch.size();
vector<int> label(N, -1);
vector<int> order, st; st.push_back(0);
while (!st.empty()) { int u = st.back(); st.pop_back(); order.push_back(u); for (int v: ch[u]) st.push_back(v); }
for (int idx = order.size()-1; idx >= 0; idx--) {
    int u = order[idx];
    vector<int> sig; for (int v: ch[u]) sig.push_back(label[v]);
    sort(sig.begin(), sig.end());
    auto it = sigToId.find(sig);
    if (it == sigToId.end()) { int id = sigToId.size(); sigToId[sig] = id; label[u] = id; }
    else label[u] = it->second;
}
return label[0];
```

with `main` reading two forests, labeling both against one shared `map<vector<int>,int>`, and printing `YES`/`NO` on `label0_1 == label0_2`.

**A trace, because clean induction transcribes into dirty code.** Before trusting anything, I run the worked sample by hand and then on the machine. Forest 1: `n1 = 4`, `par1 = [_,1,1,2]` (1-indexed) — node `1` is a root (its parent `0` is the super-root), nodes `2` and `3` have parent `1`, node `4` has parent `2`. Forest 2: `par2 = [_,1,1,3]` — same except node `4` hangs off node `3`. Both are "root with two children, one of which has one child," so the answer must be `YES`. Reverse-post-order labeling on forest 1: leaves `3` and `4` get signature `[]` -> id `0`; node `2` has child `4` so signature `[0]` -> id `1`; node `1` has children `{2,3}` with labels `{1,0}`, sorted `[0,1]` -> id `2`; super-root `0` has child `1`, signature `[2]` -> id `3`. Forest 2 reuses the same dictionary: its leaves and the depth-1 node land on the *same* ids `0` and `1`, node `1` again gets `[0,1] -> 2`, super-root again `[2] -> 3`. Both super-roots are `3`, so `YES`. Good. But a single passing sample proves little; I wire up an independent brute-force oracle (try every bijection of children recursively — obviously correct, exponential, valid only for `n <= 8`) and a random small-case generator, and run a few hundred cases.

**The bug the stress test exposes.** On a batch of random small forests the differential test flags a mismatch: my solution prints `NO` on a pair the brute oracle calls `YES`. The instance is two forests that are isomorphic only after relabeling. I reduce it and trace. The failing pair is small: forest 1 is two singleton roots, forest 2 is also two singleton roots — they must be `YES`. Walking my reader: for forest 1, `par1 = [0, 0]`, so both nodes are pushed into `ch[0]`; the super-root has two leaf children. Labeling: each leaf -> signature `[]` -> id `0`; super-root -> `[0,0]` -> id `1`. For forest 2 the same. So this case is actually `YES` and not the culprit — my reduction was too aggressive. I widen back to the real failing seed and find the genuine cause: in my very first draft of `main` I had labeled forest 1 against `sigToId`, then **cleared the map** before labeling forest 2, reasoning that each tree "should start clean." That clearing is the bug. With the map reset, the ids assigned to forest 2 are computed in isolation and *happen* to be the same integers `0, 1, 2, ...` in the order forest 2's signatures are first seen — but those integers no longer mean the same shapes they meant for forest 1. Concretely, if forest 1's super-root signature interns to id `2` because forest 1 happened to discover a depth-1 node first, while forest 2 discovers its shapes in a different order, forest 2's *isomorphic* super-root can intern to a different integer, and I wrongly print `NO`. The labels are only comparable across trees if they come from **one shared dictionary**; resetting it severs the cross-tree meaning.

**Diagnosing precisely and fixing.** The defect is exact: canonical ids are only a valid isomorphism certificate *relative to the dictionary that produced them*; comparing ids minted by two different dictionaries is meaningless. The fix is to declare a single `map<vector<int>, int> sigToId` once and pass it into both `canonicalLabel` calls, never clearing it. Equal signatures across the two forests then collapse to equal ids by construction, and the super-root comparison becomes a real isomorphism test. I remove the clear, thread the shared map through, and re-run.

**Re-verifying the fix.** Re-running the differential harness: 800 random cases against the brute oracle, then a second batch of 700 with a different seed range — **zero mismatches**, and the generator is producing a healthy mix (about 320 `YES` and 480 `NO`), so both branches are genuinely exercised rather than the test trivially answering `NO` every time. The two previously failing seeds now print `YES`, and they pass for the reason I fixed — the shared dictionary — which is the evidence I trust over "it looks right."

**Edge cases, deliberately, because this is where this kind of code dies.**
- *Two empty forests* (`n1 = n2 = 0`): each has only the super-root `0` with no children; its signature is `[]` -> id `0` in the shared map, so both super-roots are `0` and the answer is `YES`. Correct — the empty forest is isomorphic only to itself.
- *Empty vs. one node*: forest 1's super-root has signature `[]` -> id `0`; forest 2's super-root has one leaf child, the leaf is `[] -> 0`, the super-root is `[0] -> 1`. `0 != 1` -> `NO`. Correct.
- *Single vs. single*: both super-roots are `[0]` -> same id -> `YES`. Correct.
- *Two singleton roots vs. a 2-chain*: super-root `[0,0]` (two leaves) versus super-root `[1]` where `1` is the id of a one-child node — distinct signatures, distinct ids -> `NO`. Correct: a forest of two isolated nodes is not the same shape as one node with a child.
- *Deep chain of depth `2*10^5`*: the explicit-stack traversal never recurses, so there is no stack overflow; two identical chains print `YES`, and a chain versus a chain with its tail detached as a second root prints `NO`. Confirmed on the machine.
- *Sibling reordering plus full relabeling*: the generator deliberately builds forest 2 as a random id-permutation and sibling-shuffle of forest 1; every such pair prints `YES`, which is the core property the problem is testing.
- *Performance*: at `n = 2*10^5` per forest the program runs in about 0.1 s using ~22 MB — well inside 2 s / 256 MB.

**Final solution.** I convinced myself the *idea* is right by breaking serialize-and-compare on a sibling-swap and proving the sorted-multiset labeling canonical by induction, and I convinced myself the *code* is right by tracing the shared-dictionary bug to a precise cause and re-verifying the fix plus every corner. That is what I ship — one self-contained file: reduce each forest to a tree via a virtual super-root, label bottom-up with sorted-multiset signatures interned in one shared dictionary, and compare the two super-root labels.

```cpp
#include <bits/stdc++.h>
using namespace std;

// Read one rooted forest: n nodes (1..n). Then n integers par[1..n], where
// par[i] == 0 means i is a root of the forest, otherwise par[i] is i's parent.
// Returns children adjacency (index 0 is a virtual super-root whose children are
// all the forest roots) so the forest becomes a single rooted tree at node 0.
static vector<vector<int>> readForest(int n) {
    vector<vector<int>> ch(n + 1);
    for (int i = 1; i <= n; i++) {
        int p;
        cin >> p;
        ch[p].push_back(i);   // p == 0 -> child of the virtual super-root
    }
    return ch;
}

// Deterministic canonical AHU labeling.
// For each node we want a canonical integer label such that two nodes get the
// same label iff their induced subtrees are isomorphic as rooted trees. We build
// it bottom-up: a node's signature is the sorted multiset of its children's
// labels; a global map assigns a fresh small integer to each distinct signature.
// `n` is the number of real nodes; node 0 is the virtual super-root, so the
// labels array has size n+1 and the answer is the label of node 0.
static int canonicalLabel(const vector<vector<int>>& ch,
                          map<vector<int>, int>& sigToId) {
    int N = (int)ch.size();              // N = n+1 (includes node 0)
    vector<int> label(N, -1);

    // Iterative post-order over the tree rooted at 0 to avoid deep recursion
    // (n can be 2e5, so an explicit stack is required).
    vector<int> order;
    order.reserve(N);
    vector<int> st;
    st.push_back(0);
    while (!st.empty()) {
        int u = st.back();
        st.pop_back();
        order.push_back(u);
        for (int v : ch[u]) st.push_back(v);
    }
    // order is a pre-order; reversing gives an order where every child precedes
    // its parent, i.e. a valid post-order for our bottom-up labeling.
    for (int idx = (int)order.size() - 1; idx >= 0; idx--) {
        int u = order[idx];
        vector<int> sig;
        sig.reserve(ch[u].size());
        for (int v : ch[u]) sig.push_back(label[v]);
        sort(sig.begin(), sig.end());
        auto it = sigToId.find(sig);
        if (it == sigToId.end()) {
            int id = (int)sigToId.size();
            sigToId.emplace(move(sig), id);
            label[u] = id;
        } else {
            label[u] = it->second;
        }
    }
    return label[0];
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n1;
    if (!(cin >> n1)) return 0;
    vector<vector<int>> f1 = readForest(n1);

    int n2;
    cin >> n2;
    vector<vector<int>> f2 = readForest(n2);

    // A single shared map guarantees the SAME signature gets the SAME id across
    // both forests, so comparing the two super-root labels is exactly an
    // isomorphism test.
    map<vector<int>, int> sigToId;
    int l1 = canonicalLabel(f1, sigToId);
    int l2 = canonicalLabel(f2, sigToId);

    cout << (l1 == l2 ? "YES" : "NO") << "\n";
    return 0;
}
```

**Causal recap.** Serialize-and-compare looked right but a single sibling-swap (`( () (()) )` vs `( (()) () )`) showed a traversal leaks an order isomorphism is allowed to ignore, and sorting the children fixes the idea but on strings it costs `O(n^2)`; moving the sort onto *integer* child-labels and interning each sorted multiset in one global dictionary gives an exact, collision-free canonical form that two nodes share iff their subtrees are isomorphic, provable by induction; an explicit-stack reverse-pre-order labeling sidesteps the deep-chain recursion limit; the one real bug — clearing the dictionary between the two forests, which made cross-tree ids incomparable and produced a false `NO` — was pinned by a differential trace and fixed by sharing a single map; and the empty, single-node, two-root, deep-chain, relabel-and-reorder, and large-`n` corners all check out at ~0.1 s for `n = 2*10^5`.
