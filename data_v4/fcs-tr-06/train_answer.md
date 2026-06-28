**Problem.** Given two rooted forests (each a parent array, `par[i] = 0` marking a
root), decide whether they are isomorphic as rooted forests: a bijection of nodes
mapping roots to roots and preserving the parent–child relation, with siblings
freely reorderable. Print `YES` or `NO`. `n` up to `2*10^5` per forest.

**Why the obvious serialize-and-compare is wrong.** Writing each tree as a nested
bracket string and comparing for equality bakes in a sibling order. The isomorphic
trees "root with children `a` (leaf) and `b` (one child)" serialize to `( () (()) )`
or `( (()) () )` depending on which child the traversal lists first — different
strings for the same shape, so equality wrongly answers `NO`. Sorting children
before recursing fixes the *idea*, but doing it on strings makes a depth-`n` path
build `O(n^2)` characters. Discarded.

**Key idea — deterministic canonical AHU labeling.** Attach a virtual super-root
(node `0`) whose children are the forest roots, turning each forest into one rooted
tree; forest isomorphism becomes tree isomorphism at `0`. Then label every node
bottom-up with an integer such that two nodes share a label iff their subtrees are
isomorphic:

- a node's **signature** is the list of its children's labels, **sorted** (sorting
  is what makes the signature an *unordered multiset*, matching the freedom to
  reorder siblings);
- intern signatures in one global dictionary `signature -> fresh small integer`; a
  node's label is the id its signature maps to.

By induction on height, equal labels ⇔ isomorphic subtrees (leaves all have the
empty signature; internally, equal sorted child-multisets ⇔ a bijection pairing
children into isomorphic pairs). The answer is whether the two super-roots get the
same label. Interning by exact vector-equality is **collision-free by
construction** — the genuinely exact form, unlike a random 64-bit subtree hash an
adversary could collide.

**Pitfalls to get right.**
1. *One shared dictionary.* Label both forests against the **same** `map`, never
   clearing it between them. Canonical ids are only comparable relative to the
   dictionary that minted them; resetting it makes two isomorphic forests intern to
   different integers and produces a false `NO`.
2. *No recursion.* Tests include depth-`2*10^5` chains, so a recursive post-order
   overflows the stack. Compute the order with an explicit stack (pre-order, then
   iterate it in reverse — every child precedes its parent).
3. *Sort, do not just compare counts.* Comparing only the number of children, or an
   unsorted child list, is not canonical; the sorted multiset is.

**Edge cases.** Two empty forests -> `YES` (both super-roots have empty signature).
Empty vs. one node -> `NO`. Single vs. single -> `YES`. Two singleton roots vs. a
2-chain -> `NO`. Deep chain vs. identical deep chain -> `YES`; vs. tail-detached
chain -> `NO`. Full relabel + sibling shuffle of the same shape -> `YES`.

**Complexity.** `O(n log n)` time (each node's signature sorted; total signature
length is `n`), `O(n)` space. At `n = 2*10^5` per forest it runs in ~0.1 s using
~22 MB, inside 2 s / 256 MB.

**Code.**

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
