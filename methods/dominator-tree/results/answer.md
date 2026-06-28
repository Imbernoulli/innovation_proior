# Dominator tree via the Lengauer-Tarjan algorithm

## Problem

In a directed flowgraph $G$ with start vertex $s$, a vertex $u$ **dominates** $v$ if every path from $s$ to $v$ passes through $u$. The **immediate dominator** $\mathrm{idom}(v)$ is the unique dominator of $v$ (other than $v$) that is dominated by all of $v$'s other dominators — the closest bottleneck above $v$. Drawing $\mathrm{idom}(v)\to v$ for every reachable $v\neq s$ yields the **dominator tree** rooted at $s$. Goal: compute $\mathrm{idom}(v)$ for all $v$ reachable from $s$, in near-linear time.

## Key idea

Deleting each vertex and re-searching ($O(nm)$), or fixpoint-intersecting dominator sets ($O(n^2)$), wastes effort on full sets when only one parent per vertex is wanted. Instead, DFS-number the graph and route everything through the **semidominator**.

**DFS numbering.** Search from $s$, giving each vertex a preorder number $\mathrm{dfn}$, a DFS tree $T$ with parent $\mathrm{fa}(v)$, and the order array. Write $u<v$ for $\mathrm{dfn}(u)<\mathrm{dfn}(v)$. Since the tree path $s\rightsquigarrow v$ is a real $s\to v$ path, $\mathrm{idom}(v)$ is a $T$-ancestor of $v$.

**Semidominator.** $\mathrm{sdom}(v)$ is the smallest-$\mathrm{dfn}$ vertex $u$ such that some path $u\to\dots\to v$ has all *interior* vertices numbered $>v$ — the highest a "detour through later-numbered vertices" can reach. Then $\mathrm{sdom}(v)<v$, $\mathrm{sdom}(v)$ is a proper $T$-ancestor of $v$, and $\mathrm{idom}(v)$ is an ancestor of $\mathrm{sdom}(v)$.

**Computing $\mathrm{sdom}$ (no path enumeration).** For $v\neq s$,
$$\mathrm{sdom}(v)=\min\Big(\{\,w:(w,v)\in E,\ w<v\,\}\ \cup\ \{\,\mathrm{sdom}(a):a>v,\ a\text{ a }T\text{-ancestor of a predecessor }w>v\text{ of }v\,\}\Big).$$
Sweep vertices in **decreasing $\mathrm{dfn}$ order**; maintain a forest of processed vertices linked to their tree parents. The query $\mathrm{eval}(w)$ returns the processed ancestor of $w$ with minimum $\mathrm{sdom}$, or $w$ itself if no link above $w$ is installed; an unprocessed smaller predecessor contributes itself because its $\mathrm{sdom}$ is still seeded to itself, while a processed root contributes its already computed $\mathrm{sdom}$. Both families collapse to one rule:
$$\mathrm{sdom}(v)=\min_{w\in\mathrm{pred}(v)}\mathrm{sdom}(\mathrm{eval}(w)).$$
$\mathrm{eval}$ is a **path-compressing** disjoint-set query: flattening the ancestor chain while carrying the minimum-$\mathrm{sdom}$ witness makes the total cost near-linear.

**Deriving $\mathrm{idom}$ from $\mathrm{sdom}$.** Let $u=\mathrm{eval}(v)$ be the vertex of minimum semidominator on the tree stretch between $\mathrm{sdom}(v)$ (exclusive) and $v$. Then
$$\mathrm{idom}(v)=\begin{cases}\mathrm{sdom}(v) & \text{if }\mathrm{sdom}(u)=\mathrm{sdom}(v),\\ \mathrm{idom}(u) & \text{otherwise.}\end{cases}$$
The second case is **deferred** (chasing the witness $u$): place $v$ in $\mathrm{bucket}(\mathrm{sdom}(v))$, drain $\mathrm{bucket}(\mathrm{fa}(v))$ right after linking $v$, recording the witness; then a single **increasing-$\mathrm{dfn}$ pass** sets $\mathrm{idom}(v)\leftarrow\mathrm{idom}(\mathrm{idom}(v))$ wherever a witness was stored.

## Algorithm

1. DFS from $s$: assign $\mathrm{dfn}$, $\mathrm{order}$, tree parent $\mathrm{fa}$. Vertices with $\mathrm{dfn}=0$ are unreachable and ignored.
2. Sweep $v$ in decreasing $\mathrm{dfn}$: $\mathrm{sdom}(v)=\min_{w\in\mathrm{pred}(v)}\mathrm{sdom}(\mathrm{eval}(w))$; push $v$ into $\mathrm{bucket}(\mathrm{sdom}(v))$; link $v$ under $\mathrm{fa}(v)$; drain $\mathrm{bucket}(\mathrm{fa}(v))$ applying the two-case rule (deferring case two).
3. Sweep $v$ in increasing $\mathrm{dfn}$: if $\mathrm{idom}(v)\neq\mathrm{sdom}(v)$ then $\mathrm{idom}(v)\leftarrow\mathrm{idom}(\mathrm{idom}(v))$. Set $\mathrm{idom}(s)=s$.

## Code

A single self-contained C++17 program: it reads `n m` then `m` directed edges `u v` (1-based) with start vertex `s=1` from stdin, and prints `n` integers where `ans[i]` is the number of vertices dominated by `i` (its subtree size in the dominator tree; `0` if `i` is unreachable).

```cpp
// Dominator tree (Lengauer-Tarjan). Reads "n m" then m directed edges "u v"
// (1-based); start vertex s=1. Prints n integers: ans[i] = number of vertices
// vertex i dominates (its subtree size in the dominator tree), 0 if unreachable.
#include <bits/stdc++.h>
using namespace std;

const int INF = INT_MAX;

int n, m, dfc = 0;
vector<vector<int>> succ_, pred_, bucket_;
vector<int> dfn, pos_, fa, fth, sdm, mn, idm, ans_;

void dfs(int s) {
    // iterative DFS from s: assign dfn (preorder), pos[dfn]=vertex, tree parent fth
    vector<pair<int,int>> st;            // (vertex, index into succ_)
    dfn[s] = ++dfc; pos_[dfc] = s;
    st.push_back({s, 0});
    while (!st.empty()) {
        int u = st.back().first;
        int &i = st.back().second;
        bool advanced = false;
        while (i < (int)succ_[u].size()) {
            int w = succ_[u][i++];
            if (dfn[w] == 0) {
                dfn[w] = ++dfc; pos_[dfc] = w; fth[w] = u;
                st.push_back({w, 0});
                advanced = true;
                break;
            }
        }
        if (!advanced) st.pop_back();
    }
}

// disjoint-set find carrying the minimum-sdom witness through path compression
int find(int x) {
    if (fa[x] == x) return x;
    int r = find(fa[x]);
    if (dfn[sdm[mn[fa[x]]]] < dfn[sdm[mn[x]]]) mn[x] = mn[fa[x]];
    fa[x] = r;
    return r;
}

void tarjan(int s) {
    dfs(s);
    for (int i = 1; i <= n; ++i) { fa[i] = sdm[i] = mn[i] = i; }
    // decreasing dfn: semidominators, then deferred immediate dominators
    for (int i = dfc; i >= 2; --i) {
        int u = pos_[i], res = INF;
        for (int v : pred_[u]) {
            if (dfn[v] == 0) continue;
            find(v);
            if (dfn[v] < dfn[u]) res = min(res, dfn[v]);
            else res = min(res, dfn[sdm[mn[v]]]);
        }
        sdm[u] = pos_[res];
        fa[u] = fth[u];
        bucket_[sdm[u]].push_back(u);
        int p = fth[u];
        for (int v : bucket_[p]) {
            find(v);
            idm[v] = (p == sdm[mn[v]]) ? p : mn[v];
        }
        bucket_[p].clear();
    }
    // increasing dfn: resolve deferred immediate dominators
    for (int i = 2; i <= dfc; ++i) {
        int u = pos_[i];
        if (idm[u] != sdm[u]) idm[u] = idm[idm[u]];
    }
    // subtree sizes in the dominator tree: ans[i] = # vertices i dominates
    for (int i = dfc; i >= 2; --i) {
        int u = pos_[i];
        ++ans_[u];
        ans_[idm[u]] += ans_[u];
    }
    ++ans_[s];
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    if (!(cin >> n >> m)) return 0;
    succ_.assign(n + 1, {});
    pred_.assign(n + 1, {});
    bucket_.assign(n + 1, {});
    dfn.assign(n + 1, 0);
    pos_.assign(n + 1, 0);
    fa.assign(n + 1, 0);
    fth.assign(n + 1, 0);
    sdm.assign(n + 1, 0);
    mn.assign(n + 1, 0);
    idm.assign(n + 1, 0);
    ans_.assign(n + 1, 0);
    for (int i = 0; i < m; ++i) {
        int u, v; cin >> u >> v;
        succ_[u].push_back(v);
        pred_[v].push_back(u);
    }
    tarjan(1);
    for (int i = 1; i <= n; ++i) cout << ans_[i] << " \n"[i == n];
    return 0;
}
```

## Complexity

- DFS, the two sweeps, and bucketing are linear. The semidominator pass does one $\mathrm{eval}$ per edge and the idom pass one per bucketed vertex, so $O(m)$ $\mathrm{eval}$ calls. The plain link/eval version above has the classic $O(m\log n)$ near-linear bound; the balanced-link refinement improves this to $O(m\,\alpha(m,n))$.
- Memory is $O(n+m)$ — the adjacency lists plus a constant number of length-$n$ integer arrays.
- The same machinery yields the dominator tree's structure directly ($\mathrm{parent}[v]=\mathrm{idom}(v)$); subtree sizes, ancestor queries, and "how many vertices does $v$ dominate" follow from one traversal of that tree.
