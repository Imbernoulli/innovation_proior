#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    vector<int> f(n);
    for (int i = 0; i < n; i++) cin >> f[i];

    int q;
    cin >> q;
    vector<int> qs(q);          // query start node
    vector<long long> qt(q);    // query step count t
    for (int i = 0; i < q; i++) cin >> qs[i] >> qt[i];

    // ---- Step 1: locate every cycle in the functional graph. ----
    // state: 0 = unvisited, 1 = on the current walking path, 2 = finished.
    vector<int> state(n, 0);
    vector<char> onCycle(n, 0);
    vector<int> cycId(n, -1);   // which cycle a cycle-node belongs to
    vector<int> cycPos(n, 0);   // position of a cycle-node along its cycle
    vector<int> cycLen;         // length of each cycle, indexed by cycle id
    vector<vector<int>> cycNodes; // ordered nodes of each cycle

    vector<int> stkPath;        // current walking path for cycle detection
    for (int s = 0; s < n; s++) {
        if (state[s] != 0) continue;
        int v = s;
        // walk forward until we hit a node already seen
        while (state[v] == 0) {
            state[v] = 1;
            stkPath.push_back(v);
            v = f[v];
        }
        if (state[v] == 1) {
            // found a new cycle: it starts at v and runs to the end of stkPath
            int cid = (int)cycLen.size();
            vector<int> nodes;
            // pop the path back down to v, recording the cycle nodes (reversed)
            while (!stkPath.empty() && stkPath.back() != v) {
                int u = stkPath.back(); stkPath.pop_back();
                state[u] = 2;          // these were tail nodes on this walk
            }
            // now stkPath.back() == v ; collect the cycle by walking f from v
            int u = v;
            do {
                nodes.push_back(u);
                onCycle[u] = 1;
                cycId[u] = cid;
                state[u] = 2;
                u = f[u];
            } while (u != v);
            for (int i = 0; i < (int)nodes.size(); i++) cycPos[nodes[i]] = i;
            cycLen.push_back((int)nodes.size());
            cycNodes.push_back(move(nodes));
            // pop v as well (it is finished)
            stkPath.pop_back();
        }
        // mark the remaining path nodes as finished (they are tails leading
        // into an already-known structure)
        while (!stkPath.empty()) {
            int u = stkPath.back(); stkPath.pop_back();
            state[u] = 2;
        }
    }

    // ---- Step 2: build reverse edges among tail nodes (non-cycle). ----
    // We only need reverse edges that lead from a node to its tail predecessors;
    // an edge u -> f[u] is a tail edge contributing to depth structure when u is
    // NOT on a cycle. We root the tail forest at cycle nodes.
    vector<int> revHead(n, -1), revNext(n, -1);
    for (int u = 0; u < n; u++) {
        if (!onCycle[u]) {
            int p = f[u];        // u's forward target = u's parent in tail forest
            revNext[u] = revHead[p];
            revHead[p] = u;
        }
    }

    // ---- Step 3: bucket queries by their start node for offline answering. ----
    vector<int> qHead(n, -1), qNext(q, -1);
    for (int i = 0; i < q; i++) {
        int s = qs[i];
        qNext[i] = qHead[s];
        qHead[s] = i;
    }

    vector<long long> ans(q, -1);

    // Helper: answer a query whose start is at tail-depth D with cycle entry
    // node `entry`, using the live ancestor stack `stk` (stk[D] == start).
    // If t <= D the answer is a tail node stk[D - t]; else step into the cycle.
    auto answerWith = [&](int qi, int D, int entry, const vector<int>& stk) {
        long long t = qt[qi];
        if (t <= (long long)D) {
            ans[qi] = stk[(int)(D - t)];
        } else {
            long long into = t - (long long)D;          // steps once on the cycle
            int cid = cycId[entry];
            int L = cycLen[cid];
            long long pos = ((long long)cycPos[entry] + into) % L;
            ans[qi] = cycNodes[cid][(int)pos];
        }
    };

    // ---- Step 4: iterative DFS over each tail tree rooted at a cycle node. ----
    // The DFS stack `stk` holds the current root-to-node path; index 0 is the
    // cycle entry (depth 0 on the tail = the cycle node itself), index D is the
    // node currently being expanded at tail-depth D.
    vector<int> stk;             // current path of nodes (by id)
    stk.reserve(n + 1);
    // iterative frame: node + iterator over its reverse-children
    vector<int> itChild(n);      // current child pointer (revNext walk) per node

    for (int root = 0; root < n; root++) {
        if (!onCycle[root]) continue;          // start DFS only from cycle nodes
        int entry = root;                      // the cycle entry for this tree

        // First, answer queries that start exactly on this cycle node (depth 0).
        // stk just contains the root here.
        stk.clear();
        stk.push_back(root);
        for (int qi = qHead[root]; qi != -1; qi = qNext[qi]) {
            answerWith(qi, 0, entry, stk);
        }
        // descend into tail children of root
        itChild[root] = revHead[root];
        while ((int)stk.size() > 0) {
            int v = stk.back();
            int c = itChild[v];
            // advance to a child that is a tail node (all rev edges here are tails)
            if (c != -1) {
                itChild[v] = revNext[c];        // consume this child
                // push child c
                stk.push_back(c);
                int D = (int)stk.size() - 1;     // tail-depth of c
                itChild[c] = revHead[c];
                // answer queries that start at c
                for (int qi = qHead[c]; qi != -1; qi = qNext[qi]) {
                    answerWith(qi, D, entry, stk);
                }
            } else {
                stk.pop_back();                  // done with v
                if (v == root) break;            // finished this tree
            }
        }
    }

    // ---- Output ----
    string out;
    out.reserve((size_t)q * 7);
    for (int i = 0; i < q; i++) {
        out += to_string(ans[i]);
        out += '\n';
    }
    cout << out;
    return 0;
}
