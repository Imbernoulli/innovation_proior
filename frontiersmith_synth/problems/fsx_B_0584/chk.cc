#include "testlib.h"
#include <vector>
#include <algorithm>
using namespace std;

int N, M, C, E;
vector<vector<int>> pg;   // 1-indexed task -> sorted page ids

// Exact LRU simulator: capacity C over pages 0..M-1, cache empty at start,
// each task accesses its pages in increasing id order. Returns total misses.
long long simulate(const vector<int>& order) {
    vector<char> inC(M, 0);
    vector<int> prv(M, -1), nx(M, -1);
    int head = -1, tail = -1, sz = 0;
    long long miss = 0;
    auto unlink = [&](int p) {
        int a = prv[p], b = nx[p];
        if (a != -1) nx[a] = b; else head = b;
        if (b != -1) prv[b] = a; else tail = a;
        prv[p] = nx[p] = -1;
    };
    auto pushFront = [&](int p) {
        prv[p] = -1; nx[p] = head;
        if (head != -1) prv[head] = p; else tail = p;
        head = p;
    };
    for (int t : order)
        for (int p : pg[t]) {
            if (inC[p]) { unlink(p); pushFront(p); }
            else {
                miss++;
                if (sz == C) { int ev = tail; unlink(ev); inC[ev] = 0; sz--; }
                inC[p] = 1; pushFront(p); sz++;
            }
        }
    return miss;
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);
    N = inf.readInt();
    M = inf.readInt();
    C = inf.readInt();
    E = inf.readInt();
    pg.assign(N + 1, {});
    for (int i = 1; i <= N; i++) {
        int k = inf.readInt();
        for (int j = 0; j < k; j++) pg[i].push_back(inf.readInt());
        sort(pg[i].begin(), pg[i].end());
    }
    vector<pair<int,int>> edges(E);
    for (int i = 0; i < E; i++) {
        int u = inf.readInt(), v = inf.readInt();
        edges[i] = { u, v };
    }

    // participant output: a permutation of 1..N
    vector<int> order;
    order.reserve(N);
    vector<char> seen(N + 1, 0);
    vector<int> pos(N + 1, 0);
    for (int i = 0; i < N; i++) {
        int x = ouf.readInt(1, N, "task");
        if (seen[x]) quitf(_wa, "task %d repeated -> not a permutation", x);
        seen[x] = 1;
        pos[x] = i;
        order.push_back(x);
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output after the permutation");

    // precedence feasibility
    for (auto& e : edges)
        if (pos[e.first] >= pos[e.second])
            quitf(_wa, "precedence %d before %d violated", e.first, e.second);

    long long F = simulate(order);                 // participant misses
    vector<int> ident(N);
    for (int i = 0; i < N; i++) ident[i] = i + 1;
    long long B = simulate(ident);                 // reference (input order) misses
    if (B < 1) B = 1;

    double sc = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
