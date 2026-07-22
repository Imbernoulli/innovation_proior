// TIER: greedy
// The "obvious first attempt": pair up the N fixed emission ports with the N
// target cells in the SAME (input) order they were read -- no thought given
// to which port is actually close to which target -- then route each pair
// via a plain shortest path and set phases to follow it. This ignores the
// ODOMETER entirely: it never checks whether two different ports' paths pass
// through (and rotor-dilute) the same intermediate cell, so on anything but
// the smallest cases many particles cross paths, waste steps, or get
// deflected onto the wrong cell once a shared cell's rotor has already
// advanced past the intended direction.
#include <bits/stdc++.h>
using namespace std;

int L, N, sx, sy;
int DX[4] = {1, 0, -1, 0};
int DY[4] = {0, 1, 0, -1};
inline int wrapc(int v) { v %= L; if (v < 0) v += L; return v; }
inline int idxOf(int x, int y) { return y * L + x; }

int main() {
    scanf("%d %d", &L, &N);
    scanf("%d %d", &sx, &sy);
    vector<char> target(L * (size_t)L, 0);
    vector<int> targetList(N);
    for (int i = 0; i < N; i++) {
        int x, y; scanf("%d %d", &x, &y);
        target[idxOf(x, y)] = 1;
        targetList[i] = idxOf(x, y);
    }

    // fixed emission ports: BFS from source, skip source & target cells
    vector<char> seen(L * (size_t)L, 0);
    vector<int> disc;
    queue<int> q;
    int srcIdx = idxOf(sx, sy);
    seen[srcIdx] = 1; q.push(srcIdx);
    while (!q.empty() && (int)disc.size() < N) {
        int cur = q.front(); q.pop();
        int cx = cur % L, cy = cur / L;
        if (cur != srcIdx && !target[cur]) disc.push_back(cur);
        for (int d = 0; d < 4; d++) {
            int nx = wrapc(cx + DX[d]), ny = wrapc(cy + DY[d]);
            int ni = idxOf(nx, ny);
            if (!seen[ni]) { seen[ni] = 1; q.push(ni); }
        }
    }
    if (disc.empty()) disc.push_back(srcIdx);
    vector<int> portArr(N);
    for (int i = 0; i < N; i++) portArr[i] = disc[i % disc.size()];

    // naive pairing: port i (in BFS-discovery order) with the REVERSE of the
    // target list's input order -- no distance reasoning at all.
    vector<int> phase(L * (size_t)L, 0);
    for (int i = 0; i < N; i++) {
        int s = portArr[i], t = targetList[N - 1 - i];
        // plain BFS shortest path (no contention-avoidance)
        vector<int> pdist(L * (size_t)L, -1);
        vector<int> parent(L * (size_t)L, -1);
        queue<int> qq;
        pdist[s] = 0; qq.push(s);
        while (!qq.empty()) {
            int cur = qq.front(); qq.pop();
            if (cur == t) break;
            int cx = cur % L, cy = cur / L;
            for (int d = 0; d < 4; d++) {
                int nx = wrapc(cx + DX[d]), ny = wrapc(cy + DY[d]);
                int ni = idxOf(nx, ny);
                if (pdist[ni] == -1) { pdist[ni] = pdist[cur] + 1; parent[ni] = cur; qq.push(ni); }
            }
        }
        if (parent[t] != -1 || s == t) {
            vector<int> path;
            int cur = t;
            while (cur != s) { path.push_back(cur); cur = parent[cur]; }
            path.push_back(s);
            reverse(path.begin(), path.end());
            for (size_t j = 0; j + 1 < path.size(); j++) {
                int a = path[j], b = path[j + 1];
                int ax = a % L, ay = a / L, bx = b % L, by = b / L;
                for (int d = 0; d < 4; d++)
                    if (wrapc(ax + DX[d]) == bx && wrapc(ay + DY[d]) == by) { phase[a] = d; break; }
            }
        }
    }

    for (int y = 0; y < L; y++) {
        for (int x = 0; x < L; x++) printf("%d%c", phase[idxOf(x, y)], x + 1 < L ? ' ' : '\n');
    }
    return 0;
}
