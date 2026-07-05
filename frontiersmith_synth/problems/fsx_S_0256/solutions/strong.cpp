// TIER: strong
// Nearest-neighbour construction, then relocate/or-opt local search on the task order
// (asymmetric: task entry = pickup, exit = delivery), then a profitable-drop pass.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int N;
ll Hx, Hy;
vector<ll> px, py, dx, dy, w;

static inline ll dist(ll ax, ll ay, ll bx, ll by) {
    double a = (double)(ax - bx), b = (double)(ay - by);
    return (ll)llround(sqrt(a * a + b * b));
}

int main() {
    if (scanf("%d", &N) != 1) return 0;
    scanf("%lld %lld", &Hx, &Hy);
    px.resize(N + 1); py.resize(N + 1); dx.resize(N + 1); dy.resize(N + 1); w.resize(N + 1);
    for (int i = 1; i <= N; i++)
        scanf("%lld %lld %lld %lld %lld", &px[i], &py[i], &dx[i], &dy[i], &w[i]);

    // 1) nearest-neighbour over pickups
    vector<char> used(N + 1, 0);
    vector<int> ord; ord.reserve(N);
    {
        ll cx = Hx, cy = Hy;
        for (int step = 0; step < N; step++) {
            int best = -1; ll bd = LLONG_MAX;
            for (int i = 1; i <= N; i++) if (!used[i]) {
                ll dd = dist(cx, cy, px[i], py[i]);
                if (dd < bd) { bd = dd; best = i; }
            }
            used[best] = 1; ord.push_back(best);
            cx = dx[best]; cy = dy[best];
        }
    }

    // 2) relocate local search (each task moved to its best insertion gap)
    int maxpass = (int)(40000000.0 / ((double)N * (double)N + 1.0));
    if (maxpass < 3) maxpass = 3;
    if (maxpass > 30) maxpass = 30;
    bool improved = true;
    for (int pass = 0; pass < maxpass && improved; pass++) {
        improved = false;
        for (int p = 0; p < (int)ord.size(); p++) {
            int t = ord[p];
            ll lex, ley, rix, riy;
            if (p == 0) { lex = Hx; ley = Hy; } else { int a = ord[p - 1]; lex = dx[a]; ley = dy[a]; }
            if (p + 1 >= (int)ord.size()) { rix = Hx; riy = Hy; }
            else { int b = ord[p + 1]; rix = px[b]; riy = py[b]; }
            ll removed = dist(lex, ley, px[t], py[t]) + dist(dx[t], dy[t], rix, riy)
                       - dist(lex, ley, rix, riy);
            // remove t, find best insertion gap
            ord.erase(ord.begin() + p);
            ll bestAdd = LLONG_MAX; int bg = p; // default: reinsert where it was
            for (int g = 0; g <= (int)ord.size(); g++) {
                ll lx, ly, rx, ry;
                if (g == 0) { lx = Hx; ly = Hy; } else { int a = ord[g - 1]; lx = dx[a]; ly = dy[a]; }
                if (g == (int)ord.size()) { rx = Hx; ry = Hy; } else { int b = ord[g]; rx = px[b]; ry = py[b]; }
                ll add = dist(lx, ly, px[t], py[t]) + dist(dx[t], dy[t], rx, ry)
                       - dist(lx, ly, rx, ry);
                if (add < bestAdd) { bestAdd = add; bg = g; }
            }
            ord.insert(ord.begin() + bg, t);
            if (bestAdd + 1 < removed) improved = true;
        }
    }

    // 3) profitable-drop pass: skip a task if its detour exceeds its spoilage penalty
    for (int p = 0; p < (int)ord.size();) {
        int t = ord[p];
        ll lex, ley, rix, riy;
        if (p == 0) { lex = Hx; ley = Hy; } else { int a = ord[p - 1]; lex = dx[a]; ley = dy[a]; }
        if (p + 1 >= (int)ord.size()) { rix = Hx; riy = Hy; }
        else { int b = ord[p + 1]; rix = px[b]; riy = py[b]; }
        ll saved = dist(lex, ley, px[t], py[t]) + dist(dx[t], dy[t], rix, riy)
                 - dist(lex, ley, rix, riy);
        if (saved > w[t]) ord.erase(ord.begin() + p);
        else p++;
    }

    printf("%d\n", 2 * (int)ord.size());
    for (int t : ord) printf("P %d\nD %d\n", t, t);
    return 0;
}
