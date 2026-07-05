// TIER: greedy
// Nearest-neighbour construction over task pickups, serve all tasks.
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

    vector<char> used(N + 1, 0);
    vector<int> ord; ord.reserve(N);
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

    printf("%d\n", 2 * (int)ord.size());
    for (int t : ord) printf("P %d\nD %d\n", t, t);
    return 0;
}
