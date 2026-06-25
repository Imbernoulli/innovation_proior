#include <bits/stdc++.h>
using namespace std;

int n;
long long k, L;
vector<long long> p; // sorted, in [0, L)

// Greedy count starting from anchor post `start` (which is FORCED chosen):
// walk forward around the ring, take the next post whose gap from the last
// taken is >= d. Returns the number of posts taken (>= 1). The closing wrap
// gap (from the last taken back to `start`) must ALSO be >= d for the whole
// thing to form a valid cyclic placement; we enforce that by never taking a
// post that would leave a wrap gap < d, and by checking it at the end.
long long countFrom(int start, long long d) {
    long long taken = 1;
    long long lastPos = p[start];          // absolute position of last taken
    // iterate the other n-1 posts in cyclic order after `start`
    for (int step = 1; step < n; step++) {
        int idx = start + step;
        long long pos = p[idx % n] + (idx >= n ? L : 0); // unrolled position
        long long gap = pos - lastPos;
        if (gap < d) continue;             // too close, skip this post
        // would taking it leave a valid wrap back to start? wrap = (p[start]+L) - pos
        long long wrap = (p[start] + L) - pos;
        if (wrap < d) break;               // taking it (or anything further) kills the wrap
        taken++;
        lastPos = pos;
    }
    return taken;
}

// Can we choose >= k posts with every cyclic-adjacent gap >= d?
bool feasible(long long d) {
    if (d <= 0) return true;               // any selection has nonneg gaps
    // Some post must be chosen; try every post as the forced anchor.
    for (int s = 0; s < n; s++) {
        if (countFrom(s, d) >= k) return true;
    }
    return false;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    if (!(cin >> n >> k >> L)) return 0;
    p.resize(n);
    for (auto &x : p) cin >> x;
    sort(p.begin(), p.end());

    // Clearance is in [0, L/2] (k>=2 means at least two points, so the min
    // cyclic gap can be at most floor(L/2)). Binary search the largest d
    // for which a valid placement of >= k posts exists.
    long long lo = 0, hi = L / 2, ans = 0;
    while (lo <= hi) {
        long long mid = lo + (hi - lo) / 2;
        if (feasible(mid)) { ans = mid; lo = mid + 1; }
        else hi = mid - 1;
    }
    cout << ans << "\n";
    return 0;
}
