// TIER: greedy
// Textbook farthest-point (max-min) seed placement: spend the whole seed budget K,
// picking each next seed as the inside cell farthest (plain Euclidean) from all seeds
// chosen so far. Ignores zone resistance and crystallographic orientation entirely
// (always orientation 0) -- the "obvious" way to spread K sites over a region.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int H, W, K;
    long long LAMBDA;
    scanf("%d %d %d", &H, &W, &K);
    scanf("%lld", &LAMBDA);
    vector<string> grid(H);
    for (int r = 0; r < H; r++) {
        char buf[100005];
        scanf("%s", buf);
        grid[r] = buf;
    }
    vector<pair<int,int>> cells;
    for (int r = 0; r < H; r++)
        for (int c = 0; c < W; c++)
            if (grid[r][c] != '#') cells.push_back({r, c});

    int m = min((int)cells.size(), K);
    vector<pair<int,int>> seeds;
    // start from first inside cell (row-major)
    seeds.push_back(cells[0]);
    vector<long long> best(cells.size(), LLONG_MAX);
    for (size_t i = 0; i < cells.size(); i++) {
        long long dr = cells[i].first - seeds[0].first;
        long long dc = cells[i].second - seeds[0].second;
        best[i] = dr * dr + dc * dc;
    }
    while ((int)seeds.size() < m) {
        int bestIdx = -1;
        long long bestVal = -1;
        for (size_t i = 0; i < cells.size(); i++)
            if (best[i] > bestVal) { bestVal = best[i]; bestIdx = (int)i; }
        seeds.push_back(cells[bestIdx]);
        for (size_t i = 0; i < cells.size(); i++) {
            long long dr = cells[i].first - cells[bestIdx].first;
            long long dc = cells[i].second - cells[bestIdx].second;
            long long d2 = dr * dr + dc * dc;
            if (d2 < best[i]) best[i] = d2;
        }
    }
    printf("%d\n", m);
    for (auto& s : seeds) printf("%d %d 0\n", s.first, s.second);
    return 0;
}
