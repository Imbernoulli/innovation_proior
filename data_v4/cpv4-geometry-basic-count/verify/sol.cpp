#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<long long> xs(n), ys(n);
    // dedup identical points: a multiset of coincident points is one location
    set<pair<long long,long long>> seen;
    vector<pair<long long,long long>> pts;
    for (int i = 0; i < n; i++) {
        long long x, y;
        cin >> x >> y;
        if (seen.insert({x, y}).second) pts.push_back({x, y});
    }
    int m = (int)pts.size();

    // colCount[x] = number of DISTINCT points with that x; rowCount[y] similarly
    map<long long, long long> colCount, rowCount;
    for (auto &p : pts) {
        colCount[p.first]++;
        rowCount[p.second]++;
    }

    // For each point P, right triangles with the right angle AT P and legs
    // parallel to the axes = (#points sharing P's column, other than P)
    //                       * (#points sharing P's row, other than P).
    // Each such triangle has a unique right-angle vertex, so summing over P
    // counts every triangle exactly once -- no division, no double counting.
    long long answer = 0;
    for (auto &p : pts) {
        long long up = colCount[p.first] - 1;  // exclude P itself
        long long side = rowCount[p.second] - 1;
        answer += up * side;
    }

    cout << answer << "\n";
    return 0;
}
