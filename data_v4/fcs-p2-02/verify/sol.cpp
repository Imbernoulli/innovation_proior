#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;             // n = 0 (or empty input) -> answer 0

    struct Job { long long s, e, w; };
    vector<Job> job(n);
    for (auto &j : job) cin >> j.s >> j.e >> j.w;

    // Sort by finishing time (end coordinate), ascending.
    sort(job.begin(), job.end(),
         [](const Job &x, const Job &y) { return x.e < y.e; });

    // ends[i] = finishing time of the i-th job in sorted order.
    vector<long long> ends(n);
    for (int i = 0; i < n; i++) ends[i] = job[i].e;

    // best[i] = max total weight achievable using only jobs[0..i].
    // Intervals are half-open [s, e): job j is compatible with job i (j before i)
    // iff ends[j] <= starts[i]. p(i) = largest index j < i with ends[j] <= job[i].s.
    vector<long long> best(n + 1, 0); // best[0] = 0 (no jobs)
    for (int i = 0; i < n; i++) {
        // Skip job i: best[i] (using jobs[0..i-1]).
        long long skip = best[i];
        // Take job i: its weight plus best over jobs ending at or before job[i].s.
        // Find p = number of jobs (in sorted prefix [0..i-1]) whose end <= job[i].s.
        int lo = 0, hi = i; // search in ends[0..i-1]
        long long key = job[i].s;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (ends[mid] <= key) lo = mid + 1;
            else hi = mid;
        }
        // lo = count of indices in [0..i-1] with ends[] <= key, i.e. p(i)+1 in 1-based best[].
        long long take = job[i].w + best[lo];
        best[i + 1] = max(skip, take);
    }

    cout << best[n] << "\n";
    return 0;
}
