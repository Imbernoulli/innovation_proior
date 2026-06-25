#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;                 // n = 0 (or empty) -> answer 0
    vector<pair<long long,long long>> job(n);   // (deadline, processing time)
    for (int i = 0; i < n; i++) {
        long long t, d;
        cin >> t >> d;
        job[i] = {d, t};                        // sort key = deadline
    }
    sort(job.begin(), job.end());               // earliest due date first

    // Moore-Hodgson: scan in EDD order, keep a running clock and a max-heap of
    // the processing times currently scheduled. When the clock passes the
    // current deadline, evict the longest scheduled job (the exchange step).
    priority_queue<long long> heap;             // max-heap of processing times
    long long clock = 0;
    for (int i = 0; i < n; i++) {
        long long d = job[i].first, t = job[i].second;
        heap.push(t);
        clock += t;
        if (clock > d) {                        // infeasible: drop the longest job
            clock -= heap.top();
            heap.pop();
        }
    }

    cout << (long long)heap.size() << "\n";     // jobs that survive = max on-time count
    return 0;
}
