#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long C;
    if (!(cin >> n >> C)) return 0;          // empty input -> no climbers -> 0 trips
    vector<long long> w(n);
    for (auto &x : w) cin >> x;

    sort(w.begin(), w.end());                // greedy-exchange needs sorted weights

    // Two-pointer: each trip carries the heaviest remaining climber; if the
    // lightest remaining one also fits under capacity C, send the two together.
    long long trips = 0;
    int i = 0, j = n - 1;
    while (i <= j) {
        if (i != j && w[i] + w[j] <= C) {    // pair lightest with heaviest when they fit
            i++;
        }
        j--;                                  // heaviest always boards this trip
        trips++;
    }

    cout << trips << "\n";
    return 0;
}
