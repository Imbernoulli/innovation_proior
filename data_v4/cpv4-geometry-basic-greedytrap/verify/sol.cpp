#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long L;
    if (!(cin >> n >> L)) return 0;          // empty input -> nothing to do
    vector<long long> x(n);
    for (auto &v : x) cin >> v;

    if (n == 0) {                            // no houses -> no lamps needed
        cout << 0 << "\n";
        return 0;
    }

    sort(x.begin(), x.end());

    long long lamps = 0;
    int i = 0;
    while (i < n) {
        // The leftmost still-dark house is x[i]. Anchor a lamp's LEFT edge at x[i]
        // so it covers [x[i], x[i] + L]; this reaches as far right as any lamp can
        // while still covering x[i]. Cover every house inside that interval.
        long long right = x[i] + L;
        lamps++;
        while (i < n && x[i] <= right) i++;
    }

    cout << lamps << "\n";
    return 0;
}
