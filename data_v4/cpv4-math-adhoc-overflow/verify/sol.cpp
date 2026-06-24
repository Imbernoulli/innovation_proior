#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    long long sum = 0, sumsq = 0;
    for (int i = 0; i < n; i++) {
        long long x;
        cin >> x;
        sum += x;
        sumsq += x * x;            // x up to 1e4 => x*x up to 1e8, fits; long long anyway
    }
    // sum over i<j of a[i]*a[j] = (sum^2 - sum of squares) / 2
    long long answer = (sum * sum - sumsq) / 2;
    cout << answer << "\n";
    return 0;
}
