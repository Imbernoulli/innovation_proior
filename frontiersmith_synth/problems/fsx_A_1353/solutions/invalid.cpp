// TIER: invalid
// Deliberately infeasible: claims an absurd number of firings for the last
// pile while every other pile fires zero times, so it receives no cascading
// chips from anywhere. With bounded starting chips (<=300000) that pile can
// legally fire at most a few hundred thousand times -- nowhere near the
// claimed count -- so this is never realizable (and also blows the K
// budget). Must score 0.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    ll n, m, K;
    cin >> n >> m >> K;
    for (ll i = 0; i < n; i++) {
        ll v = (i + 1 == n) ? 999999999LL : 0LL;
        cout << v << " \n"[i + 1 == n];
    }
    return 0;
}
