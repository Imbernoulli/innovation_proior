// TIER: greedy
// Obvious first idea: make every cable's rest length equal to the target-shape
// distance of its endpoints (so it "wants" to be at the target), minus a small
// fixed safety margin so it actually carries some tension. Ignores stiffness
// entirely, so on nodes whose two structural cables have very different k this
// creates an unbalanced force at the target -> the equilibrium drifts off target.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(){
    int n, s, c;
    scanf("%d %d %d", &n, &s, &c);
    vector<ll> target(n);
    for (int i = 0; i < n; i++) scanf("%lld", &target[i]);
    for (int i = 0; i < s; i++) { ll u,v,d; scanf("%lld %lld %lld", &u,&v,&d); }
    const double EPS = 1.2;
    for (int i = 0; i < c; i++) {
        ll u, v, k;
        scanf("%lld %lld %lld", &u, &v, &k);
        double gap = fabs((double)target[v] - (double)target[u]);
        double rest = max(0.0, gap - EPS);
        printf("%.6f\n", rest);
    }
    return 0;
}
