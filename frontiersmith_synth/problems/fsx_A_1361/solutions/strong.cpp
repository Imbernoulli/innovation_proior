// TIER: strong
// Insight: normalize bid by job size. An honest bidder's implied rate floor(bid/s_a) is
// provably constant across auctions; a rotation ring member's is not, because it inherits
// the CURRENT winner's rate on every round it covers. Flag every bidder whose implied-rate
// set has more than one distinct value, and group all such bidders of a market together
// (a market plants at most one ring).
#include <bits/stdc++.h>
using namespace std;

int main() {
    int T;
    scanf("%d", &T);
    for (int t = 0; t < T; t++) {
        int n, m;
        scanf("%d %d", &n, &m);
        vector<long long> s(m);
        for (int a = 0; a < m; a++) scanf("%lld", &s[a]);
        vector<vector<long long>> bid(n, vector<long long>(m));
        for (int i = 0; i < n; i++)
            for (int a = 0; a < m; a++) scanf("%lld", &bid[i][a]);

        vector<int> flagged;
        for (int i = 0; i < n; i++) {
            set<long long> rates;
            for (int a = 0; a < m; a++) rates.insert(bid[i][a] / s[a]);
            if ((int)rates.size() > 1) flagged.push_back(i);
        }

        if (flagged.empty()) {
            printf("0\n");
        } else {
            printf("1\n%d", (int)flagged.size());
            for (int idx : flagged) printf(" %d", idx);
            printf("\n");
        }
    }
    return 0;
}
