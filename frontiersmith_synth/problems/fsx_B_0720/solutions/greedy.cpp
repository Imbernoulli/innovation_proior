// TIER: greedy
// The obvious two-part recipe: (1) background (cold, low-temperature) donors
// are clearly clutter -- defer them to the end of their own window, out of
// the way. That part is easy to notice. (2) Consumers need a hot donor, so
// match the scarce hot-donor pool to consumers -- but in ENCOUNTER order
// (whichever consumer's window comes first), a very natural "first come,
// first served" rule. It never asks how much a match is actually WORTH
// (price * requirement): a donor happily gets burned on a cheap consumer
// while an expensive one goes unserved, even though donors are scarce.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(){
    int N; ll H, C;
    scanf("%d %lld %lld", &N, &H, &C);
    vector<int> type(N);
    vector<ll> vol(N), e(N), l(N), p1(N), p2(N), s(N);
    for (int i = 0; i < N; i++){
        scanf("%d %lld %lld %lld %lld %lld", &type[i], &vol[i], &e[i], &l[i], &p1[i], &p2[i]);
        s[i] = e[i];
    }

    const ll HOT_CUTOFF = 50, REQ_CUTOFF = 50;
    vector<int> hotDonors, coldDonors, consumers;
    for (int i = 0; i < N; i++){
        if (type[i] == 0) (p1[i] >= HOT_CUTOFF ? hotDonors : coldDonors).push_back(i);
        else consumers.push_back(i);
    }
    for (int d : coldDonors) s[d] = l[d];        // defer clutter (the "easy" half)

    vector<int> worthy;
    for (int c : consumers) if (p1[c] >= REQ_CUTOFF) worthy.push_back(c);
    // ENCOUNTER order: earliest window first -- NOT by value
    sort(worthy.begin(), worthy.end(), [&](int a, int b){ return e[a] < e[b]; });

    vector<char> used(N, 0);
    for (int c : worthy){
        int best = -1;
        for (int d : hotDonors){
            if (used[d]) continue;
            if (e[d] > e[c] - 1) continue;
            best = d;            // first compatible donor, no value reasoning
            break;
        }
        if (best != -1){
            used[best] = 1;
            ll sd = e[c] - 1;
            if (sd < e[best]) sd = e[best];
            s[best] = sd;
            s[c] = e[c];
        } else {
            s[c] = l[c];
        }
    }
    for (int d : hotDonors) if (!used[d]) s[d] = l[d];   // leftover donors deferred late
    for (int c : consumers) if (p1[c] < REQ_CUTOFF) s[c] = l[c];   // low-value: defer late

    for (int i = 0; i < N; i++) printf("%lld\n", s[i]);
    return 0;
}
