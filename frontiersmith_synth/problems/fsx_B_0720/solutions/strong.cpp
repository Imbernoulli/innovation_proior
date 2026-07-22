// TIER: strong
// Insight: hot donors are a SCARCE shared resource, so the schedule is
// really an assignment problem -- who gets the heat -- not just a timing
// problem. Rank consumers by what serving them is actually WORTH
// (price * requirement), not by which one's window happens to come first,
// and greedily hand the pool of hot donors to the highest-value consumers
// first (an exchange argument: if a cheaper consumer were served instead of
// a pricier one that a donor could also reach, swapping them can only
// help). Whichever donor is chosen is scheduled one tick before its
// consumer's draw (minimizing what else can land between them -- preserving
// exergy), and everything irrelevant (background cold donors, consumers
// nobody could serve, low-value consumers) is deferred to the very end of
// its own window, out of the way.
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
    for (int d : coldDonors) s[d] = l[d];        // defer clutter

    vector<int> worthy;
    for (int c : consumers) if (p1[c] >= REQ_CUTOFF) worthy.push_back(c);
    // rank by VALUE = price * requirement, most valuable first
    sort(worthy.begin(), worthy.end(), [&](int a, int b){ return p1[a] * p2[a] > p1[b] * p2[b]; });

    vector<char> used(N, 0);
    for (int c : worthy){
        int best = -1;
        for (int d : hotDonors){
            if (used[d]) continue;
            if (e[d] > e[c] - 1) continue;
            if (best == -1 || p1[d] > p1[best]) best = d;   // hottest compatible donor
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
