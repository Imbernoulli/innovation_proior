// TIER: greedy
// A natural first attempt at "backward reasoning" that stops short of solving the
// real ring-wide simultaneous system: peel the update rule one layer at a time,
// using a single distinguished tap of the mask to define each output cell while
// substituting the LAYER ABOVE (the pattern being inverted) for every OTHER tap
// instead of the true unknown. This looks like genuine backward-inversion and is
// exact for a single-tap (pure shift) rule, but for multi-tap masks it silently
// mixes information across time layers, so it drifts from the true precursor and
// never reaches a perfect reconstruction.
#include <bits/stdc++.h>
using namespace std;
int N, T, B, M;
int main(){
    scanf("%d %d %d %d", &N, &T, &B, &M);
    char buf[300]; scanf("%299s", buf);
    string target(buf);

    vector<int> offs;
    for (int d = 2; d >= -2; d--) if (M & (1 << (d + 2))) offs.push_back(d);
    int dstar = offs[0]; // distinguished tap: largest offset present in the mask

    vector<char> cur(N);
    for (int i = 0; i < N; i++) cur[i] = target[i] - '0';
    for (int step = 0; step < T; step++){
        vector<char> guess(N, 0);
        for (int i = 0; i < N; i++){
            int v = cur[i];
            for (int d : offs) if (d != dstar) v ^= cur[((i + d) % N + N) % N];
            int pos = ((i + dstar) % N + N) % N;
            guess[pos] = (char)v;
        }
        cur = guess;
    }

    vector<int> chosen;
    for (int i = 0; i < N && (int)chosen.size() < B; i++) if (cur[i]) chosen.push_back(i);
    printf("%d\n", (int)chosen.size());
    for (size_t i = 0; i < chosen.size(); i++) printf("%d%c", chosen[i], i + 1 == chosen.size() ? '\n' : ' ');
    if (chosen.empty()) printf("\n");
    return 0;
}
