#include <bits/stdc++.h>
using namespace std;

/*
  Trivial baseline foil for ale-41: "first bin that currently fits, else drop".
  Same interactive protocol as sol.cpp. This is exactly the normalization
  baseline the scorer's --baseline mode computes; provided as a C++ program so
  the contrast is runnable in the same harness.
*/
int main() {
    ios::sync_with_stdio(false);
    int K; long long N;
    if (!(cin >> K >> N)) return 0;
    vector<long long> rem(K);
    for (int i = 0; i < K; i++) cin >> rem[i];
    for (long long it = 0; it < N; it++) {
        long long s, v;
        if (!(cin >> s >> v)) break;
        int choice = 0;
        for (int b = 0; b < K; b++) if (rem[b] >= s) { choice = b + 1; break; }
        if (choice >= 1) rem[choice - 1] -= s;
        cout << choice << "\n";
        cout.flush();
    }
    return 0;
}
