// TIER: trivial
// All-on every round: always feasible, equals the checker baseline B -> ratio ~0.1.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int S, G, T;
    if (scanf("%d %d %d", &S, &G, &T) != 3) return 0;
    // build the "1 2 ... S" suffix once
    string suffix;
    suffix.reserve(S * 4);
    for (int s = 1; s <= S; s++) { suffix += ' '; suffix += to_string(s); }
    for (int r = 0; r < T; r++) {
        printf("%d%s\n", S, suffix.c_str());
    }
    return 0;
}
