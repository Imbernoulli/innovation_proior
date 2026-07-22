// TIER: invalid
// Deliberately infeasible: claim ONE probe suffices and print the all-zero
// vector.  Every instance plants at least one orbit fault requiring some
// position to be 1, so the all-zero probe leaves it (and almost everything
// else) undetected -- the checker must catch this and score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int K, N;
    scanf("%d %d", &K, &N);
    for (int i = 0; i < N; i++) {
        int c; scanf("%d", &c);
        for (int k = 0; k < c; k++) { int p, v; scanf("%d %d", &p, &v); }
    }
    printf("1\n");
    string probe(K, '0');
    probe.push_back('\n');
    fwrite(probe.data(), 1, probe.size(), stdout);
    return 0;
}
