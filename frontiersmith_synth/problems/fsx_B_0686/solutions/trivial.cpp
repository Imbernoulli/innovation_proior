// TIER: trivial
#include <bits/stdc++.h>
using namespace std;

// Concatenate all patterns in input order, replacing every '?' with a fixed concrete
// letter. No merging at all -- this is exactly the checker's baseline B, so it scores
// ratio ~0.1.
int main() {
    int n;
    scanf("%d", &n);
    string S;
    S.reserve(6200);
    char buf[64];
    for (int i = 0; i < n; i++) {
        scanf("%s", buf);
        for (char *c = buf; *c; ++c) if (*c == '?') *c = 'A';
        S += buf;
    }
    printf("%d\n%s\n", (int)S.size(), S.c_str());
    return 0;
}
