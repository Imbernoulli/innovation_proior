// TIER: trivial
// The isotropic split: declare an equal growth budget in every direction
// (w[i] = 1 for all 72 bins). This is EXACTLY the checker's internal baseline
// B, so it always reproduces ratio = 0.100 -- an undirected budget has no
// notion of "toward the facet" or "toward the tip" at all, and produces the
// KPZ-universal round blob no matter how the single shared value is tuned.
#include <cstdio>
int main() {
    long long seed, M; int K;
    if (scanf("%lld %d %lld", &seed, &K, &M) != 3) return 0;
    for (int i = 0; i < K; i++) { long long ri; scanf("%lld", &ri); }
    for (int i = 0; i < 72; i++) printf("%s1", i ? " " : "");
    printf("\n");
    return 0;
}
