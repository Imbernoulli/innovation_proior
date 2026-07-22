// TIER: trivial
// Place no mirrors at all: the beam just draws the straight emitter column. This is
// exactly the checker's own baseline construction (ratio ~= 0.1 by design).
#include <cstdio>
int main() {
    int n, M, ec;
    char group[8];
    scanf("%d %d %7s", &n, &M, group);
    scanf("%d", &ec);
    printf("0\n");
    return 0;
}
