// TIER: trivial
#include "common.h"

int main() {
    Grid gr = readGrid(cin);
    int R = gr.R, C = gr.C;
    long long totalSand = 0;
    for (int r = 0; r < R; r++)
        for (int c = 0; c < C; c++)
            if (gr.g[r][c] == '.') totalSand++;
    long long target = (long long)llround(totalSand * (gr.CovMin + gr.CovMax) / 2000.0);

    // Least-effort recipe: fill sand cells in plain reading order (top rows first)
    // until the coverage floor is met. No awareness of rocks or spacing at all.
    vector<char> mask(R * C, 0);
    long long picked = 0;
    for (int r = 0; r < R && picked < target; r++)
        for (int c = 0; c < C && picked < target; c++)
            if (gr.g[r][c] == '.') { mask[r * C + c] = 1; picked++; }

    auto lines = maskToLines(gr, mask);
    printLines(lines);
    return 0;
}
