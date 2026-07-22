// TIER: greedy
#include "common.h"

int main() {
    Grid gr = readGrid(cin);
    int R = gr.R, C = gr.C;
    long long totalSand = 0;
    for (int r = 0; r < R; r++)
        for (int c = 0; c < C; c++)
            if (gr.g[r][c] == '.') totalSand++;
    long long target = (long long)llround(totalSand * (gr.CovMin + gr.CovMax) / 2000.0);

    // The obvious "textbook" raking recipe: rule a fixed number of evenly spaced
    // FULL rows, straight across the whole garden, completely ignoring where the
    // rocks are. Good global spacing in open sand, blind to obstacle geometry.
    double midFrac = (gr.CovMin + gr.CovMax) / 2000.0;
    int desiredRows = max(1, (int)llround(R * midFrac));
    vector<char> rowSel = evenRowSelection(R, desiredRows);

    vector<char> mask(R * C, 0);
    long long picked = 0;
    for (int r = 0; r < R && picked < target; r++) {
        if (!rowSel[r]) continue;
        for (int c = 0; c < C && picked < target; c++)
            if (gr.g[r][c] == '.') { mask[r * C + c] = 1; picked++; }
    }
    if (picked < target) {
        for (int r = 0; r < R && picked < target; r++)
            for (int c = 0; c < C && picked < target; c++)
                if (gr.g[r][c] == '.' && !mask[r * C + c]) { mask[r * C + c] = 1; picked++; }
    }

    auto lines = maskToLines(gr, mask);
    printLines(lines);
    return 0;
}
