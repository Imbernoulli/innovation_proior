```cpp
#include <bits/stdc++.h>
using namespace std;

static bool rowMono(const vector<vector<int>>& a, const vector<char>& activeCol, int r) {
    int color = -1;
    bool any = false;
    int C = (int)a[0].size();
    for (int c = 0; c < C; ++c) {
        if (!activeCol[c]) continue;
        any = true;
        if (color == -1) color = a[r][c];
        else if (color != a[r][c]) return false;
    }
    return any;
}

static bool colMono(const vector<vector<int>>& a, const vector<char>& activeRow, int c) {
    int color = -1;
    bool any = false;
    int R = (int)a.size();
    for (int r = 0; r < R; ++r) {
        if (!activeRow[r]) continue;
        any = true;
        if (color == -1) color = a[r][c];
        else if (color != a[r][c]) return false;
    }
    return any;
}

static bool canLeaveRows(const vector<vector<int>>& a, const vector<char>& protectRow) {
    int R = (int)a.size(), C = (int)a[0].size();
    vector<char> activeRow(R, 1), activeCol(C, 1);

    bool changed = true;
    while (changed) {
        changed = false;

        for (int r = 0; r < R; ++r) {
            if (activeRow[r] && !protectRow[r] && rowMono(a, activeCol, r)) {
                activeRow[r] = 0;
                changed = true;
            }
        }

        for (int c = 0; c < C; ++c) {
            if (activeCol[c] && colMono(a, activeRow, c)) {
                activeCol[c] = 0;
                changed = true;
            }
        }
    }

    for (int c = 0; c < C; ++c) {
        if (activeCol[c]) return false;
    }
    for (int r = 0; r < R; ++r) {
        if (activeRow[r] && !protectRow[r]) return false;
    }
    return true;
}

static vector<vector<int>> transposeGrid(const vector<vector<int>>& a) {
    int R = (int)a.size(), C = (int)a[0].size();
    vector<vector<int>> b(C, vector<int>(R));
    for (int r = 0; r < R; ++r) {
        for (int c = 0; c < C; ++c) {
            b[c][r] = a[r][c];
        }
    }
    return b;
}

static int bestLeftRows(const vector<vector<int>>& a) {
    int R = (int)a.size();
    map<vector<int>, vector<int>> classes;
    for (int r = 0; r < R; ++r) {
        classes[a[r]].push_back(r);
    }

    int best = 0;

    vector<char> none(R, 0);
    if (canLeaveRows(a, none)) best = 0;

    for (const auto& kv : classes) {
        vector<char> protect(R, 0);
        for (int r : kv.second) protect[r] = 1;
        if (canLeaveRows(a, protect)) {
            best = max(best, (int)kv.second.size());
        }
    }

    return best;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int R, C, K;
    cin >> R >> C >> K;

    vector<vector<int>> grid(R, vector<int>(C));
    for (int r = 0; r < R; ++r) {
        for (int c = 0; c < C; ++c) {
            cin >> grid[r][c];
        }
    }

    int keepRows = bestLeftRows(grid);
    int keepCols = bestLeftRows(transposeGrid(grid));
    int keep = max(keepRows, keepCols);

    if (keep == 0) {
        vector<char> none(R, 0);
        if (!canLeaveRows(grid, none)) {
            cout << -1 << '\n';
            return 0;
        }
    }

    cout << R + C - keep << '\n';
    return 0;
}
```