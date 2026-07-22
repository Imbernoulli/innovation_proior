#pragma once
#include <bits/stdc++.h>
using namespace std;

struct Grid {
    int R, C, M, Wr, CovMin, CovMax, Sc;
    vector<string> g;
};

inline Grid readGrid(istream &in) {
    Grid gr;
    in >> gr.R >> gr.C;
    in >> gr.M >> gr.Wr >> gr.CovMin >> gr.CovMax >> gr.Sc;
    gr.g.resize(gr.R);
    for (int i = 0; i < gr.R; i++) in >> gr.g[i];
    return gr;
}

static const int CH_DR[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
static const int CH_DC[8] = {-1, 0, 1, -1, 1, -1, 0, 1};

// BFS (8-directional) distance from every sand cell to the nearest rock cell.
inline vector<int> distToRock(const Grid &gr) {
    int R = gr.R, C = gr.C;
    vector<int> dist(R * C, -1);
    queue<int> q;
    for (int r = 0; r < R; r++)
        for (int c = 0; c < C; c++)
            if (gr.g[r][c] == '#') {
                for (int k = 0; k < 8; k++) {
                    int nr = r + CH_DR[k], nc = c + CH_DC[k];
                    if (nr >= 0 && nr < R && nc >= 0 && nc < C && gr.g[nr][nc] == '.' &&
                        dist[nr * C + nc] == -1) {
                        dist[nr * C + nc] = 1;
                        q.push(nr * C + nc);
                    }
                }
            }
    while (!q.empty()) {
        int cur = q.front(); q.pop();
        int r = cur / C, c = cur % C;
        for (int k = 0; k < 8; k++) {
            int nr = r + CH_DR[k], nc = c + CH_DC[k];
            if (nr >= 0 && nr < R && nc >= 0 && nc < C && gr.g[nr][nc] == '.' &&
                dist[nr * C + nc] == -1) {
                dist[nr * C + nc] = dist[cur] + 1;
                q.push(nr * C + nc);
            }
        }
    }
    int CAP = R + C + 5;
    for (auto &x : dist) if (x == -1) x = CAP;
    return dist;
}

// Turn a boolean cell mask into disjoint row-contiguous rake lines (min length 2;
// isolated single cells are dropped back to "uncovered" rather than emitted).
inline vector<vector<pair<int,int>>> maskToLines(const Grid &gr, vector<char> &mask) {
    int R = gr.R, C = gr.C;
    vector<vector<pair<int,int>>> lines;
    for (int r = 0; r < R; r++) {
        int c = 0;
        while (c < C) {
            if (gr.g[r][c] == '.' && mask[r * C + c]) {
                int c0 = c;
                while (c < C && gr.g[r][c] == '.' && mask[r * C + c]) c++;
                int len = c - c0;
                if (len >= 2) {
                    vector<pair<int,int>> ln;
                    for (int cc = c0; cc < c; cc++) ln.push_back({r + 1, cc + 1});
                    lines.push_back(ln);
                } else {
                    mask[r * C + c0] = 0;
                }
            } else c++;
        }
    }
    return lines;
}

inline void printLines(const vector<vector<pair<int,int>>> &lines) {
    cout << lines.size() << "\n";
    for (auto &ln : lines) {
        cout << ln.size() << "\n";
        for (auto &p : ln) cout << p.first << " " << p.second << "\n";
    }
}

// Evenly-spaced row selection covering `desiredRows` rows out of R (a fixed,
// deterministic "straight parallel raking" pattern independent of rock geometry).
inline vector<char> evenRowSelection(int R, int desiredRows) {
    vector<char> sel(R, 0);
    if (desiredRows <= 0) return sel;
    if (desiredRows > R) desiredRows = R;
    for (int i = 0; i < desiredRows; i++) {
        int r = (int)((i + 0.5) * R / desiredRows);
        if (r >= R) r = R - 1;
        sel[r] = 1;
    }
    return sel;
}
