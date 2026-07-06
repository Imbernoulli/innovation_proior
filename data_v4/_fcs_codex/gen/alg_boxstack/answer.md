```cpp
#include <algorithm>
#include <iostream>
#include <tuple>
#include <vector>

using namespace std;

struct Orientation {
    long long a;
    long long b;
    long long h;
    int id;
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    cin >> n;

    vector<Orientation> boxes;
    boxes.reserve(3LL * n);

    for (int i = 0; i < n; ++i) {
        long long x[3];
        cin >> x[0] >> x[1] >> x[2];

        for (int heightDim = 0; heightDim < 3; ++heightDim) {
            int p = (heightDim + 1) % 3;
            int q = (heightDim + 2) % 3;

            long long a = x[p];
            long long b = x[q];
            if (a > b) {
                swap(a, b);
            }

            boxes.push_back({a, b, x[heightDim], i});
        }
    }

    sort(boxes.begin(), boxes.end(), [](const Orientation &lhs, const Orientation &rhs) {
        return tie(lhs.a, lhs.b, lhs.h, lhs.id) < tie(rhs.a, rhs.b, rhs.h, rhs.id);
    });

    const int m = static_cast<int>(boxes.size());
    vector<long long> dp(m, 0);

    long long answer = 0;

    for (int i = 0; i < m; ++i) {
        dp[i] = boxes[i].h;

        for (int j = 0; j < i; ++j) {
            if (boxes[j].id == boxes[i].id) {
                continue;
            }

            if (boxes[j].a < boxes[i].a && boxes[j].b < boxes[i].b) {
                dp[i] = max(dp[i], dp[j] + boxes[i].h);
            }
        }

        answer = max(answer, dp[i]);
    }

    cout << answer << '\n';
    return 0;
}
```