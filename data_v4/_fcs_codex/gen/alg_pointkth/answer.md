```cpp
#include <bits/stdc++.h>
using namespace std;

struct Fenwick {
    int n;
    vector<int> bit;

    explicit Fenwick(int n = 0) : n(n), bit(n + 1, 0) {}

    void Add(int index, int value) {
        for (; index <= n; index += index & -index) {
            bit[index] += value;
        }
    }

    int SumPrefix(int index) const {
        int result = 0;
        for (; index > 0; index -= index & -index) {
            result += bit[index];
        }
        return result;
    }

    int SumRange(int left, int right) const {
        if (left > right) {
            return 0;
        }
        return SumPrefix(right) - SumPrefix(left - 1);
    }
};

struct Operation {
    int type;
    long long x;
    int k;
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int q;
    cin >> q;

    vector<Operation> operations;
    operations.reserve(q);

    vector<long long> coords;
    coords.reserve(q);

    for (int i = 0; i < q; ++i) {
        int type;
        long long x;
        cin >> type >> x;

        int k = 0;
        if (type == 3) {
            cin >> k;
        } else {
            coords.push_back(x);
        }

        operations.push_back({ type, x, k });
    }

    sort(coords.begin(), coords.end());
    coords.erase(unique(coords.begin(), coords.end()), coords.end());

    Fenwick fenwick(static_cast<int>(coords.size()));

    for (const Operation &op : operations) {
        if (op.type == 1 || op.type == 2) {
            int index = static_cast<int>(lower_bound(coords.begin(), coords.end(), op.x) - coords.begin()) + 1;
            fenwick.Add(index, op.type == 1 ? 1 : -1);
            continue;
        }

        long long x = op.x;
        int k = op.k;

        long long low = 0;
        long long high = max(llabs(coords.front() - x), llabs(coords.back() - x));

        while (low < high) {
            long long mid = low + (high - low) / 2;

            long long leftValue = x - mid;
            long long rightValue = x + mid;

            int leftIndex = static_cast<int>(lower_bound(coords.begin(), coords.end(), leftValue) - coords.begin()) + 1;
            int rightIndex = static_cast<int>(upper_bound(coords.begin(), coords.end(), rightValue) - coords.begin());

            int countInside = fenwick.SumRange(leftIndex, rightIndex);

            if (countInside >= k) {
                high = mid;
            } else {
                low = mid + 1;
            }
        }

        cout << low << '\n';
    }

    return 0;
}
```