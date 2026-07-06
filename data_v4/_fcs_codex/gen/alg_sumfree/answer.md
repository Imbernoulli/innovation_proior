```cpp
#include <bits/stdc++.h>
using namespace std;

struct Spec {
    int type = -1;
    int m = 0;
    int k = 0;
    int r = 0;
    long long value = 0;
    int count = 0;
};

static bool InMiddleInterval(int residue, int mod) {
    return 3LL * residue > mod && 3LL * residue <= 2LL * mod;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;

    vector<long long> a(n);
    long long maxValue = 0;
    for (int i = 0; i < n; ++i) {
        cin >> a[i];
        maxValue = max(maxValue, a[i]);
    }

    Spec best;

    auto improve = [&](int count, Spec spec) {
        if (count > best.count) {
            spec.count = count;
            best = spec;
        }
    };

    int upperCount = 0;
    for (long long x : a) {
        if (2LL * x > maxValue) ++upperCount;
    }
    Spec upper;
    upper.type = 0;
    improve(upperCount, upper);

    unordered_map<long long, int> frequency;
    frequency.reserve(static_cast<size_t>(n) * 2);
    for (long long x : a) {
        ++frequency[x];
    }

    for (const auto &entry : frequency) {
        Spec sameValue;
        sameValue.type = 1;
        sameValue.value = entry.first;
        improve(entry.second, sameValue);
    }

    const int MaxMod = 512;

    for (int mod = 2; mod <= MaxMod; ++mod) {
        vector<int> cnt(mod, 0);
        for (long long x : a) {
            ++cnt[static_cast<int>(x % mod)];
        }

        for (int residue = 1; residue < mod; ++residue) {
            Spec singleResidue;
            singleResidue.type = 2;
            singleResidue.m = mod;
            singleResidue.r = residue;
            improve(cnt[residue], singleResidue);
        }

        for (int multiplier = 1; multiplier < mod; ++multiplier) {
            int total = 0;

            for (int residue = 0; residue < mod; ++residue) {
                int mapped = static_cast<int>((1LL * multiplier * residue) % mod);
                if (InMiddleInterval(mapped, mod)) {
                    total += cnt[residue];
                }
            }

            Spec modularInterval;
            modularInterval.type = 3;
            modularInterval.m = mod;
            modularInterval.k = multiplier;
            improve(total, modularInterval);
        }
    }

    vector<long long> chosen;
    chosen.reserve(best.count);

    for (long long x : a) {
        bool take = false;

        if (best.type == 0) {
            take = (2LL * x > maxValue);
        } else if (best.type == 1) {
            take = (x == best.value);
        } else if (best.type == 2) {
            take = (x % best.m == best.r);
        } else if (best.type == 3) {
            int residue = static_cast<int>(x % best.m);
            int mapped = static_cast<int>((1LL * best.k * residue) % best.m);
            take = InMiddleInterval(mapped, best.m);
        }

        if (take) {
            chosen.push_back(x);
        }
    }

    cout << chosen.size() << '\n';
    for (size_t i = 0; i < chosen.size(); ++i) {
        if (i) cout << ' ';
        cout << chosen[i];
    }
    cout << '\n';

    return 0;
}
```