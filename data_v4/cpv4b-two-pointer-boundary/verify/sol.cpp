#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long D;
    if (!(cin >> n >> D)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // Count contiguous subarrays whose (max - min) < D, via a sliding window.
    // maxd holds indices with decreasing values (front = window max),
    // mind holds indices with increasing values (front = window min).
    deque<int> maxd, mind;
    long long answer = 0;
    int left = 0;
    for (int right = 0; right < n; right++) {
        while (!maxd.empty() && a[maxd.back()] <= a[right]) maxd.pop_back();
        maxd.push_back(right);
        while (!mind.empty() && a[mind.back()] >= a[right]) mind.pop_back();
        mind.push_back(right);

        // Shrink from the left until the window [left, right] satisfies max - min < D.
        // Guard left <= right so the deques are never empty when we read their fronts.
        while (left <= right && a[maxd.front()] - a[mind.front()] >= D) {
            left++;
            if (maxd.front() < left) maxd.pop_front();
            if (mind.front() < left) mind.pop_front();
        }
        // All windows ending at 'right' with start in [left, right] are stable.
        // If left == right+1 the window is empty and this contributes 0.
        answer += (long long)(right - left + 1);
    }

    cout << answer << "\n";
    return 0;
}
