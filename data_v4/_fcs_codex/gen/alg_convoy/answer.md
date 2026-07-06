```cpp
#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

struct Job {
    long long a;
    long long b;
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    cin >> n;

    vector<Job> early, late;
    early.reserve(n);
    late.reserve(n);

    for (int i = 0; i < n; ++i) {
        Job job;
        cin >> job.a >> job.b;
        if (job.a <= job.b) {
            early.push_back(job);
        } else {
            late.push_back(job);
        }
    }

    sort(early.begin(), early.end(), [](const Job &x, const Job &y) {
        return x.a < y.a;
    });

    sort(late.begin(), late.end(), [](const Job &x, const Job &y) {
        return x.b > y.b;
    });

    long long timeA = 0;
    long long timeB = 0;

    for (const Job &job : early) {
        timeA += job.a;
        timeB = max(timeB, timeA) + job.b;
    }

    for (const Job &job : late) {
        timeA += job.a;
        timeB = max(timeB, timeA) + job.b;
    }

    cout << timeB << '\n';
    return 0;
}
```