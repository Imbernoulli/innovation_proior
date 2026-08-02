# TIER: greedy
"""The obvious first pass: trust the forecast and serve every tier whenever
energy is available (reserve floor = 0 for both 'important' and 'low', every
day). This maximizes SHORT-RUN service and matches or beats every other policy
on calm days. But it never reserves anything against the forecast-uncertainty
band, so on a day where a low-sun sequence actually arrives, the battery has
already been run down serving important/low on the preceding good days -- and
critical demand itself goes unserved once the pool (battery + that day's tiny
actual PV) is smaller than critical demand. This is the trap the problem is
built to expose."""
import sys, json


def main():
    inst = json.load(sys.stdin)
    T = int(inst["T"])
    ans = {
        "reserve_important_kwh": [0.0] * T,
        "reserve_low_kwh": [0.0] * T,
    }
    print(json.dumps(ans))


if __name__ == "__main__":
    main()
