# TIER: strong
"""The insight: ASYMMETRIC hysteresis, not a bigger/smaller version of the
same symmetric window. Tripping stays on a SHORT window (react fast to a
real outage -- no reason to be slow to protect the caller). Recovering is
deliberately gated on a much LONGER, independently-sized window with a
strict success threshold: under a bursty gray-failure regime whose good/bad
sub-bursts each run only 6-9 ticks, "13 of the last 14 succeeded" cannot be
sustained by any single lucky sub-burst, so the
breaker correctly stays OPEN through the whole gray stretch instead of
flapping in and out of traffic and re-paying the trip cost over and over.
Under a genuine full recovery (>=97% success), the long window still fills
with successes soon enough -- no real recovery is meaningfully delayed.

The second half of the insight ("probe rate tied to observed recovery
time"): the probe interval for a FRESH outage episode is re-derived from the
*signal gap* of the LAST episode that fully closed -- how many ticks it took
from tripping to the first probe that came back alive (interval ~= alpha *
signal_gap) -- instead of one fixed cadence for every outage regardless of
how long outages in this dependency tend to take to show a sign of life."""
import sys


def main():
    head = sys.stdin.readline().split()
    T = int(head[0])
    sys.stdin.readline()  # outcomes, unused
    w_trip = min(3, T)
    k_trip = min(2, w_trip)
    w_recover = min(14, T)
    k_recover = min(13, w_recover)
    probe_base = min(8, T)
    # alpha = 1/2: a fresh episode's probe interval is derived from how long
    # it took the LAST episode to show its first sign of life -- fast
    # recoveries -> probe sooner next time, slow recoveries -> don't hammer
    # a still-dead dependency with wasted probes.
    print(f"{w_trip} {k_trip} {w_recover} {k_recover} {probe_base} 1 2")


if __name__ == "__main__":
    main()
