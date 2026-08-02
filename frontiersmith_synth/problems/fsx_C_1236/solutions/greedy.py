# TIER: greedy
"""The obvious first circuit breaker: a single SYMMETRIC window/threshold
pair (the textbook "N-of-M failures trips, N-of-M successes recovers"
Hystrix/resilience4j default) with a FIXED sleep/probe interval -- no
adaptation to how long recovery actually took last time. This reacts fine to
a clean, unambiguous outage (mostly-failing window trips fast, mostly-
succeeding window recovers fast). But on a genuinely noisy ~50%-success
gray-failure stretch, a short recovery window is just as easy to satisfy by
chance as the trip window is -- so it closes back into traffic, immediately
re-trips a few calls later, and keeps flapping: wasted failed calls at every
re-trip, and it never uses the fact that different outages took wildly
different real time to actually recover (it always probes at the same
fixed cadence, wasting calls during long outages and dawdling on quick
ones)."""
import sys


def main():
    head = sys.stdin.readline().split()
    T = int(head[0])
    sys.stdin.readline()  # outcomes, unused
    w_trip = min(5, T)
    k_trip = min(3, w_trip)
    w_recover = w_trip
    k_recover = k_trip
    probe_base = min(5, T)
    print(f"{w_trip} {k_trip} {w_recover} {k_recover} {probe_base} 0 1")


if __name__ == "__main__":
    main()
