# TIER: trivial
# Do-nothing baseline: predict a CONSTANT equal to the last training-day fix,
# ignoring the current entirely. This is (up to training sensor noise) the
# same "freeze the finger at the last known fix" policy the checker itself
# uses as its internal baseline B -> reproduces Ratio ~ 0.1.
import sys


def main():
    data = sys.stdin.read().split()
    T_train = int(data[0])
    # rows start at index 5: i_1 obs_1 i_2 obs_2 ...
    last_obs = int(data[5 + 2 * (T_train - 1) + 1])
    print("EXPR %d" % last_obs)


if __name__ == "__main__":
    main()
