# TIER: invalid
def main():
    # '9' is never a valid letter (alphabet is always 0..sigma-1 with sigma <= 6
    # in every generated test) -- the checker must reject this outright.
    print("9" * 40)


if __name__ == "__main__":
    main()
