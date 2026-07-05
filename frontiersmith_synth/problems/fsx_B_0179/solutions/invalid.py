# TIER: invalid
# Emits a garbage expression using a disallowed name -> the checker rejects it
# and scores 0.
def main():
    print("secret_law(x1) + os.system")


if __name__ == "__main__":
    main()
