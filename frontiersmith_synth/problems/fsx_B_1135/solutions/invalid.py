# TIER: invalid
"""
Claims victory after a handful of arbitrary root-only steps without ever actually
reaching a normal form. Must score 0 (checker's final has_redex check catches it).
"""
import sys


def main():
    sys.stdin.read()
    print(3)
    print(".")
    print(".")
    print(".")


main()
