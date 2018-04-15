#!/usr/bin/env python
"""
Static tests analysis script
"""


import glob
import pylint.lint


def main():
    """
    Main runner
    """

    python_files = glob.glob("./**/*.py", recursive=True)
    pylint.lint.Run(python_files)


if __name__ == "__main__":
    main()
