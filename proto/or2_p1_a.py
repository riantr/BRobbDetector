"""P1-a (or2): per-route harness for the orthogonality factor `t_edge / t_blend`.

Thin entry point that builds ONLY this route's panel from the shared measured
96-cell p1_ortho factorial (no detection re-run) and estimates its effect, so
`proto/or2_p1_a.py` is the one-script-per-route harness required by the goal.

Usage: python proto/or2_p1_a.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from or2_p1_slices import main

if __name__ == '__main__':
    main('p1_a')
