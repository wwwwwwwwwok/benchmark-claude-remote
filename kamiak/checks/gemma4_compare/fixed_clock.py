"""Runs a Python script with time.time() fixed at 1000.0, so two runs of a program that records wall-clock latencies
write the same bytes (regression tests; Gemma 4 work, 2026-11-12). Nothing else is changed.
usage: python -I fixed_clock.py <script> [arguments of the script ...]
"""
import runpy
import sys
import time

time.time = lambda: 1000.0
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name='__main__')
