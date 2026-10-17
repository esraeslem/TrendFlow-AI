import os
import sys

# Make the repo root importable (src/, data_generator.py, api.py) and run from it
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
