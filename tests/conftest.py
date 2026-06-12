import sys
import os
# Ensure project root is on sys.path so tests can import local modules
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root not in sys.path:
    sys.path.insert(0, root)
