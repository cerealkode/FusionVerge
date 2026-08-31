import sys
import os


# Pytest configuration
# * add project root to sys.path so all test file can import from other files directly
#   without needed to install package
sys.path.insert(0, os.path.dirname(__file__))