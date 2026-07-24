import sys
import os

# Add the src directory to the Python path so tests can import modules directly like 'import cmu_retrieve'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
