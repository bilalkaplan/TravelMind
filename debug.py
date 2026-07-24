import sys, traceback
sys.path.append('src')
from cmu_retrieve import search
try:
    search('New York')
except Exception as e:
    traceback.print_exc()
