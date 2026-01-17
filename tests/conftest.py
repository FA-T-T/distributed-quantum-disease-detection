"""
Pytest configuration and fixtures.
"""

import os
import sys

# Add code directory to path for all tests
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'code'))
