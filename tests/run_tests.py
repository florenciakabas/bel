#!/usr/bin/env python
"""
Run all unit tests for the BEL package.

This script discovers and runs all tests in the 'tests' directory.
"""

import unittest
import sys
import os

# Add parent directory to path to ensure tests can import BEL
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def run_tests():
    """Discover and run all tests."""
    loader = unittest.TestLoader()
    test_suite = loader.discover(os.path.dirname(__file__), pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)