"""
Test suite for Reverse Attribution.

Contains fixtures and test discovery helpers.
"""

def discover_tests():
    """
    Discover all test modules in this package.
    """
    import os
    from pathlib import Path
    test_dir = Path(__file__).parent
    return [f.stem for f in test_dir.glob("test_*.py")]

__all__ = ["discover_tests"]
