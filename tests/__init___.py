"""
Test suite for the Reverse Attribution framework.

Contains unit tests for:
- Core RA algorithm functionality
- Model creation and utilities
- Evaluation metrics and procedures
"""

# Test discovery helper
def discover_tests():
    """
    Helper function to discover all available tests.
    Used by pytest and other test runners.
    """
    import os
    from pathlib import Path
    
    test_files = []
    test_dir = Path(__file__).parent
    
    for file in test_dir.glob("test_*.py"):
        test_files.append(file.stem)
    
    return test_files

# Test categories
CORE_TESTS = ["test_ra_core"]
MODEL_TESTS = ["test_models"] 
EVALUATION_TESTS = ["test_evaluation"]

ALL_TEST_CATEGORIES = {
    "core": CORE_TESTS,
    "models": MODEL_TESTS,
    "evaluation": EVALUATION_TESTS
}

__all__ = [
    "discover_tests",
    "CORE_TESTS",
    "MODEL_TESTS", 
    "EVALUATION_TESTS",
    "ALL_TEST_CATEGORIES"
]

# Package info for test runners
__test_package__ = True
