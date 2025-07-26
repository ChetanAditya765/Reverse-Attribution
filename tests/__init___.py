"""
Test package for Reverse Attribution framework.
"""

# Test configuration
TEST_DATA_DIR = "test_data"
TEMP_OUTPUT_DIR = "temp_test_output"

# Test utilities
def cleanup_test_files():
    """Clean up temporary test files."""
    import shutil
    from pathlib import Path
    
    temp_dirs = [TEMP_OUTPUT_DIR, "test_figs", "temp_test_results"]
    
    for temp_dir in temp_dirs:
        if Path(temp_dir).exists():
            shutil.rmtree(temp_dir)

def create_test_environment():
    """Create test environment."""
    from pathlib import Path
    
    # Create necessary directories
    Path(TEST_DATA_DIR).mkdir(exist_ok=True)
    Path(TEMP_OUTPUT_DIR).mkdir(exist_ok=True)
