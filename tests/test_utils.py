"""
Tests for utility functions including data preprocessing, file I/O, and configuration management.
"""

import pytest
import json
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

class TestFileIOOperations:
    """Test file input/output operations."""
    
    def test_json_loading(self):
        """Test JSON file loading with various scenarios."""
        # Test valid JSON
        valid_json = {"test": "data", "number": 42}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(valid_json, f)
            temp_file = f.name
        
        try:
            with open(temp_file, 'r') as f:
                loaded_data = json.load(f)
            
            assert loaded_data == valid_json
            assert loaded_data["test"] == "data"
            assert loaded_data["number"] == 42
        finally:
            Path(temp_file).unlink()
    
    def test_json_loading_with_encoding(self):
        """Test JSON loading with UTF-8 encoding."""
        unicode_json = {"message": "Hello 世界", "emoji": "🎉"}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(unicode_json, f, ensure_ascii=False)
            temp_file = f.name
        
        try:
            with open(temp_file, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            
            assert loaded_data == unicode_json
            assert "世界" in loaded_data["message"]
            assert loaded_data["emoji"] == "🎉"
        finally:
            Path(temp_file).unlink()
    
    def test_file_discovery(self):
        """Test file discovery functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            test_files = [
                'evaluation_results.json',
                'jmlr_metrics.json',
                'training_summary.json',
                'other_file.txt'
            ]
            
            for filename in test_files:
                (temp_path / filename).touch()
            
            # Discovery logic
            json_files = list(temp_path.glob('*.json'))
            json_filenames = [f.name for f in json_files]
            
            assert 'evaluation_results.json' in json_filenames
            assert 'jmlr_metrics.json' in json_filenames
            assert 'training_summary.json' in json_filenames
            assert 'other_file.txt' not in json_filenames

class TestDataPreprocessing:
    """Test data preprocessing utilities."""
    
    def test_normalize_data(self):
        """Test data normalization functions."""
        # Test array normalization
        data = np.array([1, 2, 3, 4, 5])
        
        # Min-max normalization
        normalized = (data - data.min()) / (data.max() - data.min())
        
        assert normalized.min() == 0.0
        assert normalized.max() == 1.0
        assert len(normalized) == len(data)
    
    def test_standardize_data(self):
        """Test data standardization (z-score)."""
        data = np.random.normal(10, 3, 100)  # Mean=10, std=3
        
        # Z-score standardization
        standardized = (data - np.mean(data)) / np.std(data)
        
        assert abs(np.mean(standardized)) < 1e-10  # Mean should be ~0
        assert abs(np.std(standardized) - 1.0) < 1e-10  # Std should be ~1
    
    def test_handle_missing_values(self):
        """Test missing value handling."""
        # Create data with missing values
        data = [1.0, 2.0, np.nan, 4.0, 5.0]
        
        # Remove NaN values
        clean_data = [x for x in data if not np.isnan(x)]
        
        assert len(clean_data) == 4
        assert np.nan not in clean_data
        
        # Fill NaN with mean
        data_array = np.array(data)
        mean_val = np.nanmean(data_array)
        filled_data = np.where(np.isnan(data_array), mean_val, data_array)
        
        assert not np.any(np.isnan(filled_data))
        assert len(filled_data) == len(data)
    
    def test_data_validation(self):
        """Test data validation utilities."""
        # Test valid data
        valid_data = {
            'accuracy': 0.85,
            'f1': 0.82,
            'precision': 0.88,
            'recall': 0.79
        }
        
        # Validation checks
        for metric, value in valid_data.items():
            assert isinstance(value, (int, float))
            assert 0 <= value <= 1
        
        # Test invalid data detection
        invalid_data = {
            'accuracy': 1.5,  # Invalid: > 1
            'f1': -0.1,      # Invalid: < 0
            'precision': 'not_a_number'  # Invalid: not numeric
        }
        
        validation_errors = []
        for metric, value in invalid_data.items():
            if not isinstance(value, (int, float)):
                validation_errors.append(f"{metric}: not numeric")
            elif value < 0 or value > 1:
                validation_errors.append(f"{metric}: out of range")
        
        assert len(validation_errors) == 3  # Should catch all 3 issues

class TestConfigurationManagement:
    """Test configuration management utilities."""
    
    def test_config_loading(self):
        """Test configuration loading from various sources."""
        # Test default config
        default_config = {
            'model': 'bert-base',
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 10
        }
        
        # Test config validation
        required_keys = ['model', 'learning_rate', 'batch_size']
        
        for key in required_keys:
            assert key in default_config
        
        assert isinstance(default_config['learning_rate'], (int, float))
        assert isinstance(default_config['batch_size'], int)
        assert isinstance(default_config['epochs'], int)
    
    def test_config_merging(self):
        """Test configuration merging functionality."""
        base_config = {
            'model': 'bert-base',
            'learning_rate': 0.001,
            'batch_size': 32
        }
        
        user_config = {
            'learning_rate': 0.0005,  # Override
            'epochs': 20,             # New setting
            'dropout': 0.1            # Additional setting
        }
        
        # Merge configs (user overrides base)
        merged_config = {**base_config, **user_config}
        
        assert merged_config['model'] == 'bert-base'  # From base
        assert merged_config['learning_rate'] == 0.0005  # Overridden
        assert merged_config['epochs'] == 20  # New
        assert merged_config['dropout'] == 0.1  # Additional
        assert merged_config['batch_size'] == 32  # From base
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 10,
            'num_classes': 2
        }
        
        # Validation rules
        validation_rules = {
            'learning_rate': lambda x: 0 < x < 1,
            'batch_size': lambda x: x > 0 and isinstance(x, int),
            'epochs': lambda x: x > 0 and isinstance(x, int),
            'num_classes': lambda x: x >= 2 and isinstance(x, int)
        }
        
        validation_results = {}
        for key, rule in validation_rules.items():
            if key in config:
                validation_results[key] = rule(config[key])
            else:
                validation_results[key] = False
        
        assert all(validation_results.values())

class TestErrorHandling:
    """Test error handling utilities."""
    
    def test_graceful_error_handling(self):
        """Test graceful error handling patterns."""
        def safe_divide(a, b):
            try:
                return a / b
            except ZeroDivisionError:
                return 0.0
            except TypeError:
                return None
        
        # Test normal case
        assert safe_divide(10, 2) == 5.0
        
        # Test division by zero
        assert safe_divide(10, 0) == 0.0
        
        # Test type error
        assert safe_divide("10", 2) is None
    
    def test_input_validation(self):
        """Test input validation with error handling."""
        def validate_metric_value(value, metric_name):
            errors = []
            
            if not isinstance(value, (int, float)):
                errors.append(f"{metric_name}: must be numeric")
            elif value < 0:
                errors.append(f"{metric_name}: must be non-negative")
            elif value > 1:
                errors.append(f"{metric_name}: must not exceed 1.0")
            
            return errors
        
        # Test valid values
        assert validate_metric_value(0.85, "accuracy") == []
        assert validate_metric_value(0.0, "precision") == []
        assert validate_metric_value(1.0, "recall") == []
        
        # Test invalid values
        errors = validate_metric_value(-0.1, "f1")
        assert len(errors) == 1
        assert "non-negative" in errors[0]
        
        errors = validate_metric_value(1.5, "accuracy")
        assert len(errors) == 1
        assert "exceed 1.0" in errors[0]
        
        errors = validate_metric_value("0.85", "precision")
        assert len(errors) == 1
        assert "numeric" in errors[0]
    
    def test_file_error_handling(self):
        """Test file operation error handling."""
        def safe_load_json(filepath):
            try:
                with open(filepath, 'r') as f:
                    return json.load(f)
            except FileNotFoundError:
                return {'error': 'File not found'}
            except json.JSONDecodeError:
                return {'error': 'Invalid JSON'}
            except Exception as e:
                return {'error': f'Unexpected error: {str(e)}'}
        
        # Test file not found
        result = safe_load_json('nonexistent_file.json')
        assert 'error' in result
        assert 'not found' in result['error']
        
        # Test invalid JSON (create temp file with invalid JSON)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"invalid": json content}')
            temp_file = f.name
        
        try:
            result = safe_load_json(temp_file)
            assert 'error' in result
            assert 'JSON' in result['error']
        finally:
            Path(temp_file).unlink()

class TestUtilityHelpers:
    """Test various utility helper functions."""
    
    def test_list_operations(self):
        """Test list utility operations."""
        data = [1, 2, 3, 4, 5]
        
        # Test chunking
        def chunk_list(lst, chunk_size):
            for i in range(0, len(lst), chunk_size):
                yield lst[i:i + chunk_size]
        
        chunks = list(chunk_list(data, 2))
        assert chunks == [[1, 2], [3, 4], [5]]
        
        # Test flattening
        nested_data = [[1, 2], [3, 4], [5]]
        flattened = [item for sublist in nested_data for item in sublist]
        assert flattened == [1, 2, 3, 4, 5]
    
    def test_dictionary_operations(self):
        """Test dictionary utility operations."""
        dict1 = {'a': 1, 'b': 2, 'c': 3}
        dict2 = {'b': 20, 'd': 4}
        
        # Test deep merge
        merged = {**dict1, **dict2}
        expected = {'a': 1, 'b': 20, 'c': 3, 'd': 4}
        assert merged == expected
        
        # Test key filtering
        filtered = {k: v for k, v in dict1.items() if k in ['a', 'c']}
        assert filtered == {'a': 1, 'c': 3}
        
        # Test value filtering
        high_values = {k: v for k, v in dict1.items() if v > 1}
        assert high_values == {'b': 2, 'c': 3}
    
    def test_string_operations(self):
        """Test string utility operations."""
        # Test safe string conversion
        def safe_str(obj):
            try:
                return str(obj)
            except:
                return "N/A"
        
        assert safe_str(42) == "42"
        assert safe_str(3.14) == "3.14"
        assert safe_str([1, 2, 3]) == "[1, 2, 3]"
        
        # Test name sanitization
        def sanitize_filename(name):
            import re
            # Replace unsafe characters
            sanitized = re.sub(r'[^\w\-_\.]', '_', name)
            return sanitized
        
        assert sanitize_filename("model/name:1") == "model_name_1"
        assert sanitize_filename("test-file.json") == "test-file.json"
