"""
Training and evaluation scripts for the Reverse Attribution framework.

This package contains:
- Main orchestration script (script.py)
- Text model training (script_1.py)
- Vision model training (script_2.py) 
- Model evaluation (script_3.py)
"""

# Import main functions from each script
try:
    from .script import main as run_main_script
except ImportError:
    run_main_script = None

try:
    from .script_1 import train_text_model
except ImportError:
    train_text_model = None

try:
    from .script_2 import train_vision_model  
except ImportError:
    train_vision_model = None

try:
    from .script_3 import evaluate_all_models
except ImportError:
    evaluate_all_models = None

# Define what gets exported
__all__ = []

if run_main_script is not None:
    __all__.append("run_main_script")

if train_text_model is not None:
    __all__.append("train_text_model")

if train_vision_model is not None:
    __all__.append("train_vision_model")
    
if evaluate_all_models is not None:
    __all__.append("evaluate_all_models")

# Convenience function to list available scripts
def list_available_scripts():
    """List all available training/evaluation scripts."""
    available = []
    
    if run_main_script is not None:
        available.append("Main orchestration script")
    if train_text_model is not None:
        available.append("Text model training")  
    if train_vision_model is not None:
        available.append("Vision model training")
    if evaluate_all_models is not None:
        available.append("Model evaluation")
        
    return available

__all__.append("list_available_scripts")
