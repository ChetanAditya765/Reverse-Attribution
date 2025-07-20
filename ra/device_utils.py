import torch

# Global device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_device():
    """Get the current device"""
    return device

def to_device(obj):
    """Move object to device"""
    if isinstance(obj, (list, tuple)):
        return [o.to(device, non_blocking=True) if hasattr(o, 'to') else o for o in obj]
    if hasattr(obj, 'to'):
        return obj.to(device, non_blocking=True)
    return obj

# Print device info on import
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
