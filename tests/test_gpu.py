import torch
from ra.device_utils import device, get_device
from models.bert_sentiment import create_bert_sentiment_model
from models.resnet_cifar import resnet56_cifar

print("=== GPU Test ===")
print(f"Device: {get_device()}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Test BERT model
print("\n=== Testing BERT Model ===")
try:
    bert_model = create_bert_sentiment_model()
    print(f"BERT model device: {next(bert_model.parameters()).device}")
    
    # Test forward pass
    dummy_input = torch.randint(0, 1000, (2, 10)).to(device)
    output = bert_model(dummy_input)
    print(f"BERT output shape: {output.shape}, device: {output.device}")
    print("✅ BERT model working on GPU!")
except Exception as e:
    print(f"❌ BERT model failed: {e}")

# Test ResNet model  
print("\n=== Testing ResNet Model ===")
try:
    resnet_model = resnet56_cifar(num_classes=10)
    print(f"ResNet model device: {next(resnet_model.parameters()).device}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 32, 32).to(device)
    output = resnet_model(dummy_input)
    print(f"ResNet output shape: {output.shape}, device: {output.device}")
    print("✅ ResNet model working on GPU!")
except Exception as e:
    print(f"❌ ResNet model failed: {e}")

print("\n=== GPU Test Complete ===")
