"""
ResNet implementation specifically optimized for CIFAR-10/100 datasets.
Based on the original ResNet paper with modifications for small image sizes.
Includes ResNet-56 architecture used in the RA paper experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Dict, Any


class BasicBlock(nn.Module):
    """
    Basic ResNet block for CIFAR datasets.
    Uses 3x3 convolutions with batch normalization and ReLU activation.
    """
    expansion = 1
    
    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ):
        super(BasicBlock, self).__init__()
        
        # First convolution
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, 
            padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        
        # Second convolution
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, 
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        
        # Shortcut connection
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Shortcut connection
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Add shortcut and apply ReLU
        out += identity
        out = F.relu(out, inplace=True)
        
        return out


class Bottleneck(nn.Module):
    """
    Bottleneck block for deeper ResNet architectures.
    Uses 1x1 -> 3x3 -> 1x1 convolution pattern.
    """
    expansion = 4
    
    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ):
        super(Bottleneck, self).__init__()
        
        # 1x1 convolution
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        # 3x3 convolution
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, 
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        
        # 1x1 convolution (expansion)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        
        # Shortcut connection
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # 1x1 conv
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        
        # 3x3 conv
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out, inplace=True)
        
        # 1x1 conv (expansion)
        out = self.conv3(out)
        out = self.bn3(out)
        
        # Shortcut connection
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Add shortcut and apply ReLU
        out += identity
        out = F.relu(out, inplace=True)
        
        return out


class ResNetCIFAR(nn.Module):
    """
    ResNet architecture adapted for CIFAR datasets (32x32 images).
    
    Key differences from ImageNet ResNet:
    - First conv layer is 3x3 instead of 7x7
    - No max pooling after first conv
    - Smaller feature map sizes throughout
    """
    
    def __init__(
        self,
        block: nn.Module,
        layers: List[int],
        num_classes: int = 10,
        zero_init_residual: bool = False
    ):
        super(ResNetCIFAR, self).__init__()
        
        self.num_classes = num_classes
        self.inplanes = 16  # Start with 16 channels for CIFAR
        
        # Initial convolution layer (3x3 instead of 7x7 for CIFAR)
        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)
        
        # ResNet layers
        self.layer1 = self._make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        
        # Initialize weights
        self._init_weights(zero_init_residual)
    
    def _make_layer(
        self,
        block: nn.Module,
        planes: int,
        blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        """Create a layer with multiple blocks."""
        downsample = None
        
        # Create downsample layer if needed
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )
        
        layers = []
        # First block may have stride > 1
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        
        # Remaining blocks have stride = 1
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self, zero_init_residual: bool):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input images [batch_size, 3, 32, 32]
            
        Returns:
            logits: Classification logits [batch_size, num_classes]
        """
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        
        # ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Global pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    def get_feature_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get intermediate feature maps for analysis.
        
        Args:
            x: Input images
            
        Returns:
            Dictionary of feature maps from each layer
        """
        features = {}
        
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        features['conv1'] = x
        
        # ResNet layers
        x = self.layer1(x)
        features['layer1'] = x
        
        x = self.layer2(x)
        features['layer2'] = x
        
        x = self.layer3(x)
        features['layer3'] = x
        
        # Global pooling
        x = self.avgpool(x)
        features['avgpool'] = x
        
        return features


def resnet20_cifar(num_classes: int = 10, **kwargs) -> ResNetCIFAR:
    """ResNet-20 for CIFAR datasets."""
    return ResNetCIFAR(BasicBlock, [3, 3, 3], num_classes, **kwargs)


def resnet32_cifar(num_classes: int = 10, **kwargs) -> ResNetCIFAR:
    """ResNet-32 for CIFAR datasets."""
    return ResNetCIFAR(BasicBlock, [5, 5, 5], num_classes, **kwargs)


def resnet44_cifar(num_classes: int = 10, **kwargs) -> ResNetCIFAR:
    """ResNet-44 for CIFAR datasets."""
    return ResNetCIFAR(BasicBlock, [7, 7, 7], num_classes, **kwargs)


def resnet56_cifar(num_classes: int = 10, **kwargs) -> ResNetCIFAR:
    """
    ResNet-56 for CIFAR datasets - the architecture used in RA paper.
    
    This is the standard ResNet-56 architecture:
    - Total layers: 56 (54 conv + 2 FC)
    - Structure: [9, 9, 9] blocks per layer group
    - Parameters: ~853K for CIFAR-10
    """
    return ResNetCIFAR(BasicBlock, [9, 9, 9], num_classes, **kwargs)


def resnet110_cifar(num_classes: int = 10, **kwargs) -> ResNetCIFAR:
    """ResNet-110 for CIFAR datasets."""
    return ResNetCIFAR(BasicBlock, [18, 18, 18], num_classes, **kwargs)


def wide_resnet28_10_cifar(num_classes: int = 10) -> ResNetCIFAR:
    """
    Wide ResNet-28-10 for CIFAR datasets.
    28 layers with width multiplier of 10.
    """
    # Modify the basic ResNet to have wider layers
    model = ResNetCIFAR(BasicBlock, [4, 4, 4], num_classes)
    
    # Replace layers with wider versions
    model.inplanes = 16
    model.layer1 = model._make_layer(BasicBlock, 160, 4, stride=1)  # 16 * 10
    model.layer2 = model._make_layer(BasicBlock, 320, 4, stride=2)  # 32 * 10
    model.layer3 = model._make_layer(BasicBlock, 640, 4, stride=2)  # 64 * 10
    
    # Update classifier
    model.fc = nn.Linear(640, num_classes)
    
    return model


class ResNetCIFARTrainer:
    """
    Training utilities for ResNet CIFAR models.
    """
    
    def __init__(
        self,
        model: ResNetCIFAR,
        learning_rate: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        milestones: Optional[List[int]] = None
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        # Initialize optimizer
        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        if milestones is None:
            milestones = [100, 150]  # Default for 200 epochs
        
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=milestones,
            gamma=0.1
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        device: str = "cpu"
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        accuracy = 100.0 * correct / total
        avg_loss = total_loss / len(dataloader)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        device: str = "cpu"
    ) -> Dict[str, float]:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = 100.0 * correct / total
        avg_loss = total_loss / len(dataloader)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }


def get_model_info(model: ResNetCIFAR) -> Dict[str, Any]:
    """Get information about a ResNet CIFAR model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count layers
    total_layers = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            total_layers += 1
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'total_layers': total_layers,
        'num_classes': model.num_classes,
        'model_size_mb': total_params * 4 / (1024 * 1024)  # Assume float32
    }


if __name__ == "__main__":
    # Example usage and testing
    print("🧪 Testing ResNet CIFAR implementations")
    print("=" * 50)
    
    # Test different architectures
    models = {
        'ResNet-20': resnet20_cifar(),
        'ResNet-32': resnet32_cifar(), 
        'ResNet-44': resnet44_cifar(),
        'ResNet-56': resnet56_cifar(),
        'ResNet-110': resnet110_cifar()
    }
    
    # Test with dummy input
    dummy_input = torch.randn(4, 3, 32, 32)  # Batch of 4 CIFAR images
    
    for name, model in models.items():
        output = model(dummy_input)
        info = get_model_info(model)
        
        print(f"\n{name}:")
        print(f"  Output shape: {output.shape}")
        print(f"  Parameters: {info['total_parameters']:,}")
        print(f"  Model size: {info['model_size_mb']:.1f} MB")
        print(f"  Total layers: {info['total_layers']}")
        
        # Test feature map extraction
        feature_maps = model.get_feature_maps(dummy_input)
        print(f"  Feature maps: {list(feature_maps.keys())}")
    
    # Test with ResNet-56 (paper model)
    print(f"\n🎯 ResNet-56 (RA Paper Model) Details:")
    resnet56 = resnet56_cifar(num_classes=10)
    print(f"  Architecture: {[9, 9, 9]} blocks per layer")
    print(f"  Total depth: 56 layers (54 conv + 2 FC)")
    print(f"  Parameters: {get_model_info(resnet56)['total_parameters']:,}")
    
    # Test training setup
    trainer = ResNetCIFARTrainer(resnet56, learning_rate=0.1)
    print(f"  Optimizer: SGD (lr=0.1, momentum=0.9, wd=1e-4)")
    print(f"  Scheduler: MultiStepLR [100, 150] epochs")
    
    print(f"\n✅ All ResNet CIFAR models tested successfully!")
