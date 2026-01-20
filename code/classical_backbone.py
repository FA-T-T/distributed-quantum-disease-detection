"""
Classical backbone models for comparison with quantum models.

This module provides pure classical neural network models for baseline comparison:
- ClassicalCNN: Pure MobileNetV2 classifier
- VGGClassifier: VGG16-based classifier
- ResNetClassifier: ResNet50-based classifier

All models are implemented in PyTorch and compatible with the existing training framework.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ClassicalCNN(nn.Module):
    """
    Pure MobileNetV2 classifier.
    
    This serves as a classical baseline using only the MobileNetV2 architecture
    without any quantum components.
    
    Args:
        num_classes: Number of output classes (default: 3 for ISIC2017)
        pretrained: Whether to use pretrained ImageNet weights
    """
    
    def __init__(self, num_classes: int = 3, pretrained: bool = True):
        super(ClassicalCNN, self).__init__()
        
        # Use pretrained MobileNetV2 from torchvision
        self.mobilenet = models.mobilenet_v2(pretrained=pretrained)
        
        # Modify the classifier for our number of classes
        in_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier[1] = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 128, 128)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        return self.mobilenet(x)


class VGGClassifier(nn.Module):
    """
    VGG16-based classifier.
    
    Uses VGG16 architecture with batch normalization as the backbone
    for feature extraction and classification.
    
    Args:
        num_classes: Number of output classes (default: 3 for ISIC2017)
        pretrained: Whether to use pretrained ImageNet weights
        freeze_features: Whether to freeze feature extraction layers
    """
    
    def __init__(
        self, 
        num_classes: int = 3, 
        pretrained: bool = True,
        freeze_features: bool = False
    ):
        super(VGGClassifier, self).__init__()
        
        # Load pretrained VGG16 with batch normalization
        self.vgg = models.vgg16_bn(pretrained=pretrained)
        
        # Optionally freeze feature extraction layers
        if freeze_features:
            for param in self.vgg.features.parameters():
                param.requires_grad = False
        
        # Modify the classifier
        # VGG features output 512 channels with adaptive pooling to 7x7
        # But for 128x128 input (vs 224x224), we need to use adaptive pooling
        self.vgg.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Keep the original classifier structure
        in_features = self.vgg.classifier[0].in_features  # Should be 512 * 7 * 7
        self.vgg.classifier = nn.Sequential(
            nn.Linear(in_features, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 128, 128)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        return self.vgg(x)


class ResNetClassifier(nn.Module):
    """
    ResNet50-based classifier.
    
    Uses ResNet50 architecture as the backbone for feature extraction
    and classification.
    
    Args:
        num_classes: Number of output classes (default: 3 for ISIC2017)
        pretrained: Whether to use pretrained ImageNet weights
        freeze_features: Whether to freeze early layers
    """
    
    def __init__(
        self, 
        num_classes: int = 3, 
        pretrained: bool = True,
        freeze_features: bool = False
    ):
        super(ResNetClassifier, self).__init__()
        
        # Load pretrained ResNet50
        self.resnet = models.resnet50(pretrained=pretrained)
        
        # Optionally freeze early layers (conv1 through layer2)
        if freeze_features:
            for name, param in self.resnet.named_parameters():
                if any(layer in name for layer in ['conv1', 'bn1', 'layer1', 'layer2']):
                    param.requires_grad = False
        
        # Modify the final fully connected layer
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 128, 128)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        return self.resnet(x)


class LightweightResNet(nn.Module):
    """
    Lightweight ResNet classifier using ResNet18.
    
    A lighter alternative to ResNet50, useful for faster training and testing.
    
    Args:
        num_classes: Number of output classes (default: 3 for ISIC2017)
        pretrained: Whether to use pretrained ImageNet weights
    """
    
    def __init__(
        self, 
        num_classes: int = 3, 
        pretrained: bool = True
    ):
        super(LightweightResNet, self).__init__()
        
        # Load pretrained ResNet18
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # Modify the final fully connected layer
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 128, 128)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        return self.resnet(x)


def get_model(
    model_name: str, 
    num_classes: int = 3, 
    pretrained: bool = True,
    **kwargs
) -> nn.Module:
    """
    Factory function to get a classical model by name.
    
    Args:
        model_name: Name of the model ('mobilenet', 'vgg16', 'resnet50', 'resnet18')
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        **kwargs: Additional arguments for specific models
        
    Returns:
        The requested model
        
    Raises:
        ValueError: If model_name is not recognized
    """
    models_dict = {
        'mobilenet': ClassicalCNN,
        'vgg16': VGGClassifier,
        'resnet50': ResNetClassifier,
        'resnet18': LightweightResNet
    }
    
    if model_name.lower() not in models_dict:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(models_dict.keys())}"
        )
    
    model_class = models_dict[model_name.lower()]
    return model_class(num_classes=num_classes, pretrained=pretrained, **kwargs)


if __name__ == '__main__':
    # Test the models
    print("Testing classical backbone models...")
    
    # Test input
    batch_size = 2
    x = torch.randn(batch_size, 3, 128, 128)
    
    # Test ClassicalCNN
    print("\n1. ClassicalCNN (MobileNetV2)")
    model = ClassicalCNN(num_classes=3, pretrained=False)
    output = model(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test VGGClassifier
    print("\n2. VGGClassifier")
    model = VGGClassifier(num_classes=3, pretrained=False)
    output = model(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test ResNetClassifier
    print("\n3. ResNetClassifier (ResNet50)")
    model = ResNetClassifier(num_classes=3, pretrained=False)
    output = model(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test LightweightResNet
    print("\n4. LightweightResNet (ResNet18)")
    model = LightweightResNet(num_classes=3, pretrained=False)
    output = model(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\nAll models tested successfully!")
