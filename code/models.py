"""
Model architectures for MobileNetV2 baseline and hybrid quantum-classical models.

This module provides:
1. MobileNetV2Baseline: Pure MobileNetV2 with pretrained weights, frozen backbone, and 3-class output
2. MobileNetV2Hybrid: MobileNetV2 backbone + QNN layer + FC for 3-class classification

Both models are designed for skin cancer classification using the ISIC2017 dataset.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional

# Import QNN from mps3 for the hybrid model
from mps3 import QNN


class MobileNetV2Baseline(nn.Module):
    """
    Baseline model using PyTorch's pretrained MobileNetV2.
    
    Architecture:
    - MobileNetV2 feature extractor (pretrained on ImageNet)
    - Frozen backbone layers (optional)
    - Modified classifier for 3-class output
    
    Args:
        num_classes: Number of output classes (default: 3 for ISIC2017)
        pretrained: Whether to use pretrained ImageNet weights (default: True)
        freeze_backbone: Whether to freeze the backbone layers (default: True)
    """
    
    def __init__(
        self, 
        num_classes: int = 3, 
        pretrained: bool = True,
        freeze_backbone: bool = True
    ):
        super(MobileNetV2Baseline, self).__init__()
        
        # Load pretrained MobileNetV2 from torchvision
        if pretrained:
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
            self.mobilenet = models.mobilenet_v2(weights=weights)
        else:
            self.mobilenet = models.mobilenet_v2(weights=None)
        
        # Freeze backbone layers if specified
        if freeze_backbone:
            for param in self.mobilenet.features.parameters():
                param.requires_grad = False
        
        # Modify the classifier for our number of classes
        # Original MobileNetV2 classifier: Dropout(0.2) -> Linear(1280, 1000)
        in_features = self.mobilenet.classifier[1].in_features  # 1280
        self.mobilenet.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        return self.mobilenet(x)
    
    def get_trainable_params(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self) -> int:
        """Return the total number of parameters."""
        return sum(p.numel() for p in self.parameters())


class MobileNetV2Hybrid(nn.Module):
    """
    Hybrid Quantum-Classical model using MobileNetV2 + QNN.
    
    Architecture:
    - MobileNetV2 feature extractor (pretrained, frozen backbone)
    - Fully connected layer to map features to QNN input dimension
    - QNN layer (Distributed Quantum Neural Network with circuit cutting)
    - Final fully connected layer for 3-class classification
    
    Args:
        num_classes: Number of output classes (default: 3 for ISIC2017)
        pretrained: Whether to use pretrained ImageNet weights (default: True)
        freeze_backbone: Whether to freeze the MobileNetV2 backbone (default: True)
        qnn_input_dim: Input dimension for QNN (default: 8, matching QNN expectation)
    """
    
    def __init__(
        self, 
        num_classes: int = 3, 
        pretrained: bool = True,
        freeze_backbone: bool = True,
        qnn_input_dim: int = 8
    ):
        super(MobileNetV2Hybrid, self).__init__()
        
        self.num_classes = num_classes
        self.qnn_input_dim = qnn_input_dim
        
        # Load pretrained MobileNetV2 from torchvision
        if pretrained:
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
            self.backbone = models.mobilenet_v2(weights=weights)
        else:
            self.backbone = models.mobilenet_v2(weights=None)
        
        # Get feature dimension (before classifier)
        self.feature_dim = self.backbone.classifier[1].in_features  # 1280
        
        # Remove the original classifier - we'll use our own
        self.backbone.classifier = nn.Identity()
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # FC layer to map backbone features to QNN input dimension
        self.fc1 = nn.Linear(self.feature_dim, qnn_input_dim)
        
        # QNN layer (Distributed Quantum Neural Network)
        self.qnn = QNN()
        
        # QNN output dimension is 2^5 = 32 (from n_qubits_2 = 5)
        qnn_output_dim = 32
        
        # Final classification layer
        self.fc2 = nn.Linear(qnn_output_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid model.
        
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Extract features from MobileNetV2 backbone
        features = self.backbone(x)  # (batch_size, 1280)
        
        # Map to QNN input dimension
        qnn_input = self.fc1(features)  # (batch_size, 8)
        
        # Pass through QNN
        qnn_output = self.qnn(qnn_input)  # (batch_size, 32)
        
        # Final classification
        output = self.fc2(qnn_output)  # (batch_size, num_classes)
        
        return output
    
    def get_trainable_params(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self) -> int:
        """Return the total number of parameters."""
        return sum(p.numel() for p in self.parameters())


def get_model(
    model_type: str,
    num_classes: int = 3,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    **kwargs
) -> nn.Module:
    """
    Factory function to get a model by type.
    
    Args:
        model_type: Type of model ('baseline' or 'hybrid')
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        freeze_backbone: Whether to freeze backbone layers
        **kwargs: Additional arguments passed to the model
        
    Returns:
        The requested model instance
        
    Raises:
        ValueError: If model_type is not recognized
    """
    model_type = model_type.lower()
    
    if model_type == 'baseline':
        return MobileNetV2Baseline(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )
    elif model_type == 'hybrid':
        return MobileNetV2Hybrid(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            **kwargs
        )
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available types: 'baseline', 'hybrid'"
        )


if __name__ == '__main__':
    """Test the models."""
    print("=" * 60)
    print("Testing MobileNetV2 Models")
    print("=" * 60)
    
    # Test input
    batch_size = 2
    x = torch.randn(batch_size, 3, 128, 128)
    
    # Test MobileNetV2Baseline
    print("\n1. MobileNetV2Baseline (Pretrained=False for testing)")
    model = MobileNetV2Baseline(num_classes=3, pretrained=False, freeze_backbone=True)
    model.eval()
    
    with torch.no_grad():
        output = model(x)
    
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Total parameters: {model.get_total_params():,}")
    print(f"   Trainable parameters: {model.get_trainable_params():,}")
    
    # Test MobileNetV2Hybrid
    print("\n2. MobileNetV2Hybrid (Pretrained=False for testing)")
    model = MobileNetV2Hybrid(num_classes=3, pretrained=False, freeze_backbone=True)
    model.eval()
    
    with torch.no_grad():
        output = model(x)
    
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Total parameters: {model.get_total_params():,}")
    print(f"   Trainable parameters: {model.get_trainable_params():,}")
    
    # Test factory function
    print("\n3. Testing get_model factory function")
    baseline = get_model('baseline', num_classes=3, pretrained=False)
    hybrid = get_model('hybrid', num_classes=3, pretrained=False)
    
    print(f"   Baseline model type: {type(baseline).__name__}")
    print(f"   Hybrid model type: {type(hybrid).__name__}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
