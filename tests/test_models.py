"""
Unit tests for the model modules.
"""

import os
import sys
import pytest
import torch
import torch.nn as nn

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'code'))

from ..code.mobilnet import MobileNetV2
from ..code.mlp import MLP


class TestMobileNetV2:
    """Tests for MobileNetV2 model."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = MobileNetV2()
        assert model is not None
        assert isinstance(model, nn.Module)
    
    def test_forward_pass(self):
        """Test forward pass with correct input shape."""
        model = MobileNetV2(num_classes=8)
        model.eval()
        
        input_tensor = torch.randn(2, 3, 128, 128)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        assert output.shape == (2, 8)
    
    def test_different_num_classes(self):
        """Test model with different number of classes."""
        for num_classes in [3, 5, 10]:
            model = MobileNetV2(num_classes=num_classes)
            model.eval()
            
            input_tensor = torch.randn(1, 3, 128, 128)
            
            with torch.no_grad():
                output = model(input_tensor)
            
            assert output.shape == (1, num_classes)
    
    def test_batch_sizes(self):
        """Test model with different batch sizes."""
        model = MobileNetV2(num_classes=3)
        model.eval()
        
        for batch_size in [1, 2, 4]:
            input_tensor = torch.randn(batch_size, 3, 128, 128)
            
            with torch.no_grad():
                output = model(input_tensor)
            
            assert output.shape == (batch_size, 3)
    
    def test_gradients(self):
        """Test that gradients can be computed."""
        model = MobileNetV2(num_classes=3)
        
        input_tensor = torch.randn(2, 3, 128, 128, requires_grad=True)
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()
        
        assert input_tensor.grad is not None
        assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


class TestMLP:
    """Tests for MLP model."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = MLP()
        assert model is not None
        assert isinstance(model, nn.Module)
    
    def test_forward_pass(self):
        """Test forward pass with correct input shape."""
        model = MLP()
        model.eval()
        
        # MLP expects flattened input of 3*128*128
        input_tensor = torch.randn(2, 3, 128, 128)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        assert output.shape == (2, 6)  # MLP outputs 6 classes
    
    def test_batch_sizes(self):
        """Test model with different batch sizes."""
        model = MLP()
        model.eval()
        
        for batch_size in [1, 2, 4]:
            input_tensor = torch.randn(batch_size, 3, 128, 128)
            
            with torch.no_grad():
                output = model(input_tensor)
            
            assert output.shape == (batch_size, 6)


class TestModelParameters:
    """Tests for model parameters."""
    
    def test_mobilenet_trainable_params(self):
        """Test that MobileNetV2 has trainable parameters."""
        model = MobileNetV2()
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable_params > 0
    
    def test_mlp_trainable_params(self):
        """Test that MLP has trainable parameters."""
        model = MLP()
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable_params > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
