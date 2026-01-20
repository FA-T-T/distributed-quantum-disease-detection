"""
Unit tests for the validation module.
"""

import os
import sys
import pytest
import torch
import torch.nn as nn

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'code'))

from ..code.validation import (
    validate_image_tensor,
    validate_model_input,
    validate_model_output,
    validate_labels
)


class TestValidateImageTensor:
    """Tests for validate_image_tensor function."""
    
    def test_valid_3d_tensor(self):
        """Test with valid 3D tensor."""
        tensor = torch.randn(3, 128, 128)
        is_valid, msg = validate_image_tensor(tensor)
        
        assert is_valid is True
        assert "Valid" in msg
    
    def test_valid_4d_tensor(self):
        """Test with valid 4D tensor."""
        tensor = torch.randn(2, 3, 128, 128)
        is_valid, msg = validate_image_tensor(tensor)
        
        assert is_valid is True
    
    def test_wrong_channels(self):
        """Test with wrong number of channels."""
        tensor = torch.randn(1, 128, 128)  # 1 channel instead of 3
        is_valid, msg = validate_image_tensor(tensor, expected_channels=3)
        
        assert is_valid is False
        assert "channel" in msg.lower()
    
    def test_wrong_dimensions(self):
        """Test with wrong tensor dimensions."""
        tensor = torch.randn(100)  # 1D tensor
        is_valid, msg = validate_image_tensor(tensor)
        
        assert is_valid is False
        assert "3D or 4D" in msg
    
    def test_nan_values(self):
        """Test with NaN values."""
        tensor = torch.randn(3, 128, 128)
        tensor[0, 0, 0] = float('nan')
        is_valid, msg = validate_image_tensor(tensor)
        
        assert is_valid is False
        assert "NaN" in msg
    
    def test_inf_values(self):
        """Test with Inf values."""
        tensor = torch.randn(3, 128, 128)
        tensor[0, 0, 0] = float('inf')
        is_valid, msg = validate_image_tensor(tensor)
        
        assert is_valid is False
        assert "Inf" in msg


class TestValidateModelInput:
    """Tests for validate_model_input function."""
    
    def test_valid_input(self):
        """Test with valid input tensor."""
        tensor = torch.randn(2, 3, 128, 128)
        is_valid, msg = validate_model_input(tensor)
        
        assert is_valid is True
    
    def test_wrong_dimensions(self):
        """Test with wrong dimensions."""
        tensor = torch.randn(3, 128, 128)  # 3D instead of 4D
        is_valid, msg = validate_model_input(tensor)
        
        assert is_valid is False
        assert "4D" in msg
    
    def test_wrong_batch_size(self):
        """Test with wrong batch size."""
        tensor = torch.randn(2, 3, 128, 128)
        is_valid, msg = validate_model_input(tensor, batch_size=4)
        
        assert is_valid is False
        assert "batch size" in msg.lower()
    
    def test_wrong_channels(self):
        """Test with wrong number of channels."""
        tensor = torch.randn(2, 1, 128, 128)  # 1 channel
        is_valid, msg = validate_model_input(tensor)
        
        assert is_valid is False
        assert "channel" in msg.lower()
    
    def test_wrong_image_size(self):
        """Test with wrong image size."""
        tensor = torch.randn(2, 3, 64, 64)
        is_valid, msg = validate_model_input(tensor, image_size=128)
        
        assert is_valid is False
        assert "size" in msg.lower()


class TestValidateModelOutput:
    """Tests for validate_model_output function."""
    
    def test_valid_output(self):
        """Test with valid output tensor."""
        tensor = torch.randn(2, 3)
        is_valid, msg = validate_model_output(tensor)
        
        assert is_valid is True
    
    def test_wrong_dimensions(self):
        """Test with wrong dimensions."""
        tensor = torch.randn(2, 3, 4)  # 3D instead of 2D
        is_valid, msg = validate_model_output(tensor)
        
        assert is_valid is False
        assert "2D" in msg
    
    def test_wrong_num_classes(self):
        """Test with wrong number of classes."""
        tensor = torch.randn(2, 5)  # 5 classes instead of 3
        is_valid, msg = validate_model_output(tensor, num_classes=3)
        
        assert is_valid is False
        assert "classes" in msg.lower()
    
    def test_nan_values(self):
        """Test with NaN values."""
        tensor = torch.randn(2, 3)
        tensor[0, 0] = float('nan')
        is_valid, msg = validate_model_output(tensor)
        
        assert is_valid is False
        assert "NaN" in msg


class TestValidateLabels:
    """Tests for validate_labels function."""
    
    def test_valid_labels(self):
        """Test with valid labels."""
        labels = torch.tensor([0, 1, 2, 0, 1])
        is_valid, msg = validate_labels(labels)
        
        assert is_valid is True
    
    def test_negative_labels(self):
        """Test with negative labels."""
        labels = torch.tensor([0, -1, 2])
        is_valid, msg = validate_labels(labels)
        
        assert is_valid is False
        assert "negative" in msg.lower()
    
    def test_labels_out_of_range(self):
        """Test with labels out of range."""
        labels = torch.tensor([0, 1, 5])  # 5 is out of range for 3 classes
        is_valid, msg = validate_labels(labels, num_classes=3)
        
        assert is_valid is False
    
    def test_wrong_dimensions(self):
        """Test with wrong dimensions."""
        labels = torch.tensor([[0, 1], [2, 3]])  # 2D instead of 1D
        is_valid, msg = validate_labels(labels)
        
        assert is_valid is False
        assert "1D" in msg


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
