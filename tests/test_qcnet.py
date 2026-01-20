"""
Unit tests for the QCNet hybrid quantum-classical model.
"""

import os
import sys
import pytest
import torch
import torch.nn as nn

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'code'))

from ..code.backbone_3 import QCNet


class TestQCNet:
    """Tests for QCNet model."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = QCNet()
        assert model is not None
        assert isinstance(model, nn.Module)
    
    def test_has_components(self):
        """Test that model has required components."""
        model = QCNet()
        
        # Check for classical model
        assert hasattr(model, 'CModel')
        assert model.CModel is not None
        
        # Check for quantum model
        assert hasattr(model, 'QModel')
        assert model.QModel is not None
        
        # Check for fully connected layers
        assert hasattr(model, 'fc1')
        assert hasattr(model, 'fc2')
    
    def test_forward_pass(self):
        """Test forward pass with correct input shape."""
        model = QCNet()
        model.eval()
        
        input_tensor = torch.randn(2, 3, 128, 128)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        assert output.shape == (2, 3)
    
    def test_different_batch_sizes(self):
        """Test model with different batch sizes."""
        model = QCNet()
        model.eval()
        
        for batch_size in [1, 2, 4]:
            input_tensor = torch.randn(batch_size, 3, 128, 128)
            
            with torch.no_grad():
                output = model(input_tensor)
            
            assert output.shape == (batch_size, 3)
    
    def test_gradients(self):
        """Test that gradients can be computed."""
        model = QCNet()
        
        input_tensor = torch.randn(2, 3, 128, 128, requires_grad=True)
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()
        
        assert input_tensor.grad is not None
        # Check that at least some parameters have gradients
        assert any(p.grad is not None for p in model.parameters() if p.requires_grad)
    
    def test_trainable_parameters(self):
        """Test that model has trainable parameters."""
        model = QCNet()
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable_params > 0
        
        # Should have more than 1 million parameters (classical + quantum)
        assert trainable_params > 1_000_000
    
    def test_output_values(self):
        """Test that output values are reasonable."""
        model = QCNet()
        model.eval()
        
        input_tensor = torch.randn(2, 3, 128, 128)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        # Check for NaN or Inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_training_mode(self):
        """Test switching between train and eval modes."""
        model = QCNet()
        
        # Default should be training mode
        assert model.training
        
        # Switch to eval
        model.eval()
        assert not model.training
        
        # Switch back to train
        model.train()
        assert model.training
    
    def test_quantum_layers_exist(self):
        """Test that quantum layers are properly initialized."""
        model = QCNet()
        
        # Check that QModel has the quantum layers
        assert hasattr(model.QModel, 'QLayer_front_1')
        assert hasattr(model.QModel, 'QLayer_front_2')
        assert hasattr(model.QModel, 'QLayer_front_3')
        assert hasattr(model.QModel, 'QLayer_back_1')
        assert hasattr(model.QModel, 'QLayer_back_2')
        assert hasattr(model.QModel, 'QLayer_back_3')
        assert hasattr(model.QModel, 'QLayer_back_4')
        assert hasattr(model.QModel, 'QLayer_back_5')
        assert hasattr(model.QModel, 'QLayer_back_6')
    
    def test_shared_weights_exist(self):
        """Test that shared weights are properly initialized."""
        model = QCNet()
        
        # Check that QModel has shared weights
        assert hasattr(model.QModel, 'shared_weights_1')
        assert hasattr(model.QModel, 'shared_weights_2')
        
        # Check that they are parameters
        assert isinstance(model.QModel.shared_weights_1, nn.Parameter)
        assert isinstance(model.QModel.shared_weights_2, nn.Parameter)
        
        # Check that they require gradients
        assert model.QModel.shared_weights_1.requires_grad
        assert model.QModel.shared_weights_2.requires_grad
    
    def test_weight_initialization_range(self):
        """Test that quantum weights are initialized in proper range."""
        import numpy as np
        model = QCNet()
        
        # Check that shared weights are in [-π/4, π/4] range
        w1 = model.QModel.shared_weights_1.data
        w2 = model.QModel.shared_weights_2.data
        
        # Allow some tolerance for numerical precision
        assert (w1 >= -np.pi/4 - 0.01).all()
        assert (w1 <= np.pi/4 + 0.01).all()
        assert (w2 >= -np.pi/4 - 0.01).all()
        assert (w2 <= np.pi/4 + 0.01).all()


class TestQCNetIntegration:
    """Integration tests for QCNet."""
    
    def test_loss_computation(self):
        """Test that loss can be computed."""
        model = QCNet()
        criterion = nn.CrossEntropyLoss()
        
        input_tensor = torch.randn(2, 3, 128, 128)
        targets = torch.tensor([0, 1])
        
        outputs = model(input_tensor)
        loss = criterion(outputs, targets)
        
        assert loss.item() >= 0
        assert not torch.isnan(loss)
    
    def test_backward_pass(self):
        """Test that backward pass works."""
        model = QCNet()
        criterion = nn.CrossEntropyLoss()
        
        input_tensor = torch.randn(2, 3, 128, 128)
        targets = torch.tensor([0, 1])
        
        outputs = model(input_tensor)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Check that some gradients were computed
        gradients_computed = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters() if p.requires_grad
        )
        assert gradients_computed
    
    def test_optimizer_step(self):
        """Test that optimizer can update model parameters."""
        model = QCNet()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        input_tensor = torch.randn(2, 3, 128, 128)
        targets = torch.tensor([0, 1])
        
        # Get initial parameters
        initial_params = [p.clone() for p in model.parameters()]
        
        # Training step
        optimizer.zero_grad()
        outputs = model(input_tensor)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # Check that at least some parameters changed
        params_changed = False
        for initial, current in zip(initial_params, model.parameters()):
            if not torch.allclose(initial, current, atol=1e-6):
                params_changed = True
                break
        
        assert params_changed
    
    def test_multiple_forward_passes(self):
        """Test multiple forward passes with same model."""
        model = QCNet()
        model.eval()
        
        input_tensor = torch.randn(2, 3, 128, 128)
        
        with torch.no_grad():
            output1 = model(input_tensor)
            output2 = model(input_tensor)
        
        # Same input should give same output (deterministic)
        assert torch.allclose(output1, output2, atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
