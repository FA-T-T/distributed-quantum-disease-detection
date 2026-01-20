"""
Unit tests for MobileNetV2 baseline and hybrid models.
"""

import os
import sys
import pytest
import torch
import torch.nn as nn

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'code'))

from models import MobileNetV2Baseline, MobileNetV2Hybrid, get_model


class TestMobileNetV2Baseline:
    """Tests for MobileNetV2Baseline model."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = MobileNetV2Baseline(num_classes=3, pretrained=False)
        assert model is not None
        assert isinstance(model, nn.Module)
    
    def test_forward_pass(self):
        """Test forward pass with correct input shape."""
        model = MobileNetV2Baseline(num_classes=3, pretrained=False)
        model.eval()
        
        input_tensor = torch.randn(2, 3, 128, 128)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        assert output.shape == (2, 3)
    
    def test_different_num_classes(self):
        """Test model with different number of classes."""
        for num_classes in [2, 3, 5, 10]:
            model = MobileNetV2Baseline(num_classes=num_classes, pretrained=False)
            model.eval()
            
            input_tensor = torch.randn(1, 3, 128, 128)
            
            with torch.no_grad():
                output = model(input_tensor)
            
            assert output.shape == (1, num_classes)
    
    def test_batch_sizes(self):
        """Test model with different batch sizes."""
        model = MobileNetV2Baseline(num_classes=3, pretrained=False)
        model.eval()
        
        for batch_size in [1, 2, 4]:
            input_tensor = torch.randn(batch_size, 3, 128, 128)
            
            with torch.no_grad():
                output = model(input_tensor)
            
            assert output.shape == (batch_size, 3)
    
    def test_freeze_backbone(self):
        """Test that freeze_backbone works correctly."""
        model = MobileNetV2Baseline(num_classes=3, pretrained=False, freeze_backbone=True)
        
        # Check that feature layers are frozen
        for param in model.mobilenet.features.parameters():
            assert not param.requires_grad
        
        # Check that classifier layers are trainable
        classifier_trainable = any(
            p.requires_grad for p in model.mobilenet.classifier.parameters()
        )
        assert classifier_trainable
    
    def test_no_freeze_backbone(self):
        """Test model without freezing backbone."""
        model = MobileNetV2Baseline(num_classes=3, pretrained=False, freeze_backbone=False)
        
        # Check that feature layers are trainable
        features_trainable = any(
            p.requires_grad for p in model.mobilenet.features.parameters()
        )
        assert features_trainable
    
    def test_gradients(self):
        """Test that gradients can be computed."""
        model = MobileNetV2Baseline(num_classes=3, pretrained=False, freeze_backbone=False)
        
        input_tensor = torch.randn(2, 3, 128, 128, requires_grad=True)
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()
        
        assert input_tensor.grad is not None
        assert any(p.grad is not None for p in model.parameters() if p.requires_grad)
    
    def test_trainable_parameters(self):
        """Test parameter counting methods."""
        model = MobileNetV2Baseline(num_classes=3, pretrained=False, freeze_backbone=True)
        
        trainable_params = model.get_trainable_params()
        total_params = model.get_total_params()
        
        assert trainable_params > 0
        assert total_params > trainable_params  # Some params are frozen


class TestMobileNetV2Hybrid:
    """Tests for MobileNetV2Hybrid model."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = MobileNetV2Hybrid(num_classes=3, pretrained=False)
        assert model is not None
        assert isinstance(model, nn.Module)
    
    def test_forward_pass(self):
        """Test forward pass with correct input shape."""
        model = MobileNetV2Hybrid(num_classes=3, pretrained=False)
        model.eval()
        
        input_tensor = torch.randn(2, 3, 128, 128)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        assert output.shape == (2, 3)
    
    def test_different_num_classes(self):
        """Test model with different number of classes."""
        for num_classes in [2, 3, 5]:
            model = MobileNetV2Hybrid(num_classes=num_classes, pretrained=False)
            model.eval()
            
            input_tensor = torch.randn(1, 3, 128, 128)
            
            with torch.no_grad():
                output = model(input_tensor)
            
            assert output.shape == (1, num_classes)
    
    def test_freeze_backbone(self):
        """Test that freeze_backbone works correctly."""
        model = MobileNetV2Hybrid(num_classes=3, pretrained=False, freeze_backbone=True)
        
        # Check that backbone layers are frozen
        for param in model.backbone.parameters():
            assert not param.requires_grad
        
        # Check that QNN layers are trainable
        qnn_trainable = any(
            p.requires_grad for p in model.qnn.parameters()
        )
        assert qnn_trainable
    
    def test_qnn_component(self):
        """Test that QNN component is properly initialized."""
        model = MobileNetV2Hybrid(num_classes=3, pretrained=False)
        
        assert hasattr(model, 'qnn')
        assert hasattr(model, 'fc1')
        assert hasattr(model, 'fc2')
    
    def test_gradients(self):
        """Test that gradients can be computed for trainable parameters."""
        model = MobileNetV2Hybrid(num_classes=3, pretrained=False, freeze_backbone=True)
        
        input_tensor = torch.randn(1, 3, 128, 128)
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()
        
        # Check that QNN gradients were computed
        qnn_has_grads = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.qnn.parameters() if p.requires_grad
        )
        assert qnn_has_grads
    
    def test_trainable_parameters(self):
        """Test parameter counting methods."""
        model = MobileNetV2Hybrid(num_classes=3, pretrained=False, freeze_backbone=True)
        
        trainable_params = model.get_trainable_params()
        total_params = model.get_total_params()
        
        assert trainable_params > 0
        assert total_params > trainable_params  # Backbone is frozen


class TestGetModel:
    """Tests for the get_model factory function."""
    
    def test_baseline(self):
        """Test getting baseline model."""
        model = get_model('baseline', num_classes=3, pretrained=False)
        assert isinstance(model, MobileNetV2Baseline)
    
    def test_hybrid(self):
        """Test getting hybrid model."""
        model = get_model('hybrid', num_classes=3, pretrained=False)
        assert isinstance(model, MobileNetV2Hybrid)
    
    def test_case_insensitive(self):
        """Test that model names are case-insensitive."""
        model1 = get_model('Baseline', num_classes=3, pretrained=False)
        model2 = get_model('BASELINE', num_classes=3, pretrained=False)
        
        assert isinstance(model1, MobileNetV2Baseline)
        assert isinstance(model2, MobileNetV2Baseline)
    
    def test_invalid_model(self):
        """Test that invalid model name raises ValueError."""
        with pytest.raises(ValueError):
            get_model('invalid_model', num_classes=3, pretrained=False)


class TestModelComparison:
    """Tests comparing baseline and hybrid models."""
    
    def test_parameter_counts(self):
        """Test and compare parameter counts."""
        baseline = MobileNetV2Baseline(num_classes=3, pretrained=False, freeze_backbone=True)
        hybrid = MobileNetV2Hybrid(num_classes=3, pretrained=False, freeze_backbone=True)
        
        baseline_trainable = baseline.get_trainable_params()
        hybrid_trainable = hybrid.get_trainable_params()
        
        # Hybrid should have more trainable params due to QNN and FC layers
        assert hybrid_trainable > baseline_trainable
    
    def test_output_consistency(self):
        """Test that both models produce consistent output shapes."""
        batch_size = 2
        num_classes = 3
        input_tensor = torch.randn(batch_size, 3, 128, 128)
        
        baseline = MobileNetV2Baseline(num_classes=num_classes, pretrained=False)
        hybrid = MobileNetV2Hybrid(num_classes=num_classes, pretrained=False)
        
        baseline.eval()
        hybrid.eval()
        
        with torch.no_grad():
            baseline_output = baseline(input_tensor)
            hybrid_output = hybrid(input_tensor)
        
        assert baseline_output.shape == (batch_size, num_classes)
        assert hybrid_output.shape == (batch_size, num_classes)
        
        # Check no NaN or Inf values
        assert not torch.isnan(baseline_output).any()
        assert not torch.isnan(hybrid_output).any()
        assert not torch.isinf(baseline_output).any()
        assert not torch.isinf(hybrid_output).any()


class TestTrainingCompatibility:
    """Tests to ensure models are compatible with training pipeline."""
    
    def test_loss_computation_baseline(self):
        """Test that loss can be computed for baseline model."""
        model = MobileNetV2Baseline(num_classes=3, pretrained=False)
        criterion = nn.CrossEntropyLoss()
        
        input_tensor = torch.randn(2, 3, 128, 128)
        targets = torch.tensor([0, 1])
        
        outputs = model(input_tensor)
        loss = criterion(outputs, targets)
        
        assert loss.item() >= 0
        assert not torch.isnan(loss)
    
    def test_loss_computation_hybrid(self):
        """Test that loss can be computed for hybrid model."""
        model = MobileNetV2Hybrid(num_classes=3, pretrained=False)
        criterion = nn.CrossEntropyLoss()
        
        input_tensor = torch.randn(2, 3, 128, 128)
        targets = torch.tensor([0, 1])
        
        outputs = model(input_tensor)
        loss = criterion(outputs, targets)
        
        assert loss.item() >= 0
        assert not torch.isnan(loss)
    
    def test_optimizer_step_baseline(self):
        """Test that optimizer can update baseline model parameters."""
        model = MobileNetV2Baseline(num_classes=3, pretrained=False, freeze_backbone=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        input_tensor = torch.randn(2, 3, 128, 128)
        targets = torch.tensor([0, 1])
        
        # Get initial parameters from classifier
        initial_weight = model.mobilenet.classifier[1].weight.clone()
        
        # Training step
        optimizer.zero_grad()
        outputs = model(input_tensor)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # Check that classifier weights changed
        assert not torch.allclose(initial_weight, model.mobilenet.classifier[1].weight)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
