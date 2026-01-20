"""
Unit tests for classical backbone models.
"""

import os
import sys
import pytest
import torch
import torch.nn as nn

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'code'))

from classical_backbone import (
    ClassicalCNN,
    VGGClassifier,
    ResNetClassifier,
    LightweightResNet,
    get_model
)


class TestClassicalCNN:
    """Tests for ClassicalCNN (MobileNetV2) model."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = ClassicalCNN(num_classes=3, pretrained=False)
        assert model is not None
        assert isinstance(model, nn.Module)
    
    def test_forward_pass(self):
        """Test forward pass with correct input shape."""
        model = ClassicalCNN(num_classes=3, pretrained=False)
        model.eval()
        
        input_tensor = torch.randn(2, 3, 128, 128)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        assert output.shape == (2, 3)
    
    def test_different_num_classes(self):
        """Test model with different number of classes."""
        for num_classes in [2, 3, 5, 10]:
            model = ClassicalCNN(num_classes=num_classes, pretrained=False)
            model.eval()
            
            input_tensor = torch.randn(1, 3, 128, 128)
            
            with torch.no_grad():
                output = model(input_tensor)
            
            assert output.shape == (1, num_classes)
    
    def test_batch_sizes(self):
        """Test model with different batch sizes."""
        model = ClassicalCNN(num_classes=3, pretrained=False)
        model.eval()
        
        for batch_size in [1, 2, 4, 8]:
            input_tensor = torch.randn(batch_size, 3, 128, 128)
            
            with torch.no_grad():
                output = model(input_tensor)
            
            assert output.shape == (batch_size, 3)
    
    def test_gradients(self):
        """Test that gradients can be computed."""
        model = ClassicalCNN(num_classes=3, pretrained=False)
        
        input_tensor = torch.randn(2, 3, 128, 128, requires_grad=True)
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()
        
        assert input_tensor.grad is not None
        assert any(p.grad is not None for p in model.parameters() if p.requires_grad)
    
    def test_trainable_parameters(self):
        """Test that model has trainable parameters."""
        model = ClassicalCNN(num_classes=3, pretrained=False)
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable_params > 0


class TestVGGClassifier:
    """Tests for VGGClassifier model."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = VGGClassifier(num_classes=3, pretrained=False)
        assert model is not None
        assert isinstance(model, nn.Module)
    
    def test_forward_pass(self):
        """Test forward pass with correct input shape."""
        model = VGGClassifier(num_classes=3, pretrained=False)
        model.eval()
        
        input_tensor = torch.randn(2, 3, 128, 128)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        assert output.shape == (2, 3)
    
    def test_different_num_classes(self):
        """Test model with different number of classes."""
        for num_classes in [2, 3, 5]:
            model = VGGClassifier(num_classes=num_classes, pretrained=False)
            model.eval()
            
            input_tensor = torch.randn(1, 3, 128, 128)
            
            with torch.no_grad():
                output = model(input_tensor)
            
            assert output.shape == (1, num_classes)
    
    def test_freeze_features(self):
        """Test that freeze_features works correctly."""
        model = VGGClassifier(num_classes=3, pretrained=False, freeze_features=True)
        
        # Check that feature layers are frozen
        for param in model.vgg.features.parameters():
            assert not param.requires_grad
        
        # Check that classifier layers are trainable
        classifier_trainable = any(
            p.requires_grad for p in model.vgg.classifier.parameters()
        )
        assert classifier_trainable
    
    def test_gradients(self):
        """Test that gradients can be computed."""
        model = VGGClassifier(num_classes=3, pretrained=False)
        
        input_tensor = torch.randn(2, 3, 128, 128, requires_grad=True)
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()
        
        assert input_tensor.grad is not None
        assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


class TestResNetClassifier:
    """Tests for ResNetClassifier (ResNet50) model."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = ResNetClassifier(num_classes=3, pretrained=False)
        assert model is not None
        assert isinstance(model, nn.Module)
    
    def test_forward_pass(self):
        """Test forward pass with correct input shape."""
        model = ResNetClassifier(num_classes=3, pretrained=False)
        model.eval()
        
        input_tensor = torch.randn(2, 3, 128, 128)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        assert output.shape == (2, 3)
    
    def test_different_num_classes(self):
        """Test model with different number of classes."""
        for num_classes in [2, 3, 5]:
            model = ResNetClassifier(num_classes=num_classes, pretrained=False)
            model.eval()
            
            input_tensor = torch.randn(1, 3, 128, 128)
            
            with torch.no_grad():
                output = model(input_tensor)
            
            assert output.shape == (1, num_classes)
    
    def test_freeze_features(self):
        """Test that freeze_features works correctly."""
        model = ResNetClassifier(num_classes=3, pretrained=False, freeze_features=True)
        
        # Check that early layers are frozen
        frozen_layers = ['conv1', 'bn1', 'layer1', 'layer2']
        for name, param in model.resnet.named_parameters():
            if any(layer in name for layer in frozen_layers):
                assert not param.requires_grad, f"Parameter {name} should be frozen"
        
        # Check that later layers are trainable
        trainable_found = False
        for name, param in model.resnet.named_parameters():
            if 'layer3' in name or 'layer4' in name or 'fc' in name:
                if param.requires_grad:
                    trainable_found = True
                    break
        assert trainable_found
    
    def test_gradients(self):
        """Test that gradients can be computed."""
        model = ResNetClassifier(num_classes=3, pretrained=False)
        
        input_tensor = torch.randn(2, 3, 128, 128, requires_grad=True)
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()
        
        assert input_tensor.grad is not None
        assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


class TestLightweightResNet:
    """Tests for LightweightResNet (ResNet18) model."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = LightweightResNet(num_classes=3, pretrained=False)
        assert model is not None
        assert isinstance(model, nn.Module)
    
    def test_forward_pass(self):
        """Test forward pass with correct input shape."""
        model = LightweightResNet(num_classes=3, pretrained=False)
        model.eval()
        
        input_tensor = torch.randn(2, 3, 128, 128)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        assert output.shape == (2, 3)
    
    def test_different_num_classes(self):
        """Test model with different number of classes."""
        for num_classes in [2, 3, 5]:
            model = LightweightResNet(num_classes=num_classes, pretrained=False)
            model.eval()
            
            input_tensor = torch.randn(1, 3, 128, 128)
            
            with torch.no_grad():
                output = model(input_tensor)
            
            assert output.shape == (1, num_classes)
    
    def test_smaller_than_resnet50(self):
        """Test that ResNet18 has fewer parameters than ResNet50."""
        resnet18 = LightweightResNet(num_classes=3, pretrained=False)
        resnet50 = ResNetClassifier(num_classes=3, pretrained=False)
        
        params_18 = sum(p.numel() for p in resnet18.parameters())
        params_50 = sum(p.numel() for p in resnet50.parameters())
        
        assert params_18 < params_50


class TestGetModel:
    """Tests for the get_model factory function."""
    
    def test_mobilenet(self):
        """Test getting MobileNet model."""
        model = get_model('mobilenet', num_classes=3, pretrained=False)
        assert isinstance(model, ClassicalCNN)
    
    def test_vgg16(self):
        """Test getting VGG16 model."""
        model = get_model('vgg16', num_classes=3, pretrained=False)
        assert isinstance(model, VGGClassifier)
    
    def test_resnet50(self):
        """Test getting ResNet50 model."""
        model = get_model('resnet50', num_classes=3, pretrained=False)
        assert isinstance(model, ResNetClassifier)
    
    def test_resnet18(self):
        """Test getting ResNet18 model."""
        model = get_model('resnet18', num_classes=3, pretrained=False)
        assert isinstance(model, LightweightResNet)
    
    def test_case_insensitive(self):
        """Test that model names are case-insensitive."""
        model1 = get_model('MobileNet', num_classes=3, pretrained=False)
        model2 = get_model('MOBILENET', num_classes=3, pretrained=False)
        
        assert isinstance(model1, ClassicalCNN)
        assert isinstance(model2, ClassicalCNN)
    
    def test_invalid_model(self):
        """Test that invalid model name raises ValueError."""
        with pytest.raises(ValueError):
            get_model('invalid_model', num_classes=3, pretrained=False)
    
    def test_with_kwargs(self):
        """Test passing additional kwargs."""
        model = get_model(
            'vgg16', 
            num_classes=3, 
            pretrained=False, 
            freeze_features=True
        )
        
        assert isinstance(model, VGGClassifier)
        
        # Verify freeze_features was applied
        for param in model.vgg.features.parameters():
            assert not param.requires_grad


class TestModelComparison:
    """Tests comparing different model architectures."""
    
    def test_parameter_counts(self):
        """Test and compare parameter counts of different models."""
        models_dict = {
            'MobileNetV2': ClassicalCNN(num_classes=3, pretrained=False),
            'VGG16': VGGClassifier(num_classes=3, pretrained=False),
            'ResNet50': ResNetClassifier(num_classes=3, pretrained=False),
            'ResNet18': LightweightResNet(num_classes=3, pretrained=False)
        }
        
        param_counts = {}
        for name, model in models_dict.items():
            params = sum(p.numel() for p in model.parameters())
            param_counts[name] = params
            assert params > 0, f"{name} should have parameters"
        
        # Verify expected relationships
        assert param_counts['ResNet18'] < param_counts['ResNet50']
        assert param_counts['MobileNetV2'] < param_counts['VGG16']
    
    def test_output_consistency(self):
        """Test that all models produce consistent output shapes."""
        batch_size = 4
        num_classes = 3
        input_tensor = torch.randn(batch_size, 3, 128, 128)
        
        models_list = [
            ClassicalCNN(num_classes=num_classes, pretrained=False),
            VGGClassifier(num_classes=num_classes, pretrained=False),
            ResNetClassifier(num_classes=num_classes, pretrained=False),
            LightweightResNet(num_classes=num_classes, pretrained=False)
        ]
        
        for model in models_list:
            model.eval()
            with torch.no_grad():
                output = model(input_tensor)
            
            assert output.shape == (batch_size, num_classes)
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()


class TestTrainingCompatibility:
    """Tests to ensure models are compatible with training pipeline."""
    
    def test_loss_computation(self):
        """Test that loss can be computed for all models."""
        models_list = [
            ClassicalCNN(num_classes=3, pretrained=False),
            VGGClassifier(num_classes=3, pretrained=False),
            ResNetClassifier(num_classes=3, pretrained=False),
            LightweightResNet(num_classes=3, pretrained=False)
        ]
        
        criterion = nn.CrossEntropyLoss()
        input_tensor = torch.randn(2, 3, 128, 128)
        targets = torch.tensor([0, 1])
        
        for model in models_list:
            outputs = model(input_tensor)
            loss = criterion(outputs, targets)
            
            assert loss.item() >= 0
            assert not torch.isnan(loss)
    
    def test_backward_pass(self):
        """Test that backward pass works for all models."""
        models_list = [
            ClassicalCNN(num_classes=3, pretrained=False),
            VGGClassifier(num_classes=3, pretrained=False),
            ResNetClassifier(num_classes=3, pretrained=False),
            LightweightResNet(num_classes=3, pretrained=False)
        ]
        
        criterion = nn.CrossEntropyLoss()
        input_tensor = torch.randn(2, 3, 128, 128)
        targets = torch.tensor([0, 1])
        
        for model in models_list:
            model.train()
            outputs = model(input_tensor)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Check that gradients were computed
            gradients_computed = any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in model.parameters() if p.requires_grad
            )
            assert gradients_computed
    
    def test_optimizer_step(self):
        """Test that optimizer can update model parameters."""
        model = ClassicalCNN(num_classes=3, pretrained=False)
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
            if not torch.allclose(initial, current):
                params_changed = True
                break
        
        assert params_changed


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
