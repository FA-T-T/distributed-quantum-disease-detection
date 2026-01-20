"""
Unit tests for the data re-uploading QNN module.
"""

import os
import sys
import pytest
import torch
import torch.nn as nn
import numpy as np

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'code'))

from ..code.data_reuploading import (
    SingleQubitReuploadingQNN,
    EfficientMultiQubitEncoding,
    DataEncodingTorchLayer,
    generate_trigonometric_data,
    train_single_qubit_qnn,
    benchmark_encoding_efficiency,
)


class TestSingleQubitReuploadingQNN:
    """Tests for single-qubit data re-uploading QNN."""
    
    def test_initialization(self):
        """Test model initialization with default parameters."""
        model = SingleQubitReuploadingQNN()
        assert model is not None
        assert isinstance(model, nn.Module)
        assert model.n_layers == 3
    
    def test_initialization_custom_layers(self):
        """Test model initialization with custom number of layers."""
        for n_layers in [1, 2, 5]:
            model = SingleQubitReuploadingQNN(n_layers=n_layers)
            assert model.n_layers == n_layers
    
    def test_forward_pass_1d(self):
        """Test forward pass with 1D input."""
        model = SingleQubitReuploadingQNN(n_layers=2)
        model.eval()
        
        x = torch.tensor([0.0, 0.5, 1.0])
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (3,)
        # Output should be expectation values in [-1, 1]
        assert torch.all(output >= -1) and torch.all(output <= 1)
    
    def test_forward_pass_2d(self):
        """Test forward pass with 2D input."""
        model = SingleQubitReuploadingQNN(n_layers=2)
        model.eval()
        
        x = torch.tensor([[0.0], [0.5], [1.0]])
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (3,)
    
    def test_gradient_computation(self):
        """Test that gradients can be computed."""
        model = SingleQubitReuploadingQNN(n_layers=2)
        
        x = torch.tensor([0.5, 1.0], requires_grad=False)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        # Check that model parameters have gradients
        for param in model.parameters():
            assert param.grad is not None
    
    def test_trainable_parameters(self):
        """Test number of trainable parameters."""
        n_layers = 3
        model = SingleQubitReuploadingQNN(n_layers=n_layers)
        
        # Should have n_layers * 2 parameters (RY and RZ per layer)
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params == n_layers * 2


class TestEfficientMultiQubitEncoding:
    """Tests for efficient multi-qubit encoding."""
    
    def test_initialization(self):
        """Test model initialization with default parameters."""
        model = EfficientMultiQubitEncoding()
        assert model is not None
        assert isinstance(model, nn.Module)
        assert model.n_qubits == 4
        assert model.n_layers == 2
    
    def test_initialization_custom_params(self):
        """Test model initialization with custom parameters."""
        model = EfficientMultiQubitEncoding(n_qubits=6, n_layers=3)
        assert model.n_qubits == 6
        assert model.n_layers == 3
    
    def test_forward_pass(self):
        """Test forward pass with correct input shape."""
        n_qubits = 4
        model = EfficientMultiQubitEncoding(n_qubits=n_qubits, n_layers=2)
        model.eval()
        
        batch_size = 3
        x = torch.randn(batch_size, n_qubits)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (batch_size, n_qubits)
        # Output should be expectation values in [-1, 1]
        assert torch.all(output >= -1) and torch.all(output <= 1)
    
    def test_forward_pass_1d(self):
        """Test forward pass with 1D input (single sample)."""
        n_qubits = 4
        model = EfficientMultiQubitEncoding(n_qubits=n_qubits, n_layers=2)
        model.eval()
        
        x = torch.randn(n_qubits)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (1, n_qubits)
    
    def test_different_batch_sizes(self):
        """Test with different batch sizes."""
        n_qubits = 4
        model = EfficientMultiQubitEncoding(n_qubits=n_qubits)
        model.eval()
        
        for batch_size in [1, 2, 5]:
            x = torch.randn(batch_size, n_qubits)
            
            with torch.no_grad():
                output = model(x)
            
            assert output.shape == (batch_size, n_qubits)
    
    def test_entanglement_patterns(self):
        """Test different entanglement patterns."""
        for pattern in ["linear", "circular"]:
            model = EfficientMultiQubitEncoding(
                n_qubits=4, 
                n_layers=2,
                entangle=True,
                entangle_pattern=pattern
            )
            model.eval()
            
            x = torch.randn(2, 4)
            
            with torch.no_grad():
                output = model(x)
            
            assert output.shape == (2, 4)
    
    def test_no_entanglement(self):
        """Test encoding without entanglement."""
        model = EfficientMultiQubitEncoding(
            n_qubits=4,
            n_layers=2,
            entangle=False
        )
        model.eval()
        
        x = torch.randn(2, 4)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (2, 4)
    
    def test_num_trainable_params(self):
        """Test efficient parameter count."""
        n_qubits = 4
        n_layers = 2
        model = EfficientMultiQubitEncoding(n_qubits=n_qubits, n_layers=n_layers)
        
        # Should have exactly n_layers * n_qubits parameters
        assert model.num_trainable_params == n_qubits * n_layers
    
    def test_gradient_computation(self):
        """Test that gradients can be computed."""
        model = EfficientMultiQubitEncoding(n_qubits=4, n_layers=2)
        
        x = torch.randn(2, 4)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        for param in model.parameters():
            assert param.grad is not None


class TestDataEncodingTorchLayer:
    """Tests for the TorchLayer wrapper."""
    
    def test_initialization(self):
        """Test model initialization."""
        layer = DataEncodingTorchLayer(input_dim=8, n_qubits=4)
        assert layer is not None
        assert isinstance(layer, nn.Module)
    
    def test_forward_pass(self):
        """Test forward pass."""
        layer = DataEncodingTorchLayer(
            input_dim=8,
            n_qubits=4,
            n_layers=2
        )
        layer.eval()
        
        x = torch.randn(3, 8)
        
        with torch.no_grad():
            output = layer(x)
        
        assert output.shape == (3, 4)  # Default output_dim = n_qubits
    
    def test_custom_output_dim(self):
        """Test with custom output dimension."""
        layer = DataEncodingTorchLayer(
            input_dim=8,
            n_qubits=4,
            output_dim=3
        )
        layer.eval()
        
        x = torch.randn(2, 8)
        
        with torch.no_grad():
            output = layer(x)
        
        assert output.shape == (2, 3)
    
    def test_same_input_output_dim(self):
        """Test when input_dim equals n_qubits."""
        layer = DataEncodingTorchLayer(
            input_dim=4,
            n_qubits=4
        )
        layer.eval()
        
        x = torch.randn(2, 4)
        
        with torch.no_grad():
            output = layer(x)
        
        assert output.shape == (2, 4)
    
    def test_normalization_disabled(self):
        """Test with input normalization disabled."""
        layer = DataEncodingTorchLayer(
            input_dim=4,
            n_qubits=4,
            normalize_input=False
        )
        layer.eval()
        
        x = torch.randn(2, 4)
        
        with torch.no_grad():
            output = layer(x)
        
        assert output.shape == (2, 4)
    
    def test_quantum_params_count(self):
        """Test quantum parameter count."""
        layer = DataEncodingTorchLayer(
            input_dim=8,
            n_qubits=4,
            n_layers=2
        )
        
        assert layer.num_quantum_params == 4 * 2  # n_qubits * n_layers
    
    def test_total_params_count(self):
        """Test total parameter count."""
        layer = DataEncodingTorchLayer(
            input_dim=8,
            n_qubits=4,
            n_layers=2,
            output_dim=3
        )
        
        # Quantum params + preprocessing (8*4 + 4 bias) + postprocessing (4*3 + 3 bias)
        # = 8 + 32 + 4 + 12 + 3 = 59
        total = layer.total_params
        assert total > layer.num_quantum_params
    
    def test_gradient_computation(self):
        """Test that gradients flow through the layer."""
        layer = DataEncodingTorchLayer(
            input_dim=8,
            n_qubits=4,
            output_dim=3
        )
        
        x = torch.randn(2, 8, requires_grad=True)
        output = layer(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None


class TestDataGeneration:
    """Tests for data generation utilities."""
    
    def test_generate_sin_data(self):
        """Test sine data generation."""
        x, y = generate_trigonometric_data(n_samples=50, function="sin")
        
        assert x.shape == (50,)
        assert y.shape == (50,)
        assert torch.allclose(y, torch.sin(x), atol=1e-6)
    
    def test_generate_cos_data(self):
        """Test cosine data generation."""
        x, y = generate_trigonometric_data(n_samples=50, function="cos")
        
        assert x.shape == (50,)
        assert y.shape == (50,)
        assert torch.allclose(y, torch.cos(x), atol=1e-6)
    
    def test_generate_combined_data(self):
        """Test combined function data generation."""
        x, y = generate_trigonometric_data(n_samples=50, function="combined")
        
        assert x.shape == (50,)
        assert y.shape == (50,)
    
    def test_generate_noisy_data(self):
        """Test data generation with noise."""
        x, y = generate_trigonometric_data(n_samples=100, function="sin", noise_std=0.1)
        
        assert x.shape == (100,)
        assert y.shape == (100,)
        # Should not be exactly equal to sin due to noise
        assert not torch.allclose(y, torch.sin(x), atol=1e-6)
    
    def test_invalid_function(self):
        """Test that invalid function raises error."""
        with pytest.raises(ValueError):
            generate_trigonometric_data(function="invalid")


class TestTraining:
    """Tests for training utilities."""
    
    def test_train_single_qubit_qnn(self):
        """Test training function."""
        model = SingleQubitReuploadingQNN(n_layers=2)
        x_train, y_train = generate_trigonometric_data(n_samples=20, function="sin")
        
        losses = train_single_qubit_qnn(
            model, x_train, y_train, 
            epochs=10, lr=0.1, verbose=False
        )
        
        assert len(losses) == 10
        # Loss should generally decrease (allowing for some variation)
        assert losses[-1] <= losses[0] * 2  # Not strictly decreasing but shouldn't explode


class TestBenchmarking:
    """Tests for benchmarking utilities."""
    
    def test_benchmark_encoding_efficiency(self):
        """Test benchmark function returns expected results."""
        results = benchmark_encoding_efficiency(
            n_qubits_list=[2],
            n_layers_list=[1],
            n_samples=2
        )
        
        assert len(results) == 1
        key = "qubits=2, layers=1"
        assert key in results
        assert "n_qubits" in results[key]
        assert "n_layers" in results[key]
        assert "num_params" in results[key]
        assert "time_per_sample" in results[key]


class TestIntegration:
    """Integration tests for the data re-uploading module."""
    
    def test_full_training_pipeline(self):
        """Test complete training pipeline for single-qubit QNN."""
        # Generate data
        x_train, y_train = generate_trigonometric_data(n_samples=30, function="sin")
        
        # Create model
        model = SingleQubitReuploadingQNN(n_layers=3)
        
        # Train
        losses = train_single_qubit_qnn(
            model, x_train, y_train,
            epochs=50, lr=0.1, verbose=False
        )
        
        # Verify training improved the model
        assert losses[-1] < losses[0]
        
        # Test inference
        model.eval()
        with torch.no_grad():
            predictions = model(x_train)
        
        # Should have reasonable predictions
        assert predictions.shape == y_train.shape
    
    def test_encoding_layer_in_sequential(self):
        """Test that encoding layer works in nn.Sequential."""
        model = nn.Sequential(
            nn.Linear(10, 4),
            DataEncodingTorchLayer(
                input_dim=4,
                n_qubits=4,
                n_layers=1,
                output_dim=3
            )
        )
        
        x = torch.randn(2, 10)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (2, 3)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
