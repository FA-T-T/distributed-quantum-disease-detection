"""
Data Re-uploading Quantum Neural Network implementation.

This module implements:
1. Single-qubit QNN with data-reuploading for fitting trigonometric functions
2. Multi-qubit efficient data encoding layer
3. TorchLayer wrapper for integration with PyTorch
"""

import pennylane as qml
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List


# ============================================================================
# Single-Qubit Data Re-uploading QNN
# ============================================================================

def create_single_qubit_device(wires: int = 1):
    """Create a single qubit quantum device."""
    return qml.device('default.qubit', wires=wires)


def single_qubit_reuploading_circuit(x, weights, n_layers: int):
    """
    Single-qubit data re-uploading circuit for function fitting.
    
    Each layer consists of:
    - RY rotation with trainable weight
    - RZ rotation with trainable weight  
    - Data encoding: RY(x)
    
    Args:
        x: Input data (scalar)
        weights: Trainable weights, shape (n_layers, 2)
        n_layers: Number of layers
    
    Returns:
        Expectation value of Pauli-Z
    """
    for layer in range(n_layers):
        # Trainable rotations
        qml.RY(weights[layer, 0], wires=0)
        qml.RZ(weights[layer, 1], wires=0)
        # Data encoding (re-uploading)
        qml.RY(x, wires=0)
    
    return qml.expval(qml.PauliZ(0))


class SingleQubitReuploadingQNN(nn.Module):
    """
    Single-qubit QNN with data re-uploading for fitting trigonometric functions.
    
    This implements the universal approximation capability using a single qubit
    with multiple layers of data re-uploading.
    
    Args:
        n_layers: Number of re-uploading layers (default: 3)
    """
    
    def __init__(self, n_layers: int = 3):
        super().__init__()
        self.n_layers = n_layers
        
        # Create quantum device
        self.dev = create_single_qubit_device(wires=1)
        
        # Create QNode - PennyLane TorchLayer requires first arg named 'inputs'
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, weights):
            return single_qubit_reuploading_circuit(inputs, weights, n_layers)
        
        # Create TorchLayer with correct weight shapes
        weight_shapes = {"weights": (n_layers, 2)}
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the single-qubit re-uploading QNN.
        
        Args:
            x: Input tensor of shape (batch_size,) or (batch_size, 1)
        
        Returns:
            Output tensor of shape (batch_size,)
        """
        if x.dim() == 2:
            x = x.squeeze(-1)
        
        # Process each input
        outputs = []
        for xi in x:
            out = self.qlayer(xi)
            outputs.append(out)
        
        return torch.stack(outputs)


# ============================================================================
# Multi-Qubit Efficient Data Encoding
# ============================================================================

def create_multi_qubit_device(n_qubits: int):
    """Create a multi-qubit quantum device."""
    return qml.device('default.qubit', wires=n_qubits)


def efficient_encoding_block(x, weights, qubit: int, layer: int):
    """
    Efficient encoding block for a single qubit.
    
    Uses only 2 trainable parameters per qubit per layer:
    - One RY rotation (trainable)
    - One data-dependent RZ rotation
    
    Args:
        x: Input data for this qubit
        weights: Trainable weights for this layer
        qubit: Target qubit index
        layer: Layer index
    """
    # Trainable rotation
    qml.RY(weights[layer, qubit], wires=qubit)
    # Direct data encoding (without additional scaling)
    qml.RZ(x, wires=qubit)


def entangling_layer(n_qubits: int, pattern: str = "linear"):
    """
    Create entanglement between qubits.
    
    Args:
        n_qubits: Number of qubits
        pattern: Entanglement pattern ('linear', 'circular', 'all2all')
    """
    if pattern == "linear":
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
    elif pattern == "circular":
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])
    elif pattern == "all2all":
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                qml.CNOT(wires=[i, j])


class EfficientMultiQubitEncoding(nn.Module):
    """
    Efficient multi-qubit data encoding using data re-uploading.
    
    This encoding method is designed to be parameter-efficient while
    maintaining high accuracy. It uses:
    - One trainable RY per qubit per layer
    - Data-dependent RZ rotations
    - Linear entanglement (uses fewer gates than all-to-all)
    
    Total trainable parameters: n_layers * n_qubits (very efficient!)
    
    Args:
        n_qubits: Number of qubits (default: 4)
        n_layers: Number of encoding layers (default: 2)
        entangle: Whether to include entangling layers (default: True)
        entangle_pattern: Entanglement pattern ('linear', 'circular', 'all2all')
    """
    
    def __init__(
        self, 
        n_qubits: int = 4, 
        n_layers: int = 2,
        entangle: bool = True,
        entangle_pattern: str = "linear"
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.entangle = entangle
        self.entangle_pattern = entangle_pattern
        
        # Create quantum device
        self.dev = create_multi_qubit_device(n_qubits)
        
        # Create QNode for the encoding circuit
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, weights):
            return self._build_circuit(inputs, weights)
        
        # Weight shapes: (n_layers, n_qubits) for efficient encoding
        weight_shapes = {"weights": (n_layers, n_qubits)}
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
    
    def _build_circuit(self, inputs, weights):
        """
        Build the quantum circuit.
        
        Note: If inputs has fewer elements than n_qubits, data is reused
        cyclically across qubits (data re-uploading pattern).
        """
        for layer in range(self.n_layers):
            # Encode data into each qubit
            for qubit in range(self.n_qubits):
                # Cyclically reuse data if fewer inputs than qubits
                data_idx = qubit % len(inputs) if len(inputs) < self.n_qubits else qubit
                efficient_encoding_block(inputs[data_idx], weights, qubit, layer)
            
            # Apply entanglement (except after last layer)
            if self.entangle and layer < self.n_layers - 1:
                entangling_layer(self.n_qubits, self.entangle_pattern)
        
        # Return expectation values for all qubits
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the efficient multi-qubit encoding.
        
        Args:
            x: Input tensor of shape (batch_size, n_features)
        
        Returns:
            Output tensor of shape (batch_size, n_qubits)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        outputs = []
        for xi in x:
            out = self.qlayer(xi)
            # Stack the list of expectation values into a tensor
            if isinstance(out, (list, tuple)):
                out = torch.stack(out)
            outputs.append(out)
        
        return torch.stack(outputs)
    
    @property
    def num_trainable_params(self) -> int:
        """Return the number of trainable parameters."""
        return self.n_layers * self.n_qubits


# ============================================================================
# TorchLayer Wrapper for Data Encoding
# ============================================================================

class DataEncodingTorchLayer(nn.Module):
    """
    A TorchLayer wrapper that provides a flexible interface for data encoding.
    
    This layer can be easily integrated into any PyTorch model and provides:
    - Configurable number of qubits and layers
    - Optional classical pre/post-processing layers
    - Built-in normalization for input data
    
    Args:
        input_dim: Dimension of input features
        n_qubits: Number of qubits to use
        n_layers: Number of encoding layers
        output_dim: Optional output dimension (if None, outputs n_qubits)
        normalize_input: Whether to normalize input to [-π, π]
        entangle: Whether to use entanglement
        entangle_pattern: Type of entanglement pattern
    """
    
    def __init__(
        self,
        input_dim: int,
        n_qubits: int = 4,
        n_layers: int = 2,
        output_dim: Optional[int] = None,
        normalize_input: bool = True,
        entangle: bool = True,
        entangle_pattern: str = "linear"
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.output_dim = output_dim or n_qubits
        self.normalize_input = normalize_input
        
        # Classical preprocessing to map input_dim to n_qubits
        if input_dim != n_qubits:
            self.preprocess = nn.Linear(input_dim, n_qubits)
        else:
            self.preprocess = nn.Identity()
        
        # Quantum encoding layer
        self.encoding = EfficientMultiQubitEncoding(
            n_qubits=n_qubits,
            n_layers=n_layers,
            entangle=entangle,
            entangle_pattern=entangle_pattern
        )
        
        # Classical postprocessing
        if self.output_dim != n_qubits:
            self.postprocess = nn.Linear(n_qubits, self.output_dim)
        else:
            self.postprocess = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the data encoding layer.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Preprocess
        x = self.preprocess(x)
        
        # Normalize to [-π, π] if enabled
        if self.normalize_input:
            x = torch.tanh(x) * np.pi
        
        # Quantum encoding
        x = self.encoding(x)
        
        # Postprocess
        x = self.postprocess(x)
        
        return x
    
    @property
    def num_quantum_params(self) -> int:
        """Return the number of quantum trainable parameters."""
        return self.encoding.num_trainable_params
    
    @property  
    def total_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Utility Functions for Training and Testing
# ============================================================================

def generate_trigonometric_data(
    n_samples: int = 100,
    function: str = "sin",
    noise_std: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate training data for fitting trigonometric functions.
    
    Args:
        n_samples: Number of samples
        function: Trigonometric function ('sin', 'cos', 'combined')
        noise_std: Standard deviation of Gaussian noise
    
    Returns:
        Tuple of (x_data, y_data) tensors
    """
    x = torch.linspace(-np.pi, np.pi, n_samples)
    
    if function == "sin":
        y = torch.sin(x)
    elif function == "cos":
        y = torch.cos(x)
    elif function == "combined":
        y = 0.5 * torch.sin(x) + 0.5 * torch.cos(2 * x)
    else:
        raise ValueError(f"Unknown function: {function}")
    
    if noise_std > 0:
        y = y + torch.randn_like(y) * noise_std
    
    return x, y


def train_single_qubit_qnn(
    model: SingleQubitReuploadingQNN,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int = 100,
    lr: float = 0.1,
    verbose: bool = True
) -> List[float]:
    """
    Train a single-qubit QNN to fit data.
    
    Args:
        model: The QNN model
        x_train: Training input data
        y_train: Training target data
        epochs: Number of training epochs
        lr: Learning rate
        verbose: Whether to print training progress
    
    Returns:
        List of loss values during training
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        predictions = model(x_train)
        loss = criterion(predictions, y_train)
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if verbose and (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")
    
    return losses


def benchmark_encoding_efficiency(
    n_qubits_list: List[int] = [2, 4, 6, 8],
    n_layers_list: List[int] = [1, 2, 3],
    n_samples: int = 10
) -> dict:
    """
    Benchmark the efficiency of the encoding method.
    
    Args:
        n_qubits_list: List of qubit counts to test
        n_layers_list: List of layer counts to test
        n_samples: Number of samples for timing
    
    Returns:
        Dictionary with benchmark results
    """
    import time
    
    results = {}
    
    for n_qubits in n_qubits_list:
        for n_layers in n_layers_list:
            key = f"qubits={n_qubits}, layers={n_layers}"
            
            # Create model
            model = EfficientMultiQubitEncoding(
                n_qubits=n_qubits,
                n_layers=n_layers
            )
            
            # Generate random input
            x = torch.randn(n_samples, n_qubits)
            
            # Time forward pass
            start_time = time.time()
            with torch.no_grad():
                _ = model(x)
            end_time = time.time()
            
            results[key] = {
                "n_qubits": n_qubits,
                "n_layers": n_layers,
                "num_params": model.num_trainable_params,
                "time_per_sample": (end_time - start_time) / n_samples,
            }
    
    return results


if __name__ == "__main__":
    # Demo: Fit a sine function with single-qubit QNN
    print("=" * 60)
    print("Single-Qubit Data Re-uploading QNN Demo")
    print("=" * 60)
    
    # Generate data
    x_train, y_train = generate_trigonometric_data(n_samples=50, function="sin")
    
    # Create and train model
    model = SingleQubitReuploadingQNN(n_layers=3)
    print(f"\nNumber of trainable parameters: {sum(p.numel() for p in model.parameters())}")
    
    losses = train_single_qubit_qnn(model, x_train, y_train, epochs=100, lr=0.1)
    
    print(f"\nFinal loss: {losses[-1]:.6f}")
    
    # Demo: Multi-qubit encoding
    print("\n" + "=" * 60)
    print("Multi-Qubit Efficient Encoding Demo")
    print("=" * 60)
    
    encoding = EfficientMultiQubitEncoding(n_qubits=4, n_layers=2)
    print(f"\nNumber of trainable parameters: {encoding.num_trainable_params}")
    
    # Test with random data
    x_test = torch.randn(5, 4)
    with torch.no_grad():
        output = encoding(x_test)
    print(f"Input shape: {x_test.shape}")
    print(f"Output shape: {output.shape}")
    
    # Demo: TorchLayer wrapper
    print("\n" + "=" * 60)
    print("DataEncodingTorchLayer Demo")
    print("=" * 60)
    
    layer = DataEncodingTorchLayer(
        input_dim=8,
        n_qubits=4,
        n_layers=2,
        output_dim=3
    )
    print(f"\nQuantum parameters: {layer.num_quantum_params}")
    print(f"Total parameters: {layer.total_params}")
    
    x_test = torch.randn(5, 8)
    with torch.no_grad():
        output = layer(x_test)
    print(f"Input shape: {x_test.shape}")
    print(f"Output shape: {output.shape}")
