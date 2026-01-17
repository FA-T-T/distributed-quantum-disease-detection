"""
Validation module for the Quantum-Classical hybrid model.

This module provides functions to validate model inputs, outputs,
tensor dimensions, and data integrity.
"""

import os
import sys
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
import numpy as np

# Add code directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def validate_image_tensor(
    tensor: torch.Tensor,
    expected_channels: int = 3,
    expected_size: Optional[Tuple[int, int]] = None
) -> Tuple[bool, str]:
    """
    Validate an image tensor.
    
    Args:
        tensor: The image tensor to validate.
        expected_channels: Expected number of channels.
        expected_size: Optional expected (height, width).
        
    Returns:
        Tuple of (is_valid, message).
    """
    # Check tensor type
    if not isinstance(tensor, torch.Tensor):
        return False, f"Expected torch.Tensor, got {type(tensor)}"
    
    # Check dimensions (should be 3D or 4D)
    if tensor.dim() not in [3, 4]:
        return False, f"Expected 3D or 4D tensor, got {tensor.dim()}D"
    
    # Get channel dimension
    if tensor.dim() == 3:
        channels, height, width = tensor.shape
    else:  # 4D
        batch, channels, height, width = tensor.shape
    
    # Check channels
    if channels != expected_channels:
        return False, f"Expected {expected_channels} channels, got {channels}"
    
    # Check size if specified
    if expected_size is not None:
        if (height, width) != expected_size:
            return False, f"Expected size {expected_size}, got ({height}, {width})"
    
    # Check for NaN or Inf
    if torch.isnan(tensor).any():
        return False, "Tensor contains NaN values"
    if torch.isinf(tensor).any():
        return False, "Tensor contains Inf values"
    
    return True, "Valid image tensor"


def validate_model_input(
    tensor: torch.Tensor,
    batch_size: Optional[int] = None,
    image_size: int = 128
) -> Tuple[bool, str]:
    """
    Validate input tensor for the QCNet model.
    
    Args:
        tensor: Input tensor to validate.
        batch_size: Optional expected batch size.
        image_size: Expected image size.
        
    Returns:
        Tuple of (is_valid, message).
    """
    # Check dimensions
    if tensor.dim() != 4:
        return False, f"Expected 4D tensor (batch, channels, height, width), got {tensor.dim()}D"
    
    batch, channels, height, width = tensor.shape
    
    # Check batch size
    if batch_size is not None and batch != batch_size:
        return False, f"Expected batch size {batch_size}, got {batch}"
    
    # Check channels
    if channels != 3:
        return False, f"Expected 3 channels (RGB), got {channels}"
    
    # Check image size
    if height != image_size or width != image_size:
        return False, f"Expected image size {image_size}x{image_size}, got {height}x{width}"
    
    return True, "Valid model input"


def validate_model_output(
    tensor: torch.Tensor,
    num_classes: int = 3,
    batch_size: Optional[int] = None
) -> Tuple[bool, str]:
    """
    Validate output tensor from the QCNet model.
    
    Args:
        tensor: Output tensor to validate.
        num_classes: Expected number of classes.
        batch_size: Optional expected batch size.
        
    Returns:
        Tuple of (is_valid, message).
    """
    # Check dimensions
    if tensor.dim() != 2:
        return False, f"Expected 2D tensor (batch, classes), got {tensor.dim()}D"
    
    batch, classes = tensor.shape
    
    # Check batch size
    if batch_size is not None and batch != batch_size:
        return False, f"Expected batch size {batch_size}, got {batch}"
    
    # Check number of classes
    if classes != num_classes:
        return False, f"Expected {num_classes} classes, got {classes}"
    
    # Check for NaN or Inf
    if torch.isnan(tensor).any():
        return False, "Output contains NaN values"
    if torch.isinf(tensor).any():
        return False, "Output contains Inf values"
    
    return True, "Valid model output"


def validate_forward_pass(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 128, 128),
    expected_output_classes: int = 3,
    device: Optional[torch.device] = None
) -> Tuple[bool, str, Optional[torch.Tensor]]:
    """
    Validate a complete forward pass through the model.
    
    Args:
        model: The model to validate.
        input_shape: Input tensor shape.
        expected_output_classes: Expected number of output classes.
        device: Device to use.
        
    Returns:
        Tuple of (is_valid, message, output_tensor).
    """
    if device is None:
        device = torch.device('cpu')
    
    try:
        # Move model to device
        model = model.to(device)
        model.eval()
        
        # Create input tensor
        input_tensor = torch.randn(*input_shape).to(device)
        
        # Validate input
        is_valid, msg = validate_model_input(input_tensor)
        if not is_valid:
            return False, f"Invalid input: {msg}", None
        
        # Forward pass
        with torch.no_grad():
            output = model(input_tensor)
        
        # Validate output
        is_valid, msg = validate_model_output(
            output,
            num_classes=expected_output_classes,
            batch_size=input_shape[0]
        )
        if not is_valid:
            return False, f"Invalid output: {msg}", None
        
        return True, "Forward pass successful", output
        
    except Exception as e:
        return False, f"Forward pass failed: {str(e)}", None


def validate_labels(
    labels: torch.Tensor,
    num_classes: int = 3
) -> Tuple[bool, str]:
    """
    Validate label tensor.
    
    Args:
        labels: Label tensor to validate.
        num_classes: Number of classes.
        
    Returns:
        Tuple of (is_valid, message).
    """
    # Check dimensions
    if labels.dim() != 1:
        return False, f"Expected 1D tensor, got {labels.dim()}D"
    
    # Check label range
    min_label = labels.min().item()
    max_label = labels.max().item()
    
    if min_label < 0:
        return False, f"Labels should be non-negative, got minimum {min_label}"
    
    if max_label >= num_classes:
        return False, f"Labels should be < {num_classes}, got maximum {max_label}"
    
    return True, "Valid labels"


def validate_dataset(
    dataset,
    num_classes: int = 3,
    image_size: int = 128
) -> Dict[str, Any]:
    """
    Validate a dataset.
    
    Args:
        dataset: The dataset to validate.
        num_classes: Expected number of classes.
        image_size: Expected image size.
        
    Returns:
        Dictionary with validation results.
    """
    results = {
        'is_valid': True,
        'total_samples': len(dataset),
        'valid_samples': 0,
        'invalid_samples': 0,
        'errors': []
    }
    
    # Check a sample of the dataset
    check_count = min(10, len(dataset))
    
    for i in range(check_count):
        try:
            image, label = dataset[i]
            
            # Validate image
            is_valid, msg = validate_image_tensor(
                image,
                expected_channels=3,
                expected_size=(image_size, image_size)
            )
            
            if not is_valid:
                results['invalid_samples'] += 1
                results['errors'].append(f"Sample {i}: {msg}")
            else:
                # Validate label
                if label < 0 or label >= num_classes:
                    results['invalid_samples'] += 1
                    results['errors'].append(f"Sample {i}: Invalid label {label}")
                else:
                    results['valid_samples'] += 1
                    
        except Exception as e:
            results['invalid_samples'] += 1
            results['errors'].append(f"Sample {i}: {str(e)}")
    
    if results['invalid_samples'] > 0:
        results['is_valid'] = False
    
    return results


def run_all_validations(
    model: nn.Module,
    device: Optional[torch.device] = None,
    verbose: bool = True
) -> bool:
    """
    Run all validation checks.
    
    Args:
        model: The model to validate.
        device: Device to use.
        verbose: Whether to print results.
        
    Returns:
        True if all validations pass.
    """
    if device is None:
        device = torch.device('cpu')
    
    all_passed = True
    
    if verbose:
        print("\n" + "=" * 50)
        print("Running Model Validations")
        print("=" * 50)
    
    # Test different batch sizes
    test_cases = [
        (1, 3, 128, 128),
        (2, 3, 128, 128),
        (4, 3, 128, 128),
    ]
    
    for input_shape in test_cases:
        is_valid, msg, _ = validate_forward_pass(
            model, input_shape, expected_output_classes=3, device=device
        )
        
        if verbose:
            status = "✓" if is_valid else "✗"
            print(f"{status} Input shape {input_shape}: {msg}")
        
        if not is_valid:
            all_passed = False
    
    if verbose:
        print("=" * 50)
        if all_passed:
            print("All validations passed!")
        else:
            print("Some validations failed!")
    
    return all_passed


if __name__ == '__main__':
    print("Testing validation module...")
    
    # Test tensor validations
    valid_tensor = torch.randn(2, 3, 128, 128)
    is_valid, msg = validate_model_input(valid_tensor)
    print(f"Valid input test: {is_valid} - {msg}")
    
    invalid_tensor = torch.randn(2, 1, 128, 128)  # Wrong channels
    is_valid, msg = validate_model_input(invalid_tensor)
    print(f"Invalid input test: {is_valid} - {msg}")
    
    # Test with actual model
    from backbone_3 import QCNet
    
    model = QCNet()
    all_passed = run_all_validations(model, verbose=True)
    
    print(f"\nValidation module test: {'PASSED' if all_passed else 'FAILED'}")
