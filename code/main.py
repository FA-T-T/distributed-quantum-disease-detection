"""
Main entry point for the Distributed Quantum Disease Detection system.

This script provides a unified interface for training, testing,
and inference with multiple model types for skin cancer classification:
- QCNet: Original quantum-classical hybrid network
- Baseline: MobileNetV2 with pretrained weights and frozen backbone (classical baseline)
- Hybrid: MobileNetV2 + QNN layer (quantum-classical hybrid)
"""

import os
import sys
import argparse
from typing import Optional

import torch

# Add code directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backbone_3 import QCNet
from models import MobileNetV2Baseline, MobileNetV2Hybrid, get_model as get_mobilenet_model
from data_loader import get_dummy_loaders, get_isic2017_loaders
from train import train, load_checkpoint
from test import evaluate, predict
from preprocessing import ISIC2017_CLASSES, NUM_CLASSES, DEFAULT_IMAGE_SIZE


def print_banner():
    """Print application banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║     Distributed Quantum Disease Detection System                  ║
║     Quantum Splitting CNN for Skin Cancer Classification          ║
╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def get_device(device_name: Optional[str] = None) -> torch.device:
    """
    Get the appropriate device.
    
    Args:
        device_name: Optional device name ('cpu' or 'cuda').
        
    Returns:
        torch.device instance.
    """
    if device_name:
        return torch.device(device_name)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def cmd_train(args):
    """Handle train command."""
    # Check model type and use appropriate training script
    if hasattr(args, 'model') and args.model in ['baseline', 'hybrid']:
        from train_models import main as train_models_main
        
        # Build argument list for train_models script
        sys.argv = ['train_models.py']
        sys.argv.extend(['--model', args.model])
        
        if args.data_dir:
            sys.argv.extend(['--data-dir', args.data_dir])
        if args.epochs:
            sys.argv.extend(['--epochs', str(args.epochs)])
        if args.batch_size:
            sys.argv.extend(['--batch-size', str(args.batch_size)])
        if args.learning_rate:
            sys.argv.extend(['--learning-rate', str(args.learning_rate)])
        if args.save_dir:
            sys.argv.extend(['--save-dir', args.save_dir])
        if args.use_dummy:
            sys.argv.append('--use-dummy')
        if args.device:
            sys.argv.extend(['--device', args.device])
        
        train_models_main()
    else:
        # Default to original QCNet training
        from train import main as train_main
        
        # Build argument list for train script
        sys.argv = ['train.py']
        
        if args.data_dir:
            sys.argv.extend(['--data-dir', args.data_dir])
        if args.epochs:
            sys.argv.extend(['--epochs', str(args.epochs)])
        if args.batch_size:
            sys.argv.extend(['--batch-size', str(args.batch_size)])
        if args.learning_rate:
            sys.argv.extend(['--learning-rate', str(args.learning_rate)])
        if args.save_dir:
            sys.argv.extend(['--save-dir', args.save_dir])
        if args.use_dummy:
            sys.argv.append('--use-dummy')
        if args.device:
            sys.argv.extend(['--device', args.device])
        
        train_main()


def cmd_test(args):
    """Handle test command."""
    from test import main as test_main
    
    # Build argument list for test script
    sys.argv = ['test.py']
    
    if args.model_path:
        sys.argv.extend(['--model-path', args.model_path])
    if args.data_dir:
        sys.argv.extend(['--data-dir', args.data_dir])
    if args.batch_size:
        sys.argv.extend(['--batch-size', str(args.batch_size)])
    if args.use_dummy:
        sys.argv.append('--use-dummy')
    if args.device:
        sys.argv.extend(['--device', args.device])
    if args.save_results:
        sys.argv.extend(['--save-results', args.save_results])
    
    test_main()


def cmd_info(args):
    """Handle info command - show model information."""
    print("\nModel Information")
    print("=" * 50)
    
    device = get_device(args.device)
    print(f"Device: {device}")
    
    model_type = getattr(args, 'model', 'qcnet')
    
    if model_type == 'baseline':
        print(f"\nModel Architecture: MobileNetV2Baseline")
        model = MobileNetV2Baseline(num_classes=NUM_CLASSES, pretrained=False)
        print(f"  - Classical backbone: MobileNetV2 (pretrained on ImageNet)")
        print(f"  - Frozen backbone: Yes (only classifier trainable)")
        print(f"  - Number of classes: {NUM_CLASSES}")
    elif model_type == 'hybrid':
        print(f"\nModel Architecture: MobileNetV2Hybrid")
        model = MobileNetV2Hybrid(num_classes=NUM_CLASSES, pretrained=False)
        print(f"  - Classical backbone: MobileNetV2 (pretrained on ImageNet)")
        print(f"  - Quantum network: Distributed QNN with circuit cutting")
        print(f"  - Number of qubits: 8 (4 + 5 with overlap)")
        print(f"  - Number of classes: {NUM_CLASSES}")
    else:
        print(f"\nModel Architecture: QCNet")
        model = QCNet()
        print(f"  - Classical backbone: MobileNetV2")
        print(f"  - Quantum network: Distributed QNN with circuit cutting")
        print(f"  - Number of qubits: 8 (4 + 5 with overlap)")
        print(f"  - Number of classes: {NUM_CLASSES}")
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nParameter Count:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    
    print(f"\nExpected Input:")
    print(f"  - Shape: (batch_size, 3, {DEFAULT_IMAGE_SIZE}, {DEFAULT_IMAGE_SIZE})")
    print(f"  - Type: torch.FloatTensor")
    
    print(f"\nExpected Output:")
    print(f"  - Shape: (batch_size, {NUM_CLASSES})")
    print(f"  - Type: torch.FloatTensor (logits)")
    
    print(f"\nClasses:")
    for i, class_name in enumerate(ISIC2017_CLASSES):
        print(f"  - {i}: {class_name}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    try:
        test_input = torch.randn(1, 3, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE).to(device)
        with torch.no_grad():
            output = model(test_input)
        print(f"  - Input shape: {test_input.shape}")
        print(f"  - Output shape: {output.shape}")
        print("  - Forward pass: SUCCESS")
    except Exception as e:
        print(f"  - Forward pass: FAILED ({e})")


def cmd_validate(args):
    """Handle validate command - validate tensor dimensions."""
    print("\nTensor Dimension Validation")
    print("=" * 50)
    
    device = get_device(args.device)
    print(f"Device: {device}")
    
    model_type = getattr(args, 'model', 'qcnet')
    
    if model_type == 'baseline':
        print(f"\nModel: MobileNetV2Baseline")
        model = MobileNetV2Baseline(num_classes=NUM_CLASSES, pretrained=False)
    elif model_type == 'hybrid':
        print(f"\nModel: MobileNetV2Hybrid")
        model = MobileNetV2Hybrid(num_classes=NUM_CLASSES, pretrained=False)
    else:
        print(f"\nModel: QCNet")
        model = QCNet()
    
    model = model.to(device)
    model.eval()
    
    test_cases = [
        (1, 3, 128, 128),   # Single sample
        (2, 3, 128, 128),   # Small batch
        (4, 3, 128, 128),   # Medium batch
    ]
    
    print("\nValidating tensor shapes:")
    all_passed = True
    
    for input_shape in test_cases:
        try:
            test_input = torch.randn(*input_shape).to(device)
            with torch.no_grad():
                output = model(test_input)
            
            expected_output_shape = (input_shape[0], NUM_CLASSES)
            actual_output_shape = tuple(output.shape)
            
            if actual_output_shape == expected_output_shape:
                status = "✓ PASS"
            else:
                status = f"✗ FAIL (expected {expected_output_shape})"
                all_passed = False
            
            print(f"  Input {input_shape} -> Output {actual_output_shape} {status}")
        except Exception as e:
            print(f"  Input {input_shape} -> ERROR: {e}")
            all_passed = False
    
    if all_passed:
        print("\n✓ All tensor dimension validations passed!")
    else:
        print("\n✗ Some validations failed.")
    
    return all_passed


def main():
    """Main entry point."""
    print_banner()
    
    parser = argparse.ArgumentParser(
        description='Distributed Quantum Disease Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--model', type=str, default='qcnet',
                              choices=['qcnet', 'baseline', 'hybrid'],
                              help='Model type to train (default: qcnet)')
    train_parser.add_argument('--data-dir', type=str, default='./data',
                              help='Path to data directory')
    train_parser.add_argument('--epochs', type=int, default=10,
                              help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, default=8,
                              help='Batch size')
    train_parser.add_argument('--learning-rate', type=float, default=0.001,
                              help='Learning rate')
    train_parser.add_argument('--save-dir', type=str, default='./checkpoints',
                              help='Directory to save checkpoints')
    train_parser.add_argument('--use-dummy', action='store_true',
                              help='Use dummy data for testing')
    train_parser.add_argument('--device', type=str, default=None,
                              help='Device to use (cpu/cuda)')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test the model')
    test_parser.add_argument('--model-path', type=str, default='./checkpoints/best_model.pth',
                             help='Path to model checkpoint')
    test_parser.add_argument('--data-dir', type=str, default='./data',
                             help='Path to data directory')
    test_parser.add_argument('--batch-size', type=int, default=8,
                             help='Batch size')
    test_parser.add_argument('--use-dummy', action='store_true',
                             help='Use dummy data for testing')
    test_parser.add_argument('--device', type=str, default=None,
                             help='Device to use (cpu/cuda)')
    test_parser.add_argument('--save-results', type=str, default=None,
                             help='Path to save results')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show model information')
    info_parser.add_argument('--model', type=str, default='qcnet',
                             choices=['qcnet', 'baseline', 'hybrid'],
                             help='Model type to show info for (default: qcnet)')
    info_parser.add_argument('--device', type=str, default=None,
                             help='Device to use (cpu/cuda)')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate tensor dimensions')
    validate_parser.add_argument('--model', type=str, default='qcnet',
                                 choices=['qcnet', 'baseline', 'hybrid'],
                                 help='Model type to validate (default: qcnet)')
    validate_parser.add_argument('--device', type=str, default=None,
                                 help='Device to use (cpu/cuda)')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        cmd_train(args)
    elif args.command == 'test':
        cmd_test(args)
    elif args.command == 'info':
        cmd_info(args)
    elif args.command == 'validate':
        cmd_validate(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
