"""
Testing and evaluation script for the Quantum-Classical hybrid model.

This module provides functionality to evaluate the QCNet model
and compute various metrics for skin cancer classification.
"""

import os
import sys
import argparse
from typing import Optional, Dict, Any, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from tqdm import tqdm

# Add code directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backbone_3 import QCNet
from data_loader import get_dummy_loaders, get_isic2017_loaders
from preprocessing import ISIC2017_CLASSES, NUM_CLASSES


def predict(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Make predictions on a dataset.
    
    Args:
        model: The model to use for predictions.
        data_loader: Data loader for the dataset.
        device: Device to use.
        verbose: Whether to show progress.
        
    Returns:
        Tuple of (predictions, true_labels, probabilities).
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probs = []
    
    if verbose:
        pbar = tqdm(data_loader, desc="Predicting")
    else:
        pbar = data_loader
    
    with torch.no_grad():
        for inputs, targets in pbar:
            inputs = inputs.to(device)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(targets.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return (
        np.array(all_predictions),
        np.array(all_labels),
        np.array(all_probs)
    )


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    class_names: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate the model and compute metrics.
    
    Args:
        model: The model to evaluate.
        data_loader: Data loader for the test set.
        device: Device to use.
        class_names: Optional list of class names.
        verbose: Whether to print results.
        
    Returns:
        Dictionary containing evaluation metrics.
    """
    if class_names is None:
        class_names = ISIC2017_CLASSES
    
    # Get predictions
    predictions, labels, probs = predict(model, data_loader, device, verbose)
    
    # Compute metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(labels, predictions, average=None, zero_division=0)
    recall_per_class = recall_score(labels, predictions, average=None, zero_division=0)
    f1_per_class = f1_score(labels, predictions, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # Classification report
    report = classification_report(
        labels, predictions,
        target_names=class_names[:len(set(labels))],
        zero_division=0
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': predictions,
        'labels': labels,
        'probabilities': probs
    }
    
    if verbose:
        print("\n" + "=" * 50)
        print("Evaluation Results")
        print("=" * 50)
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Precision (weighted): {precision * 100:.2f}%")
        print(f"Recall (weighted): {recall * 100:.2f}%")
        print(f"F1 Score (weighted): {f1 * 100:.2f}%")
        print("\nClassification Report:")
        print(report)
        print("\nConfusion Matrix:")
        print(cm)
    
    return metrics


def test_model(
    model_path: str,
    data_loader: DataLoader,
    device: Optional[torch.device] = None,
    class_names: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Load a model from checkpoint and evaluate it.
    
    Args:
        model_path: Path to model checkpoint.
        data_loader: Data loader for test set.
        device: Device to use. If None, auto-detect.
        class_names: Optional list of class names.
        verbose: Whether to print results.
        
    Returns:
        Dictionary containing evaluation metrics.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if verbose:
        print(f"Loading model from: {model_path}")
    
    # Create model
    model = QCNet()
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    
    if verbose:
        print("Model loaded successfully!")
    
    # Evaluate
    return evaluate(model, data_loader, device, class_names, verbose)


def compute_accuracy(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device
) -> float:
    """
    Compute accuracy on a dataset.
    
    Args:
        model: The model.
        data_loader: Data loader.
        device: Device to use.
        
    Returns:
        Accuracy as a float between 0 and 1.
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return correct / total


def compute_loss(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """
    Compute loss on a dataset.
    
    Args:
        model: The model.
        data_loader: Data loader.
        criterion: Loss function.
        device: Device to use.
        
    Returns:
        Average loss.
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    
    return total_loss / len(data_loader)


def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(description='Test QCNet for skin cancer classification')
    parser.add_argument('--model-path', type=str, default='./checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Path to data directory')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--use-dummy', action='store_true',
                        help='Use dummy data for testing')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cpu/cuda)')
    parser.add_argument('--save-results', type=str, default=None,
                        help='Path to save results')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("QCNet Testing")
    print("=" * 50)
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Load data
    if args.use_dummy:
        print("Using dummy data for testing...")
        loaders = get_dummy_loaders(
            num_samples=30,
            batch_size=args.batch_size
        )
    else:
        print(f"Loading data from: {args.data_dir}")
        loaders = get_isic2017_loaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size
        )
        
        if not loaders:
            print("No data found. Using dummy data instead.")
            loaders = get_dummy_loaders(
                num_samples=30,
                batch_size=args.batch_size
            )
    
    test_loader = loaders.get('test')
    
    if test_loader is None:
        print("Error: No test data available.")
        return
    
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Check if model exists
    if os.path.exists(args.model_path):
        # Load and test existing model
        metrics = test_model(
            model_path=args.model_path,
            data_loader=test_loader,
            device=device,
            verbose=True
        )
    else:
        print(f"Model not found at {args.model_path}")
        print("Creating new model and running inference...")
        
        model = QCNet()
        model = model.to(device)
        
        metrics = evaluate(
            model=model,
            data_loader=test_loader,
            device=device,
            verbose=True
        )
    
    # Save results if requested
    if args.save_results:
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        results = {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1': float(metrics['f1']),
            'confusion_matrix': metrics['confusion_matrix'].tolist(),
            'classification_report': metrics['classification_report']
        }
        
        with open(args.save_results, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {args.save_results}")
    
    print("\nTesting complete!")


if __name__ == '__main__':
    main()
