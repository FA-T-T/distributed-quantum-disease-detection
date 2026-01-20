"""
Training script for the Quantum-Classical hybrid model.

This module provides functionality to train the QCNet model
for skin cancer classification.
"""

import os
import sys
import argparse
import time
from typing import Optional, Dict, Any, Tuple
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Add code directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backbone_3 import QCNet
from data_loader import get_dummy_loaders, get_isic2017_loaders, DummyDataset


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    verbose: bool = True
) -> Tuple[float, float]:
    """
    Train the model for one epoch.
    
    Args:
        model: The model to train.
        train_loader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to use (cpu/cuda).
        epoch: Current epoch number.
        verbose: Whether to print progress.
        
    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    if verbose:
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    else:
        pbar = train_loader
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent gradient explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if verbose:
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    verbose: bool = True
) -> Tuple[float, float]:
    """
    Validate the model.
    
    Args:
        model: The model to validate.
        val_loader: Validation data loader.
        criterion: Loss function.
        device: Device to use.
        verbose: Whether to print progress.
        
    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    avg_loss = running_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    if verbose:
        print(f"Validation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    return avg_loss, accuracy


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 10,
    learning_rate: float = 0.001,
    device: Optional[torch.device] = None,
    save_dir: str = './checkpoints',
    save_best: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Train the model.
    
    Args:
        model: The model to train.
        train_loader: Training data loader.
        val_loader: Optional validation data loader.
        epochs: Number of training epochs.
        learning_rate: Learning rate (base learning rate for classical layers).
        device: Device to use. If None, auto-detect.
        save_dir: Directory to save checkpoints.
        save_best: Whether to save the best model.
        verbose: Whether to print progress.
        
    Returns:
        Dictionary containing training history.
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if verbose:
        print(f"Training on device: {device}")
    
    # Move model to device
    model = model.to(device)
    
    # Loss function and optimizer with differentiated learning rates
    criterion = nn.CrossEntropyLoss()
    
    # Use differentiated learning rates for quantum and classical layers
    # Quantum layers typically need higher learning rates due to smaller gradients
    optimizer_params = []
    
    # Check if model has QCNet structure with quantum layers
    if hasattr(model, 'QModel') and hasattr(model, 'CModel'):
        optimizer_params = [
            {'params': model.CModel.parameters(), 'lr': learning_rate * 0.1},  # Classical: 1e-4 if lr=1e-3
            {'params': model.QModel.parameters(), 'lr': learning_rate * 10},   # Quantum: 1e-2 if lr=1e-3
            {'params': model.fc2.parameters(), 'lr': learning_rate}            # Final layer: 1e-3
        ]
        if hasattr(model, 'fc1'):
            optimizer_params.append({'params': model.fc1.parameters(), 'lr': learning_rate})
        
        if verbose:
            print("Using differentiated learning rates:")
            print(f"  Classical layers: {learning_rate * 0.1:.2e}")
            print(f"  Quantum layers: {learning_rate * 10:.2e}")
            print(f"  Final layers: {learning_rate:.2e}")
    else:
        optimizer_params = model.parameters()
        if verbose:
            print(f"Using uniform learning rate: {learning_rate:.2e}")
    
    optimizer = optim.Adam(optimizer_params, lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=verbose
    )
    
    # Create save directory
    if save_best:
        os.makedirs(save_dir, exist_ok=True)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'best_val_acc': 0.0,
        'best_epoch': 0
    }
    
    best_val_acc = 0.0
    
    for epoch in range(1, epochs + 1):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, verbose
        )
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Validate
        if val_loader is not None:
            val_loss, val_acc = validate(model, val_loader, criterion, device, verbose)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Save best model
            if save_best and val_acc > best_val_acc:
                best_val_acc = val_acc
                history['best_val_acc'] = val_acc
                history['best_epoch'] = epoch
                
                checkpoint_path = os.path.join(save_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss
                }, checkpoint_path)
                
                if verbose:
                    print(f"Saved best model with val_acc: {val_acc:.2f}%")
        
        if verbose:
            print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}, "
                  f"Train Acc: {train_acc:.2f}%")
    
    return history


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    loss: float,
    path: str
) -> None:
    """
    Save a training checkpoint.
    
    Args:
        model: The model.
        optimizer: The optimizer.
        epoch: Current epoch.
        loss: Current loss.
        path: Path to save checkpoint.
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, path)


def load_checkpoint(
    model: nn.Module,
    path: str,
    optimizer: Optional[optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Load a training checkpoint.
    
    Args:
        model: The model to load weights into.
        path: Path to checkpoint.
        optimizer: Optional optimizer to load state.
        device: Device to load to.
        
    Returns:
        Checkpoint dictionary.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train QCNet for skin cancer classification')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Path to data directory')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--use-dummy', action='store_true',
                        help='Use dummy data for testing')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cpu/cuda)')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Print training progress')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("QCNet Training")
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
            num_samples=50,
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
                num_samples=50,
                batch_size=args.batch_size
            )
    
    train_loader = loaders.get('train')
    val_loader = loaders.get('val')
    
    if train_loader is None:
        print("Error: No training data available.")
        return
    
    print(f"Training samples: {len(train_loader.dataset)}")
    if val_loader:
        print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    print("Creating QCNet model...")
    model = QCNet()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train
    print("\nStarting training...")
    start_time = time.time()
    
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device,
        save_dir=args.save_dir,
        save_best=True,
        verbose=args.verbose
    )
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print(f"Total training time: {elapsed_time:.2f} seconds")
    print(f"Final training accuracy: {history['train_acc'][-1]:.2f}%")
    
    if history['val_acc']:
        print(f"Final validation accuracy: {history['val_acc'][-1]:.2f}%")
        print(f"Best validation accuracy: {history['best_val_acc']:.2f}% (epoch {history['best_epoch']})")


if __name__ == '__main__':
    main()
