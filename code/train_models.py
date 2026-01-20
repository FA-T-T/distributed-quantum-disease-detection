"""
Training script for MobileNetV2 baseline and hybrid quantum-classical models.

This module provides functionality to train:
1. MobileNetV2Baseline: Pure classical model with pretrained weights
2. MobileNetV2Hybrid: Hybrid model with QNN layer

Usage:
    # Train baseline model
    python train_models.py --model baseline --epochs 10 --use-dummy
    
    # Train hybrid model
    python train_models.py --model hybrid --epochs 10 --use-dummy
"""

import os
import sys
import argparse
import time
from typing import Optional, Dict, Any, Tuple, Literal
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from tqdm import tqdm
import numpy as np

# Add code directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import MobileNetV2Baseline, MobileNetV2Hybrid, get_model
from data_loader import get_dummy_loaders, get_isic2017_loaders


def get_balanced_sampler(
    dataset,
    strategy: Literal["oversample", "undersample"] = "oversample",
    random_seed: int = 42
) -> Tuple[Optional[WeightedRandomSampler], Optional[Subset], int]:
    """
    Create a balanced sampler for handling imbalanced datasets.
    """
    # Try to get labels from dataset attributes first
    if hasattr(dataset, 'labels'):
        labels = np.array([int(l) for l in dataset.labels])
    elif hasattr(dataset, 'targets'):
        labels = np.array([int(t) for t in dataset.targets])
    else:
        # Fallback: iterate through dataset
        labels = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            labels.append(int(label))
        labels = np.array(labels)
    
    class_counts = Counter(labels)
    num_classes = len(class_counts)
    
    if strategy == "oversample":
        class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
        sample_weights = [class_weights[label] for label in labels]
        sample_weights = torch.DoubleTensor(sample_weights)
        
        max_count = max(class_counts.values())
        num_samples = max_count * num_classes
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=num_samples,
            replacement=True
        )
        return sampler, None, num_samples
    
    elif strategy == "undersample":
        min_count = min(class_counts.values())
        class_indices = {cls: np.where(labels == cls)[0] for cls in class_counts.keys()}
        
        rng = np.random.default_rng(random_seed)
        selected_indices = []
        for cls, indices in class_indices.items():
            sampled = rng.choice(indices, size=min_count, replace=False)
            selected_indices.extend(sampled)
        
        rng.shuffle(selected_indices)
        subset = Subset(dataset, selected_indices)
        return None, subset, len(selected_indices)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def create_balanced_loader(
    train_loader: DataLoader,
    balance_strategy: Literal["oversample", "undersample", "none"] = "oversample"
) -> Tuple[DataLoader, int]:
    """Create a balanced DataLoader from an existing DataLoader."""
    if balance_strategy == "none":
        return train_loader, len(train_loader.dataset)
    
    dataset = train_loader.dataset
    batch_size = train_loader.batch_size
    num_workers = train_loader.num_workers
    
    sampler, subset, num_samples = get_balanced_sampler(dataset, balance_strategy)
    
    if balance_strategy == "oversample":
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True
        )
        return loader, num_samples
    else:  # undersample
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        return loader, num_samples


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    verbose: bool = True
) -> Tuple[float, float]:
    """Train the model for one epoch."""
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
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
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
    """Validate the model."""
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
    verbose: bool = True,
    balance_strategy: Literal["oversample", "undersample", "none"] = "oversample",
    model_type: str = "baseline"
) -> Dict[str, Any]:
    """
    Train the model.
    
    Args:
        model: The model to train.
        train_loader: Training data loader.
        val_loader: Optional validation data loader.
        epochs: Number of training epochs.
        learning_rate: Learning rate.
        device: Device to use.
        save_dir: Directory to save checkpoints.
        save_best: Whether to save the best model.
        verbose: Whether to print progress.
        balance_strategy: Strategy for handling imbalanced data.
        model_type: Type of model ('baseline' or 'hybrid').
        
    Returns:
        Dictionary containing training history.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if verbose:
        print(f"Training on device: {device}")
    
    # Apply data balancing strategy
    if balance_strategy != "none":
        if verbose:
            original_size = len(train_loader.dataset)
            print(f"Applying data balancing strategy: {balance_strategy}")
        train_loader, balanced_samples = create_balanced_loader(train_loader, balance_strategy)
        if verbose:
            print(f"  Original samples: {original_size}, Balanced samples per epoch: {balanced_samples}")
    
    model = model.to(device)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Configure optimizer based on model type
    if model_type == "hybrid" and hasattr(model, 'qnn'):
        # Use differentiated learning rates for hybrid model
        optimizer_params = [
            {'params': model.backbone.parameters(), 'lr': learning_rate * 0.1},  # Frozen backbone (still set LR)
            {'params': model.fc1.parameters(), 'lr': learning_rate},
            {'params': model.qnn.parameters(), 'lr': learning_rate * 10},  # Higher LR for quantum
            {'params': model.fc2.parameters(), 'lr': learning_rate}
        ]
        if verbose:
            print("Using differentiated learning rates for hybrid model:")
            print(f"  Classical backbone: {learning_rate * 0.1:.2e}")
            print(f"  Quantum layers: {learning_rate * 10:.2e}")
            print(f"  FC layers: {learning_rate:.2e}")
    else:
        optimizer_params = filter(lambda p: p.requires_grad, model.parameters())
        if verbose:
            print(f"Using uniform learning rate: {learning_rate:.2e}")
    
    optimizer = optim.Adam(optimizer_params, lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    if save_best:
        os.makedirs(save_dir, exist_ok=True)
    
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
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, verbose
        )
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        if val_loader is not None:
            val_loss, val_acc = validate(model, val_loader, criterion, device, verbose)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            scheduler.step(val_loss)
            
            if save_best and val_acc > best_val_acc:
                best_val_acc = val_acc
                history['best_val_acc'] = val_acc
                history['best_epoch'] = epoch
                
                checkpoint_path = os.path.join(save_dir, f'{model_type}_best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'model_type': model_type
                }, checkpoint_path)
                
                if verbose:
                    print(f"Saved best {model_type} model with val_acc: {val_acc:.2f}%")
        
        if verbose:
            print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}, "
                  f"Train Acc: {train_acc:.2f}%")
    
    return history


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description='Train MobileNetV2 baseline or hybrid model for skin cancer classification'
    )
    parser.add_argument('--model', type=str, default='baseline',
                        choices=['baseline', 'hybrid'],
                        help='Model type to train (default: baseline)')
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
    parser.add_argument('--no-pretrained', action='store_true',
                        help='Do not use pretrained weights')
    parser.add_argument('--no-freeze', action='store_true',
                        help='Do not freeze backbone layers')
    parser.add_argument('--balance-strategy', type=str, default='oversample',
                        choices=['oversample', 'undersample', 'none'],
                        help='Strategy for handling imbalanced data')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"Training MobileNetV2 {args.model.capitalize()} Model")
    print("=" * 60)
    
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
    print(f"\nCreating {args.model.capitalize()} model...")
    model = get_model(
        model_type=args.model,
        num_classes=3,
        pretrained=not args.no_pretrained,
        freeze_backbone=not args.no_freeze
    )
    
    # Print parameter count
    total_params = model.get_total_params()
    trainable_params = model.get_trainable_params()
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
        verbose=True,
        balance_strategy=args.balance_strategy,
        model_type=args.model
    )
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Model type: {args.model.capitalize()}")
    print(f"Total training time: {elapsed_time:.2f} seconds")
    print(f"Final training accuracy: {history['train_acc'][-1]:.2f}%")
    
    if history['val_acc']:
        print(f"Final validation accuracy: {history['val_acc'][-1]:.2f}%")
        print(f"Best validation accuracy: {history['best_val_acc']:.2f}% (epoch {history['best_epoch']})")


if __name__ == '__main__':
    main()
