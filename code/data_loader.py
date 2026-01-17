"""
Data loading module for ISIC2017 skin cancer dataset.

This module provides DataLoader utilities and data splitting functions
for training, validation, and testing.
"""

import os
from typing import Tuple, Optional, Dict, Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
import numpy as np

from preprocessing import (
    ISIC2017Dataset,
    create_dummy_dataset,
    get_default_transforms,
    DEFAULT_IMAGE_SIZE,
    NUM_CLASSES
)


def create_data_loaders(
    dataset: Dataset,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    num_workers: int = 0,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders from a dataset.
    
    Args:
        dataset: The full dataset.
        batch_size: Batch size for data loaders.
        train_ratio: Ratio of training data.
        val_ratio: Ratio of validation data.
        test_ratio: Ratio of test data.
        num_workers: Number of worker processes for data loading.
        random_seed: Random seed for reproducibility.
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
        
    Raises:
        ValueError: If ratios don't sum to 1.0.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Set random seed for reproducibility
    generator = torch.Generator().manual_seed(random_seed)
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def create_single_loader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0
) -> DataLoader:
    """
    Create a single data loader from a dataset.
    
    Args:
        dataset: The dataset.
        batch_size: Batch size.
        shuffle: Whether to shuffle data.
        num_workers: Number of worker processes.
        
    Returns:
        DataLoader instance.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


class DummyDataset(Dataset):
    """
    Dummy dataset for testing purposes.
    
    This dataset generates random images and labels
    without requiring actual image files.
    """
    
    def __init__(
        self,
        num_samples: int = 100,
        image_size: int = DEFAULT_IMAGE_SIZE,
        num_classes: int = NUM_CLASSES,
        random_seed: int = 42
    ):
        """
        Initialize the dummy dataset.
        
        Args:
            num_samples: Number of samples.
            image_size: Image size.
            num_classes: Number of classes.
            random_seed: Random seed for reproducibility.
        """
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        
        # Set seed for reproducibility
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        # Generate random data
        self.images = torch.randn(num_samples, 3, image_size, image_size)
        self.labels = torch.randint(0, num_classes, (num_samples,))
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.images[idx], int(self.labels[idx])


def get_isic2017_loaders(
    data_dir: str,
    train_gt_path: Optional[str] = None,
    val_gt_path: Optional[str] = None,
    test_gt_path: Optional[str] = None,
    batch_size: int = 32,
    image_size: int = DEFAULT_IMAGE_SIZE,
    num_workers: int = 0
) -> Dict[str, DataLoader]:
    """
    Get data loaders for ISIC2017 dataset.
    
    Expected directory structure:
    data_dir/
        train/
            images/
        val/
            images/
        test/
            images/
    
    Args:
        data_dir: Root data directory.
        train_gt_path: Path to training ground truth CSV.
        val_gt_path: Path to validation ground truth CSV.
        test_gt_path: Path to test ground truth CSV.
        batch_size: Batch size.
        image_size: Target image size.
        num_workers: Number of data loading workers.
        
    Returns:
        Dictionary with 'train', 'val', and 'test' data loaders.
    """
    loaders = {}
    
    # Check for separate train/val/test directories
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    
    if os.path.exists(train_dir):
        train_dataset = ISIC2017Dataset(
            train_dir,
            ground_truth_path=train_gt_path,
            image_size=image_size,
            is_training=True
        )
        loaders['train'] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
    
    if os.path.exists(val_dir):
        val_dataset = ISIC2017Dataset(
            val_dir,
            ground_truth_path=val_gt_path,
            image_size=image_size,
            is_training=False
        )
        loaders['val'] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    if os.path.exists(test_dir):
        test_dataset = ISIC2017Dataset(
            test_dir,
            ground_truth_path=test_gt_path,
            image_size=image_size,
            is_training=False
        )
        loaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    # If no separate directories, use single directory with splits
    if not loaders:
        dataset = ISIC2017Dataset(
            data_dir,
            ground_truth_path=train_gt_path,
            image_size=image_size,
            is_training=True
        )
        
        if len(dataset) > 0:
            train_loader, val_loader, test_loader = create_data_loaders(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers
            )
            loaders = {
                'train': train_loader,
                'val': val_loader,
                'test': test_loader
            }
    
    return loaders


def get_dummy_loaders(
    num_samples: int = 100,
    batch_size: int = 32,
    image_size: int = DEFAULT_IMAGE_SIZE,
    num_classes: int = NUM_CLASSES
) -> Dict[str, DataLoader]:
    """
    Get dummy data loaders for testing.
    
    Args:
        num_samples: Number of samples.
        batch_size: Batch size.
        image_size: Image size.
        num_classes: Number of classes.
        
    Returns:
        Dictionary with 'train', 'val', and 'test' data loaders.
    """
    dataset = DummyDataset(
        num_samples=num_samples,
        image_size=image_size,
        num_classes=num_classes
    )
    
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset,
        batch_size=batch_size
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


if __name__ == '__main__':
    # Test the data loading module
    print("Testing data loading module...")
    
    # Test dummy dataset
    dummy_dataset = DummyDataset(num_samples=100)
    print(f"Dummy dataset size: {len(dummy_dataset)}")
    
    image, label = dummy_dataset[0]
    print(f"Sample - Image shape: {image.shape}, Label: {label}")
    
    # Test data loaders
    loaders = get_dummy_loaders(num_samples=100, batch_size=10)
    print(f"Created loaders: {list(loaders.keys())}")
    
    for name, loader in loaders.items():
        print(f"{name} loader: {len(loader)} batches")
    
    # Test a batch
    train_loader = loaders['train']
    batch_images, batch_labels = next(iter(train_loader))
    print(f"Batch - Images shape: {batch_images.shape}, Labels shape: {batch_labels.shape}")
    
    print("Data loading module tests passed!")
