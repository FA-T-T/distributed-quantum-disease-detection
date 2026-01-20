"""
Unit tests for the data_loader module.
"""

import os
import sys
import pytest
import torch
from torch.utils.data import DataLoader

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'code'))

from ..code.data_loader import (
    DummyDataset,
    create_data_loaders,
    create_single_loader,
    get_dummy_loaders
)
from ..code.preprocessing import NUM_CLASSES, DEFAULT_IMAGE_SIZE


class TestDummyDataset:
    """Tests for DummyDataset class."""
    
    def test_initialization(self):
        """Test dataset initialization."""
        dataset = DummyDataset(num_samples=50)
        
        assert len(dataset) == 50
    
    def test_getitem(self):
        """Test getting items from dataset."""
        dataset = DummyDataset(num_samples=10)
        
        image, label = dataset[0]
        
        assert isinstance(image, torch.Tensor)
        assert isinstance(label, int)
        assert image.shape == (3, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
        assert 0 <= label < NUM_CLASSES
    
    def test_custom_parameters(self):
        """Test dataset with custom parameters."""
        dataset = DummyDataset(
            num_samples=20,
            image_size=64,
            num_classes=5
        )
        
        assert len(dataset) == 20
        
        image, label = dataset[0]
        assert image.shape == (3, 64, 64)
        assert 0 <= label < 5
    
    def test_reproducibility(self):
        """Test that dataset is reproducible with same seed."""
        dataset1 = DummyDataset(num_samples=10, random_seed=42)
        dataset2 = DummyDataset(num_samples=10, random_seed=42)
        
        image1, label1 = dataset1[0]
        image2, label2 = dataset2[0]
        
        assert torch.allclose(image1, image2)
        assert label1 == label2


class TestCreateDataLoaders:
    """Tests for create_data_loaders function."""
    
    def test_default_split_ratios(self):
        """Test data loaders with default split ratios."""
        dataset = DummyDataset(num_samples=100)
        train_loader, val_loader, test_loader = create_data_loaders(dataset)
        
        # Check that loaders are created
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)
        
        # Check split sizes (70/15/15)
        assert len(train_loader.dataset) == 70
        assert len(val_loader.dataset) == 15
        assert len(test_loader.dataset) == 15
    
    def test_custom_split_ratios(self):
        """Test data loaders with custom split ratios."""
        dataset = DummyDataset(num_samples=100)
        train_loader, val_loader, test_loader = create_data_loaders(
            dataset,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1
        )
        
        assert len(train_loader.dataset) == 80
        assert len(val_loader.dataset) == 10
        assert len(test_loader.dataset) == 10
    
    def test_invalid_ratios(self):
        """Test that invalid ratios raise error."""
        dataset = DummyDataset(num_samples=100)
        
        with pytest.raises(ValueError):
            create_data_loaders(
                dataset,
                train_ratio=0.5,
                val_ratio=0.3,
                test_ratio=0.3  # Sum > 1
            )
    
    def test_batch_iteration(self):
        """Test iterating through batches."""
        dataset = DummyDataset(num_samples=100)
        train_loader, _, _ = create_data_loaders(dataset, batch_size=10)
        
        batch_images, batch_labels = next(iter(train_loader))
        
        assert batch_images.shape[0] <= 10  # Batch size
        assert batch_images.shape[1:] == (3, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
        assert batch_labels.shape[0] <= 10


class TestCreateSingleLoader:
    """Tests for create_single_loader function."""
    
    def test_basic_creation(self):
        """Test creating a single data loader."""
        dataset = DummyDataset(num_samples=50)
        loader = create_single_loader(dataset, batch_size=10)
        
        assert isinstance(loader, DataLoader)
        assert len(loader) == 5  # 50 samples / 10 batch size
    
    def test_shuffle(self):
        """Test shuffle option."""
        dataset = DummyDataset(num_samples=50, random_seed=42)
        
        loader_shuffled = create_single_loader(dataset, batch_size=10, shuffle=True)
        loader_not_shuffled = create_single_loader(dataset, batch_size=10, shuffle=False)
        
        assert isinstance(loader_shuffled, DataLoader)
        assert isinstance(loader_not_shuffled, DataLoader)


class TestGetDummyLoaders:
    """Tests for get_dummy_loaders function."""
    
    def test_creates_all_loaders(self):
        """Test that all loaders are created."""
        loaders = get_dummy_loaders(num_samples=100)
        
        assert 'train' in loaders
        assert 'val' in loaders
        assert 'test' in loaders
    
    def test_loader_types(self):
        """Test that loaders are DataLoader instances."""
        loaders = get_dummy_loaders(num_samples=100)
        
        for name, loader in loaders.items():
            assert isinstance(loader, DataLoader), f"{name} is not a DataLoader"
    
    def test_custom_parameters(self):
        """Test with custom parameters."""
        loaders = get_dummy_loaders(
            num_samples=50,
            batch_size=5,
            image_size=64,
            num_classes=5
        )
        
        train_loader = loaders['train']
        batch_images, batch_labels = next(iter(train_loader))
        
        assert batch_images.shape[0] <= 5
        assert batch_images.shape[1:] == (3, 64, 64)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
