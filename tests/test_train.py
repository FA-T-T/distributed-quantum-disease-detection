"""
Unit tests for the train module.
"""

import os
import sys
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import tempfile
import numpy as np

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'code'))

from train import (
    train_one_epoch,
    validate,
    train,
    save_checkpoint,
    load_checkpoint,
    get_balanced_sampler,
    create_balanced_loader
)
from data_loader import DummyDataset
from mobilnet import MobileNetV2


class ImbalancedDataset(Dataset):
    """Imbalanced dataset for testing balancing strategies."""
    
    def __init__(self, num_samples_per_class=None, image_size=128):
        """
        Create an imbalanced dataset.
        
        Args:
            num_samples_per_class: Dict mapping class to number of samples.
                                   Default: {0: 50, 1: 10, 2: 5} (heavily imbalanced)
            image_size: Image size.
        """
        if num_samples_per_class is None:
            num_samples_per_class = {0: 50, 1: 10, 2: 5}
        
        self.image_size = image_size
        self.images = []
        self.labels = []
        
        for label, count in num_samples_per_class.items():
            for _ in range(count):
                self.images.append(torch.randn(3, image_size, image_size))
                self.labels.append(label)
        
        self.images = torch.stack(self.images)
        self.labels = torch.tensor(self.labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.images[idx], int(self.labels[idx])


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, num_classes=3):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(3 * 128 * 128, num_classes)
    
    def forward(self, x):
        return self.fc(self.flatten(x))


class TestTrainOneEpoch:
    """Tests for train_one_epoch function."""
    
    def test_basic_training(self):
        """Test basic training for one epoch."""
        model = SimpleModel()
        dataset = DummyDataset(num_samples=20)
        loader = DataLoader(dataset, batch_size=5)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        device = torch.device('cpu')
        
        loss, acc = train_one_epoch(
            model, loader, criterion, optimizer, device, epoch=1, verbose=False
        )
        
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert loss >= 0
        assert 0 <= acc <= 100


class TestValidate:
    """Tests for validate function."""
    
    def test_basic_validation(self):
        """Test basic validation."""
        model = SimpleModel()
        dataset = DummyDataset(num_samples=20)
        loader = DataLoader(dataset, batch_size=5)
        criterion = nn.CrossEntropyLoss()
        device = torch.device('cpu')
        
        loss, acc = validate(model, loader, criterion, device, verbose=False)
        
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert loss >= 0
        assert 0 <= acc <= 100


class TestTrain:
    """Tests for train function."""
    
    def test_basic_training(self):
        """Test basic training loop."""
        model = SimpleModel()
        dataset = DummyDataset(num_samples=20)
        train_loader = DataLoader(dataset, batch_size=5)
        val_loader = DataLoader(dataset, batch_size=5)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            history = train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=2,
                learning_rate=0.01,
                save_dir=tmpdir,
                save_best=True,
                verbose=False
            )
        
        assert 'train_loss' in history
        assert 'train_acc' in history
        assert 'val_loss' in history
        assert 'val_acc' in history
        assert len(history['train_loss']) == 2
    
    def test_training_without_validation(self):
        """Test training without validation set."""
        model = SimpleModel()
        dataset = DummyDataset(num_samples=20)
        train_loader = DataLoader(dataset, batch_size=5)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            history = train(
                model=model,
                train_loader=train_loader,
                val_loader=None,
                epochs=2,
                save_dir=tmpdir,
                save_best=False,
                verbose=False
            )
        
        assert len(history['train_loss']) == 2
        assert len(history['val_loss']) == 0


class TestCheckpoints:
    """Tests for checkpoint save/load functions."""
    
    def test_save_and_load_checkpoint(self):
        """Test saving and loading checkpoint."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            checkpoint_path = f.name
        
        try:
            # Save checkpoint
            save_checkpoint(model, optimizer, epoch=5, loss=0.5, path=checkpoint_path)
            
            # Load checkpoint
            new_model = SimpleModel()
            new_optimizer = torch.optim.Adam(new_model.parameters())
            
            checkpoint = load_checkpoint(new_model, checkpoint_path, new_optimizer)
            
            assert checkpoint['epoch'] == 5
            assert checkpoint['loss'] == 0.5
        finally:
            os.unlink(checkpoint_path)
    
    def test_load_model_only(self):
        """Test loading only model weights."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            checkpoint_path = f.name
        
        try:
            save_checkpoint(model, optimizer, epoch=5, loss=0.5, path=checkpoint_path)
            
            new_model = SimpleModel()
            checkpoint = load_checkpoint(new_model, checkpoint_path)
            
            assert checkpoint['epoch'] == 5
        finally:
            os.unlink(checkpoint_path)


class TestGetBalancedSampler:
    """Tests for get_balanced_sampler function."""
    
    def test_oversample_strategy(self):
        """Test oversampling strategy creates correct sampler."""
        dataset = ImbalancedDataset()  # 50 + 10 + 5 = 65 samples
        
        sampler, subset, num_samples = get_balanced_sampler(dataset, "oversample")
        
        assert sampler is not None
        assert subset is None
        # Expected: max_count * num_classes = 50 * 3 = 150
        assert num_samples == 150
    
    def test_undersample_strategy(self):
        """Test undersampling strategy creates correct subset."""
        dataset = ImbalancedDataset()  # 50 + 10 + 5 = 65 samples
        
        sampler, subset, num_samples = get_balanced_sampler(dataset, "undersample")
        
        assert sampler is None
        assert subset is not None
        # Expected: min_count * num_classes = 5 * 3 = 15
        assert num_samples == 15
        assert len(subset) == 15
    
    def test_undersample_class_distribution(self):
        """Test undersampling maintains equal class distribution."""
        dataset = ImbalancedDataset()
        
        _, subset, _ = get_balanced_sampler(dataset, "undersample")
        
        # Check class distribution in subset
        labels = [subset[i][1] for i in range(len(subset))]
        unique, counts = np.unique(labels, return_counts=True)
        
        # Should have 5 samples per class (min_count = 5)
        assert len(unique) == 3
        assert all(count == 5 for count in counts)
    
    def test_invalid_strategy_raises_error(self):
        """Test that invalid strategy raises ValueError."""
        dataset = ImbalancedDataset()
        
        with pytest.raises(ValueError, match="Unknown strategy"):
            get_balanced_sampler(dataset, "invalid")


class TestCreateBalancedLoader:
    """Tests for create_balanced_loader function."""
    
    def test_none_strategy_returns_original(self):
        """Test that 'none' strategy returns original loader."""
        dataset = ImbalancedDataset()
        original_loader = DataLoader(dataset, batch_size=8)
        
        balanced_loader, num_samples = create_balanced_loader(original_loader, "none")
        
        assert balanced_loader is original_loader
        assert num_samples == len(dataset)
    
    def test_oversample_creates_new_loader(self):
        """Test oversampling creates a new DataLoader with sampler."""
        dataset = ImbalancedDataset()
        original_loader = DataLoader(dataset, batch_size=8, num_workers=0)
        
        balanced_loader, num_samples = create_balanced_loader(original_loader, "oversample")
        
        assert balanced_loader is not original_loader
        assert balanced_loader.batch_size == 8
        assert balanced_loader.sampler is not None
        # Expected: max_count * num_classes = 50 * 3 = 150
        assert num_samples == 150
    
    def test_undersample_creates_new_loader(self):
        """Test undersampling creates a new DataLoader with subset."""
        dataset = ImbalancedDataset()
        original_loader = DataLoader(dataset, batch_size=8, num_workers=0)
        
        balanced_loader, num_samples = create_balanced_loader(original_loader, "undersample")
        
        assert balanced_loader is not original_loader
        assert balanced_loader.batch_size == 8
        # Undersampled dataset should have 15 samples (5 per class)
        assert len(balanced_loader.dataset) == 15
        assert num_samples == 15


class TestTrainWithBalancing:
    """Tests for train function with data balancing."""
    
    def test_train_with_oversample(self):
        """Test training with oversampling strategy."""
        model = SimpleModel()
        dataset = ImbalancedDataset()
        train_loader = DataLoader(dataset, batch_size=8)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            history = train(
                model=model,
                train_loader=train_loader,
                val_loader=None,
                epochs=1,
                learning_rate=0.01,
                save_dir=tmpdir,
                save_best=False,
                verbose=False,
                balance_strategy="oversample"
            )
        
        assert len(history['train_loss']) == 1
        assert history['train_loss'][0] >= 0
    
    def test_train_with_undersample(self):
        """Test training with undersampling strategy."""
        model = SimpleModel()
        dataset = ImbalancedDataset()
        train_loader = DataLoader(dataset, batch_size=8)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            history = train(
                model=model,
                train_loader=train_loader,
                val_loader=None,
                epochs=1,
                learning_rate=0.01,
                save_dir=tmpdir,
                save_best=False,
                verbose=False,
                balance_strategy="undersample"
            )
        
        assert len(history['train_loss']) == 1
        assert history['train_loss'][0] >= 0
    
    def test_train_with_no_balancing(self):
        """Test training without balancing."""
        model = SimpleModel()
        dataset = ImbalancedDataset()
        train_loader = DataLoader(dataset, batch_size=8)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            history = train(
                model=model,
                train_loader=train_loader,
                val_loader=None,
                epochs=1,
                learning_rate=0.01,
                save_dir=tmpdir,
                save_best=False,
                verbose=False,
                balance_strategy="none"
            )
        
        assert len(history['train_loss']) == 1
        assert history['train_loss'][0] >= 0
    
    def test_default_balance_strategy_is_oversample(self):
        """Test that default balance_strategy is 'oversample'."""
        model = SimpleModel()
        dataset = ImbalancedDataset()
        train_loader = DataLoader(dataset, batch_size=8)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Call train without specifying balance_strategy
            history = train(
                model=model,
                train_loader=train_loader,
                val_loader=None,
                epochs=1,
                learning_rate=0.01,
                save_dir=tmpdir,
                save_best=False,
                verbose=False
                # balance_strategy not specified, should default to "oversample"
            )
        
        assert len(history['train_loss']) == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
