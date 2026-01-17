"""
Unit tests for the train module.
"""

import os
import sys
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tempfile

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'code'))

from train import (
    train_one_epoch,
    validate,
    train,
    save_checkpoint,
    load_checkpoint
)
from data_loader import DummyDataset
from mobilnet import MobileNetV2


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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
