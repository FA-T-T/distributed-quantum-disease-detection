"""
Unit tests for the test module.
"""

import os
import sys
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'code'))

from test import (
    predict,
    evaluate,
    compute_accuracy,
    compute_loss
)
from ..code.data_loader import DummyDataset


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, num_classes=3):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(3 * 128 * 128, num_classes)
    
    def forward(self, x):
        return self.fc(self.flatten(x))


class TestPredict:
    """Tests for predict function."""
    
    def test_basic_prediction(self):
        """Test basic prediction."""
        model = SimpleModel()
        dataset = DummyDataset(num_samples=20)
        loader = DataLoader(dataset, batch_size=5)
        device = torch.device('cpu')
        
        predictions, labels, probs = predict(model, loader, device, verbose=False)
        
        assert len(predictions) == 20
        assert len(labels) == 20
        assert len(probs) == 20
        assert probs.shape[1] == 3  # 3 classes


class TestEvaluate:
    """Tests for evaluate function."""
    
    def test_basic_evaluation(self):
        """Test basic evaluation."""
        model = SimpleModel()
        dataset = DummyDataset(num_samples=20)
        loader = DataLoader(dataset, batch_size=5)
        device = torch.device('cpu')
        
        metrics = evaluate(model, loader, device, verbose=False)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'confusion_matrix' in metrics
        
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1'] <= 1


class TestComputeAccuracy:
    """Tests for compute_accuracy function."""
    
    def test_accuracy_computation(self):
        """Test accuracy computation."""
        model = SimpleModel()
        dataset = DummyDataset(num_samples=20)
        loader = DataLoader(dataset, batch_size=5)
        device = torch.device('cpu')
        
        accuracy = compute_accuracy(model, loader, device)
        
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 1


class TestComputeLoss:
    """Tests for compute_loss function."""
    
    def test_loss_computation(self):
        """Test loss computation."""
        model = SimpleModel()
        dataset = DummyDataset(num_samples=20)
        loader = DataLoader(dataset, batch_size=5)
        criterion = nn.CrossEntropyLoss()
        device = torch.device('cpu')
        
        loss = compute_loss(model, loader, criterion, device)
        
        assert isinstance(loss, float)
        assert loss >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
