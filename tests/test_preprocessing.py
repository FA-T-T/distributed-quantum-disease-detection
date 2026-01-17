"""
Unit tests for the preprocessing module.
"""

import os
import sys
import pytest
import torch
import numpy as np
from PIL import Image
import tempfile

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'code'))

from preprocessing import (
    get_default_transforms,
    load_image,
    preprocess_image,
    create_dummy_dataset,
    ISIC2017_CLASSES,
    NUM_CLASSES,
    DEFAULT_IMAGE_SIZE
)


class TestGetDefaultTransforms:
    """Tests for get_default_transforms function."""
    
    def test_training_transforms(self):
        """Test that training transforms include augmentation."""
        transform = get_default_transforms(128, is_training=True)
        assert transform is not None
        
        # Create a dummy image and apply transform
        image = Image.new('RGB', (200, 200), color='red')
        tensor = transform(image)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 128, 128)
    
    def test_test_transforms(self):
        """Test that test transforms don't include augmentation."""
        transform = get_default_transforms(128, is_training=False)
        assert transform is not None
        
        # Create a dummy image and apply transform
        image = Image.new('RGB', (200, 200), color='blue')
        tensor = transform(image)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 128, 128)
    
    def test_different_image_sizes(self):
        """Test transforms with different image sizes."""
        for size in [64, 128, 224]:
            transform = get_default_transforms(size, is_training=False)
            image = Image.new('RGB', (300, 300), color='green')
            tensor = transform(image)
            
            assert tensor.shape == (3, size, size)


class TestLoadImage:
    """Tests for load_image function."""
    
    def test_load_valid_image(self):
        """Test loading a valid image file."""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            image = Image.new('RGB', (100, 100), color='white')
            image.save(f.name)
            
            loaded_image = load_image(f.name)
            assert isinstance(loaded_image, Image.Image)
            assert loaded_image.size == (100, 100)
            
            os.unlink(f.name)
    
    def test_load_nonexistent_image(self):
        """Test loading a non-existent image file."""
        with pytest.raises(FileNotFoundError):
            load_image('/nonexistent/path/image.jpg')


class TestPreprocessImage:
    """Tests for preprocess_image function."""
    
    def test_preprocess_with_default_transform(self):
        """Test preprocessing with default transform."""
        image = Image.new('RGB', (200, 200), color='red')
        tensor = preprocess_image(image)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
    
    def test_preprocess_with_custom_size(self):
        """Test preprocessing with custom image size."""
        image = Image.new('RGB', (200, 200), color='blue')
        tensor = preprocess_image(image, image_size=64)
        
        assert tensor.shape == (3, 64, 64)


class TestCreateDummyDataset:
    """Tests for create_dummy_dataset function."""
    
    def test_default_parameters(self):
        """Test dummy dataset creation with default parameters."""
        images, labels = create_dummy_dataset()
        
        assert images.shape == (100, 3, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
        assert labels.shape == (100,)
        assert labels.min() >= 0
        assert labels.max() < NUM_CLASSES
    
    def test_custom_parameters(self):
        """Test dummy dataset creation with custom parameters."""
        images, labels = create_dummy_dataset(
            num_samples=50,
            image_size=64,
            num_classes=5
        )
        
        assert images.shape == (50, 3, 64, 64)
        assert labels.shape == (50,)
        assert labels.max() < 5


class TestConstants:
    """Tests for module constants."""
    
    def test_isic2017_classes(self):
        """Test that ISIC2017 classes are correctly defined."""
        assert len(ISIC2017_CLASSES) == 3
        assert 'melanoma' in ISIC2017_CLASSES
        assert 'seborrheic_keratosis' in ISIC2017_CLASSES
        assert 'nevus' in ISIC2017_CLASSES
    
    def test_num_classes(self):
        """Test that NUM_CLASSES matches ISIC2017_CLASSES length."""
        assert NUM_CLASSES == len(ISIC2017_CLASSES)
    
    def test_default_image_size(self):
        """Test default image size."""
        assert DEFAULT_IMAGE_SIZE == 128


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
