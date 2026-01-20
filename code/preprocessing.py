"""
Data preprocessing module for ISIC2017 skin cancer dataset.

This module provides functions to preprocess images and labels
for the skin cancer classification task.
"""

import os
from typing import Tuple, Optional, Callable, List

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


# Default image size for the model
DEFAULT_IMAGE_SIZE = 128

# ISIC2017 has 3 classes: melanoma, seborrheic_keratosis, nevus
ISIC2017_CLASSES = ['melanoma', 'seborrheic_keratosis', 'nevus']
NUM_CLASSES = 3


def get_default_transforms(
    image_size: int = DEFAULT_IMAGE_SIZE,
    is_training: bool = True
) -> transforms.Compose:
    """
    Get default image transforms for preprocessing.
    
    Args:
        image_size: Target image size (width and height).
        is_training: If True, apply data augmentation.
        
    Returns:
        A torchvision transforms composition.
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def load_image(image_path: str) -> Image.Image:
    """
    Load an image from disk.
    
    Args:
        image_path: Path to the image file.
        
    Returns:
        PIL Image object.
        
    Raises:
        FileNotFoundError: If image file does not exist.
        ValueError: If file is not a valid image.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    try:
        image = Image.open(image_path).convert('RGB')
        return image
    except Exception as e:
        raise ValueError(f"Failed to load image {image_path}: {e}")


def preprocess_image(
    image: Image.Image,
    transform: Optional[transforms.Compose] = None,
    image_size: int = DEFAULT_IMAGE_SIZE
) -> torch.Tensor:
    """
    Preprocess a single image.
    
    Args:
        image: PIL Image object.
        transform: Optional custom transform. If None, default is used.
        image_size: Target image size.
        
    Returns:
        Preprocessed image tensor of shape (3, image_size, image_size).
    """
    if transform is None:
        transform = get_default_transforms(image_size, is_training=False)
    
    return transform(image)


def parse_isic2017_ground_truth(
    ground_truth_path: str
) -> dict:
    """
    Parse ISIC2017 ground truth CSV file.
    
    The ISIC2017 ground truth file has format:
    image_id,melanoma,seborrheic_keratosis
    
    If melanoma=1, label is 0 (melanoma)
    If seborrheic_keratosis=1, label is 1 (seborrheic_keratosis)
    Otherwise, label is 2 (nevus)
    
    Args:
        ground_truth_path: Path to ground truth CSV file.
        
    Returns:
        Dictionary mapping image_id to class label (0, 1, or 2).
    """
    import csv
    
    labels = {}
    
    if not os.path.exists(ground_truth_path):
        raise FileNotFoundError(f"Ground truth file not found: {ground_truth_path}")
    
    with open(ground_truth_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row['image_id']
            melanoma = float(row['melanoma'])
            seborrheic_keratosis = float(row['seborrheic_keratosis'])
            
            if melanoma == 1.0:
                labels[image_id] = 0  # melanoma
            elif seborrheic_keratosis == 1.0:
                labels[image_id] = 1  # seborrheic_keratosis
            else:
                labels[image_id] = 2  # nevus
    
    return labels


class ISIC2017Dataset(Dataset):
    """
    Dataset class for ISIC2017 skin cancer dataset.
    
    Expected directory structure:
    data_dir/
        images/
            ISIC_0000000.jpg
            ISIC_0000001.jpg
            ...
        ISIC-2017_Training_Part3_GroundTruth.csv (or similar)
    
    Attributes:
        data_dir: Root directory of the dataset.
        transform: Optional transform to apply to images.
        image_paths: List of paths to image files.
        labels: Dictionary mapping image IDs to labels.
    """
    
    def __init__(
        self,
        data_dir: str,
        ground_truth_path: Optional[str] = None,
        transform: Optional[Callable] = None,
        image_size: int = DEFAULT_IMAGE_SIZE,
        is_training: bool = True
    ):
        """
        Initialize the ISIC2017 dataset.
        
        Args:
            data_dir: Directory containing images folder.
            ground_truth_path: Path to ground truth CSV. If None, will search
                              in data_dir for common naming patterns.
            transform: Optional custom transform.
            image_size: Target image size.
            is_training: Whether this is training data (enables augmentation).
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.is_training = is_training
        
        # Set up transforms
        if transform is not None:
            self.transform = transform
        else:
            self.transform = get_default_transforms(image_size, is_training)
        
        # Find images directory
        images_dir = os.path.join(data_dir, 'images')
        if not os.path.exists(images_dir):
            images_dir = data_dir  # Images might be directly in data_dir
        
        # Load image paths
        self.image_paths = self._find_images(images_dir)
        
        # Load labels if ground truth is provided
        self.labels = {}
        if ground_truth_path is not None:
            self.labels = parse_isic2017_ground_truth(ground_truth_path)
        else:
            # Try to find ground truth file
            gt_path = self._find_ground_truth(data_dir)
            if gt_path is not None:
                self.labels = parse_isic2017_ground_truth(gt_path)
    
    def _find_images(self, images_dir: str) -> List[str]:
        """Find all image files in directory."""
        image_extensions = {'.jpg', '.jpeg'}
        image_paths = []
        
        if os.path.exists(images_dir):
            for filename in sorted(os.listdir(images_dir)):
                ext = os.path.splitext(filename)[1].lower()
                if ext in image_extensions:
                    image_paths.append(os.path.join(images_dir, filename))
        
        return image_paths
    
    def _find_ground_truth(self, data_dir: str) -> Optional[str]:
        """Try to find ground truth file in data directory."""
        common_patterns = [
            'ISIC-2017_Training_Part3_GroundTruth.csv',
            'ISIC-2017_Validation_Part3_GroundTruth.csv',
            'ISIC-2017_Test_v2_Part3_GroundTruth.csv',
            'ground_truth.csv',
            'labels.csv'
        ]
        
        for pattern in common_patterns:
            path = os.path.join(data_dir, pattern)
            if os.path.exists(path):
                return path
        
        return None
    
    def _get_image_id(self, image_path: str) -> str:
        """Extract image ID from file path."""
        filename = os.path.basename(image_path)
        # Remove extension
        image_id = os.path.splitext(filename)[0]
        return image_id
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index.
            
        Returns:
            Tuple of (image_tensor, label).
            If no labels are available, label will be -1.
        """
        image_path = self.image_paths[idx]
        
        # Load and preprocess image
        image = load_image(image_path)
        image_tensor = self.transform(image)
        
        # Get label
        image_id = self._get_image_id(image_path)
        label = self.labels.get(image_id, -1)
        
        return image_tensor, label


def create_dummy_dataset(
    num_samples: int = 100,
    image_size: int = DEFAULT_IMAGE_SIZE,
    num_classes: int = NUM_CLASSES
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a dummy dataset for testing purposes.
    
    Args:
        num_samples: Number of samples to generate.
        image_size: Image size.
        num_classes: Number of classes.
        
    Returns:
        Tuple of (images, labels) tensors.
    """
    images = torch.randn(num_samples, 3, image_size, image_size)
    labels = torch.randint(0, num_classes, (num_samples,))
    
    return images, labels


if __name__ == '__main__':
    # Test the preprocessing module
    print("Testing preprocessing module...")
    
    # Test transforms
    train_transform = get_default_transforms(128, is_training=True)
    test_transform = get_default_transforms(128, is_training=False)
    print(f"Training transforms: {train_transform}")
    print(f"Test transforms: {test_transform}")
    
    # Test dummy dataset creation
    images, labels = create_dummy_dataset(10, 128, 3)
    print(f"Dummy dataset - Images shape: {images.shape}, Labels shape: {labels.shape}")
    
    print("Preprocessing module tests passed!")
