# Distributed Quantum Disease Detection

Quantum Splitting Convolutional Neural Network-Based Distributed Quantum Disease Detection Model

## Overview

This project implements a hybrid quantum-classical neural network for skin cancer classification using the ISIC2017 dataset. The model combines:

- **Classical backbone**: MobileNetV2 for feature extraction
- **Quantum neural network**: Distributed QNN with circuit cutting for enhanced classification

## Project Structure

```
distributed-quantum-disease-detection/
├── code/
│   ├── backbone_3.py      # QCNet hybrid model combining classical and quantum networks
│   ├── mobilnet.py        # MobileNetV2 classical backbone
│   ├── mps3.py            # Distributed QNN with circuit cutting
│   ├── qnn.py             # Alternative QNN implementation
│   ├── no_cut_qnn.py      # QNN without circuit cutting
│   ├── mlp.py             # Simple MLP baseline
│   ├── matrix.py          # Confusion matrix visualization
│   ├── preprocessing.py   # Data preprocessing utilities for ISIC2017
│   ├── data_loader.py     # Data loading and dataset classes
│   ├── train.py           # Training script
│   ├── test.py            # Testing and evaluation script
│   ├── validation.py      # Model and tensor validation utilities
│   └── main.py            # Main entry point with CLI
├── data/                  # Dataset directory (ISIC2017)
├── tests/                 # Unit tests
├── requirements.txt       # Python dependencies
└── README.md
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.11+
- PyTorch 2.0+
- PennyLane 0.33+
- pennylane-lightning (for quantum simulation)
- torchvision
- scikit-learn
- matplotlib
- Pillow
- tqdm
- pandas

## Usage

### Quick Start

```bash
# Show model information
cd code
python main.py info

# Validate tensor dimensions
python main.py validate

# Train with dummy data (for testing)
python main.py train --use-dummy --epochs 5

# Test with dummy data
python main.py test --use-dummy
```

### Training

```bash
cd code
python train.py --data-dir ../data --epochs 20 --batch-size 8 --learning-rate 0.001
```

Options:
- `--data-dir`: Path to ISIC2017 dataset
- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Batch size (default: 8)
- `--learning-rate`: Learning rate (default: 0.001)
- `--save-dir`: Directory to save checkpoints (default: ./checkpoints)
- `--use-dummy`: Use dummy data for testing

### Testing

```bash
cd code
python test.py --model-path ./checkpoints/best_model.pth --data-dir ../data
```

Options:
- `--model-path`: Path to trained model checkpoint
- `--data-dir`: Path to test data
- `--batch-size`: Batch size (default: 8)
- `--save-results`: Path to save evaluation results (JSON)

## Dataset: ISIC2017

The model is designed for the ISIC2017 Skin Lesion Analysis Challenge dataset with 3 classes:
- **Melanoma** (class 0)
- **Seborrheic keratosis** (class 1)
- **Nevus** (class 2)

### Dataset Structure

```
data/
├── train/
│   └── images/
├── val/
│   └── images/
├── test/
│   └── images/
├── ISIC-2017_Training_Part3_GroundTruth.csv
├── ISIC-2017_Validation_Part3_GroundTruth.csv
└── ISIC-2017_Test_v2_Part3_GroundTruth.csv
```

## Model Architecture

### QCNet (Quantum-Classical Network)

1. **MobileNetV2 backbone**: Extracts 8 features from 128x128 RGB images
2. **Distributed QNN**: 
   - Front circuit: 4 qubits for initial processing
   - Back circuit: 5 qubits with overlapping qubit for distributed computation
   - Circuit cutting technique for efficient quantum-classical splitting
3. **Classification head**: Maps 32-dimensional quantum output to 3 classes

### Input/Output Specifications

- **Input**: `(batch_size, 3, 128, 128)` - RGB images
- **Output**: `(batch_size, 3)` - Logits for 3 classes

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ -v --cov=code
```

## License

This project is for research and educational purposes.

## Citation

If you use this code, please cite:
```
Quantum Splitting Convolutional Neural Network-Based Distributed Quantum Disease Detection Model
```
