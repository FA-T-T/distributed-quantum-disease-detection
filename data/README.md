# ISIC2017 Skin Cancer Dataset

This directory is for the ISIC2017 skin cancer dataset.

## Download Instructions

1. Visit the ISIC 2017 Challenge website: https://challenge.isic-archive.com/data/#2017
2. Download the following files:
   - ISIC-2017_Training_Data.zip
   - ISIC-2017_Training_Part3_GroundTruth.csv
   - ISIC-2017_Validation_Data.zip
   - ISIC-2017_Validation_Part3_GroundTruth.csv
   - ISIC-2017_Test_v2_Data.zip
   - ISIC-2017_Test_v2_Part3_GroundTruth.csv

## Expected Directory Structure

After downloading and extracting, organize the data as follows:

```
data/
├── train/
│   └── images/
│       ├── ISIC_0000000.jpg
│       ├── ISIC_0000001.jpg
│       └── ...
├── val/
│   └── images/
│       ├── ISIC_0000000.jpg
│       └── ...
├── test/
│   └── images/
│       ├── ISIC_0000000.jpg
│       └── ...
├── ISIC-2017_Training_Part3_GroundTruth.csv
├── ISIC-2017_Validation_Part3_GroundTruth.csv
└── ISIC-2017_Test_v2_Part3_GroundTruth.csv
```

## Ground Truth Format

The ground truth CSV files have the following format:
```
image_id,melanoma,seborrheic_keratosis
ISIC_0000000,0.0,0.0
ISIC_0000001,1.0,0.0
...
```

- `melanoma=1.0`: Melanoma (class 0)
- `seborrheic_keratosis=1.0`: Seborrheic keratosis (class 1)
- Both are 0: Nevus (class 2)
