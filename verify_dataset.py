#!/usr/bin/env python3

"""
Test script to verify the FaceForensics++ dataset setup works correctly.
"""

import sys
import os
sys.path.append('.')

try:
    from dataset.dataset_faceforensics import FaceForensics
    from dataset.transform import xception_default_data_transforms
    print("✓ Successfully imported dataset classes")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

def test_dataset():
    print("\nTesting FaceForensics dataset...")
    
    # Check if data directory exists
    data_dir = './data/FF++'
    if not os.path.exists(data_dir):
        print(f"✗ Data directory not found: {data_dir}")
        print("Please download the FaceForensics++ dataset first.")
        return False
    
    if not os.path.exists(os.path.join(data_dir, 'real')):
        print(f"✗ 'real' folder not found in {data_dir}")
        print("Please extract the 'real' folder from the dataset.")
        return False
    
    if not os.path.exists(os.path.join(data_dir, 'fake')):
        print(f"✗ 'fake' folder not found in {data_dir}")
        print("Please extract the 'fake' folder from the dataset.")
        return False
    
    try:
        # Test dataset creation
        transforms = xception_default_data_transforms
        train_dataset = FaceForensics(
            root_dir=data_dir, 
            mode='Train', 
            transform=transforms['train']
        )
        
        val_dataset = FaceForensics(
            root_dir=data_dir, 
            mode='Test', 
            transform=transforms['val']
        )
        
        print(f"✓ Train dataset created successfully: {len(train_dataset)} samples")
        print(f"✓ Val dataset created successfully: {len(val_dataset)} samples")
        
        if len(train_dataset) == 0:
            print("✗ No training samples found. Please check if images are in the correct folders.")
            return False
        
        # Test loading a sample
        sample = train_dataset[0]
        if isinstance(sample, tuple) and len(sample) == 2:
            image_dict, label = sample
            if 'image' in image_dict:
                print(f"✓ Sample loaded successfully. Image shape: {image_dict['image'].shape}, Label: {label}")
            else:
                print("✗ Sample format incorrect: missing 'image' key")
                return False
        else:
            print(f"✗ Unexpected sample format: {type(sample)}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error creating dataset: {e}")
        return False

def main():
    print("FaceForensics++ Dataset Setup Verification")
    print("=" * 50)
    
    if test_dataset():
        print("\n🎉 Dataset setup successful!")
        print("\nYou can now run training with:")
        print("python train_CNN.py --sub_dataset FaceForensics --model_name xception --batch_size 8 --epoches 10")
    else:
        print("\n❌ Dataset setup failed!")
        print("\nPlease follow these steps:")
        print("1. Download the dataset from: https://www.kaggle.com/datasets/hungle3401/faceforensic")
        print("2. Extract the 'real' folder to: ./data/FF++/real/")
        print("3. Extract the 'fake' folder to: ./data/FF++/fake/")
        print("4. Run this verification script again")

if __name__ == "__main__":
    main()
