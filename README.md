# ISTVT-Official
Implementation of ISTVT: Interpretable Spatial-Temporal Video Transformer for Deepfake Detection.

## Dataset Setup

### Download FaceForensics++ Dataset
- Visit: https://www.kaggle.com/datasets/hungle3401/faceforensics?resource=download-directory
- Download the dataset manually
- Extract to `./data/FF++/` directory

   ```
   ./data/FF++/
   ├── real/           # Real video files
   │   ├── video1.mp4
   │   ├── video2.mp4
   │   └── ...
   └── fake/           # Fake video files
       ├── video1.mp4
       ├── video2.mp4
       └── ...
   ```

## Quick Start

### Training
Train the model with FaceForensics++ dataset using 20 videos:
```bash
cd /home/jjbigdub/gitrepo/2023-TIFS-ISTVT
python3 train_CNN.py --model_name xception --batch_size 4 --epoches 10 --sub_dataset FaceForensics --sequence_length 4 --num_videos 20
```

### Visualization
Generate LRP (Layer-wise Relevance Propagation) attribution visualizations:
```bash
cd /home/jjbigdub/gitrepo/2023-TIFS-ISTVT
python3 visualize_rel.py --model_name xception --sub_dataset FaceForensics --model_path ./output/xception/best.pkl
```

## Parameters

### Training Parameters
- `--num_videos`: Number of videos to use for training/testing (default: 10)
- `--model_name`: Model architecture (e.g., xception)
- `--batch_size`: Batch size for training
- `--epoches`: Number of training epochs
- `--sequence_length`: Length of video sequences
- `--sub_dataset`: Dataset to use (FaceForensics, Celeb, OULU)

### Visualization Parameters
- `--model_path`: Path to trained model weights
- `--sub_dataset`: Dataset to use for visualization
- `--batch_size`: Batch size for visualization (default: 16)

## Output
- Training: Model weights saved in `./output/[model_name]/`
- Visualization: Attribution maps saved in `./visualize/faceforensics/`
