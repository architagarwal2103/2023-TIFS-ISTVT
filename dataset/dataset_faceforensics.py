import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import json
from PIL import Image
import random
import glob

class FaceForensics(Dataset):
    def __init__(self, root_dir='./data/FF++', mode='Train', transform=None, 
                 num_multi=0, quality='hq', manipulation_types=['fake'],
                 shuffle_min_slice=1, require_idx=False, compress_param=0.8, pair_return=False, fixed_qual=True, seq_len=1, num_videos=10, **kwargs):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.num_multi = num_multi
        self.quality = quality
        self.manipulation_types = manipulation_types
        self.shuffle_min_slice = shuffle_min_slice
        self.require_idx = require_idx
        self.compress_param = compress_param
        self.pair_return = pair_return
        self.fixed_qual = fixed_qual
        self.seq_len = seq_len  # Sequence length for video sequences
        self.num_videos = num_videos  # Number of videos to use
        self.frames_per_video = max(10, seq_len * 2)  # Extract enough frames for sequences
        
        # Simple paths for real and fake folders
        self.real_path = os.path.join(root_dir, 'real')
        self.fake_path = os.path.join(root_dir, 'fake')
        
        self.data_list = []
        self._load_video_data()
        
    def _load_video_data(self):
        """Load video file paths and create frame samples"""
        # Load real videos - limit to specified number of videos
        if os.path.exists(self.real_path):
            real_videos = glob.glob(os.path.join(self.real_path, '*.mp4'))[:self.num_videos]
            print(f"Found {len(real_videos)} real videos (limited to {self.num_videos})")
            for video_path in real_videos:
                video_id = os.path.basename(video_path)
                # Extract frames and add to data list
                frames = self._extract_frames_from_video(video_path)
                for frame_idx, frame in enumerate(frames):
                    self.data_list.append({
                        'frame': frame,
                        'label': 0,  # Real
                        'manipulation': 'original',
                        'video_id': video_id,
                        'frame_idx': frame_idx
                    })
        
        # Load fake videos - limit to specified number of videos
        if os.path.exists(self.fake_path):
            fake_videos = glob.glob(os.path.join(self.fake_path, '*.mp4'))[:self.num_videos]
            print(f"Found {len(fake_videos)} fake videos (limited to {self.num_videos})")
            for video_path in fake_videos:
                video_id = os.path.basename(video_path)
                # Extract frames and add to data list
                frames = self._extract_frames_from_video(video_path)
                for frame_idx, frame in enumerate(frames):
                    self.data_list.append({
                        'frame': frame,
                        'label': 1,  # Fake
                        'manipulation': 'deepfake',
                        'video_id': video_id,
                        'frame_idx': frame_idx
                    })
        
        print(f"Total frames extracted: {len(self.data_list)}")
        
        # Split data based on mode
        random.shuffle(self.data_list)
        total_len = len(self.data_list)
        if self.mode == 'Train':
            self.data_list = self.data_list[:int(0.7 * total_len)]
        elif self.mode == 'Val':
            self.data_list = self.data_list[int(0.7 * total_len):int(0.85 * total_len)]
        else:  # Test
            self.data_list = self.data_list[int(0.85 * total_len):]
    
    def _extract_frames_from_video(self, video_path):
        """Extract frames from video file"""
        frames = []
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Warning: Could not open video {video_path}")
                return frames
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                cap.release()
                return frames
            
            # Sample frames uniformly across the video
            frame_indices = np.linspace(0, total_frames - 1, min(self.frames_per_video, total_frames), dtype=int)
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                else:
                    print(f"Warning: Could not read frame {frame_idx} from {video_path}")
            
            cap.release()
        except Exception as e:
            print(f"Error processing video {video_path}: {str(e)}")
        
        return frames
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        if self.seq_len > 1:
            # Return a sequence of frames
            return self._get_sequence_item(idx)
        else:
            # Return a single frame
            return self._get_single_item(idx)
    
    def _get_single_item(self, idx):
        item = self.data_list[idx]
        
        # Get the preprocessed frame
        image = item['frame']
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        else:
            # Default preprocessing
            image = cv2.resize(image, (299, 299))
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)
        
        label = torch.tensor(item['label'], dtype=torch.long)
        
        if self.require_idx:
            # Return dummy indices for jigsaw models
            idx_data = [torch.randint(0, 9, (1,)) for _ in range(2)]
            return ({'image': image}, idx_data, label)
        else:
            return ({'image': image}, label)
    
    def _get_sequence_item(self, idx):
        # For sequence data, we need to get multiple frames from the same video
        base_item = self.data_list[idx]
        video_id = base_item['video_id']
        
        # Find all frames from the same video
        video_frames = [item for item in self.data_list if item['video_id'] == video_id]
        
        # If we don't have enough frames, repeat the available ones
        while len(video_frames) < self.seq_len:
            video_frames.extend(video_frames)
        
        # Sample seq_len frames
        if len(video_frames) >= self.seq_len:
            # Sample evenly across the video
            indices = np.linspace(0, len(video_frames) - 1, self.seq_len, dtype=int)
            selected_frames = [video_frames[i]['frame'] for i in indices]
        else:
            selected_frames = [frame['frame'] for frame in video_frames[:self.seq_len]]
        
        # Process each frame
        sequence_images = []
        for frame in selected_frames:
            if self.transform:
                processed_frame = self.transform(frame)
            else:
                # Default preprocessing
                processed_frame = cv2.resize(frame, (299, 299))
                processed_frame = processed_frame.astype(np.float32) / 255.0
                processed_frame = torch.from_numpy(processed_frame).permute(2, 0, 1)
            sequence_images.append(processed_frame)
        
        # Stack into sequence tensor: (seq_len, channels, height, width)
        image_sequence = torch.stack(sequence_images, dim=0)
        
        label = torch.tensor(base_item['label'], dtype=torch.long)
        
        if self.require_idx:
            # Return dummy indices for jigsaw models
            idx_data = [torch.randint(0, 9, (1,)) for _ in range(2)]
            return ({'image': image_sequence}, idx_data, label)
        else:
            return ({'image': image_sequence}, label)

# Create aliases for compatibility
Celeb = FaceForensics
VideoSeqDataset = FaceForensics
