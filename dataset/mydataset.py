# Compatibility imports for existing datasets
import cv2
import torch
import numpy as np
from .dataset_faceforensics import FaceForensics

class MyDataset(FaceForensics):
    """Legacy dataset class for compatibility"""
    def __init__(self, index_range=None, transform=None, get_triplet='False', subset='Classic', 
                 require_landmarks=False, use_white_list=False, num_multi=0):
        super().__init__(mode='Train' if index_range else 'Test', transform=transform, num_multi=num_multi)

class VideoSeqDataset(FaceForensics):
    """Video sequence dataset for compatibility"""
    def __init__(self, quality='hq', transform=None, get_triplet='False', subset=None, 
                 require_landmarks=False, num_multi=0, shuffle_min_slice=1, require_idx=False,
                 random_compress=False, compress_param=0.8, size=300, mode='Train',
                 dataset_len=60000, frame_type='normal', diverse_quality=False, seq_len=4,
                 return_fake_type=False):
        super().__init__(mode=mode, transform=transform, num_multi=num_multi, quality=quality,
                        shuffle_min_slice=shuffle_min_slice, require_idx=require_idx,
                        compress_param=compress_param)
        self.seq_len = seq_len
        self.return_fake_type = return_fake_type
        
    def __getitem__(self, idx):
        if self.seq_len > 1:
            # Return a sequence of frames
            sequences = []
            labels = []
            
            # Get multiple frames from the same video or similar frames
            base_item = self.data_list[idx]
            video_id = base_item['video_id']
            
            # Find other frames from the same video
            same_video_frames = [i for i, item in enumerate(self.data_list) 
                               if item['video_id'] == video_id]
            
            # If we don't have enough frames from the same video, use random frames
            if len(same_video_frames) < self.seq_len:
                frame_indices = [idx] + [idx] * (self.seq_len - 1)  # Repeat the same frame
            else:
                # Sample seq_len frames from the same video
                if len(same_video_frames) >= self.seq_len:
                    frame_indices = sorted(same_video_frames[:self.seq_len])
                else:
                    frame_indices = same_video_frames + [same_video_frames[-1]] * (self.seq_len - len(same_video_frames))
            
            # Collect frames
            frame_tensors = []
            for frame_idx in frame_indices:
                item = self.data_list[frame_idx]
                image = item['frame']
                
                # Apply transforms if provided
                if self.transform:
                    image = self.transform(image)
                else:
                    # Default preprocessing
                    image = cv2.resize(image, (299, 299))
                    image = image.astype(np.float32) / 255.0
                    image = torch.from_numpy(image).permute(2, 0, 1)
                
                frame_tensors.append(image)
            
            # Stack frames into sequence: (seq_len, channels, height, width)
            sequence_tensor = torch.stack(frame_tensors, dim=0)
            label = torch.tensor(base_item['label'], dtype=torch.long)
            
            if self.require_idx:
                # Return dummy indices for jigsaw models
                idx_data = [torch.randint(0, 9, (1,)) for _ in range(2)]
                return ({'image': sequence_tensor}, idx_data, label)
            else:
                return ({'image': sequence_tensor}, label)
        else:
            # Return single frame (original behavior)
            return super().__getitem__(idx)

class OULU(FaceForensics):
    """OULU dataset for compatibility"""
    def __init__(self, num_multi=0, mode='Train', shuffle_min_slice=1):
        super().__init__(mode=mode, num_multi=num_multi, shuffle_min_slice=shuffle_min_slice)
