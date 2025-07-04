# OULU dataset for compatibility
from .dataset_faceforensics import FaceForensics

class OULU(FaceForensics):
    """OULU dataset for compatibility"""
    def __init__(self, num_multi=0, mode='Train', shuffle_min_slice=1):
        super().__init__(mode=mode, num_multi=num_multi, shuffle_min_slice=shuffle_min_slice)
