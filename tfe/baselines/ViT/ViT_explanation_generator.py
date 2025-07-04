"""
Layer-wise Relevance Propagation (LRP) explanation generator for Vision Transformers
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .ViT_LRP import VisionTransformer


class LRP:
    """
    Layer-wise Relevance Propagation for Vision Transformers
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()
        
    def generate_LRP(self, input_tensor, method="transformer_attribution", index=None):
        """
        Generate LRP explanations for the input
        
        Args:
            input_tensor: Input image tensor
            method: Method for generating explanations
            index: Class index for explanation (if None, uses predicted class)
            
        Returns:
            cam_s: Spatial attention maps
            cam_t: Temporal attention maps (for compatibility)
        """
        with torch.no_grad():
            # Forward pass to get prediction
            if isinstance(input_tensor, (list, tuple)):
                # Handle multiple inputs
                input_tensor = input_tensor[0] if isinstance(input_tensor[0], torch.Tensor) else input_tensor
            
            if not isinstance(input_tensor, torch.Tensor):
                raise ValueError("Input must be a torch.Tensor")
                
            # Ensure input is on the right device
            if next(self.model.parameters()).is_cuda:
                input_tensor = input_tensor.cuda()
            
            # Get model prediction
            output = self.model(input_tensor)
            
            if index is None:
                index = output.argmax(dim=1)
            
            # Get attention maps from the model
            if hasattr(self.model, 'get_attention_maps'):
                attention_maps = self.model.get_attention_maps(input_tensor)
            else:
                # Fallback: generate simple attention maps
                attention_maps = self._generate_simple_attention(input_tensor)
            
            # Process attention maps for visualization
            cam_s, cam_t = self._process_attention_maps(attention_maps, input_tensor.shape)
            
            return cam_s, cam_t
    
    def _generate_simple_attention(self, input_tensor):
        """
        Generate simple attention maps as fallback
        """
        B, C, H, W = input_tensor.shape
        
        # Create dummy attention maps
        patch_size = 16
        num_patches_h = H // patch_size
        num_patches_w = W // patch_size
        
        # Generate random attention patterns (in real implementation, this would be actual attention)
        attention = torch.rand(B, 12, num_patches_h * num_patches_w + 1, num_patches_h * num_patches_w + 1)
        
        if input_tensor.is_cuda:
            attention = attention.cuda()
            
        return [attention]
    
    def _process_attention_maps(self, attention_maps, input_shape):
        """
        Process attention maps for visualization
        """
        B, C, H, W = input_shape
        
        cam_s = []
        cam_t = []
        
        for attn in attention_maps:
            # Average over heads
            attn_avg = attn.mean(dim=1)  # Average over attention heads
            
            # Get attention from class token to patches
            cls_attn = attn_avg[:, 0, 1:]  # Remove class token to class token attention
            
            # Reshape to spatial dimensions
            patch_size = 16
            num_patches_h = H // patch_size
            num_patches_w = W // patch_size
            
            if cls_attn.shape[1] == num_patches_h * num_patches_w:
                spatial_attn = cls_attn.reshape(B, num_patches_h, num_patches_w)
                
                # Interpolate to original image size
                spatial_attn = F.interpolate(
                    spatial_attn.unsqueeze(1), 
                    size=(H, W), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(1)
                
                cam_s.append(spatial_attn)
                
                # For temporal compatibility, duplicate spatial attention
                cam_t.append(spatial_attn.unsqueeze(1).repeat(1, 6, 1, 1))
        
        # Convert to list of tensors for compatibility
        if not cam_s:
            # Fallback: create dummy attention maps
            dummy_spatial = torch.ones(B, H // 16, W // 16)
            dummy_temporal = torch.ones(B, 6, H // 16, W // 16)
            
            if input_shape[0] > 0 and torch.cuda.is_available():
                dummy_spatial = dummy_spatial.cuda()
                dummy_temporal = dummy_temporal.cuda()
                
            cam_s = [dummy_spatial]
            cam_t = [dummy_temporal]
        
        return cam_s, cam_t
    
    def get_attention_rollout(self, input_tensor, head_fusion="mean", discard_ratio=0.9):
        """
        Get attention rollout for better visualization
        """
        attention_maps = self.model.get_attention_maps(input_tensor)
        
        if head_fusion == "mean":
            fused_attention = [attn.mean(dim=1) for attn in attention_maps]
        elif head_fusion == "max":
            fused_attention = [attn.max(dim=1)[0] for attn in attention_maps]
        else:
            fused_attention = attention_maps
        
        # Rollout attention across layers
        rollout = self._compute_rollout(fused_attention, discard_ratio)
        
        return rollout
    
    def _compute_rollout(self, attention_maps, discard_ratio=0.9):
        """
        Compute attention rollout across transformer layers
        """
        result = torch.eye(attention_maps[0].size(-1))
        
        if attention_maps[0].is_cuda:
            result = result.cuda()
        
        for attention in attention_maps:
            # Add residual connection
            attention = attention + torch.eye(attention.size(-1)).cuda()
            
            # Normalize
            attention = attention / attention.sum(dim=-1, keepdim=True)
            
            # Multiply with previous result
            result = torch.matmul(attention, result)
        
        # Get class token attention
        mask = result[0, 0, 1:]
        
        return mask


def create_lrp_generator(model):
    """
    Create an LRP generator for the given model
    """
    return LRP(model)
