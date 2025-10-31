"""
Utilities for loading visual grounding datasets (RefCOCO, Flickr30k Entities)
and mapping bounding boxes to visual tokens.
"""

import json
import os
import os.path as osp
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from PIL import Image


class RefCOCOLoader:
    """
    Loader for RefCOCO dataset with phrase-to-bounding-box alignment.
    Supports RefCOCO, RefCOCO+, and RefCOCOg variants.
    """
    
    def __init__(self, dataset_path: str, split: str = 'val'):
        """
        Args:
            dataset_path: Path to RefCOCO dataset root directory
            split: Dataset split ('train', 'val', 'test')
        """
        self.dataset_path = dataset_path
        self.split = split
        self.samples = self._load_samples()
    
    def _load_samples(self) -> List[Dict]:
        """Load RefCOCO samples from JSON file."""
        # Try different possible file patterns
        possible_paths = [
            osp.join(self.dataset_path, f'{self.split}.json'),
            osp.join(self.dataset_path, 'annotations', f'{self.split}.json'),
            osp.join(self.dataset_path, f'refcoco_{self.split}.json'),
        ]
        
        samples = []
        for path in possible_paths:
            if osp.exists(path):
                with open(path, 'r') as f:
                    samples = json.load(f)
                print(f"Loaded {len(samples)} samples from {path}")
                return samples
        
        # If no file found, return empty list (will use dummy data for testing)
        print(f"Warning: No RefCOCO data found in {self.dataset_path}")
        return []
    
    def get_sample(self, idx: int) -> Dict:
        """
        Get a sample by index.
        
        Returns:
            Dict with keys:
                - 'image_id': Image identifier
                - 'image_path': Path to image file
                - 'query': Query phrase (e.g., "the woman in red dress")
                - 'bbox': Ground-truth bounding box [x, y, w, h]
                - 'ref_id': Reference ID
        """
        if not self.samples:
            # Return dummy sample for testing
            return self._get_dummy_sample()
        
        sample = self.samples[idx]
        
        # Normalize sample format (different RefCOCO versions may have different keys)
        normalized = {
            'image_id': sample.get('image_id', sample.get('img_id')),
            'image_path': sample.get('image_path', sample.get('file_name')),
            'query': sample.get('sent', sample.get('caption', sample.get('phrase'))),
            'bbox': sample.get('bbox', sample.get('box')),
            'ref_id': sample.get('ref_id', sample.get('ann_id', idx)),
        }
        
        return normalized
    
    def _get_dummy_sample(self) -> Dict:
        """Generate a dummy sample for testing when dataset is not available."""
        return {
            'image_id': 0,
            'image_path': 'dummy.jpg',
            'query': 'the woman in red dress',
            'bbox': [100, 100, 200, 200],  # [x, y, w, h]
            'ref_id': 0,
        }
    
    def __len__(self) -> int:
        return len(self.samples) if self.samples else 1
    
    def random_sample(self) -> Dict:
        """Get a random sample from the dataset."""
        import random
        idx = random.randint(0, len(self) - 1)
        return self.get_sample(idx)


class BoundingBoxMapper:
    """
    Maps bounding boxes to visual tokens based on patch positions.
    """
    
    def __init__(self, image_size: Tuple[int, int], patch_size: int = 14, num_patches: int = 24):
        """
        Args:
            image_size: (height, width) of the image
            patch_size: Size of each patch in pixels (default 14 for CLIP ViT)
            num_patches: Number of patches per side (default 24 for 336x336 image)
        """
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        
        # Calculate actual image size used for patch extraction
        self.effective_size = patch_size * num_patches
    
    def bbox_to_token_indices(self, bbox: List[float], return_masks: bool = False) -> Dict[str, torch.Tensor]:
        """
        Map a bounding box to visual token indices.
        
        Args:
            bbox: Bounding box in format [x, y, width, height]
            return_masks: If True, return boolean masks for relevant/irrelevant tokens
        
        Returns:
            Dict with:
                - 'relevant_indices': Tensor of token indices whose centers fall inside bbox
                - 'irrelevant_indices': Tensor of token indices whose centers fall outside bbox
                - 'relevant_mask': (Optional) Boolean mask for relevant tokens
                - 'irrelevant_mask': (Optional) Boolean mask for irrelevant tokens
        """
        x, y, w, h = bbox
        img_h, img_w = self.image_size
        
        # Scale bbox to effective image size
        scale_x = self.effective_size / img_w
        scale_y = self.effective_size / img_h
        
        x_scaled = x * scale_x
        y_scaled = y * scale_y
        w_scaled = w * scale_x
        h_scaled = h * scale_y
        
        # Calculate patch grid positions
        patch_centers_x = torch.arange(self.num_patches) * self.patch_size + self.patch_size / 2
        patch_centers_y = torch.arange(self.num_patches) * self.patch_size + self.patch_size / 2
        
        # Create meshgrid of patch centers
        grid_y, grid_x = torch.meshgrid(patch_centers_y, patch_centers_x, indexing='ij')
        
        # Check which patches have centers inside the bbox
        inside_x = (grid_x >= x_scaled) & (grid_x < x_scaled + w_scaled)
        inside_y = (grid_y >= y_scaled) & (grid_y < y_scaled + h_scaled)
        inside_bbox = inside_x & inside_y
        
        # Convert 2D mask to 1D token indices
        # Flatten in row-major order to match typical vision transformer token ordering
        inside_bbox_flat = inside_bbox.flatten()
        relevant_indices = torch.nonzero(inside_bbox_flat, as_tuple=True)[0]
        irrelevant_indices = torch.nonzero(~inside_bbox_flat, as_tuple=True)[0]
        
        result = {
            'relevant_indices': relevant_indices,
            'irrelevant_indices': irrelevant_indices,
        }
        
        if return_masks:
            result['relevant_mask'] = inside_bbox_flat
            result['irrelevant_mask'] = ~inside_bbox_flat
        
        return result
    
    def get_patch_coordinates(self, token_idx: int) -> Tuple[float, float]:
        """
        Get the (x, y) center coordinates of a patch given its token index.
        
        Args:
            token_idx: Flattened token index (0 to num_patches^2 - 1)
        
        Returns:
            (x, y) center coordinates in the effective image space
        """
        row = token_idx // self.num_patches
        col = token_idx % self.num_patches
        
        x = col * self.patch_size + self.patch_size / 2
        y = row * self.patch_size + self.patch_size / 2
        
        return (x, y)
    
    def visualize_mapping(self, bbox: List[float], relevant_indices: torch.Tensor) -> np.ndarray:
        """
        Create a visualization of the bbox-to-token mapping.
        
        Args:
            bbox: Bounding box [x, y, w, h]
            relevant_indices: Token indices classified as relevant
        
        Returns:
            RGB image array showing the mapping
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        # Draw grid
        for i in range(self.num_patches + 1):
            pos = i * self.patch_size
            ax.axhline(y=pos, color='gray', linewidth=0.5, alpha=0.3)
            ax.axvline(x=pos, color='gray', linewidth=0.5, alpha=0.3)
        
        # Highlight relevant patches
        for idx in relevant_indices:
            row = idx // self.num_patches
            col = idx % self.num_patches
            rect = patches.Rectangle(
                (col * self.patch_size, row * self.patch_size),
                self.patch_size, self.patch_size,
                linewidth=1, edgecolor='green', facecolor='green', alpha=0.3
            )
            ax.add_patch(rect)
        
        # Draw bounding box
        x, y, w, h = bbox
        img_h, img_w = self.image_size
        scale_x = self.effective_size / img_w
        scale_y = self.effective_size / img_h
        
        rect = patches.Rectangle(
            (x * scale_x, y * scale_y),
            w * scale_x, h * scale_y,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        
        ax.set_xlim(0, self.effective_size)
        ax.set_ylim(self.effective_size, 0)
        ax.set_aspect('equal')
        ax.set_title('Bounding Box to Token Mapping')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        
        # Convert plot to numpy array
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        return img
