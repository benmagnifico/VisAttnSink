"""
Attention Capture Module for Visual Attention Sink Analysis

This module captures attention weights during model inference for later visualization
and analysis of the visual attention sink phenomenon.
"""

import os
import pickle
import torch
from collections import defaultdict
from typing import Dict, List, Optional
import numpy as np


class AttentionCapture:
    """
    Captures and stores attention weights during LLaVA model inference.
    Supports capturing attention patterns for visual attention sink analysis.
    """
    
    def __init__(self, save_dir: str = "F_visualizations"):
        self.save_dir = save_dir
        self.attention_data = defaultdict(dict)
        self.metadata = {}
        self.enabled = False
        os.makedirs(save_dir, exist_ok=True)
        
    def activate(self):
        """Enable attention capture"""
        self.enabled = True
        
    def deactivate(self):
        """Disable attention capture"""
        self.enabled = False
        
    def is_active(self):
        """Check if capture is active"""
        return self.enabled
    
    def clear(self):
        """Clear captured attention data"""
        self.attention_data = defaultdict(dict)
        self.metadata = {}
        
    def store_attention(
        self, 
        layer_idx: int, 
        attention_weights: torch.Tensor,
        token_idx: int = -1
    ):
        """
        Store attention weights for a specific layer and token.
        
        Args:
            layer_idx: Index of the transformer layer
            attention_weights: Attention tensor [batch, heads, query, key]
            token_idx: Index of the generated token (-1 for prefill phase)
        """
        if not self.enabled:
            return
            
        # Detach and move to CPU to save memory
        attn_cpu = attention_weights.detach().cpu()
        
        if token_idx not in self.attention_data[layer_idx]:
            self.attention_data[layer_idx][token_idx] = []
        
        self.attention_data[layer_idx][token_idx].append(attn_cpu)
    
    def store_metadata(
        self,
        question_id: int,
        question: str,
        image_path: str,
        response: str,
        vis_token_start: int,
        vis_token_end: int,
        total_tokens: int
    ):
        """
        Store metadata about the inference run.
        
        Args:
            question_id: Unique identifier for the question
            question: The question text
            image_path: Path to the image
            response: Model's response
            vis_token_start: Start index of visual tokens
            vis_token_end: End index of visual tokens
            total_tokens: Total number of tokens in sequence
        """
        self.metadata = {
            'question_id': question_id,
            'question': question,
            'image_path': image_path,
            'response': response,
            'vis_token_start': vis_token_start,
            'vis_token_end': vis_token_end,
            'total_tokens': total_tokens,
            'num_layers': len(self.attention_data) if self.attention_data else 0
        }
    
    def save(self, filename: str):
        """
        Save captured attention data and metadata to file.
        
        Args:
            filename: Name of the file to save (without path)
        """
        if not self.attention_data:
            print("No attention data to save")
            return
            
        save_path = os.path.join(self.save_dir, filename)
        
        # Convert defaultdict to regular dict for pickling
        save_data = {
            'attention': dict(self.attention_data),
            'metadata': self.metadata
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Attention data saved to {save_path}")
        
    @staticmethod
    def load(filepath: str) -> Dict:
        """
        Load saved attention data from file.
        
        Args:
            filepath: Path to the saved attention file
            
        Returns:
            Dictionary containing attention data and metadata
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data
    
    def get_attention_summary(self) -> Dict:
        """
        Get a summary of captured attention data.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.attention_data:
            return {"status": "No data captured"}
        
        summary = {
            "num_layers": len(self.attention_data),
            "layers": list(self.attention_data.keys()),
        }
        
        if self.metadata:
            summary.update({
                "question_id": self.metadata.get('question_id'),
                "vis_token_range": (
                    self.metadata.get('vis_token_start'),
                    self.metadata.get('vis_token_end')
                ),
                "total_tokens": self.metadata.get('total_tokens')
            })
        
        return summary


# Global singleton instance
_attention_capture = None


def get_attention_capture(save_dir: str = "F_visualizations") -> AttentionCapture:
    """Get or create the global attention capture instance"""
    global _attention_capture
    if _attention_capture is None:
        _attention_capture = AttentionCapture(save_dir)
    return _attention_capture
