"""
Attention Visualizer for Visual Attention Sink Analysis

This module visualizes attention patterns to identify and analyze the visual attention
sink phenomenon in Large Multimodal Models.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Optional, Tuple
import torch


class AttentionVisualizer:
    """
    Visualizes attention patterns and highlights visual attention sinks.
    """
    
    def __init__(self, save_dir: str = "F_visualizations"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Create custom colormap for attention visualization
        self.cmap = LinearSegmentedColormap.from_list(
            'attention', ['white', 'lightblue', 'blue', 'darkblue']
        )
    
    def visualize_attention_heatmap(
        self,
        attention_weights: np.ndarray,
        layer_idx: int,
        head_idx: int,
        vis_token_start: int,
        vis_token_end: int,
        save_name: str,
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10)
    ):
        """
        Create heatmap visualization of attention weights.
        
        Args:
            attention_weights: Attention matrix [query_len, key_len]
            layer_idx: Layer index
            head_idx: Attention head index
            vis_token_start: Start index of visual tokens
            vis_token_end: End index of visual tokens
            save_name: Filename to save the visualization
            title: Optional title for the plot
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap
        im = ax.imshow(attention_weights, cmap=self.cmap, aspect='auto', interpolation='nearest')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attention Weight', rotation=270, labelpad=20)
        
        # Highlight visual token region
        vis_width = vis_token_end - vis_token_start
        rect = patches.Rectangle(
            (vis_token_start - 0.5, -0.5),
            vis_width, len(attention_weights),
            linewidth=2, edgecolor='red', facecolor='none',
            label='Visual Tokens'
        )
        ax.add_patch(rect)
        
        # Vertical rectangle for keys
        rect_h = patches.Rectangle(
            (-0.5, vis_token_start - 0.5),
            attention_weights.shape[1], vis_width,
            linewidth=2, edgecolor='orange', facecolor='none',
            linestyle='--', alpha=0.5
        )
        ax.add_patch(rect_h)
        
        # Labels and title
        ax.set_xlabel('Key Position', fontsize=12)
        ax.set_ylabel('Query Position', fontsize=12)
        
        if title is None:
            title = f'Attention Heatmap - Layer {layer_idx}, Head {head_idx}'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved attention heatmap to {save_path}")
    
    def visualize_visual_attention_distribution(
        self,
        attention_weights: np.ndarray,
        layer_idx: int,
        vis_token_start: int,
        vis_token_end: int,
        save_name: str,
        num_heads_to_show: int = 8
    ):
        """
        Visualize attention distribution to visual tokens across heads.
        
        Args:
            attention_weights: Attention tensor [num_heads, query_len, key_len]
            layer_idx: Layer index
            vis_token_start: Start index of visual tokens
            vis_token_end: End index of visual tokens
            save_name: Filename to save
            num_heads_to_show: Number of attention heads to display
        """
        num_heads = min(num_heads_to_show, attention_weights.shape[0])
        
        # Calculate attention to visual tokens for each head
        vis_attention = attention_weights[:, :, vis_token_start:vis_token_end]
        vis_attention_sum = vis_attention.sum(axis=2)  # Sum over visual tokens
        
        # Create subplot for each head
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i in range(num_heads):
            ax = axes[i]
            query_positions = np.arange(vis_attention_sum.shape[1])
            
            ax.plot(query_positions, vis_attention_sum[i], linewidth=2)
            ax.axvspan(vis_token_start, vis_token_end, alpha=0.2, color='red', 
                      label='Visual Token Region')
            
            ax.set_xlabel('Query Position', fontsize=10)
            ax.set_ylabel('Total Attention to Visual Tokens', fontsize=10)
            ax.set_title(f'Head {i}', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            if i == 0:
                ax.legend(loc='best', fontsize=8)
        
        plt.suptitle(f'Visual Attention Distribution - Layer {layer_idx}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visual attention distribution to {save_path}")
    
    def identify_attention_sinks(
        self,
        attention_weights: np.ndarray,
        vis_token_start: int,
        vis_token_end: int,
        threshold: float = 0.1
    ) -> Tuple[List[int], np.ndarray]:
        """
        Identify visual attention sink tokens.
        
        Args:
            attention_weights: Attention matrix [query_len, key_len]
            vis_token_start: Start index of visual tokens
            vis_token_end: End index of visual tokens
            threshold: Threshold for identifying sink tokens
            
        Returns:
            Tuple of (sink_token_indices, attention_to_each_visual_token)
        """
        # Calculate attention to each visual token
        vis_attention = attention_weights[:, vis_token_start:vis_token_end]
        attention_per_vis_token = vis_attention.sum(axis=0)
        
        # Normalize
        attention_per_vis_token = attention_per_vis_token / attention_per_vis_token.sum()
        
        # Identify sink tokens (those receiving more than threshold of total attention)
        sink_indices = np.where(attention_per_vis_token > threshold)[0]
        sink_indices = sink_indices + vis_token_start  # Adjust to absolute positions
        
        return sink_indices.tolist(), attention_per_vis_token
    
    def visualize_sink_tokens(
        self,
        attention_weights: np.ndarray,
        layer_idx: int,
        head_idx: int,
        vis_token_start: int,
        vis_token_end: int,
        save_name: str,
        threshold: float = 0.1
    ):
        """
        Visualize identified attention sink tokens.
        
        Args:
            attention_weights: Attention matrix [query_len, key_len]
            layer_idx: Layer index
            head_idx: Head index
            vis_token_start: Start index of visual tokens
            vis_token_end: End index of visual tokens
            save_name: Filename to save
            threshold: Threshold for identifying sinks
        """
        sink_indices, attention_dist = self.identify_attention_sinks(
            attention_weights, vis_token_start, vis_token_end, threshold
        )
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Attention heatmap with sink highlights
        im = ax1.imshow(attention_weights, cmap=self.cmap, aspect='auto', interpolation='nearest')
        
        # Highlight sink tokens
        for sink_idx in sink_indices:
            rect = patches.Rectangle(
                (sink_idx - 0.5, -0.5),
                1, len(attention_weights),
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax1.add_patch(rect)
        
        # Highlight visual region
        vis_width = vis_token_end - vis_token_start
        rect_region = patches.Rectangle(
            (vis_token_start - 0.5, -0.5),
            vis_width, len(attention_weights),
            linewidth=1, edgecolor='orange', facecolor='none',
            linestyle='--', alpha=0.5
        )
        ax1.add_patch(rect_region)
        
        ax1.set_xlabel('Key Position', fontsize=12)
        ax1.set_ylabel('Query Position', fontsize=12)
        ax1.set_title(f'Attention Sinks - Layer {layer_idx}, Head {head_idx}', 
                     fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=ax1, label='Attention Weight')
        
        # Plot 2: Bar chart of attention distribution over visual tokens
        vis_token_positions = np.arange(vis_token_start, vis_token_end)
        colors = ['red' if i in sink_indices else 'blue' for i in vis_token_positions]
        
        ax2.bar(range(len(attention_dist)), attention_dist, color=colors, alpha=0.7)
        ax2.axhline(y=threshold, color='black', linestyle='--', 
                   label=f'Sink Threshold ({threshold})')
        ax2.set_xlabel('Visual Token Index (relative)', fontsize=12)
        ax2.set_ylabel('Normalized Attention', fontsize=12)
        ax2.set_title('Attention Distribution over Visual Tokens', 
                     fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved sink visualization to {save_path}")
        print(f"Identified {len(sink_indices)} sink tokens at positions: {sink_indices}")
    
    def visualize_layer_comparison(
        self,
        attention_data: Dict,
        vis_token_start: int,
        vis_token_end: int,
        save_name: str,
        layers_to_compare: Optional[List[int]] = None
    ):
        """
        Compare attention patterns across different layers.
        
        Args:
            attention_data: Dictionary of attention weights by layer
            vis_token_start: Start index of visual tokens
            vis_token_end: End index of visual tokens
            save_name: Filename to save
            layers_to_compare: List of layer indices to compare (default: first 4 and last 4)
        """
        if layers_to_compare is None:
            all_layers = sorted(attention_data.keys())
            if len(all_layers) > 8:
                layers_to_compare = all_layers[:4] + all_layers[-4:]
            else:
                layers_to_compare = all_layers
        
        num_layers = len(layers_to_compare)
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for idx, layer_idx in enumerate(layers_to_compare):
            if idx >= 8:  # Max 8 subplots
                break
                
            ax = axes[idx]
            
            # Get attention for this layer (average over heads)
            attn = attention_data[layer_idx]
            if isinstance(attn, list):
                attn = attn[0]  # Take first token if multiple
            if isinstance(attn, torch.Tensor):
                attn = attn.numpy()
            
            # Average over heads
            if len(attn.shape) == 4:  # [batch, heads, query, key]
                attn = attn[0].mean(axis=0)  # Average over heads
            elif len(attn.shape) == 3:  # [heads, query, key]
                attn = attn.mean(axis=0)
            
            im = ax.imshow(attn, cmap=self.cmap, aspect='auto', interpolation='nearest')
            
            # Highlight visual region
            vis_width = vis_token_end - vis_token_start
            rect = patches.Rectangle(
                (vis_token_start - 0.5, -0.5),
                vis_width, attn.shape[0],
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
            
            ax.set_title(f'Layer {layer_idx}', fontsize=11, fontweight='bold')
            ax.set_xlabel('Key Position', fontsize=9)
            ax.set_ylabel('Query Position', fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Hide unused subplots
        for idx in range(num_layers, 8):
            axes[idx].axis('off')
        
        plt.suptitle('Attention Patterns Across Layers', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved layer comparison to {save_path}")
