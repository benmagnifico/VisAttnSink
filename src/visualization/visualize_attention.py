#!/usr/bin/env python3
"""
Visualization Script for Visual Attention Sink Analysis

This script processes saved attention data and generates comprehensive visualizations
of the visual attention sink phenomenon in LLaVA-1.5-7B model.

Usage:
    python src/visualization/visualize_attention.py --attention_file path/to/attention.pkl
    python src/visualization/visualize_attention.py --attention_file path/to/attention.pkl --layer 15 --head 10
"""

import argparse
import os
import sys
import pickle
from pathlib import Path

import numpy as np
import torch

from attention_capture import AttentionCapture
from attention_visualizer import AttentionVisualizer


def load_attention_data(filepath: str):
    """Load attention data from pickle file"""
    print(f"Loading attention data from {filepath}...")
    data = AttentionCapture.load(filepath)
    print(f"Loaded data with metadata: {data.get('metadata', {})}")
    return data


def visualize_all(
    attention_data: dict,
    output_dir: str,
    layer_idx: int = None,
    head_idx: int = None,
    threshold: float = 0.1
):
    """
    Generate all visualizations for the attention data.
    
    Args:
        attention_data: Loaded attention data dictionary
        output_dir: Directory to save visualizations
        layer_idx: Specific layer to visualize (None for all)
        head_idx: Specific head to visualize (None for all)
        threshold: Threshold for identifying attention sinks
    """
    visualizer = AttentionVisualizer(save_dir=output_dir)
    
    metadata = attention_data.get('metadata', {})
    attention = attention_data.get('attention', {})
    
    vis_token_start = metadata.get('vis_token_start', 0)
    vis_token_end = metadata.get('vis_token_end', 576)
    question_id = metadata.get('question_id', 'unknown')
    
    print(f"\nVisualization Parameters:")
    print(f"  Visual tokens: [{vis_token_start}, {vis_token_end})")
    print(f"  Question ID: {question_id}")
    print(f"  Number of layers: {len(attention)}")
    
    # Determine which layers to visualize
    layers_to_viz = [layer_idx] if layer_idx is not None else sorted(attention.keys())
    
    for layer in layers_to_viz:
        if layer not in attention:
            print(f"Warning: Layer {layer} not found in attention data")
            continue
        
        print(f"\nProcessing Layer {layer}...")
        layer_attention = attention[layer]
        
        # Get attention for the first token (prefill phase, token_idx = -1)
        if -1 in layer_attention:
            attn_tensor = layer_attention[-1]
            if isinstance(attn_tensor, list):
                attn_tensor = attn_tensor[0]
        else:
            # Try first available token
            first_token = min(layer_attention.keys())
            attn_tensor = layer_attention[first_token]
            if isinstance(attn_tensor, list):
                attn_tensor = attn_tensor[0]
        
        if isinstance(attn_tensor, torch.Tensor):
            attn_tensor = attn_tensor.numpy()
        
        # attn_tensor shape: [batch, heads, query, key] or [heads, query, key]
        if len(attn_tensor.shape) == 4:
            attn_tensor = attn_tensor[0]  # Remove batch dimension
        
        num_heads = attn_tensor.shape[0]
        print(f"  Number of heads: {num_heads}")
        print(f"  Attention shape: {attn_tensor.shape}")
        
        # Determine which heads to visualize
        heads_to_viz = [head_idx] if head_idx is not None else range(min(8, num_heads))
        
        # Visualization 1: Individual head heatmaps
        for head in heads_to_viz:
            if head >= num_heads:
                continue
            
            attn_matrix = attn_tensor[head]  # [query, key]
            
            # Heatmap visualization
            save_name = f"heatmap_qid{question_id}_layer{layer}_head{head}.png"
            visualizer.visualize_attention_heatmap(
                attn_matrix,
                layer,
                head,
                vis_token_start,
                vis_token_end,
                save_name
            )
            
            # Sink visualization
            save_name = f"sinks_qid{question_id}_layer{layer}_head{head}.png"
            visualizer.visualize_sink_tokens(
                attn_matrix,
                layer,
                head,
                vis_token_start,
                vis_token_end,
                save_name,
                threshold=threshold
            )
        
        # Visualization 2: Visual attention distribution across heads
        save_name = f"vis_attention_dist_qid{question_id}_layer{layer}.png"
        visualizer.visualize_visual_attention_distribution(
            attn_tensor,
            layer,
            vis_token_start,
            vis_token_end,
            save_name,
            num_heads_to_show=8
        )
    
    # Visualization 3: Layer comparison (if multiple layers)
    if layer_idx is None and len(attention) > 1:
        print("\nGenerating layer comparison visualization...")
        # Prepare attention data for comparison
        layer_attention_dict = {}
        for layer in attention.keys():
            layer_attn = attention[layer]
            if -1 in layer_attn:
                layer_attention_dict[layer] = layer_attn[-1]
            else:
                first_token = min(layer_attn.keys())
                layer_attention_dict[layer] = layer_attn[first_token]
        
        save_name = f"layer_comparison_qid{question_id}.png"
        visualizer.visualize_layer_comparison(
            layer_attention_dict,
            vis_token_start,
            vis_token_end,
            save_name
        )
    
    print(f"\n✓ All visualizations saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize attention patterns and identify visual attention sinks"
    )
    parser.add_argument(
        "--attention_file",
        type=str,
        required=True,
        help="Path to the saved attention data file (.pkl)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="F_visualizations",
        help="Directory to save visualizations (default: F_visualizations)"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Specific layer to visualize (default: all layers)"
    )
    parser.add_argument(
        "--head",
        type=int,
        default=None,
        help="Specific attention head to visualize (default: first 8 heads)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Threshold for identifying attention sink tokens (default: 0.1)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.attention_file):
        print(f"Error: Attention file not found: {args.attention_file}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and visualize
    attention_data = load_attention_data(args.attention_file)
    visualize_all(
        attention_data,
        args.output_dir,
        layer_idx=args.layer,
        head_idx=args.head,
        threshold=args.threshold
    )


if __name__ == "__main__":
    main()
