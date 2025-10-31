#!/usr/bin/env python3
"""
Test script for visualization utilities

This script tests the attention capture and visualization functionality
with synthetic data to ensure everything works correctly.
"""

import os
import sys
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.visualization.attention_capture import AttentionCapture, get_attention_capture
from src.visualization.attention_visualizer import AttentionVisualizer


def create_synthetic_attention(
    num_heads=32,
    num_queries=50,
    num_keys=600,
    vis_start=10,
    vis_end=586
):
    """
    Create synthetic attention weights with some visual attention sinks.
    
    Args:
        num_heads: Number of attention heads
        num_queries: Number of query tokens
        num_keys: Number of key tokens
        vis_start: Start index of visual tokens
        vis_end: End index of visual tokens
    
    Returns:
        Attention tensor [num_heads, num_queries, num_keys]
    """
    # Initialize with random attention
    attention = np.random.rand(num_heads, num_queries, num_keys)
    
    # Create visual attention sinks at specific positions
    sink_positions = [vis_start + 50, vis_start + 150, vis_start + 300]
    
    for head_idx in range(num_heads):
        for query_idx in range(num_queries):
            # Add attention sinks
            for sink_pos in sink_positions:
                if sink_pos < vis_end:
                    # Make some visual tokens receive high attention
                    attention[head_idx, query_idx, sink_pos] += 0.5 * (1 + np.random.rand())
    
    # Normalize attention weights
    attention = attention / attention.sum(axis=2, keepdims=True)
    
    return attention


def test_attention_capture():
    """Test attention capture functionality"""
    print("\n" + "="*60)
    print("Testing Attention Capture...")
    print("="*60)
    
    # Initialize capture
    capture = AttentionCapture(save_dir="/tmp/test_visualizations")
    capture.activate()
    
    # Create synthetic attention
    attention = create_synthetic_attention()
    attention_tensor = torch.from_numpy(attention).float()
    
    # Store attention for multiple layers
    for layer_idx in range(8):
        # Add batch dimension: [1, heads, query, key]
        attn_with_batch = attention_tensor.unsqueeze(0)
        capture.store_attention(layer_idx, attn_with_batch, token_idx=-1)
    
    # Store metadata
    capture.store_metadata(
        question_id=999,
        question="Test question: Is there a cat?",
        image_path="test_image.jpg",
        response="Test response: Yes, there is a cat.",
        vis_token_start=10,
        vis_token_end=586,
        total_tokens=600
    )
    
    # Save
    capture.save("test_attention.pkl")
    
    # Load and verify
    loaded_data = AttentionCapture.load("/tmp/test_visualizations/test_attention.pkl")
    
    print(f"✓ Captured attention for {len(loaded_data['attention'])} layers")
    print(f"✓ Metadata: {loaded_data['metadata']['question_id']}")
    print(f"✓ Visual tokens: [{loaded_data['metadata']['vis_token_start']}, {loaded_data['metadata']['vis_token_end']})")
    
    return loaded_data


def test_visualization(attention_data):
    """Test visualization functionality"""
    print("\n" + "="*60)
    print("Testing Visualizations...")
    print("="*60)
    
    visualizer = AttentionVisualizer(save_dir="/tmp/test_visualizations")
    
    metadata = attention_data['metadata']
    attention = attention_data['attention']
    
    vis_start = metadata['vis_token_start']
    vis_end = metadata['vis_token_end']
    
    # Test 1: Heatmap visualization
    print("\n1. Testing heatmap visualization...")
    layer_attn = attention[0][-1]  # Layer 0, token -1
    if isinstance(layer_attn, list):
        layer_attn = layer_attn[0]
    if isinstance(layer_attn, torch.Tensor):
        layer_attn = layer_attn.numpy()
    
    # Remove batch dimension if present
    if len(layer_attn.shape) == 4:
        layer_attn = layer_attn[0]
    
    attn_matrix = layer_attn[0]  # First head
    visualizer.visualize_attention_heatmap(
        attn_matrix,
        layer_idx=0,
        head_idx=0,
        vis_token_start=vis_start,
        vis_token_end=vis_end,
        save_name="test_heatmap.png"
    )
    print("✓ Heatmap visualization created")
    
    # Test 2: Sink visualization
    print("\n2. Testing sink visualization...")
    visualizer.visualize_sink_tokens(
        attn_matrix,
        layer_idx=0,
        head_idx=0,
        vis_token_start=vis_start,
        vis_token_end=vis_end,
        save_name="test_sinks.png",
        threshold=0.01  # Lower threshold for synthetic data
    )
    print("✓ Sink visualization created")
    
    # Test 3: Visual attention distribution
    print("\n3. Testing visual attention distribution...")
    visualizer.visualize_visual_attention_distribution(
        layer_attn,
        layer_idx=0,
        vis_token_start=vis_start,
        vis_token_end=vis_end,
        save_name="test_vis_dist.png",
        num_heads_to_show=8
    )
    print("✓ Visual attention distribution created")
    
    # Test 4: Layer comparison
    print("\n4. Testing layer comparison...")
    layer_dict = {}
    for layer_idx in attention.keys():
        layer_dict[layer_idx] = attention[layer_idx][-1]
    
    visualizer.visualize_layer_comparison(
        layer_dict,
        vis_token_start=vis_start,
        vis_token_end=vis_end,
        save_name="test_layer_comparison.png"
    )
    print("✓ Layer comparison created")
    
    # Test 5: Identify sinks
    print("\n5. Testing sink identification...")
    sink_indices, attention_dist = visualizer.identify_attention_sinks(
        attn_matrix,
        vis_start,
        vis_end,
        threshold=0.01
    )
    print(f"✓ Identified {len(sink_indices)} sink tokens at positions: {sink_indices[:5]}...")
    
    return True


def main():
    print("\n" + "="*60)
    print("Visual Attention Sink Utilities Test")
    print("="*60)
    
    try:
        # Test attention capture
        attention_data = test_attention_capture()
        
        # Test visualizations
        test_visualization(attention_data)
        
        print("\n" + "="*60)
        print("✅ All tests passed!")
        print("="*60)
        print(f"\nTest visualizations saved to: /tmp/test_visualizations/")
        print("\nYou can view the generated images to verify the visualizations.")
        
        return 0
    
    except Exception as e:
        print("\n" + "="*60)
        print("❌ Test failed!")
        print("="*60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
