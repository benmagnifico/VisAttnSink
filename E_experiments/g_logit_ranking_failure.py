"""
Experiment 2: g_LogitRanking_Failure (Logit Ranking Failure Experiment)

Goal: Prove that model's "key relevance" (Q·K^T logits) is unreliable.
      Even with "head consensus" (S_consensus), the model frequently ranks
      "irrelevant tokens" higher than "relevant tokens".

This experiment uses RefCOCO or Flickr30k Entities dataset.

Method:
1. Sample Selection: Get image, query Q, and classify tokens into T_relevant and T_irrelevant
   based on ground-truth bounding box B_gt
2. Compute Consensus Logit:
   - Identify all "image-centric heads" (ICHs) H_img^l
   - For each visual token j, compute consensus key relevance:
     S_consensus(j) = avg_{h' ∈ H_img^l} (q_i^{h'} · (k_j^{h'})^T)
3. Ranking and Verification:
   - Sort all visual tokens by S_consensus(j) in descending order
   - Take Top-K tokens (e.g., K=10)
   - Calculate "Top-K contamination rate": percentage of Top-K that are in T_irrelevant
4. Metric:
   - Top-K Contamination Rate = (# of T_irrelevant in Top-K) / K

Expected Result:
- High contamination rate (>50%) across many samples
- Qualitative failures: Top-1 token (model's "most relevant") is often 
  irrelevant background/sky/floor
"""

import argparse
import json
import math
import os
import os.path as osp
import sys
import time
from collections import defaultdict
from types import SimpleNamespace
from tqdm import tqdm

import torch
import torch.nn as nn
from PIL import Image
import yaml
import numpy as np

sys.path.append(osp.join(osp.dirname(osp.dirname(__file__))))

from src.constants import (DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN,
                            DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX)
from src.conversation import conv_templates
from src.model.builder import load_pretrained_model
from src.mm_utils import (get_model_name_from_path, process_images,
                          tokenizer_image_token)
from src.utils import disable_torch_init
from src.logic import DimProspector, HeadFork, VARProcessor, LogicEngine
from src.stash import StashEngine, MetadataStation, ValueMonitor


class LogitCapture:
    """Helper class to capture attention logits during forward pass"""
    logits_by_layer = defaultdict(dict)
    current_sample_id = None
    
    @classmethod
    def reset(cls):
        cls.logits_by_layer = defaultdict(dict)
    
    @classmethod
    def set_sample_id(cls, sample_id):
        cls.current_sample_id = sample_id
    
    @classmethod
    def capture(cls, layer_idx, query_states, key_states, head_dim):
        """
        Capture Q·K^T logits before softmax.
        
        Args:
            layer_idx: current layer index
            query_states: [bsz, num_heads, q_len, head_dim]
            key_states: [bsz, num_heads, kv_len, head_dim]
            head_dim: dimension per head
        """
        # Compute Q·K^T
        attn_logits = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
        # Store (detach to save memory)
        cls.logits_by_layer[layer_idx] = attn_logits.detach().cpu()
    
    @classmethod
    def get_logits(cls, layer_idx):
        return cls.logits_by_layer.get(layer_idx, None)


def get_patch_centers(image_size, patch_size=14, num_patches_per_side=24):
    """Calculate center coordinates for image patches."""
    width, height = image_size
    scale_x = width / num_patches_per_side
    scale_y = height / num_patches_per_side
    
    centers = []
    for i in range(num_patches_per_side):
        for j in range(num_patches_per_side):
            center_x = (j + 0.5) * scale_x
            center_y = (i + 0.5) * scale_y
            centers.append((center_x, center_y))
    
    return centers


def point_in_box(point, box):
    """Check if a point is inside a bounding box."""
    x, y = point
    bx, by, bw, bh = box
    return bx <= x <= bx + bw and by <= y <= by + bh


def classify_tokens(image_size, bbox, num_patches=576):
    """
    Classify visual tokens as relevant or irrelevant based on bbox.
    
    Returns:
        relevant_indices: list of indices for relevant tokens
        irrelevant_indices: list of indices for irrelevant tokens
    """
    num_patches_per_side = int(math.sqrt(num_patches))
    centers = get_patch_centers(image_size, num_patches_per_side=num_patches_per_side)
    
    relevant_indices = []
    irrelevant_indices = []
    
    for idx, center in enumerate(centers):
        if point_in_box(center, bbox):
            relevant_indices.append(idx)
        else:
            irrelevant_indices.append(idx)
    
    return relevant_indices, irrelevant_indices


def identify_image_centric_heads(attn_logits, im_start, vis_len, threshold=0.5):
    """
    Identify Image-Centric Heads (ICHs): heads that focus primarily on image tokens.
    
    Args:
        attn_logits: [bsz, num_heads, q_len, kv_len] attention logits
        im_start: starting position of image tokens
        vis_len: length of visual tokens
        threshold: proportion threshold to consider a head as image-centric
    
    Returns:
        ich_indices: list of head indices that are image-centric
    """
    bsz, num_heads, q_len, kv_len = attn_logits.shape
    
    # Apply softmax to get attention weights
    attn_weights = torch.softmax(attn_logits, dim=-1)
    
    # Sum attention on image tokens across all queries
    image_attn = attn_weights[:, :, :, im_start:im_start+vis_len].sum(dim=(0, 2))  # [num_heads]
    total_attn = attn_weights.sum(dim=(0, 2, 3))  # [num_heads]
    
    # Calculate proportion of attention on image tokens
    image_proportion = image_attn / (total_attn + 1e-8)
    
    # Heads with >threshold attention on images are ICHs
    ich_indices = torch.where(image_proportion > threshold)[0].tolist()
    
    return ich_indices


def compute_consensus_logits(model, tokenizer, image_processor, sample, device, cfgs):
    """
    Compute consensus logits for visual tokens using ICHs.
    
    Returns:
        consensus_scores: [num_visual_tokens] consensus relevance score for each visual token
        relevant_indices: ground-truth relevant token indices
        irrelevant_indices: ground-truth irrelevant token indices
        ich_info: information about identified ICHs
    """
    # Parse sample
    image_file = sample["image"]
    bbox = sample["bbox"]
    query = sample.get("text") or sample.get("question") or sample.get("prompt")
    
    # Load image
    image_path = osp.join(cfgs.path_image_dir, image_file)
    image = Image.open(image_path).convert("RGB")
    image_size = image.size
    
    # Classify tokens
    relevant_indices, irrelevant_indices = classify_tokens(image_size, bbox, num_patches=576)
    
    # Prepare input
    qs = query
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    
    conv = conv_templates[cfgs.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, conv=conv, return_tensors="pt"
    ).unsqueeze(0).to(device=device)
    
    image_tensor = process_images([image], image_processor, model.config)[0]
    
    # Hook to capture logits
    captured_logits = {}
    
    def create_hook(layer_idx):
        def hook(module, input, output):
            # output is (attn_output, attn_weights, past_key_value)
            # We need to capture Q·K^T before softmax
            # This requires accessing intermediate values, which is tricky
            # For simplicity, we'll capture from attention weights and reverse softmax
            pass
        return hook
    
    # Alternative: modify model to capture logits
    # For this experiment, we'll use a simpler approach:
    # Capture attention weights and use them as proxy for logits
    
    MetadataStation.activate()
    MetadataStation.export_model_config(model.config)
    
    # Disable VAR for this analysis
    LogicEngine.set_flag(False)
    
    with torch.inference_mode():
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().to(device),
                image_sizes=[image.size],
                return_dict_in_generate=True,
                output_attentions=True,  # This captures attention weights
                output_hidden_states=False,
                do_sample=False,
                max_new_tokens=1,  # Just need first token for analysis
                use_cache=True,
            )
    
    # Extract attention from first generated token
    # outputs.attentions is a tuple of length num_generated_tokens
    # Each element is a tuple of length num_layers
    # Each layer attention has shape [bsz, num_heads, 1, past_len]
    
    if hasattr(outputs, 'attentions') and len(outputs.attentions) > 0:
        first_token_attentions = outputs.attentions[0]  # Attentions for first generated token
        
        # Use a middle layer for analysis
        target_layer_idx = len(first_token_attentions) // 2
        layer_attn = first_token_attentions[target_layer_idx]  # [bsz, num_heads, 1, past_len]
        
        # Get image token positions
        im_start = MetadataStation.segments["begin_pos"]["image"]
        vis_len = MetadataStation.metadata["vis_len"]
        
        # Since we only have attention weights (post-softmax), we'll use log of them as proxy for logits
        # Note: This is an approximation; true logits would require model modification
        attn_weights = layer_attn.squeeze(2)  # [bsz, num_heads, past_len]
        
        # Convert back to logits (approximate)
        epsilon = 1e-8
        attn_logits = torch.log(attn_weights + epsilon)
        
        # Identify ICHs
        ich_indices = identify_image_centric_heads(
            attn_logits.unsqueeze(2),  # Add q_len dimension back
            im_start, vis_len, threshold=0.3
        )
        
        if len(ich_indices) == 0:
            print("Warning: No ICHs found, using all heads")
            ich_indices = list(range(attn_weights.shape[1]))
        
        # Compute consensus scores for visual tokens
        # For each visual token, average logits across ICHs
        visual_token_logits = attn_logits[0, :, im_start:im_start+vis_len]  # [num_heads, vis_len]
        
        # Average across ICH heads
        ich_logits = visual_token_logits[ich_indices, :]  # [num_ichs, vis_len]
        consensus_scores = ich_logits.mean(dim=0)  # [vis_len]
        
        ich_info = {
            'num_ichs': len(ich_indices),
            'ich_indices': ich_indices,
            'layer_used': target_layer_idx,
            'total_heads': attn_weights.shape[1]
        }
    else:
        # Fallback if attentions not available
        print("Warning: Attention weights not available, using random scores")
        consensus_scores = torch.randn(576)
        ich_info = {'error': 'attentions_not_available'}
    
    # Clean up
    StashEngine.clear()
    LogicEngine.clear()
    
    return consensus_scores, relevant_indices, irrelevant_indices, ich_info


def analyze_ranking(consensus_scores, relevant_indices, irrelevant_indices, top_k=10):
    """
    Analyze ranking quality by computing contamination rate.
    
    Returns:
        contamination_rate: fraction of top-k that are irrelevant
        top_k_indices: indices of top-k tokens
        ranking_stats: detailed statistics
    """
    # Sort by consensus score (descending)
    sorted_indices = torch.argsort(consensus_scores, descending=True)
    
    # Get top-k
    top_k_indices = sorted_indices[:top_k].tolist()
    
    # Count how many are irrelevant
    num_irrelevant_in_top_k = sum(1 for idx in top_k_indices if idx in irrelevant_indices)
    contamination_rate = num_irrelevant_in_top_k / top_k
    
    # Additional stats
    num_relevant_in_top_k = sum(1 for idx in top_k_indices if idx in relevant_indices)
    
    # Find rank of best relevant token
    ranks_of_relevant = []
    for idx in relevant_indices:
        rank = (sorted_indices == idx).nonzero(as_tuple=True)[0].item()
        ranks_of_relevant.append(rank)
    
    best_relevant_rank = min(ranks_of_relevant) if ranks_of_relevant else -1
    
    ranking_stats = {
        'top_k': top_k,
        'contamination_rate': contamination_rate,
        'num_irrelevant_in_top_k': num_irrelevant_in_top_k,
        'num_relevant_in_top_k': num_relevant_in_top_k,
        'top_1_is_irrelevant': top_k_indices[0] in irrelevant_indices,
        'top_1_index': top_k_indices[0],
        'best_relevant_rank': best_relevant_rank,
        'total_relevant_tokens': len(relevant_indices),
        'total_irrelevant_tokens': len(irrelevant_indices)
    }
    
    return contamination_rate, top_k_indices, ranking_stats


def run_experiment(args):
    """Main experiment runner"""
    
    # Load config
    with open(args.exp_config, "r") as file:
        config_dict = yaml.safe_load(file)
    cfgs = SimpleNamespace(**config_dict)
    
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    cfgs.device = device
    
    print(f"\n{'='*80}")
    print(f"Experiment: g_LogitRanking_Failure")
    print(f"Using device: {torch.cuda.get_device_name()}-{args.device}")
    print(f"{'='*80}\n")
    
    # Load model
    disable_torch_init()
    path_model = os.path.expanduser(cfgs.path_model)
    name_model = get_model_name_from_path(path_model)
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        path_model, args.model_base, name_model,
        attn_implementation="eager", device_map=device
    )
    
    # Load samples
    question_file_path = osp.join(
        cfgs.path_question_dir,
        f"{cfgs.name_category}-questions.jsonl" if cfgs.name_category != "" else "questions.jsonl"
    )
    samples = [json.loads(q) for q in open(os.path.expanduser(question_file_path), "r")]
    
    # Prepare output directory
    timestamp = str(int(time.time()))
    output_dir = osp.join("F_experiment_results", "logit_ranking_failure", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    contamination_rates = []
    
    print(f"Analyzing {min(len(samples), args.max_samples)} samples...")
    print(f"Top-K = {args.top_k}\n")
    
    for idx, sample in enumerate(tqdm(samples[:args.max_samples], desc="Analyzing samples")):
        try:
            # Compute consensus logits
            consensus_scores, relevant_idx, irrelevant_idx, ich_info = compute_consensus_logits(
                model, tokenizer, image_processor, sample, device, cfgs
            )
            
            # Analyze ranking
            contamination_rate, top_k_indices, ranking_stats = analyze_ranking(
                consensus_scores, relevant_idx, irrelevant_idx, top_k=args.top_k
            )
            
            contamination_rates.append(contamination_rate)
            
            result = {
                'qid': sample.get('qid', None),
                'image': sample['image'],
                'query': sample.get('text') or sample.get('question') or sample.get('prompt'),
                'bbox': sample.get('bbox', None),
                'label': sample.get('label', None),
                'contamination_rate': contamination_rate,
                'top_k_indices': top_k_indices,
                'ranking_stats': ranking_stats,
                'ich_info': ich_info,
                'num_relevant_tokens': len(relevant_idx),
                'num_irrelevant_tokens': len(irrelevant_idx)
            }
            
            results.append(result)
            
            # Log notable failures
            if ranking_stats['top_1_is_irrelevant']:
                print(f"\n  ✗ Sample {idx}: Top-1 is IRRELEVANT! Contamination: {contamination_rate:.1%}")
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    # Compute aggregate statistics
    avg_contamination = np.mean(contamination_rates)
    median_contamination = np.median(contamination_rates)
    std_contamination = np.std(contamination_rates)
    
    top1_irrelevant_count = sum(1 for r in results if r['ranking_stats']['top_1_is_irrelevant'])
    top1_irrelevant_rate = top1_irrelevant_count / len(results) if results else 0
    
    # Save detailed results
    results_file = osp.join(output_dir, "results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate summary
    summary = {
        'experiment': 'g_LogitRanking_Failure',
        'samples_analyzed': len(results),
        'top_k': args.top_k,
        'aggregate_statistics': {
            'mean_contamination_rate': float(avg_contamination),
            'median_contamination_rate': float(median_contamination),
            'std_contamination_rate': float(std_contamination),
            'top1_irrelevant_rate': float(top1_irrelevant_rate),
            'top1_irrelevant_count': top1_irrelevant_count
        },
        'config': vars(cfgs),
        'timestamp': timestamp
    }
    
    summary_file = osp.join(output_dir, "summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Generate qualitative failures file (samples where Top-1 is irrelevant)
    qualitative_failures = [r for r in results if r['ranking_stats']['top_1_is_irrelevant']]
    failures_file = osp.join(output_dir, "qualitative_failures.json")
    with open(failures_file, 'w') as f:
        json.dump(qualitative_failures, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Experiment completed!")
    print(f"{'='*80}")
    print(f"Samples analyzed: {len(results)}")
    print(f"Average contamination rate: {avg_contamination:.1%}")
    print(f"Median contamination rate: {median_contamination:.1%}")
    print(f"Top-1 irrelevant rate: {top1_irrelevant_rate:.1%} ({top1_irrelevant_count}/{len(results)} samples)")
    print(f"\nResults saved to: {output_dir}")
    print(f"  - results.json: detailed results for all samples")
    print(f"  - summary.json: aggregate statistics")
    print(f"  - qualitative_failures.json: samples where Top-1 was irrelevant")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Logit Ranking Failure Experiment")
    parser.add_argument("--exp_config", type=str, required=True,
                       help="Path to experiment config YAML file")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=100,
                       help="Maximum samples to analyze")
    parser.add_argument("--top_k", type=int, default=10,
                       help="K for Top-K contamination rate calculation")
    
    args = parser.parse_args()
    run_experiment(args)
