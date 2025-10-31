"""
Experiment 1: g_HarmfulRecycling_Intervention (Harmful Recycling Intervention Experiment)

Goal: Prove that (1) "relevant tokens" can be misidentified as "sink tokens"; 
      (2) applying VAR in these cases actively harms model performance.

This experiment requires a visual grounding dataset with phrase-to-bounding-box alignment,
such as RefCOCO or Flickr30k Entities.

Method:
1. Sample Selection: Randomly select samples from RefCOCO dataset with query phrase Q
   and ground-truth bounding box B_gt
2. Define "truly relevant" vs "truly irrelevant":
   - T_relevant: visual tokens whose center points fall inside B_gt
   - T_irrelevant: visual tokens whose center points fall outside B_gt
3. Find "Relevant Sink Tokens": Tokens j_R where:
   - j_R ∈ T_relevant (actually relevant)
   - φ(x_{j_R}) >= τ (mistakenly identified as sink)
4. Intervention Experiment: Run three modes:
   - Mode A (Baseline): Normal LMM model
   - Mode B (VAR - paper method): Apply VAR which incorrectly recycles from j_R
   - Mode C (Control - Ideal VAR): Manually remove j_R from sink list, only recycle
     from truly irrelevant sinks (j ∈ T_irrelevant and φ(x_j) >= τ)

Expected Result:
- Mode A and C should give correct answers
- Mode B performance should significantly drop (hallucinations or "can't see clearly")
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
import copy

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


class InterventionVARProcessor(VARProcessor):
    """
    Extended VAR Processor with intervention capabilities.
    Can exclude specific tokens from the sink list for controlled experiments.
    """
    excluded_indices = []
    
    @classmethod
    def set_excluded_indices(cls, indices):
        """Set indices to exclude from sink recycling (for Mode C)"""
        cls.excluded_indices = indices
    
    @classmethod
    def clear_excluded_indices(cls):
        """Clear excluded indices"""
        cls.excluded_indices = []
    
    @classmethod
    def attn_redist(cls, attention_map, layer_idx):
        """
        Modified attention redistribution that respects excluded indices.
        """
        p = cls.p
        
        if cls.except_last_layer and cls.current_decoder_layer == cls.model_config.num_hidden_layers - 1:
            return attention_map    
    
        im, pa = MetadataStation.segments["begin_pos"]["image"], MetadataStation.metadata["vis_len"]
        coord = HeadFork.forked_head[layer_idx]
        indices = cls.__base__.indices[layer_idx]
        
        # Apply exclusion filter for Mode C
        if len(cls.excluded_indices) > 0:
            excluded_tensor = torch.tensor(cls.excluded_indices, device=indices.device)
            mask = ~torch.isin(indices, excluded_tensor)
            indices = indices[mask]
        
        if len(coord) > 0:
            model_head_num = MetadataStation.model_config["num_attention_heads"]
            for h in range(model_head_num):
                query_coord = coord[coord[:, 1]==h][:,2]
                query_coord = query_coord[im+pa<=query_coord] if ValueMonitor.get_output_token_count() < 0 else query_coord
                bsz_coord = coord[coord[:, 1] == h][:,0][:len(query_coord)]
                head_coord = coord[coord[:, 1]==h][:,1][:len(query_coord)]

                if not query_coord.shape[0] or not head_coord.shape[0]:
                    continue

                selected_attn_map = attention_map[bsz_coord, head_coord, query_coord, :].clone()
                indices = indices.to(selected_attn_map.device)
                vis_indices = indices[(im<=indices) & (indices<im+pa)]
                text_indices = indices[~torch.isin(indices, vis_indices)]

                copied_attention_map = copy.deepcopy(selected_attn_map.detach())

                selected_attn_map[:, text_indices] *= p
                selected_attn_map[:, vis_indices] *= p 

                weight_budget_vis = copied_attention_map[:, vis_indices].sum(dim=1) * (1 - p)
                weight_budget_text = copied_attention_map[:, text_indices].sum(dim=1) * (1 - p)

                copied_attention_map[:, vis_indices] *= 0  
                ratios_vis = copied_attention_map[:, im:im+pa] / copied_attention_map[:, im:im+pa].sum(dim=1, keepdim=True).to(selected_attn_map.dtype)                
                selected_attn_map[:, im:im+pa] += (weight_budget_vis + weight_budget_text).view(-1,1) * ratios_vis
                attention_map[bsz_coord, head_coord, query_coord, :] = selected_attn_map
        return attention_map


def get_patch_centers(image_size, patch_size=14, num_patches_per_side=24):
    """
    Calculate center coordinates for image patches.
    
    Args:
        image_size: (width, height) of original image
        patch_size: size of each patch in the vision encoder (default 14 for CLIP)
        num_patches_per_side: number of patches per side after encoding (default 24 for 336x336 input)
    
    Returns:
        centers: List of (x, y) center coordinates in original image space
    """
    width, height = image_size
    
    # Calculate scaling factor from patch space to image space
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
    """
    Check if a point is inside a bounding box.
    
    Args:
        point: (x, y) coordinates
        box: [x, y, w, h] bounding box
    
    Returns:
        bool: True if point is inside box
    """
    x, y = point
    bx, by, bw, bh = box
    return bx <= x <= bx + bw and by <= y <= by + bh


def classify_tokens(image_size, bbox, num_patches=576):
    """
    Classify visual tokens as relevant or irrelevant based on bbox.
    
    Args:
        image_size: (width, height) of image
        bbox: [x, y, w, h] ground truth bounding box
        num_patches: total number of visual patches (default 576 = 24x24)
    
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


def find_relevant_sink_tokens(model, tokenizer, image_processor, sample, device, cfgs):
    """
    Find tokens that are both relevant (in bbox) and identified as sink (φ(x) >= τ).
    
    Returns:
        relevant_sink_tokens: list of token indices that are relevant sinks
        all_sink_tokens: list of all identified sink token indices
        relevant_indices: list of relevant token indices
        irrelevant_indices: list of irrelevant token indices
    """
    # Parse sample
    image_file = sample["image"]
    bbox = sample["bbox"]  # [x, y, w, h]
    query = sample.get("text") or sample.get("question") or sample.get("prompt")
    
    # Load image
    image_path = osp.join(cfgs.path_image_dir, image_file)
    image = Image.open(image_path).convert("RGB")
    image_size = image.size
    
    # Classify tokens based on bbox
    relevant_indices, irrelevant_indices = classify_tokens(
        image_size, bbox, num_patches=576
    )
    
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
    
    # Run model with DimProspector active to identify sink tokens
    LogicEngine.activate(tau=cfgs.tau, rho=cfgs.rho, summ=cfgs.summ, p=cfgs.p, 
                        except_last_layer=cfgs.except_last_layer)
    
    with torch.inference_mode():
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().to(device),
                image_sizes=[image.size],
                return_dict_in_generate=True,
                output_attentions=False,
                output_hidden_states=False,
                do_sample=False,
                max_new_tokens=cfgs.max_new_tokens,
                use_cache=True,
            )
    
    # Get sink indices from DimProspector
    # Collect all sink indices across layers
    all_sink_indices_by_layer = DimProspector.indices
    
    # For simplicity, use indices from a middle layer (e.g., layer 16 for 32-layer model)
    target_layer = len(all_sink_indices_by_layer) // 2 if len(all_sink_indices_by_layer) > 0 else 0
    sink_indices_raw = all_sink_indices_by_layer.get(target_layer, torch.tensor([]))
    
    # Convert to list and adjust for image token offset
    im_start = MetadataStation.segments["begin_pos"]["image"]
    vis_len = MetadataStation.metadata["vis_len"]
    
    sink_indices = []
    for idx in sink_indices_raw:
        idx_val = idx.item()
        # Check if this is a visual token
        if im_start <= idx_val < im_start + vis_len:
            # Convert to relative index within visual tokens
            rel_idx = idx_val - im_start
            sink_indices.append(rel_idx)
    
    # Find intersection: relevant tokens that are also sinks
    relevant_sink_tokens = list(set(relevant_indices) & set(sink_indices))
    
    # Clean up
    StashEngine.clear()
    LogicEngine.clear()
    
    return relevant_sink_tokens, sink_indices, relevant_indices, irrelevant_indices


def run_inference_mode(model, tokenizer, image_processor, sample, device, cfgs, 
                       mode='A', excluded_indices=None):
    """
    Run inference in one of three modes.
    
    Args:
        mode: 'A' (Baseline), 'B' (VAR), or 'C' (Ideal VAR)
        excluded_indices: For mode C, indices to exclude from sink recycling
    
    Returns:
        response: generated text response
    """
    # Parse sample
    image_file = sample["image"]
    query = sample.get("text") or sample.get("question") or sample.get("prompt")
    
    # Load image
    image_path = osp.join(cfgs.path_image_dir, image_file)
    image = Image.open(image_path).convert("RGB")
    
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
    
    # Configure model based on mode
    if mode == 'A':
        # Mode A: Baseline - no VAR
        LogicEngine.set_flag(False)
        InterventionVARProcessor.clear_excluded_indices()
    elif mode == 'B':
        # Mode B: VAR - apply standard VAR
        LogicEngine.activate(tau=cfgs.tau, rho=cfgs.rho, summ=cfgs.summ, p=cfgs.p,
                           except_last_layer=cfgs.except_last_layer)
        InterventionVARProcessor.clear_excluded_indices()
    elif mode == 'C':
        # Mode C: Ideal VAR - apply VAR but exclude relevant sink tokens
        LogicEngine.activate(tau=cfgs.tau, rho=cfgs.rho, summ=cfgs.summ, p=cfgs.p,
                           except_last_layer=cfgs.except_last_layer)
        if excluded_indices is not None:
            # Convert relative indices to absolute indices
            im_start = MetadataStation.segments["begin_pos"]["image"]
            abs_excluded = [im_start + idx for idx in excluded_indices]
            InterventionVARProcessor.set_excluded_indices(abs_excluded)
    
    # Run inference
    with torch.inference_mode():
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().to(device),
                image_sizes=[image.size],
                return_dict_in_generate=True,
                output_attentions=False,
                output_hidden_states=False,
                do_sample=False,
                max_new_tokens=cfgs.max_new_tokens,
                use_cache=True,
            )
    
    response = tokenizer.batch_decode(outputs.sequences)[0]
    
    # Clean up
    StashEngine.clear()
    LogicEngine.clear()
    InterventionVARProcessor.clear_excluded_indices()
    
    return response


def run_experiment(args):
    """Main experiment runner"""
    
    # Load config
    with open(args.exp_config, "r") as file:
        config_dict = yaml.safe_load(file)
    cfgs = SimpleNamespace(**config_dict)
    
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    cfgs.device = device
    
    print(f"\n{'='*80}")
    print(f"Experiment: g_HarmfulRecycling_Intervention")
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
    
    # Activate metadata tracking
    MetadataStation.activate()
    MetadataStation.export_model_config(model.config)
    
    # Replace VARProcessor with InterventionVARProcessor
    import src.logic as logic_module
    logic_module.VARProcessor = InterventionVARProcessor
    
    # Load samples
    question_file_path = osp.join(
        cfgs.path_question_dir,
        f"{cfgs.name_category}-questions.jsonl" if cfgs.name_category != "" else "questions.jsonl"
    )
    samples = [json.loads(q) for q in open(os.path.expanduser(question_file_path), "r")]
    
    # Prepare output directory
    timestamp = str(int(time.time()))
    output_dir = osp.join("F_experiment_results", "harmful_recycling_intervention", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    samples_with_relevant_sinks = []
    
    print(f"Searching for samples with relevant sink tokens...")
    print(f"Total samples to examine: {len(samples)}\n")
    
    # Phase 1: Find samples with relevant sink tokens
    for idx, sample in enumerate(tqdm(samples[:args.max_samples], desc="Finding relevant sinks")):
        try:
            relevant_sinks, all_sinks, relevant_idx, irrelevant_idx = find_relevant_sink_tokens(
                model, tokenizer, image_processor, sample, device, cfgs
            )
            
            if len(relevant_sinks) > 0:
                samples_with_relevant_sinks.append({
                    'sample': sample,
                    'relevant_sinks': relevant_sinks,
                    'all_sinks': all_sinks,
                    'relevant_indices': relevant_idx,
                    'irrelevant_indices': irrelevant_idx
                })
                print(f"\n✓ Found sample {idx} with {len(relevant_sinks)} relevant sink token(s)!")
                
                if len(samples_with_relevant_sinks) >= args.num_intervention_samples:
                    break
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    print(f"\n{'='*80}")
    print(f"Found {len(samples_with_relevant_sinks)} samples with relevant sink tokens")
    print(f"{'='*80}\n")
    
    # Phase 2: Run intervention experiments on found samples
    for sample_data in tqdm(samples_with_relevant_sinks, desc="Running interventions"):
        sample = sample_data['sample']
        relevant_sinks = sample_data['relevant_sinks']
        
        print(f"\nProcessing sample: {sample.get('qid', 'unknown')}")
        print(f"  Relevant sink tokens: {len(relevant_sinks)}")
        
        # Run three modes
        response_A = run_inference_mode(model, tokenizer, image_processor, sample,
                                       device, cfgs, mode='A')
        response_B = run_inference_mode(model, tokenizer, image_processor, sample,
                                       device, cfgs, mode='B')
        response_C = run_inference_mode(model, tokenizer, image_processor, sample,
                                       device, cfgs, mode='C', 
                                       excluded_indices=relevant_sinks)
        
        result = {
            'qid': sample.get('qid', None),
            'image': sample['image'],
            'query': sample.get('text') or sample.get('question') or sample.get('prompt'),
            'bbox': sample.get('bbox', None),
            'label': sample.get('label', None),
            'relevant_sink_tokens': relevant_sinks,
            'num_relevant_sinks': len(relevant_sinks),
            'num_all_sinks': len(sample_data['all_sinks']),
            'num_relevant_total': len(sample_data['relevant_indices']),
            'response_mode_A_baseline': response_A,
            'response_mode_B_var': response_B,
            'response_mode_C_ideal_var': response_C,
        }
        
        results.append(result)
        
        # Print results for this sample
        print(f"\n  Mode A (Baseline): {response_A[:100]}...")
        print(f"  Mode B (VAR):      {response_B[:100]}...")
        print(f"  Mode C (Ideal):    {response_C[:100]}...")
    
    # Save results
    results_file = osp.join(output_dir, "results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate summary
    summary = {
        'experiment': 'g_HarmfulRecycling_Intervention',
        'total_samples_examined': args.max_samples,
        'samples_with_relevant_sinks': len(samples_with_relevant_sinks),
        'intervention_experiments_run': len(results),
        'config': vars(cfgs),
        'timestamp': timestamp
    }
    
    summary_file = osp.join(output_dir, "summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Experiment completed!")
    print(f"Results saved to: {output_dir}")
    print(f"  - results.json: detailed results for each sample")
    print(f"  - summary.json: experiment summary")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Harmful Recycling Intervention Experiment")
    parser.add_argument("--exp_config", type=str, required=True,
                       help="Path to experiment config YAML file")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=100,
                       help="Maximum samples to examine for relevant sinks")
    parser.add_argument("--num_intervention_samples", type=int, default=10,
                       help="Number of samples to run full intervention on")
    
    args = parser.parse_args()
    run_experiment(args)
