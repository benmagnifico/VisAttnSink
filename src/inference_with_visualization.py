"""
Inference with Attention Capture for Visual Attention Sink Analysis

This script extends the standard inference to capture and save attention patterns
for visualization and analysis of the visual attention sink phenomenon.

Usage:
    python src/inference_with_visualization.py --device 0 --exp_config A_exps/lv1.5_7b_viz.yml
"""

import argparse
import gc
import json
import math
import os
import os.path as osp
from pprint import pprint
import pickle
import sys
sys.path.append(osp.join(osp.dirname(osp.dirname(__file__))))

import time
from types import SimpleNamespace

import torch
from PIL import Image
from tqdm import tqdm
import yaml

from src.constants import (DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN,
                            DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX)
from src.conversation import conv_templates
from src.model.builder import load_pretrained_model
from src.mm_utils import (get_model_name_from_path, process_images,
                           tokenizer_image_token)
from src.utils import disable_torch_init

from src.logic import DimProspector, HeadFork, VARProcessor, LogicEngine
from src.stash import StashEngine, MetadataStation
from src.visualization.attention_capture import get_attention_capture
from src.visualization.attention_visualizer import AttentionVisualizer


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model_with_visualization(args):
    """
    Run model inference with attention capture for visualization.
    """
    with open(args.exp_config, "r") as file:
        config_dict = yaml.safe_load(file)
    cfgs = SimpleNamespace(**config_dict)

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    cfgs.device = device
    
    print("\n\n\n")
    print(f"Using device: {torch.cuda.get_device_name()}-{args.device}")
    pprint(vars(cfgs))
    print("\n\n\n")

    disable_torch_init()
    path_model = os.path.expanduser(cfgs.path_model)
    name_model = get_model_name_from_path(path_model)
    cfgs.name_model = name_model
    
    # Load model
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        path_model, 
        args.model_base, 
        name_model, 
        attn_implementation="eager",  # Required for attention capture
        device_map=device
    )

    # Activate StashEngine
    MetadataStation.activate()
    MetadataStation.export_model_config(model.config)
    
    # Activate logic if specified
    if getattr(cfgs, "logic", 0) == 1:
        LogicEngine.activate(
            tau=cfgs.tau, 
            rho=cfgs.rho, 
            summ=cfgs.summ, 
            p=cfgs.p, 
            except_last_layer=cfgs.except_last_layer
        )

    # Initialize attention capture
    attention_capture = get_attention_capture(save_dir="F_visualizations")
    visualizer = AttentionVisualizer(save_dir="F_visualizations")
    
    # Enable attention capture
    capture_enabled = getattr(cfgs, "capture_attention", True)
    if capture_enabled:
        attention_capture.activate()
        print("Attention capture ENABLED")
    
    # Load questions
    question_file_path = osp.join(
        cfgs.path_question_dir, 
        f"{cfgs.name_category}-questions.jsonl" if cfgs.name_category != "" else "questions.jsonl"
    )
    questions = [json.loads(q) for q in open(os.path.expanduser(question_file_path), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    
    # Limit number of questions for visualization
    max_questions = getattr(cfgs, "max_viz_questions", 5)
    questions = questions[:max_questions]
    
    answer_file_ver = f"[{cfgs.name_daset}-{cfgs.name_category}]{cfgs.name_exp}-viz-{str(int(time.time()))}"
    answers_file = osp.join("E_answers", cfgs.name_model, f"{answer_file_ver}.jsonl")
    answers_file = os.path.expanduser(answers_file)
    os.makedirs(osp.dirname(answers_file), exist_ok=True)

    file_mode = "w"
    with open(answers_file, file_mode) as ans_file:
        setattr(model, "tokenizer", tokenizer)
        
        for line in tqdm(questions):
            try:    
                qid = int(line.get("qid", None) or line.get("question_id", None))
                gt_label = line.get("label", None) or line.get("answer", None) or line.get("gt-label", None)

                image_file = line["image"]
                _, img_ext = osp.splitext(image_file)
                if img_ext is None or img_ext == "":
                    image_file = f"{image_file}.jpg"

                qs = line.get("text") or line.get("question") or line.get("prompt")
                assert qs is not None
                cur_prompt = qs
                
                if model.config.mm_use_im_start_end:
                    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

                conv = conv_templates[cfgs.conv_mode].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                # Clear previous attention data
                attention_capture.clear()
                
                # Prepare input
                input_ids = tokenizer_image_token(
                    prompt, 
                    tokenizer, 
                    IMAGE_TOKEN_INDEX, 
                    conv=conv, 
                    return_tensors="pt"
                ).unsqueeze(0).to(device=device)
                
                image = Image.open(os.path.join(cfgs.path_image_dir, image_file)).convert("RGB")
                image_tensor = process_images([image], image_processor, model.config)[0]
                
                # Calculate visual token positions
                # In LLaVA, image tokens are inserted at the position of IMAGE_TOKEN_INDEX
                image_token_positions = (input_ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)
                if len(image_token_positions[1]) > 0:
                    vis_token_start = image_token_positions[1][0].item()
                    # LLaVA-1.5 uses 576 visual tokens (24x24 patches)
                    vis_token_end = vis_token_start + 576
                else:
                    vis_token_start = 0
                    vis_token_end = 576
                
                # Run inference with attention capture
                with torch.inference_mode():
                    with torch.no_grad():
                        setattr(model, "tokenizer", tokenizer)
                        outputs = model.generate(
                            input_ids,
                            images=image_tensor.unsqueeze(0).half().to(device),
                            image_sizes=[image.size],
                            return_dict_in_generate=True,
                            output_attentions=True,
                            output_hidden_states=True,
                            do_sample=False,
                            max_new_tokens=cfgs.max_new_tokens,
                            use_cache=True,
                        )
                
                # Process attention outputs
                if capture_enabled and hasattr(outputs, 'attentions') and outputs.attentions:
                    print(f"\nProcessing attention for question {qid}...")
                    
                    # outputs.attentions is a tuple of tuples
                    # First level: generation steps
                    # Second level: layers
                    for step_idx, step_attentions in enumerate(outputs.attentions):
                        for layer_idx, layer_attention in enumerate(step_attentions):
                            attention_capture.store_attention(
                                layer_idx=layer_idx,
                                attention_weights=layer_attention,
                                token_idx=step_idx - 1  # -1 for prefill, 0+ for generation
                            )
                
                generated_texts = tokenizer.batch_decode(outputs.sequences)[0]

                # Store metadata
                if capture_enabled:
                    attention_capture.store_metadata(
                        question_id=qid,
                        question=cur_prompt,
                        image_path=image_file,
                        response=generated_texts,
                        vis_token_start=vis_token_start,
                        vis_token_end=vis_token_end,
                        total_tokens=input_ids.shape[1]
                    )
                    
                    # Save attention data
                    attn_filename = f"attention_qid{qid}.pkl"
                    attention_capture.save(attn_filename)
                    
                    # Generate visualizations immediately for this question
                    print(f"Generating visualizations for question {qid}...")
                    try:
                        # Get the saved data
                        attn_data = attention_capture.load(
                            os.path.join("F_visualizations", attn_filename)
                        )
                        
                        # Generate visualization for middle layer (layer 15 for 32-layer model)
                        middle_layer = model.config.num_hidden_layers // 2
                        
                        if middle_layer in attn_data['attention']:
                            layer_attn = attn_data['attention'][middle_layer]
                            if -1 in layer_attn:  # Prefill phase
                                attn_tensor = layer_attn[-1]
                                if isinstance(attn_tensor, list):
                                    attn_tensor = attn_tensor[0]
                                if isinstance(attn_tensor, torch.Tensor):
                                    attn_tensor = attn_tensor.numpy()
                                
                                # Remove batch dimension if present
                                if len(attn_tensor.shape) == 4:
                                    attn_tensor = attn_tensor[0]
                                
                                # Visualize first head
                                attn_matrix = attn_tensor[0]
                                visualizer.visualize_sink_tokens(
                                    attn_matrix,
                                    middle_layer,
                                    0,
                                    vis_token_start,
                                    vis_token_end,
                                    f"preview_qid{qid}_layer{middle_layer}_head0.png",
                                    threshold=0.1
                                )
                    except Exception as viz_error:
                        print(f"Warning: Could not generate preview visualization: {viz_error}")

                # Write answer
                ans_file.write(json.dumps(
                    {"question_id": qid, 
                     "prompt": cur_prompt, 
                     "label": gt_label, 
                     "response": generated_texts, 
                     "image": image_file, 
                     "model_id": name_model}
                ) + "\n")
                ans_file.flush()
                
                del outputs
                StashEngine.clear()
                LogicEngine.clear()
                
            except Exception as e:
                print(f"Error processing question {qid}: {e}")
                import traceback
                traceback.print_exc()
                continue

    print(f"\n✓ Inference complete. Answers saved to {answers_file}")
    print(f"✓ Attention visualizations saved to F_visualizations/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--exp_config", type=str, default=None)
    
    args = parser.parse_args()
    
    eval_model_with_visualization(args)
