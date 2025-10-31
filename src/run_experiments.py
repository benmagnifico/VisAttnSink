#!/usr/bin/env python3
"""
Main script to run VAR assumption validation experiments.

Experiments:
1. g_HarmfulRecycling_Intervention: Demonstrates that relevant tokens can be
   misclassified as sink tokens, causing VAR to harm performance.
   
2. g_LogitRanking_Failure: Demonstrates that key-based relevance (Q·K^T) is
   unreliable for ranking token importance.
"""

import argparse
import os
import os.path as osp
import sys
import yaml
from types import SimpleNamespace

import torch
from PIL import Image

# Add src to path
sys.path.append(osp.dirname(osp.dirname(__file__)))

from src.model.builder import load_pretrained_model
from src.mm_utils import get_model_name_from_path
from src.utils import disable_torch_init
from src.logic import LogicEngine, DimProspector, HeadFork, VARProcessor
from src.stash import MetadataStation

from src.experiments import (
    RefCOCOLoader,
    BoundingBoxMapper,
    HarmfulRecyclingExperiment,
    LogitRankingExperiment,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Run VAR assumption validation experiments')
    
    parser.add_argument(
        '--experiment',
        type=str,
        required=True,
        choices=['harmful_recycling', 'logit_ranking', 'both'],
        help='Which experiment to run'
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to LLaVA model checkpoint'
    )
    
    parser.add_argument(
        '--model_base',
        type=str,
        default=None,
        help='Base model path (if using LoRA)'
    )
    
    parser.add_argument(
        '--dataset_path',
        type=str,
        default='C_datasets/refcoco',
        help='Path to RefCOCO dataset'
    )
    
    parser.add_argument(
        '--dataset_split',
        type=str,
        default='val',
        choices=['train', 'val', 'test'],
        help='Dataset split to use'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='E_experiments',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='0',
        help='CUDA device number'
    )
    
    parser.add_argument(
        '--tau',
        type=float,
        default=20.0,
        help='Threshold for sink token detection'
    )
    
    parser.add_argument(
        '--num_search_samples',
        type=int,
        default=100,
        help='Number of samples to search for harmful recycling experiment'
    )
    
    parser.add_argument(
        '--num_intervention_samples',
        type=int,
        default=5,
        help='Number of samples to run intervention on'
    )
    
    parser.add_argument(
        '--num_logit_samples',
        type=int,
        default=50,
        help='Number of samples for logit ranking experiment'
    )
    
    parser.add_argument(
        '--layers',
        type=str,
        default=None,
        help='Comma-separated layer indices to analyze (e.g., "10,20,30")'
    )
    
    return parser.parse_args()


def load_model(args):
    """Load the LLaVA model."""
    print("Loading model...")
    
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    
    disable_torch_init()
    path_model = os.path.expanduser(args.model_path)
    name_model = get_model_name_from_path(path_model)
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        path_model,
        args.model_base,
        name_model,
        attn_implementation="eager",
        device_map=device
    )
    
    # Activate metadata station
    MetadataStation.activate()
    MetadataStation.export_model_config(model.config)
    
    # Activate logic engine with specified tau
    LogicEngine.activate(
        tau=args.tau,
        rho=0.5,
        summ=0.2,
        p=0.6,
        except_last_layer=True
    )
    
    print(f"Model loaded on {device}")
    print(f"Model: {name_model}")
    print(f"Tau: {args.tau}")
    
    return model, tokenizer, image_processor, device


def run_harmful_recycling_experiment(args, model, tokenizer, image_processor, device):
    """Run Experiment 1: Harmful Recycling Intervention."""
    print("\n" + "="*80)
    print("EXPERIMENT 1: g_HarmfulRecycling_Intervention")
    print("="*80)
    print("\nObjective: Prove that 'relevant tokens' can be misclassified as 'sink tokens',")
    print("and that applying VAR in such cases actively harms model performance.\n")
    
    # Load dataset
    dataset_loader = RefCOCOLoader(args.dataset_path, args.dataset_split)
    
    # Create experiment
    output_dir = osp.join(args.output_dir, 'harmful_recycling')
    experiment = HarmfulRecyclingExperiment(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        dataset_loader=dataset_loader,
        tau=args.tau,
        device=device,
        output_dir=output_dir
    )
    
    # Run experiment
    results = experiment.run_full_experiment(
        num_search_samples=args.num_search_samples,
        num_intervention_samples=args.num_intervention_samples
    )
    
    print("\n" + "="*80)
    print("EXPERIMENT 1 COMPLETED")
    print("="*80)
    print(f"Status: {results['status']}")
    print(f"Samples searched: {results.get('num_samples_searched', 0)}")
    print(f"Samples with relevant sinks: {results.get('num_samples_with_relevant_sinks', 0)}")
    print(f"Interventions run: {results.get('num_interventions_run', 0)}")
    
    if results.get('results'):
        print("\nSample Results:")
        for i, result in enumerate(results['results'][:3]):  # Show first 3
            print(f"\n  Sample {i+1}:")
            print(f"    Query: {result['sample_info']['query']}")
            print(f"    Relevant sink tokens: {result['sample_info']['num_relevant_sinks']}")
            print(f"    Mode A (Baseline): {result['mode_a_baseline']['output'][:100]}...")
            print(f"    Mode B (VAR): {result['mode_b_var']['output'][:100]}...")
            print(f"    Mode C (Control): {result['mode_c_control']['output'][:100]}...")
    
    return results


def run_logit_ranking_experiment(args, model, tokenizer, image_processor, device):
    """Run Experiment 2: Logit Ranking Failure."""
    print("\n" + "="*80)
    print("EXPERIMENT 2: g_LogitRanking_Failure")
    print("="*80)
    print("\nObjective: Prove that 'key relevance' (Q·K^T logits) is unreliable,")
    print("even when using consensus across Image-Centric Heads (ICHs).\n")
    
    # Load dataset
    dataset_loader = RefCOCOLoader(args.dataset_path, args.dataset_split)
    
    # Parse layers to analyze
    layers_to_analyze = None
    if args.layers:
        layers_to_analyze = [int(x.strip()) for x in args.layers.split(',')]
        print(f"Analyzing layers: {layers_to_analyze}")
    
    # Create experiment
    output_dir = osp.join(args.output_dir, 'logit_ranking')
    experiment = LogitRankingExperiment(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        dataset_loader=dataset_loader,
        device=device,
        output_dir=output_dir
    )
    
    # Run experiment
    results = experiment.run_full_experiment(
        num_samples=args.num_logit_samples,
        layers_to_analyze=layers_to_analyze,
        k_values=[5, 10, 20, 50]
    )
    
    print("\n" + "="*80)
    print("EXPERIMENT 2 COMPLETED")
    print("="*80)
    print(f"Status: {results['status']}")
    print(f"Samples analyzed: {results.get('num_samples_analyzed', 0)}")
    
    if 'aggregate_stats' in results:
        print("\nAggregate Statistics:")
        for key, stats in results['aggregate_stats'].items():
            print(f"\n  {key}:")
            print(f"    Average pollution rate: {stats['avg_pollution_rate']:.2%}")
            print(f"    Std deviation: {stats['std_pollution_rate']:.2%}")
            print(f"    Top-1 relevant rate: {stats['top1_relevant_rate']:.2%}")
    
    return results


def main():
    args = parse_args()
    
    print("="*80)
    print("VAR ASSUMPTION VALIDATION EXPERIMENTS")
    print("="*80)
    print(f"\nExperiment: {args.experiment}")
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset_path} ({args.dataset_split} split)")
    print(f"Output: {args.output_dir}")
    
    # Load model
    model, tokenizer, image_processor, device = load_model(args)
    
    # Run experiments
    if args.experiment in ['harmful_recycling', 'both']:
        run_harmful_recycling_experiment(args, model, tokenizer, image_processor, device)
    
    if args.experiment in ['logit_ranking', 'both']:
        run_logit_ranking_experiment(args, model, tokenizer, image_processor, device)
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*80)


if __name__ == '__main__':
    main()
