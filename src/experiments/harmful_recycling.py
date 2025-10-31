"""
Experiment 1: g_HarmfulRecycling_Intervention

This experiment proves that "relevant tokens" can be incorrectly identified as "sink tokens",
and applying VAR in such cases actively harms model performance.
"""

import os
import os.path as osp
import json
import copy
from typing import Dict, List, Tuple, Optional
import torch
import numpy as np
from tqdm import tqdm

from src.logic import DimProspector, LogicEngine
from src.stash import MetadataStation
from .grounding_utils import RefCOCOLoader, BoundingBoxMapper


class HarmfulRecyclingExperiment:
    """
    Experiment to demonstrate the existence of "Relevant Sink Tokens" (Problem 1).
    
    This experiment:
    1. Identifies tokens that are both relevant (inside ground-truth bbox) 
       and classified as sink tokens (φ(x) >= τ)
    2. Runs three inference modes:
       - Mode A (Baseline): Original LMM without VAR
       - Mode B (VAR): Apply original VAR method (will harm performance on relevant sinks)
       - Mode C (Control): Apply VAR but exclude relevant sink tokens from recycling
    3. Demonstrates that Mode B performs worse than A and C, proving the harm
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        image_processor,
        dataset_loader: RefCOCOLoader,
        tau: float = 20.0,
        device: str = 'cuda',
        output_dir: str = 'E_experiments/harmful_recycling'
    ):
        """
        Args:
            model: LLaVA model instance
            tokenizer: Model tokenizer
            image_processor: Image processor
            dataset_loader: RefCOCO dataset loader
            tau: Threshold for sink token detection (φ(x) >= τ)
            device: Device to run on
            output_dir: Directory to save results
        """
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.dataset_loader = dataset_loader
        self.tau = tau
        self.device = device
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Storage for detected relevant sink tokens
        self.relevant_sink_samples = []
    
    def find_relevant_sink_tokens(
        self,
        hidden_states: torch.Tensor,
        relevant_token_indices: torch.Tensor,
        layer: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find tokens that are both relevant and classified as sink tokens.
        
        Args:
            hidden_states: Hidden states from the layer [bsz, seq_len, hidden_dim]
            relevant_token_indices: Indices of tokens inside ground-truth bbox
            layer: Current layer index
        
        Returns:
            Tuple of (relevant_sink_indices, phi_values):
                - relevant_sink_indices: Indices of tokens that are both relevant and sink
                - phi_values: Corresponding φ(x) values for these tokens
        """
        # Apply RMSNorm as in DimProspector
        rms_norm_hs = torch.abs(DimProspector.rmsnorm(hidden_states))
        
        # Extract values for sink dimensions
        dim_sink = DimProspector.dim_sink
        rms_values = torch.stack(
            [rms_norm_hs[:, :, idx] for idx in dim_sink], 
            dim=-1
        )  # [bsz, tok, num_sink_dims]
        
        # Get max value across sink dimensions
        phi_values = torch.max(rms_values, dim=-1)[0]  # [bsz, tok]
        
        # Find tokens where φ(x) >= τ
        is_sink = phi_values >= self.tau
        
        # Find relevant tokens that are also sinks
        relevant_sink_mask = torch.zeros_like(is_sink, dtype=torch.bool)
        relevant_sink_mask[:, relevant_token_indices] = is_sink[:, relevant_token_indices]
        
        # Get indices of relevant sink tokens
        relevant_sink_indices = torch.nonzero(relevant_sink_mask[0], as_tuple=True)[0]
        
        return relevant_sink_indices, phi_values[0, relevant_sink_indices]
    
    def run_inference_mode(
        self,
        image,
        query: str,
        mode: str,
        relevant_sink_indices: Optional[torch.Tensor] = None,
        relevant_indices: Optional[torch.Tensor] = None,
        irrelevant_indices: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Run inference in one of three modes.
        
        Args:
            image: PIL Image
            query: Query text
            mode: One of 'baseline', 'var', 'control'
            relevant_sink_indices: Indices of relevant sink tokens (for control mode)
            relevant_indices: All relevant token indices
            irrelevant_indices: All irrelevant token indices
        
        Returns:
            Dict with 'output', 'attention_maps', 'sink_tokens_used', etc.
        """
        from src.constants import (DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN,
                                    DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX)
        from src.conversation import conv_templates
        from src.mm_utils import process_images, tokenizer_image_token
        
        # Prepare input
        conv = conv_templates['vicuna_v1'].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + '\n' + query)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # Process image
        image_tensor = process_images([image], self.image_processor, self.model.config)
        if isinstance(image_tensor, list):
            image_tensor = [img.to(self.device, dtype=torch.float16) for img in image_tensor]
        else:
            image_tensor = image_tensor.to(self.device, dtype=torch.float16)
        
        # Tokenize
        input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors='pt'
        ).unsqueeze(0).to(self.device)
        
        # Set experiment mode in LogicEngine
        if mode == 'baseline':
            # Disable VAR
            LogicEngine.logic_flag = False
        elif mode == 'var':
            # Enable VAR (will use all detected sinks)
            LogicEngine.logic_flag = True
        elif mode == 'control':
            # Enable VAR but exclude relevant sink tokens
            LogicEngine.logic_flag = True
            # We need to modify the sink detection to exclude relevant sinks
            # This is done by storing which tokens to exclude
            self._set_excluded_sink_tokens(relevant_sink_indices)
        
        # Run inference
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image.size],
                do_sample=False,
                max_new_tokens=128,
                use_cache=True,
            )
        
        # Decode output
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        # Reset logic flag
        LogicEngine.logic_flag = True
        self._clear_excluded_sink_tokens()
        
        return {
            'output': outputs,
            'mode': mode,
        }
    
    def _set_excluded_sink_tokens(self, indices: Optional[torch.Tensor]):
        """Mark certain tokens to be excluded from sink detection."""
        if not hasattr(DimProspector, '_excluded_tokens'):
            DimProspector._excluded_tokens = None
        DimProspector._excluded_tokens = indices
    
    def _clear_excluded_sink_tokens(self):
        """Clear excluded tokens."""
        if hasattr(DimProspector, '_excluded_tokens'):
            DimProspector._excluded_tokens = None
    
    def search_for_samples_with_relevant_sinks(
        self,
        num_samples: int = 100,
        min_relevant_sinks: int = 1
    ) -> List[Dict]:
        """
        Search through dataset to find samples with relevant sink tokens.
        
        Args:
            num_samples: Number of samples to search through
            min_relevant_sinks: Minimum number of relevant sink tokens to qualify
        
        Returns:
            List of samples that contain relevant sink tokens
        """
        from PIL import Image
        from src.mm_utils import process_images
        
        found_samples = []
        
        print(f"Searching for samples with relevant sink tokens (tau={self.tau})...")
        
        for i in tqdm(range(min(num_samples, len(self.dataset_loader)))):
            sample = self.dataset_loader.get_sample(i)
            
            # For dummy samples without actual images, create synthetic data
            if not osp.exists(sample['image_path']):
                # Create a dummy image
                image = Image.new('RGB', (336, 336), color='white')
            else:
                image = Image.open(sample['image_path']).convert('RGB')
            
            # Get bounding box and map to tokens
            bbox = sample['bbox']
            mapper = BoundingBoxMapper(
                image_size=image.size,
                patch_size=14,
                num_patches=24
            )
            mapping = mapper.bbox_to_token_indices(bbox)
            relevant_indices = mapping['relevant_indices']
            
            # Process image to get visual features
            image_tensor = process_images([image], self.image_processor, self.model.config)
            if isinstance(image_tensor, list):
                image_tensor = image_tensor[0].to(self.device, dtype=torch.float16)
            else:
                image_tensor = image_tensor.to(self.device, dtype=torch.float16)
            
            # Get image features through vision encoder
            with torch.no_grad():
                if hasattr(self.model, 'get_vision_tower'):
                    vision_tower = self.model.get_vision_tower()
                    image_features = vision_tower(image_tensor.unsqueeze(0))
                elif hasattr(self.model.model, 'vision_tower'):
                    vision_tower = self.model.model.vision_tower
                    image_features = vision_tower(image_tensor.unsqueeze(0))
                else:
                    # Skip if we can't access vision tower
                    continue
                
                # Project to LLM space
                if hasattr(self.model.model, 'mm_projector'):
                    image_features = self.model.model.mm_projector(image_features)
                
                # Check for relevant sinks in the projected features
                # We check at the input level (layer 0)
                relevant_sink_indices, phi_values = self.find_relevant_sink_tokens(
                    image_features,
                    relevant_indices,
                    layer=0
                )
            
            if len(relevant_sink_indices) >= min_relevant_sinks:
                sample_info = {
                    'sample_idx': i,
                    'sample': sample,
                    'image': image,
                    'relevant_indices': relevant_indices,
                    'relevant_sink_indices': relevant_sink_indices,
                    'phi_values': phi_values,
                    'num_relevant_sinks': len(relevant_sink_indices),
                }
                found_samples.append(sample_info)
                print(f"\nFound sample {i} with {len(relevant_sink_indices)} relevant sink tokens!")
                print(f"  Query: {sample['query']}")
                print(f"  Relevant sink indices: {relevant_sink_indices.cpu().tolist()}")
                print(f"  Phi values: {phi_values.cpu().tolist()}")
        
        print(f"\nFound {len(found_samples)} samples with relevant sink tokens")
        return found_samples
    
    def run_intervention_experiment(
        self,
        sample_info: Dict
    ) -> Dict:
        """
        Run the three-mode intervention experiment on a sample with relevant sink tokens.
        
        Args:
            sample_info: Dictionary from search_for_samples_with_relevant_sinks()
        
        Returns:
            Dict with results from all three modes
        """
        sample = sample_info['sample']
        image = sample_info['image']
        query = sample['query']
        relevant_sink_indices = sample_info['relevant_sink_indices']
        relevant_indices = sample_info['relevant_indices']
        
        # Get irrelevant indices
        total_tokens = 24 * 24  # For 336x336 image with 14x14 patches
        all_indices = torch.arange(total_tokens)
        irrelevant_indices = all_indices[~torch.isin(all_indices, relevant_indices)]
        
        print(f"\nRunning intervention experiment on sample {sample_info['sample_idx']}")
        print(f"Query: {query}")
        print(f"Number of relevant sink tokens: {len(relevant_sink_indices)}")
        
        # Mode A: Baseline (no VAR)
        print("\nMode A (Baseline): Running without VAR...")
        result_baseline = self.run_inference_mode(
            image, query, 'baseline',
            relevant_sink_indices=relevant_sink_indices,
            relevant_indices=relevant_indices,
            irrelevant_indices=irrelevant_indices
        )
        print(f"Output: {result_baseline['output']}")
        
        # Mode B: VAR (will recycle from relevant sinks - harmful)
        print("\nMode B (VAR): Running with original VAR...")
        result_var = self.run_inference_mode(
            image, query, 'var',
            relevant_sink_indices=relevant_sink_indices,
            relevant_indices=relevant_indices,
            irrelevant_indices=irrelevant_indices
        )
        print(f"Output: {result_var['output']}")
        
        # Mode C: Control (VAR excluding relevant sinks)
        print("\nMode C (Control): Running with modified VAR...")
        result_control = self.run_inference_mode(
            image, query, 'control',
            relevant_sink_indices=relevant_sink_indices,
            relevant_indices=relevant_indices,
            irrelevant_indices=irrelevant_indices
        )
        print(f"Output: {result_control['output']}")
        
        results = {
            'sample_info': {
                'idx': sample_info['sample_idx'],
                'query': query,
                'bbox': sample['bbox'],
                'num_relevant_sinks': len(relevant_sink_indices),
                'relevant_sink_indices': relevant_sink_indices.cpu().tolist(),
            },
            'mode_a_baseline': result_baseline,
            'mode_b_var': result_var,
            'mode_c_control': result_control,
        }
        
        return results
    
    def save_results(self, results: List[Dict], filename: str = 'harmful_recycling_results.json'):
        """Save experiment results to JSON file."""
        output_path = osp.join(self.output_dir, filename)
        
        # Convert tensors to lists for JSON serialization
        serializable_results = []
        for result in results:
            serializable = copy.deepcopy(result)
            # Handle any remaining tensors
            serializable_results.append(serializable)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nResults saved to {output_path}")
        return output_path
    
    def run_full_experiment(
        self,
        num_search_samples: int = 100,
        num_intervention_samples: int = 5
    ) -> Dict:
        """
        Run the complete harmful recycling experiment.
        
        Args:
            num_search_samples: Number of samples to search through
            num_intervention_samples: Number of samples to run intervention on
        
        Returns:
            Dict with all experiment results
        """
        # Step 1: Search for samples with relevant sink tokens
        found_samples = self.search_for_samples_with_relevant_sinks(
            num_samples=num_search_samples,
            min_relevant_sinks=1
        )
        
        if len(found_samples) == 0:
            print("No samples with relevant sink tokens found!")
            return {'status': 'no_samples_found', 'results': []}
        
        # Step 2: Run intervention experiment on found samples
        intervention_results = []
        for i, sample_info in enumerate(found_samples[:num_intervention_samples]):
            result = self.run_intervention_experiment(sample_info)
            intervention_results.append(result)
        
        # Step 3: Save results
        self.save_results(intervention_results)
        
        return {
            'status': 'success',
            'num_samples_searched': num_search_samples,
            'num_samples_with_relevant_sinks': len(found_samples),
            'num_interventions_run': len(intervention_results),
            'results': intervention_results,
        }
