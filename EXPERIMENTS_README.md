# VAR Assumption Validation Experiments

This directory contains two experiments designed to validate assumptions made in the paper "SEE WHAT YOU ARE TOLD: VISUAL ATTENTION SINK IN LARGE MULTIMODAL MODELS" (arXiv:2503.03321v1).

## Overview

The experiments demonstrate two critical problems with the VAR (Visual Attention Redistribution) method:

### **Experiment 1: g_HarmfulRecycling_Intervention**

**Problem**: The paper assumes that "sink tokens" (φ(x) ≥ τ) and "relevant tokens" are mutually exclusive. 

**Hypothesis**: They can overlap, creating **"Relevant Sink Tokens"** that are incorrectly penalized by VAR.

**Method**:
1. Use RefCOCO dataset with ground-truth bounding boxes
2. Identify tokens inside the bbox as "truly relevant" (T_relevant)
3. Find tokens where: token ∈ T_relevant AND φ(x) ≥ τ (relevant sinks)
4. Run three inference modes:
   - **Mode A (Baseline)**: Original LMM without VAR
   - **Mode B (VAR)**: Apply original VAR (will harm performance on relevant sinks)
   - **Mode C (Control)**: Apply VAR but exclude relevant sinks from recycling

**Expected Result**: Mode B performs worse than A and C, proving that VAR harms performance when it incorrectly identifies relevant tokens as sinks.

### **Experiment 2: g_LogitRanking_Failure**

**Problem**: Follow-up methods (like HCR) assume "key relevance" (Q·K^T) can correctly rank token importance.

**Hypothesis**: Key relevance is unreliable and cannot distinguish relevant from irrelevant tokens.

**Method**:
1. Use RefCOCO dataset with ground-truth bounding boxes
2. Identify Image-Centric Heads (ICHs) that focus on image tokens
3. Compute consensus key relevance: S_consensus(j) = avg over ICHs of (q_i · k_j^T)
4. Rank all visual tokens by consensus scores (descending)
5. Measure **Top-K Pollution Rate** = (# irrelevant tokens in Top-K) / K

**Expected Result**: High pollution rates (>50%) prove that Q·K^T logits are unreliable for ranking token importance.

## Installation

Ensure you have the VisAttnSink environment set up:

```bash
# Create conda environment
conda create -n VisAttnSink python=3.10
conda activate VisAttnSink

# Install dependencies
pip install -r env_pip.txt
```

## Dataset Preparation

The experiments use RefCOCO or similar visual grounding datasets with phrase-to-bounding-box alignment.

### RefCOCO Dataset Structure

```
C_datasets/
└── refcoco/
    ├── val.json          # Validation split annotations
    ├── train.json        # Training split annotations
    └── images/           # Image files
        ├── 000001.jpg
        ├── 000002.jpg
        └── ...
```

Each annotation entry should contain:
```json
{
  "image_id": 123,
  "image_path": "images/000001.jpg",
  "query": "the woman in red dress",
  "bbox": [100, 100, 200, 200],  // [x, y, width, height]
  "ref_id": 456
}
```

**Note**: If you don't have RefCOCO, the code will use dummy samples for testing purposes.

## Usage

### Quick Start

Run both experiments with default settings:

```bash
python src/run_experiments.py \
  --experiment both \
  --model_path YOUR_PATH_TO_LLAVA_MODEL \
  --dataset_path C_datasets/refcoco \
  --output_dir E_experiments
```

### Run Individual Experiments

**Experiment 1: Harmful Recycling**

```bash
python src/run_experiments.py \
  --experiment harmful_recycling \
  --model_path YOUR_PATH_TO_LLAVA_MODEL \
  --dataset_path C_datasets/refcoco \
  --tau 20.0 \
  --num_search_samples 100 \
  --num_intervention_samples 5 \
  --output_dir E_experiments
```

**Experiment 2: Logit Ranking**

```bash
python src/run_experiments.py \
  --experiment logit_ranking \
  --model_path YOUR_PATH_TO_LLAVA_MODEL \
  --dataset_path C_datasets/refcoco \
  --num_logit_samples 50 \
  --layers 10,20,30 \
  --output_dir E_experiments
```

### Configuration Files

You can also use YAML configuration files:

```bash
# Edit configuration
vim A_exps/harmful_recycling.yml

# Run using configuration (not yet implemented, use command line for now)
```

## Command-Line Arguments

### Common Arguments

- `--experiment`: Which experiment to run (`harmful_recycling`, `logit_ranking`, or `both`)
- `--model_path`: Path to LLaVA model checkpoint (required)
- `--model_base`: Base model path if using LoRA (optional)
- `--dataset_path`: Path to RefCOCO dataset (default: `C_datasets/refcoco`)
- `--dataset_split`: Dataset split to use (`train`, `val`, or `test`, default: `val`)
- `--output_dir`: Output directory for results (default: `E_experiments`)
- `--device`: CUDA device number (default: `0`)

### Experiment 1 Specific

- `--tau`: Threshold for sink token detection (default: `20.0`)
- `--num_search_samples`: Number of samples to search through (default: `100`)
- `--num_intervention_samples`: Number of samples to run intervention on (default: `5`)

### Experiment 2 Specific

- `--num_logit_samples`: Number of samples to analyze (default: `50`)
- `--layers`: Comma-separated layer indices to analyze (default: all layers)

## Output

### Experiment 1 Output

Results are saved to `E_experiments/harmful_recycling/harmful_recycling_results.json`:

```json
{
  "sample_info": {
    "idx": 0,
    "query": "the woman in red dress",
    "bbox": [100, 100, 200, 200],
    "num_relevant_sinks": 3,
    "relevant_sink_indices": [45, 67, 89]
  },
  "mode_a_baseline": {
    "output": "She is smiling and waving at the camera.",
    "mode": "baseline"
  },
  "mode_b_var": {
    "output": "I cannot see what she is doing.",
    "mode": "var"
  },
  "mode_c_control": {
    "output": "She is smiling and waving at the camera.",
    "mode": "control"
  }
}
```

**Key Findings**:
- Mode B (VAR) produces worse outputs than Mode A and C
- This proves that recycling attention from relevant sink tokens harms performance

### Experiment 2 Output

Results are saved to `E_experiments/logit_ranking/logit_ranking_results.json`:

```json
{
  "aggregate_statistics": {
    "top5": {
      "avg_pollution_rate": 0.62,
      "std_pollution_rate": 0.15,
      "top1_relevant_rate": 0.35
    },
    "top10": {
      "avg_pollution_rate": 0.58,
      "std_pollution_rate": 0.12,
      "top1_relevant_rate": 0.35
    }
  }
}
```

**Key Findings**:
- High pollution rates (>50%) show that Q·K^T logits are unreliable
- Even the Top-1 token is often irrelevant (relevance rate < 50%)
- This proves that key-based ranking cannot identify truly relevant tokens

## Interpretation of Results

### Success Criteria for Experiment 1

✅ **Problem 1 Confirmed** if:
1. Found samples with relevant sink tokens (tokens where relevant AND φ(x) ≥ τ)
2. Mode B (VAR) produces significantly worse outputs than Mode A (Baseline)
3. Mode C (Control) produces outputs similar to Mode A

### Success Criteria for Experiment 2

✅ **Problem 2 Confirmed** if:
1. Average Top-K pollution rate > 50%
2. Top-1 relevant rate < 50%
3. Many qualitative failure cases where Top-1 token is clearly irrelevant

## Advanced Usage

### Analyzing Specific Layers

To analyze only specific layers in Experiment 2:

```bash
python src/run_experiments.py \
  --experiment logit_ranking \
  --model_path YOUR_MODEL \
  --layers 5,10,15,20,25,30 \
  --num_logit_samples 100
```

### Custom Tau Values

To test different sink detection thresholds:

```bash
for tau in 10 15 20 25 30; do
  python src/run_experiments.py \
    --experiment harmful_recycling \
    --model_path YOUR_MODEL \
    --tau $tau \
    --output_dir E_experiments/tau_${tau}
done
```

### Using Different Datasets

The experiments support any dataset with phrase-to-bounding-box alignment. To use Flickr30k Entities:

```bash
python src/run_experiments.py \
  --experiment both \
  --model_path YOUR_MODEL \
  --dataset_path C_datasets/flickr30k \
  --dataset_split test
```

## Code Structure

```
src/experiments/
├── __init__.py                  # Module exports
├── grounding_utils.py           # RefCOCO loader and bbox-to-token mapping
├── harmful_recycling.py         # Experiment 1 implementation
└── logit_ranking.py             # Experiment 2 implementation

src/run_experiments.py           # Main experiment runner script

A_exps/
├── harmful_recycling.yml        # Config for Experiment 1
└── logit_ranking.yml            # Config for Experiment 2
```

## Troubleshooting

### No RefCOCO dataset available

If you don't have the RefCOCO dataset, the code will use dummy samples for testing. This is useful for:
- Verifying the code runs correctly
- Testing the experimental pipeline
- Debugging

However, for actual research results, you need real data.

### CUDA out of memory

Reduce batch processing or use a smaller model:
```bash
python src/run_experiments.py \
  --experiment harmful_recycling \
  --num_search_samples 10 \
  --num_intervention_samples 1
```

### Model not loading

Ensure you have the correct model path and have downloaded the model:
```bash
# Example for LLaVA-1.5-7B
git lfs install
git clone https://huggingface.co/liuhaotian/llava-v1.5-7b
```

## Citation

If you use these experiments in your research, please cite:

```bibtex
@article{var2025,
  title={SEE WHAT YOU ARE TOLD: VISUAL ATTENTION SINK IN LARGE MULTIMODAL MODELS},
  author={},
  journal={arXiv preprint arXiv:2503.03321},
  year={2025}
}
```

## License

This code is licensed under Apache 2.0, consistent with the main VisAttnSink repository.

## Contact

For questions or issues, please open an issue on the repository.
