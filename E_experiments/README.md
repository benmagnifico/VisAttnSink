# VAR Critique Experiments

This directory contains two experiments designed to demonstrate fundamental weaknesses in the VAR (Visual Attention Recycling) approach proposed in paper 2503.03321v1.

## Overview

These experiments are designed as **motivation experiments** that can be directly used in a follow-up paper's "Motivation Analysis" section.

### Experiment 1: g_HarmfulRecycling_Intervention
**Hypothesis**: The paper assumes "sink tokens" (φ(x) ≥ τ) and "relevant tokens" are mutually exclusive. This experiment proves they can overlap, creating **"Relevant Sink Tokens"** that VAR mistakenly penalizes.

**Expected Outcome**: Proves that VAR's sink detection mechanism (based on φ(x)) is **unsafe** and can actively harm performance by recycling attention from important tokens.

### Experiment 2: g_LogitRanking_Failure  
**Hypothesis**: Follow-up methods (like HCR) assume "key relevance" (Q·K^T) can correctly rank token importance. This experiment proves it **cannot**.

**Expected Outcome**: Proves that budget allocation mechanisms based on Q·K^T logits (whether single-head LSR or multi-head HCR) are **unreliable** and frequently rank irrelevant tokens higher than relevant ones.

## Requirements

### Dataset
Both experiments require a visual grounding dataset with phrase-to-bounding-box alignment:
- **RefCOCO** (recommended)
- **Flickr30k Entities**

The dataset should be formatted with questions in JSONL format, where each line contains:
```json
{
  "qid": 12345,
  "image": "image_name.jpg",
  "text": "the woman in the red dress",
  "bbox": [x, y, width, height],
  "label": "ground truth answer"
}
```

### Model
- LLaVA-1.5-7B or similar vision-language model
- The model should be compatible with the existing VisAttnSink codebase

## Setup

1. **Install dependencies** (if not already done):
```bash
conda create -n VisAttnSink -f env_conda.txt
pip install -r env_pip.txt
```

2. **Prepare dataset**:
   - Download RefCOCO dataset
   - Convert to the required JSONL format with bounding box annotations
   - Update paths in configuration files

3. **Update configuration files**:
   - Edit `A_exps/exp1_harmful_recycling.yml`
   - Edit `A_exps/exp2_logit_ranking.yml`
   - Set correct paths for:
     - `path_image_dir`: Directory containing images
     - `path_question_dir`: Directory containing question JSONL files
     - `path_model`: Path to LLaVA model checkpoint

## Running Experiments

### Experiment 1: Harmful Recycling Intervention

This experiment searches for samples where relevant tokens are misidentified as sinks, then runs three intervention modes to demonstrate the harm.

```bash
python E_experiments/g_harmful_recycling_intervention.py \
  --exp_config A_exps/exp1_harmful_recycling.yml \
  --device 0 \
  --max_samples 100 \
  --num_intervention_samples 10
```

**Arguments**:
- `--exp_config`: Path to experiment configuration file
- `--device`: GPU device ID
- `--max_samples`: Maximum samples to examine for finding relevant sink tokens
- `--num_intervention_samples`: Number of samples to run full intervention (3 modes) on

**Output**:
- `F_experiment_results/harmful_recycling_intervention/{timestamp}/results.json`: Detailed results showing responses from all three modes
- `F_experiment_results/harmful_recycling_intervention/{timestamp}/summary.json`: Experiment summary with statistics

**Three Modes**:
- **Mode A (Baseline)**: Standard model without VAR
- **Mode B (VAR)**: Standard VAR that incorrectly recycles from relevant sinks
- **Mode C (Ideal VAR)**: Modified VAR that excludes relevant sinks from recycling

**Expected Result**: Mode B should show degraded performance compared to Modes A and C, proving that VAR's sink detection is harmful.

### Experiment 2: Logit Ranking Failure

This experiment analyzes whether Q·K^T logits can correctly rank token importance.

```bash
python E_experiments/g_logit_ranking_failure.py \
  --exp_config A_exps/exp2_logit_ranking.yml \
  --device 0 \
  --max_samples 100 \
  --top_k 10
```

**Arguments**:
- `--exp_config`: Path to experiment configuration file
- `--device`: GPU device ID
- `--max_samples`: Maximum samples to analyze
- `--top_k`: K for Top-K contamination rate calculation

**Output**:
- `F_experiment_results/logit_ranking_failure/{timestamp}/results.json`: Per-sample contamination rates and rankings
- `F_experiment_results/logit_ranking_failure/{timestamp}/summary.json`: Aggregate statistics
- `F_experiment_results/logit_ranking_failure/{timestamp}/qualitative_failures.json`: Cases where Top-1 token was irrelevant

**Metrics**:
- **Top-K Contamination Rate**: Percentage of Top-K tokens that are irrelevant (should be >50%)
- **Top-1 Irrelevant Rate**: Percentage of samples where the highest-ranked token is irrelevant

**Expected Result**: High contamination rates (>50%) proving that Q·K^T logits are unreliable for ranking token importance.

## Understanding the Results

### Experiment 1 Results

The results will show three responses for each sample:
1. **Mode A (Baseline)**: Correct answer (e.g., "She is smiling")
2. **Mode B (VAR)**: Degraded answer (e.g., "I can't see clearly" or hallucinations)
3. **Mode C (Ideal VAR)**: Correct answer, similar to baseline

This demonstrates that VAR's harm comes specifically from recycling attention from relevant sinks.

### Experiment 2 Results

Key statistics to look for:
- **Mean Contamination Rate**: Average percentage of Top-K that are irrelevant (expect >50%)
- **Top-1 Irrelevant Rate**: How often the "most relevant" token is actually irrelevant (expect >30%)
- **Qualitative Failures**: Specific examples where the model ranks background/sky/floor as most relevant

These results prove that relying on Q·K^T for budget allocation is fundamentally flawed.

## Implementation Details

### Experiment 1: How Intervention Works

1. **Token Classification**: Each visual token is classified as "relevant" or "irrelevant" based on whether its center point falls inside the ground-truth bounding box.

2. **Sink Detection**: The standard VAR mechanism (DimProspector) identifies sink tokens using φ(x) ≥ τ.

3. **Finding Relevant Sinks**: The intersection of relevant tokens and sink tokens gives us "Relevant Sink Tokens" - tokens that are actually important but mistakenly flagged as sinks.

4. **Mode C Implementation**: Uses `InterventionVARProcessor`, which extends `VARProcessor` with the ability to exclude specific indices from recycling.

### Experiment 2: How Ranking Analysis Works

1. **ICH Identification**: Image-Centric Heads (ICHs) are identified as attention heads that focus >30% of their attention on image tokens.

2. **Consensus Computation**: For each visual token j, compute:
   ```
   S_consensus(j) = average over ICHs of (q_i^h · k_j^h)
   ```

3. **Ranking**: Sort all visual tokens by S_consensus in descending order.

4. **Contamination**: Count how many of Top-K tokens fall outside the ground-truth bounding box.

## Troubleshooting

### "No ICHs found" warning
If the model doesn't have clear image-centric heads, try:
- Lowering the ICH threshold (currently 0.3)
- Using different layers for analysis
- The code will fall back to using all heads

### "No relevant sink tokens found"
This is actually expected for most samples. The experiment will search through many samples to find the problematic cases. If you can't find any after 100+ samples:
- Try adjusting τ (tau) parameter
- Try different datasets (RefCOCO vs Flickr30k)
- Check that bounding boxes are correctly formatted

### Model compatibility issues
These experiments are designed for LLaVA-1.5-7B. For other models:
- Update the attention mechanism hooks if needed
- Adjust `num_patches` parameter based on vision encoder
- Verify that MetadataStation correctly tracks image token positions

## Citation

If you use these experiments in your research, please cite the original VAR paper and acknowledge this experimental framework:

```bibtex
@article{var2024,
  title={See What You Are Told: Visual Attention Sink in Large Multimodal Models},
  journal={arXiv preprint arXiv:2503.03321v1},
  year={2024}
}
```

## Contact

For questions or issues with these experiments, please open an issue on the repository.
