# Quick Reference Guide for VAR Critique Experiments

## Files Overview

### Experiment Scripts
- `E_experiments/g_harmful_recycling_intervention.py` - Experiment 1: Proves VAR can harm performance
- `E_experiments/g_logit_ranking_failure.py` - Experiment 2: Proves Q·K^T logits are unreliable
- `E_experiments/visualize_results.py` - Generate HTML/PDF visualizations from results
- `E_experiments/prepare_refcoco_data.py` - Helper to prepare dataset in required format
- `E_experiments/run_experiments.sh` - Convenience launcher script

### Configuration Files
- `A_exps/exp1_harmful_recycling.yml` - Config for Experiment 1
- `A_exps/exp2_logit_ranking.yml` - Config for Experiment 2

## Key Classes and Functions

### Experiment 1: g_HarmfulRecycling_Intervention

**Main Classes:**
- `InterventionVARProcessor`: Extended VAR processor with exclusion capabilities
  - `set_excluded_indices()`: Exclude specific tokens from recycling (Mode C)
  - `attn_redist()`: Modified attention redistribution

**Key Functions:**
- `classify_tokens(image_size, bbox)`: Classify tokens as relevant/irrelevant based on bbox
- `find_relevant_sink_tokens()`: Find tokens that are both relevant AND sinks
- `run_inference_mode(mode='A'|'B'|'C')`: Run inference in one of three modes

**Three Modes:**
1. **Mode A (Baseline)**: `LogicEngine.set_flag(False)` - no VAR
2. **Mode B (VAR)**: `LogicEngine.activate()` - standard VAR
3. **Mode C (Ideal VAR)**: `LogicEngine.activate() + set_excluded_indices()` - VAR excluding relevant sinks

### Experiment 2: g_LogitRanking_Failure

**Main Classes:**
- `LogitCapture`: Helper to capture attention logits during forward pass

**Key Functions:**
- `identify_image_centric_heads()`: Find ICHs (heads focusing on image tokens)
- `compute_consensus_logits()`: Compute S_consensus for each visual token
- `analyze_ranking()`: Calculate contamination rate and ranking statistics

**Key Metrics:**
- **Contamination Rate**: (# irrelevant in Top-K) / K
- **Top-1 Irrelevant Rate**: % of samples where highest-ranked token is irrelevant

## Command Line Usage

### Basic Usage

```bash
# Experiment 1 - Find and test relevant sink tokens
python E_experiments/g_harmful_recycling_intervention.py \
  --exp_config A_exps/exp1_harmful_recycling.yml \
  --device 0 \
  --max_samples 100 \
  --num_intervention_samples 10

# Experiment 2 - Analyze logit ranking quality
python E_experiments/g_logit_ranking_failure.py \
  --exp_config A_exps/exp2_logit_ranking.yml \
  --device 0 \
  --max_samples 100 \
  --top_k 10

# Visualize results
python E_experiments/visualize_results.py \
  --exp1_results F_experiment_results/harmful_recycling_intervention/12345 \
  --exp2_results F_experiment_results/logit_ranking_failure/67890
```

### Using the Launcher Script

```bash
# Run Experiment 1 only
./E_experiments/run_experiments.sh 1 --device 0 --max_samples 50

# Run Experiment 2 only
./E_experiments/run_experiments.sh 2 --device 0 --top_k 10

# Run both experiments
./E_experiments/run_experiments.sh both --device 0
```

## Data Format Requirements

### Input Format (JSONL)

Each line in the questions file should be:
```json
{
  "qid": 12345,
  "image": "COCO_train2014_000000123456.jpg",
  "text": "the woman in the red dress",
  "bbox": [x, y, width, height],
  "label": "person"
}
```

### Output Format

**Experiment 1 Output:**
```
F_experiment_results/harmful_recycling_intervention/{timestamp}/
├── results.json          # Detailed results with all 3 mode responses
├── summary.json          # Experiment summary statistics
└── visualizations/
    ├── report.html       # Interactive HTML report
    └── text_report.txt   # Plain text report
```

**Experiment 2 Output:**
```
F_experiment_results/logit_ranking_failure/{timestamp}/
├── results.json                # Per-sample contamination rates
├── summary.json                # Aggregate statistics
├── qualitative_failures.json   # Cases where Top-1 was irrelevant
└── visualizations/
    ├── report.html             # Interactive HTML report
    ├── text_report.txt         # Plain text report
    └── contamination_histogram.png  # Distribution plot
```

## Modifying for Different Models

### Vision Encoder Changes

If your model uses different patch sizes or grid sizes:
```python
# In both experiment scripts, update:
num_patches_per_side = 24  # Change this for different grid sizes
num_patches = 576          # Change this for different total patches

# Example: For 448x448 input with 14x14 patches:
# num_patches_per_side = 32
# num_patches = 1024
```

### Attention Mechanism Changes

If your model has different attention structure:
```python
# Update in InterventionVARProcessor.attn_redist()
# and in compute_consensus_logits()
# to match your model's attention tensor shapes
```

## Common Parameters

### VAR Parameters (in config YAML)
- `tau`: Threshold for sink detection (default: 20)
- `rho`: Maximum proportion for sink tokens (default: 0.5)
- `p`: Budget retention rate (default: 0.6)
- `summ`: Minimum summation threshold (default: 0.2)

### Experiment Parameters
- `max_samples`: Number of samples to examine (Exp 1 & 2)
- `num_intervention_samples`: Number to run full intervention (Exp 1 only)
- `top_k`: K for Top-K contamination rate (Exp 2 only)

## Troubleshooting

### Issue: "No relevant sink tokens found"
**Solution**: This is expected for most samples. The experiment searches through many samples to find problematic cases.
- Try increasing `--max_samples` to 200-500
- Try adjusting `tau` parameter (lower values = more sinks detected)

### Issue: "No ICHs found"
**Solution**: Model may not have clear image-centric heads at default threshold.
- Lower ICH threshold from 0.3 to 0.2 in `identify_image_centric_heads()`
- Try using different layers for analysis

### Issue: "Attention weights not available"
**Solution**: Model generation doesn't return attentions.
- Ensure `output_attentions=True` in generate call
- Use `attn_implementation="eager"` when loading model
- Check model's `forward()` method supports attention output

## Expected Results

### Experiment 1
- Should find 5-20% of samples have relevant sink tokens
- Mode B performance should degrade compared to A and C
- Examples: Hallucinations, "can't see clearly" responses in Mode B

### Experiment 2
- Mean contamination rate: 50-70% (proves Q·K^T is unreliable)
- Top-1 irrelevant rate: 30-50%
- Many qualitative failures showing background ranked highest

## Performance Tips

1. **Use GPU**: Both experiments are GPU-intensive
2. **Batch size**: Keep at 1 for cleaner intervention analysis
3. **Start small**: Test with `--max_samples 10` first
4. **Save often**: Results are saved incrementally (JSONL format)

## Citation

When using these experiments in your research:

```bibtex
@article{var_critique_2024,
  title={Critique of Visual Attention Recycling: Demonstrating Fundamental Limitations},
  note={VAR Critique Experiments},
  year={2024}
}
```
