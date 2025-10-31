# Quick Start Guide: VAR Assumption Validation Experiments

This guide will help you quickly run the VAR assumption validation experiments.

## Prerequisites

1. **Environment Setup**
   ```bash
   conda create -n VisAttnSink python=3.10
   conda activate VisAttnSink
   pip install -r env_pip.txt
   ```

2. **Model Download**
   ```bash
   # Download LLaVA-1.5-7B (example)
   git lfs install
   git clone https://huggingface.co/liuhaotian/llava-v1.5-7b
   ```

3. **Dataset Setup** (Optional - will use dummy data if not available)
   ```
   C_datasets/
   └── refcoco/
       ├── val.json
       └── images/
   ```

## Running Experiments

### Option 1: Quick Test (No Dataset Required)

Run a quick test to verify the code works:

```bash
python src/test_experiments.py
```

This will test the basic functionality without requiring a full model or dataset.

### Option 2: Run Both Experiments

```bash
python src/run_experiments.py \
  --experiment both \
  --model_path /path/to/llava-v1.5-7b \
  --dataset_path C_datasets/refcoco \
  --output_dir E_experiments
```

Or using the shell script:

```bash
bash B_scripts/run_experiments.sh \
  --model /path/to/llava-v1.5-7b \
  --experiment both
```

### Option 3: Run Individual Experiments

**Experiment 1: Harmful Recycling Intervention**

```bash
python src/run_experiments.py \
  --experiment harmful_recycling \
  --model_path /path/to/llava-v1.5-7b \
  --tau 20.0 \
  --num_search_samples 100 \
  --num_intervention_samples 5
```

**Experiment 2: Logit Ranking Failure**

```bash
python src/run_experiments.py \
  --experiment logit_ranking \
  --model_path /path/to/llava-v1.5-7b \
  --num_logit_samples 50 \
  --layers 10,20,30
```

## Understanding Results

### Experiment 1 Results

Results saved to: `E_experiments/harmful_recycling/harmful_recycling_results.json`

**What to look for:**
- Samples with `num_relevant_sinks > 0` (proves Problem 1 exists)
- `mode_b_var` output should be worse than `mode_a_baseline` and `mode_c_control`
- Examples of hallucinations or performance degradation in Mode B

**Example successful result:**
```json
{
  "sample_info": {"num_relevant_sinks": 3},
  "mode_a_baseline": {"output": "She is smiling."},
  "mode_b_var": {"output": "I cannot see clearly."},  // ← Degraded!
  "mode_c_control": {"output": "She is smiling."}
}
```

### Experiment 2 Results

Results saved to: `E_experiments/logit_ranking/logit_ranking_results.json`

**What to look for:**
- `avg_pollution_rate > 0.5` (proves logits are unreliable)
- `top1_relevant_rate < 0.5` (even Top-1 is often wrong)
- Consistent results across multiple K values

**Example successful result:**
```json
{
  "aggregate_statistics": {
    "top10": {
      "avg_pollution_rate": 0.62,  // ← 62% irrelevant!
      "top1_relevant_rate": 0.35   // ← Top-1 wrong 65% of time
    }
  }
}
```

## Common Issues

### Issue 1: No Dataset Available

**Solution:** The code will use dummy data automatically. For testing, this is fine. For real results, download RefCOCO.

### Issue 2: CUDA Out of Memory

**Solution:** Reduce the number of samples:
```bash
python src/run_experiments.py \
  --experiment harmful_recycling \
  --model_path /path/to/model \
  --num_search_samples 10 \
  --num_intervention_samples 1
```

### Issue 3: Model Not Loading

**Solution:** Ensure you have the correct path and have downloaded the model:
```bash
ls -l /path/to/llava-v1.5-7b  # Should show model files
```

## Next Steps

1. **Review Results**: Check the output JSON files in `E_experiments/`
2. **Read Documentation**: See `EXPERIMENTS_README.md` for detailed information
3. **Customize**: Modify parameters in `A_exps/*.yml` or use command-line options
4. **Analyze**: Use the results to motivate your D-VAR approach

## Help

For detailed documentation:
- **EXPERIMENTS_README.md** - Complete experiment guide
- **IMPLEMENTATION_SUMMARY.md** - Technical implementation details
- **README.md** - Main repository documentation

For command-line help:
```bash
python src/run_experiments.py --help
bash B_scripts/run_experiments.sh --help
```

## Expected Runtime

- **Experiment 1**: 
  - Search phase: ~10-30 minutes (100 samples)
  - Intervention phase: ~5-10 minutes (5 samples)
  
- **Experiment 2**:
  - Analysis: ~20-40 minutes (50 samples)

*Times vary based on GPU and model size*

## Success Criteria Checklist

### Experiment 1 ✓
- [ ] Found samples with relevant sink tokens
- [ ] Mode B shows degraded output
- [ ] Mode C shows similar output to Mode A
- [ ] Clear qualitative differences

### Experiment 2 ✓
- [ ] Pollution rate > 50%
- [ ] Top-1 relevant rate < 50%
- [ ] Consistent across layers
- [ ] Qualitative failure examples

If you can check all boxes, your experiments successfully validate the problems with VAR!
