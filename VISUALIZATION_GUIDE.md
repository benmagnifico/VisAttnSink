# Visual Attention Sink Visualization for LLaVA-1.5-7B

This guide explains how to reproduce the experimental results and visualize the visual attention sink phenomenon from the paper "SEE WHAT YOU ARE TOLD: VISUAL ATTENTION SINK IN LARGE MULTIMODAL MODELS".

## Overview

The visual attention sink phenomenon refers to the observation that certain visual tokens in multimodal models receive disproportionately high attention from text tokens, acting as "attention sinks". This repository provides tools to:

1. Capture attention patterns during LLaVA-1.5-7B inference
2. Visualize attention heatmaps and identify sink tokens
3. Analyze attention distribution across layers and heads

## Quick Start

### Step 1: Environment Setup

Follow the main repository README to set up the environment:

```bash
conda create -n VisAttnSink python=3.11
conda activate VisAttnSink
pip install -r env_pip.txt
```

### Step 2: Prepare Dataset

Download the POPE dataset or use your own dataset. The expected structure is:

```
D_datasets/
└── POPE/
    ├── Questions/
    │   └── random-questions.jsonl
    └── Images/
        ├── image1.jpg
        ├── image2.jpg
        └── ...
```

Question file format (JSONL):
```json
{"question_id": 1, "image": "COCO_val2014_000000001268.jpg", "text": "Is there a cat in the image?", "label": "yes"}
```

### Step 3: Download LLaVA-1.5-7B Model

Download the pretrained LLaVA-1.5-7B model:

```bash
# Using huggingface-cli
huggingface-cli download liuhaotian/llava-v1.5-7b --local-dir /path/to/llava-v1.5-7b
```

Or use the model directly from HuggingFace Hub by setting:
```yaml
path_model: liuhaotian/llava-v1.5-7b
```

### Step 4: Configure Experiment

Edit `A_exps/lv1.5_7b_viz.yml`:

```yaml
name_exp: lv1.5_7b_visualization
name_daset: POPE
name_category: random

# Update these paths
path_image_dir: /path/to/coco/val2014
path_question_dir: D_datasets/POPE/Questions
path_model: /path/to/llava-v1.5-7b

conv_mode: vicuna_v1

# Visual Attention Sink Logic Parameters
logic: 1
tau: 20      # Threshold for identifying sink tokens
rho: 0.5     # Maximum attention portion to visual sinks
p: 0.6       # Attention redistribution factor
summ: 0.2    # Minimum total attention to image

# Generation Parameters
max_new_tokens: 128
except_last_layer: 1

# Visualization Parameters
capture_attention: true
max_viz_questions: 5  # Number of questions to visualize
```

### Step 5: Run Inference with Visualization

```bash
python src/inference_with_visualization.py \
    --device 0 \
    --exp_config A_exps/lv1.5_7b_viz.yml
```

This will:
- Run inference on the specified questions
- Capture attention patterns at each layer
- Save attention data to `F_visualizations/attention_qid*.pkl`
- Generate preview visualizations automatically
- Save answers to `E_answers/`

### Step 6: Generate Comprehensive Visualizations

After inference, generate additional visualizations:

```bash
# Visualize all layers and heads for a specific question
python src/visualization/visualize_attention.py \
    --attention_file F_visualizations/attention_qid1.pkl \
    --output_dir F_visualizations

# Visualize specific layer and head
python src/visualization/visualize_attention.py \
    --attention_file F_visualizations/attention_qid1.pkl \
    --layer 15 \
    --head 10 \
    --threshold 0.1
```

## Understanding the Visualizations

### 1. Attention Heatmaps
- **File**: `heatmap_qid*_layer*_head*.png`
- **Description**: Shows the full attention matrix
- **Red box**: Visual token region
- **Orange dashed box**: Attention from visual tokens

### 2. Attention Sink Visualization
- **File**: `sinks_qid*_layer*_head*.png`
- **Left panel**: Heatmap with sink tokens highlighted in red
- **Right panel**: Bar chart showing attention distribution over visual tokens
- **Red bars**: Identified sink tokens (above threshold)

### 3. Visual Attention Distribution
- **File**: `vis_attention_dist_qid*_layer*.png`
- **Description**: Shows how much attention each query position pays to visual tokens across 8 heads
- **Red region**: Visual token positions

### 4. Layer Comparison
- **File**: `layer_comparison_qid*.png`
- **Description**: Compares attention patterns across different layers (first 4 and last 4)

## Key Parameters

### Visual Attention Sink Logic

- **tau** (default: 20): Threshold for identifying visual attention sink tokens based on RMS norm values
- **rho** (default: 0.5): Maximum allowed portion of attention to visual sink tokens
- **p** (default: 0.6): Attention redistribution factor (how much to reduce sink attention)
- **summ** (default: 0.2): Minimum total attention that must be directed to the image
- **except_last_layer** (default: 1): Whether to skip attention redistribution in the last layer

### Visualization

- **threshold** (default: 0.1): Normalized attention threshold for identifying sink tokens in visualizations

## Example Workflow

Here's a complete example workflow:

```bash
# 1. Set up environment
conda activate VisAttnSink

# 2. Run inference with visualization (5 questions)
python src/inference_with_visualization.py \
    --device 0 \
    --exp_config A_exps/lv1.5_7b_viz.yml

# 3. Generate detailed visualizations for question 1
python src/visualization/visualize_attention.py \
    --attention_file F_visualizations/attention_qid1.pkl \
    --output_dir F_visualizations/detailed

# 4. Visualize specific layer (layer 15, middle of the model)
python src/visualization/visualize_attention.py \
    --attention_file F_visualizations/attention_qid1.pkl \
    --layer 15 \
    --output_dir F_visualizations/layer15
```

## Output Structure

After running the complete workflow:

```
F_visualizations/
├── attention_qid1.pkl                           # Saved attention data
├── attention_qid2.pkl
├── ...
├── preview_qid1_layer15_head0.png              # Quick preview
├── heatmap_qid1_layer15_head0.png              # Full heatmap
├── sinks_qid1_layer15_head0.png                # Sink analysis
├── vis_attention_dist_qid1_layer15.png         # Distribution across heads
└── layer_comparison_qid1.png                   # Cross-layer comparison

E_answers/
└── llava-v1.5-7b/
    └── [POPE-random]lv1.5_7b_visualization-viz-*.jsonl
```

## Interpreting Results

### Visual Attention Sink Phenomenon

1. **Identification**: Look for visual tokens (red regions) that consistently receive high attention across many query positions
2. **Layer Patterns**: Early layers may show different sink patterns than later layers
3. **Head Specialization**: Different attention heads may focus on different visual regions or have different sink patterns

### Expected Observations

Based on the paper, you should observe:
- Certain visual tokens consistently act as attention sinks across multiple heads
- The attention sink phenomenon is more pronounced in middle-to-late layers
- The logic mechanism (when enabled) redistributes attention from sink tokens to other visual tokens

## Troubleshooting

### Out of Memory
- Reduce `max_viz_questions` in the config
- Use a smaller batch size
- Visualize specific layers instead of all layers

### No Attention Data Captured
- Ensure `capture_attention: true` in config
- Check that `attn_implementation="eager"` is used (required for attention capture)
- Verify `output_attentions=True` in generate call

### Visualization Errors
- Check that attention data file exists and is not corrupted
- Verify visual token positions are correct (should be around 576 tokens for LLaVA-1.5)
- Ensure matplotlib and other visualization dependencies are installed

## Citation

If you use this code for your research, please cite:

```bibtex
@inproceedings{visual-attention-sink,
  title={See What You Are Told: Visual Attention Sink in Large Multimodal Models},
  author={[Authors]},
  booktitle={[Conference]},
  year={2024}
}
```

## Additional Resources

- [LLaVA Repository](https://github.com/haotian-liu/LLaVA)
- [Paper on arXiv](https://arxiv.org/abs/2503.03321)
- [POPE Dataset](https://github.com/AoiDragon/POPE)
