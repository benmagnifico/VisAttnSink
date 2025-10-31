# Example: Visualizing Visual Attention Sink in LLaVA-1.5-7B

This example demonstrates how to reproduce the visual attention sink phenomenon described in the paper "SEE WHAT YOU ARE TOLD: VISUAL ATTENTION SINK IN LARGE MULTIMODAL MODELS".

## What is Visual Attention Sink?

The visual attention sink phenomenon refers to the observation that:
1. Certain visual tokens in multimodal models receive disproportionately high attention from text tokens
2. These "sink" tokens act as focal points for attention, even when they may not be semantically relevant
3. This phenomenon can lead to:
   - Reduced attention to other important visual features
   - Potential object hallucination issues
   - Suboptimal visual grounding

## Step-by-Step Example

### 1. Setup

First, ensure you have the environment set up:

```bash
conda activate VisAttnSink
cd /path/to/VisAttnSink
```

### 2. Prepare a Sample Image and Question

Create a simple test case:

**Image**: A photo of a cat sitting on a table
**Question**: "Is there a dog in the image?"
**Expected Answer**: "No"

Save this as `D_datasets/SAMPLE/Images/cat_on_table.jpg`

Create a question file `D_datasets/SAMPLE/Questions/test-questions.jsonl`:
```json
{"question_id": 1, "image": "cat_on_table.jpg", "text": "Is there a dog in the image?", "label": "no"}
```

### 3. Configure Experiment

Create `A_exps/example_viz.yml`:

```yaml
name_exp: example_visualization
name_daset: SAMPLE
name_category: test

path_image_dir: D_datasets/SAMPLE/Images
path_question_dir: D_datasets/SAMPLE/Questions
path_model: liuhaotian/llava-v1.5-7b

conv_mode: vicuna_v1

# Enable visual attention sink logic
logic: 1
tau: 20      # Sink identification threshold
rho: 0.5     # Max attention to sinks
p: 0.6       # Redistribution factor
summ: 0.2    # Min total image attention

max_new_tokens: 128
except_last_layer: 1

# Enable attention capture
capture_attention: true
max_viz_questions: 1
```

### 4. Run Inference with Visualization

```bash
python src/inference_with_visualization.py \
    --device 0 \
    --exp_config A_exps/example_viz.yml
```

This will:
- Load the LLaVA-1.5-7B model
- Process the question with attention capture enabled
- Save attention weights to `F_visualizations/attention_qid1.pkl`
- Generate preview visualization
- Save the model's response

### 5. Analyze the Results

#### 5.1 Check the Model's Response

```bash
cat E_answers/llava-v1.5-7b/*example_visualization*.jsonl
```

Example output:
```json
{"question_id": 1, "prompt": "Is there a dog in the image?", "label": "no", "response": "No, there is no dog in the image. The image shows a cat sitting on a table.", "image": "cat_on_table.jpg", "model_id": "llava-v1.5-7b"}
```

#### 5.2 Examine Attention Patterns

Generate comprehensive visualizations:

```bash
python src/visualization/visualize_attention.py \
    --attention_file F_visualizations/attention_qid1.pkl \
    --output_dir F_visualizations/example
```

This creates:
1. **Heatmaps** (`heatmap_qid1_layer*_head*.png`) - Full attention matrices
2. **Sink Analysis** (`sinks_qid1_layer*_head*.png`) - Identified attention sinks
3. **Distribution Plots** (`vis_attention_dist_qid1_layer*.png`) - Attention to visual tokens across heads
4. **Layer Comparison** (`layer_comparison_qid1.png`) - How attention evolves across layers

### 6. Interpret the Visualizations

#### Understanding the Heatmap

```
       Key Tokens →
Query  ┌─────────────────────────────┐
Tokens │ System | <img> | Question  │
  ↓    │        |       |           │
       │--------┼-------┼-----------│
       │ Low    │ High  │ Medium    │
       │ Attn   │ Attn  │ Attn      │
       └─────────────────────────────┘
```

- **Red box**: Visual token region (typically 576 tokens for LLaVA-1.5)
- **Bright areas**: High attention weights
- **Dark areas**: Low attention weights

#### Identifying Attention Sinks

In the sink visualization, look for:
- **Red vertical lines** in the heatmap: These are sink tokens
- **Red bars** in the bar chart: Tokens receiving > threshold attention
- **Concentration**: If attention is concentrated on a few tokens (sinks) vs. distributed

Example interpretation:
```
Visual Token Index:  0   50  100 150 200 250 300 350 400 450 500 550
Attention:          [██  ████  █   ███  ██  ████  █   █   ██  ██  █ ]
                          ↑              ↑
                       Sinks            Sinks
```

### 7. Compare With and Without VAR (Visual Attention Redistribution)

#### Without VAR (logic: 0)

Edit config to disable the logic:
```yaml
logic: 0
```

Run inference again and compare:

**Without VAR**: 
- Strong concentration on a few visual tokens (attention sinks)
- May lead to hallucination (e.g., seeing a "dog" due to focusing on irrelevant features)

**With VAR** (logic: 1):
- More distributed attention across visual tokens
- Better visual grounding
- Reduced hallucination

### 8. Analyze Specific Layers and Heads

Focus on middle layers (where the phenomenon is most pronounced):

```bash
# Analyze layer 15 (middle of 32 layers)
python src/visualization/visualize_attention.py \
    --attention_file F_visualizations/attention_qid1.pkl \
    --layer 15 \
    --output_dir F_visualizations/layer15

# Analyze specific head that shows strong sink behavior
python src/visualization/visualize_attention.py \
    --attention_file F_visualizations/attention_qid1.pkl \
    --layer 15 \
    --head 10 \
    --threshold 0.05
```

### 9. Expected Observations

Based on the paper, you should observe:

1. **Early Layers (0-10)**:
   - Relatively distributed attention
   - Less pronounced sink behavior

2. **Middle Layers (10-22)**:
   - Clear visual attention sinks emerge
   - Some tokens receive > 10% of total attention
   - Sink pattern varies by head

3. **Late Layers (22-32)**:
   - Continued sink behavior
   - May be modulated by the VAR mechanism (if enabled)

4. **Effect of VAR**:
   - Reduced attention concentration
   - More uniform distribution over visual tokens
   - Improved performance on hallucination benchmarks

## Common Patterns

### Pattern 1: Corner Sinks
Some models show attention sinks at image corners or edges.

### Pattern 2: Central Sinks
Attention may concentrate on central image regions.

### Pattern 3: Patch Boundary Sinks
Sinks may appear at patch boundaries (artifacts of vision encoder).

## Troubleshooting

### Issue: No clear sinks visible
- Try lowering the threshold: `--threshold 0.05`
- Check if you're looking at the right layer (try layer 15-20)
- Visualize more heads to find sink behavior

### Issue: Too many sinks
- Increase the threshold: `--threshold 0.15`
- This may indicate the model is working correctly (distributed attention)

### Issue: Visualizations are too small
- Edit the visualization code to increase `figsize`
- Use higher DPI: modify `dpi=300` in visualization functions

## Next Steps

1. **Try Different Questions**: Test with questions known to cause hallucination
2. **Compare Models**: Run same questions on different model sizes
3. **Analyze Failure Cases**: Focus on questions where the model hallucinates
4. **Parameter Tuning**: Experiment with different τ, ρ, p values

## References

- Paper: "See What You Are Told: Visual Attention Sink in Large Multimodal Models"
- arXiv: https://arxiv.org/abs/2503.03321
- Original LLaVA: https://github.com/haotian-liu/LLaVA
