# Paper Implementation Notes

## Paper: "SEE WHAT YOU ARE TOLD: VISUAL ATTENTION SINK IN LARGE MULTIMODAL MODELS"

### Key Findings from the Paper

1. **Visual Attention Sink Phenomenon**
   - Certain visual tokens consistently receive disproportionately high attention
   - These "sink" tokens may not be semantically relevant
   - Can lead to object hallucination and reduced visual grounding

2. **Model Tested**
   - LLaVA-1.5-7B (primary model)
   - Also tested on LLaVA-1.5-13B and LLaVA-1.6-Vicuna-13B

3. **Experimental Setup**
   - Vision Encoder: CLIP ViT-L/14
   - Visual Tokens: 576 tokens (24×24 patches from 336×336 images)
   - LLM: Llama-2-7B

### VAR (Visual Attention Redistribution) Mechanism

The paper proposes a mechanism to address visual attention sinks:

#### Parameters (from paper)
- **τ (tau)**: Threshold for identifying sink tokens based on RMS norm
  - Default: 20
  - Used to detect which tokens are "sink candidates"
  
- **ρ (rho)**: Maximum allowed attention portion to visual sinks
  - Default: 0.5
  - If attention to sinks exceeds this, redistribution occurs
  
- **p**: Attention redistribution factor
  - Default: 0.6
  - Determines how much attention is reduced from sinks
  
- **summ**: Minimum total attention to image
  - Default: 0.2
  - Ensures sufficient attention remains on visual content

#### Algorithm
1. **Identify Sink Tokens** (DimProspector)
   - Calculate RMS norm for each visual token's hidden state
   - Select tokens with specific dimensions exceeding τ
   - Dimensions used: [2533, 1415] for Llama-2-7B

2. **Detect Problematic Heads** (HeadFork)
   - For each attention head:
     - Calculate portion of attention to sink tokens
     - Check if portion ≤ ρ
     - Check if total attention to image ≥ summ
   - Mark heads meeting both conditions for redistribution

3. **Redistribute Attention** (VARProcessor)
   - For marked heads:
     - Reduce attention to sink tokens by factor p
     - Redistribute freed attention to non-sink visual tokens
     - Maintain attention distribution proportions

### Datasets Used in Paper

1. **POPE (Polling-based Object Probing Evaluation)**
   - Tests object hallucination
   - Categories: random, popular, adversarial
   - Binary yes/no questions

2. **MMBench**
   - Multi-modal understanding benchmark
   - Various vision-language tasks

3. **SEED-Bench**
   - Comprehensive evaluation suite
   - Tests multiple capabilities

### Evaluation Metrics

1. **Accuracy**: For yes/no questions (POPE)
2. **F1 Score**: Particularly for POPE
3. **MMBench Score**: Standard MMBench evaluation
4. **SEED Score**: Standard SEED-Bench evaluation

### Expected Results (from paper)

On POPE dataset:
- **Without VAR**: Lower accuracy, higher hallucination
- **With VAR**: Improved accuracy, reduced hallucination
  - Random: ~87% → ~89%
  - Popular: ~82% → ~85%
  - Adversarial: ~77% → ~81%

### Implementation Checklist

- [x] DimProspector: Identify sink tokens using RMS norm
- [x] HeadFork: Detect heads with problematic attention patterns
- [x] VARProcessor: Redistribute attention from sinks
- [x] Visualization: Capture and visualize attention patterns
- [x] Configuration: Proper parameter settings (τ=20, ρ=0.5, p=0.6, summ=0.2)

### Reproduction Steps

1. **Prepare POPE Dataset**
   ```bash
   # Download COCO val2014 images
   # Download POPE questions from official repo
   ```

2. **Configure Experiment**
   ```yaml
   logic: 1
   tau: 20
   rho: 0.5
   p: 0.6
   summ: 0.2
   except_last_layer: 1
   ```

3. **Run Baseline (without VAR)**
   ```yaml
   logic: 0
   ```

4. **Run with VAR**
   ```yaml
   logic: 1
   ```

5. **Compare Results**
   - Accuracy improvement
   - F1 score improvement
   - Visual attention distribution changes

### Visualization Focus

Key things to visualize:
1. **Attention Heatmaps** at layer 15-20 (middle layers)
2. **Sink Token Patterns** across different heads
3. **Before/After VAR** comparison
4. **Layer-wise Evolution** of attention patterns

### Critical Implementation Details

1. **Visual Token Range**
   - For LLaVA-1.5: typically tokens at positions [im_start:im_start+576]
   - Need to track where image tokens are inserted

2. **Layer Selection**
   - Apply VAR starting from layer 2 (except first 2 layers)
   - Optionally skip last layer

3. **Dimension Selection**
   - Llama-2-7B: dimensions [2533, 1415]
   - Llama-2-13B: dimensions [2100, 4743]

4. **Attention Implementation**
   - Must use "eager" attention (not SDPA) to capture weights
   - Need to enable `output_attentions=True`

### References

- Paper arXiv: https://arxiv.org/abs/2503.03321
- OpenReview: https://openreview.net/forum?id=7uDI7w5RQA
- Original LLaVA: https://github.com/haotian-liu/LLaVA
- POPE Dataset: https://github.com/AoiDragon/POPE
