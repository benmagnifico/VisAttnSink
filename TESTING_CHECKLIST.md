# Testing and Validation Checklist

This document provides a checklist for validating the visual attention sink visualization implementation.

## Prerequisites Checklist

- [ ] Conda environment created and activated
- [ ] All dependencies installed from `env_pip.txt`
- [ ] PyTorch with CUDA support installed (if using GPU)
- [ ] LLaVA-1.5-7B model downloaded or accessible via HuggingFace Hub
- [ ] Test dataset prepared (POPE or sample dataset)
- [ ] GPU available with sufficient memory (recommended: 24GB+)

## Code Validation

### Syntax Check
- [x] All Python files compile without syntax errors
- [x] Import statements are correct
- [x] Module paths are properly configured

### File Structure
- [x] `src/visualization/__init__.py` exists
- [x] `src/visualization/attention_capture.py` exists
- [x] `src/visualization/attention_visualizer.py` exists
- [x] `src/visualization/visualize_attention.py` exists
- [x] `src/inference_with_visualization.py` exists
- [x] `src/utils/prepare_sample_dataset.py` exists
- [x] Configuration files in `A_exps/` exist
- [x] Documentation files exist

## Functional Testing

### Unit Tests

#### Test 1: Attention Capture Module
```bash
python3 -c "from src.visualization.attention_capture import AttentionCapture; print('✓ Import successful')"
```
- [ ] Module imports successfully
- [ ] Can create AttentionCapture instance
- [ ] Can activate/deactivate capture
- [ ] Can store metadata
- [ ] Can save/load pickle files

#### Test 2: Attention Visualizer Module
```bash
python3 -c "from src.visualization.attention_visualizer import AttentionVisualizer; print('✓ Import successful')"
```
- [ ] Module imports successfully
- [ ] Can create AttentionVisualizer instance
- [ ] Matplotlib backend works
- [ ] Can create synthetic visualizations

#### Test 3: Visualization Script
```bash
python src/visualization/visualize_attention.py --help
```
- [ ] Script runs without errors
- [ ] Help message displays correctly
- [ ] Arguments are properly defined

### Integration Tests

#### Test 4: Sample Dataset Preparation
```bash
python src/utils/prepare_sample_dataset.py --num_questions 3 --output_dir /tmp/test_dataset --create_config
```
- [ ] Creates directory structure
- [ ] Generates JSONL file with questions
- [ ] Creates sample config file
- [ ] Output messages are informative

#### Test 5: Visualization with Synthetic Data
```bash
# This requires numpy, torch, matplotlib
python src/visualization/test_visualization.py
```
- [ ] Creates synthetic attention data
- [ ] Generates all visualization types
- [ ] Saves files to output directory
- [ ] No errors during execution

### End-to-End Testing (Requires Model)

#### Test 6: Inference with Visualization
```bash
python src/inference_with_visualization.py \
    --device 0 \
    --exp_config A_exps/lv1.5_7b_viz.yml
```
- [ ] Model loads successfully
- [ ] Inference completes without errors
- [ ] Attention data is captured
- [ ] Pickle files are created in F_visualizations/
- [ ] Preview visualizations are generated
- [ ] Answers are saved to E_answers/

#### Test 7: Visualization Generation
```bash
python src/visualization/visualize_attention.py \
    --attention_file F_visualizations/attention_qid1.pkl \
    --output_dir F_visualizations/test
```
- [ ] Loads attention data successfully
- [ ] Generates heatmaps
- [ ] Generates sink visualizations
- [ ] Generates distribution plots
- [ ] Generates layer comparison
- [ ] All images are created without errors

## Output Validation

### Attention Data Files
- [ ] `F_visualizations/attention_qid*.pkl` files exist
- [ ] Files contain both 'attention' and 'metadata' keys
- [ ] Metadata includes all required fields (question_id, question, image_path, etc.)
- [ ] Attention data has correct shape [batch, heads, query, key]

### Visualization Files
- [ ] Heatmap images are generated
- [ ] Sink visualization images are generated
- [ ] Distribution plots are generated
- [ ] Layer comparison images are generated
- [ ] Images are viewable and correctly formatted
- [ ] Visual token regions are highlighted correctly

### Answer Files
- [ ] JSONL files created in E_answers/
- [ ] Each line contains required fields (question_id, prompt, response, etc.)
- [ ] Responses are coherent and relevant to questions

## Quality Checks

### Visual Attention Sink Detection
- [ ] Sink tokens are identified in visualizations
- [ ] Sink positions make sense (within visual token range)
- [ ] Multiple sinks can be detected when present
- [ ] Threshold parameter affects detection appropriately

### VAR Mechanism
- [ ] With `logic: 0`, baseline attention patterns are captured
- [ ] With `logic: 1`, VAR mechanism is applied
- [ ] Attention redistribution is visible in visualizations
- [ ] Results differ between VAR enabled/disabled

### Visualization Quality
- [ ] Heatmaps are clear and readable
- [ ] Color schemes are appropriate
- [ ] Labels and titles are informative
- [ ] Legends are present where needed
- [ ] Visual token regions are clearly marked
- [ ] Resolution is sufficient (150 DPI)

## Performance Validation

### Memory Usage
- [ ] Model loads without OOM errors
- [ ] Attention capture doesn't cause OOM
- [ ] Visualization generation completes successfully
- [ ] Memory is properly cleared between questions

### Processing Time
- [ ] Inference completes in reasonable time
- [ ] Attention capture doesn't significantly slow inference
- [ ] Visualization generation is reasonably fast
- [ ] Batch processing works for multiple questions

## Documentation Validation

### README Files
- [ ] Main README.md is clear and comprehensive
- [ ] VISUALIZATION_GUIDE.md provides complete instructions
- [ ] EXAMPLE.md walkthrough is easy to follow
- [ ] README_CN.md (Chinese) is accurate translation
- [ ] PAPER_NOTES.md correctly summarizes paper

### Code Documentation
- [ ] All functions have docstrings
- [ ] Complex logic is commented
- [ ] Module purposes are clear
- [ ] Examples are provided where helpful

### Configuration Files
- [ ] YAML configs have all required fields
- [ ] Comments explain parameters
- [ ] Default values match paper specifications
- [ ] Examples are provided

## Reproducibility Checklist

### Paper Results Reproduction
- [ ] Parameters match paper (τ=20, ρ=0.5, p=0.6, summ=0.2)
- [ ] Visual token dimensions correct for model
- [ ] Layer selection matches paper
- [ ] Can run on POPE dataset
- [ ] Results are comparable to paper

### Documentation Completeness
- [ ] Installation steps are complete
- [ ] All dependencies are listed
- [ ] Dataset preparation is explained
- [ ] Configuration is documented
- [ ] Usage examples are provided
- [ ] Troubleshooting tips are included

## Known Limitations

Document any known issues or limitations:

1. **Environment Requirements**
   - Requires specific PyTorch version for compatibility
   - Needs sufficient GPU memory (24GB recommended)
   - Some dependencies may conflict with other packages

2. **Dataset Requirements**
   - POPE dataset requires manual download
   - Image paths must be correctly configured
   - Question format must match expected schema

3. **Visualization Constraints**
   - Large models may be slow to process
   - Many layers/heads generate many files
   - High-resolution images require more memory

## Future Improvements

Potential enhancements:

- [ ] Add unit tests with pytest
- [ ] Add CI/CD pipeline
- [ ] Support more model architectures
- [ ] Add interactive visualizations
- [ ] Create web-based viewer
- [ ] Add statistical analysis tools
- [ ] Support batch visualization generation
- [ ] Add comparison tools for different runs

## Sign-off

- [ ] All critical tests passed
- [ ] Documentation is complete
- [ ] Code is ready for use
- [ ] Known issues are documented

**Tested by**: _________________
**Date**: _________________
**Notes**: _________________
