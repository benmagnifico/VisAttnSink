#!/bin/bash
#
# Quick Start Script for Visual Attention Sink Visualization
#
# This script helps you quickly set up and run the visualization experiments
# for the LLaVA-1.5-7B model.
#

set -e  # Exit on error

echo "============================================================"
echo "Visual Attention Sink Visualization - Quick Start"
echo "============================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Step 1: Check environment
print_info "Checking environment..."
if ! command -v python &> /dev/null; then
    echo "Error: Python not found. Please install Python 3.11+"
    exit 1
fi

if ! command -v conda &> /dev/null; then
    echo "Warning: Conda not found. Make sure you have the required packages installed."
fi

print_success "Environment check passed"
echo ""

# Step 2: Prepare sample dataset
print_info "Preparing sample dataset..."
python src/utils/prepare_sample_dataset.py \
    --num_questions 5 \
    --output_dir D_datasets/SAMPLE \
    --create_config

print_success "Sample dataset prepared"
echo ""

# Step 3: Instructions for adding images
echo "============================================================"
echo "IMPORTANT: Add your test images"
echo "============================================================"
echo ""
echo "Before running inference, please add your test images to:"
echo "  D_datasets/SAMPLE/Images/"
echo ""
echo "Required images:"
echo "  - sample_image_1.jpg"
echo "  - sample_image_2.jpg"
echo "  - sample_image_3.jpg"
echo "  - sample_image_4.jpg"
echo "  - sample_image_5.jpg"
echo ""
echo "You can use any JPEG images for testing."
echo ""
read -p "Press Enter when you have added the images..."

# Step 4: Update config with model path
echo ""
print_info "Please update the model path in A_exps/sample_viz.yml"
echo ""
echo "Edit the file and set:"
echo "  path_model: /path/to/llava-v1.5-7b"
echo ""
echo "Or use the HuggingFace Hub model directly:"
echo "  path_model: liuhaotian/llava-v1.5-7b"
echo ""
read -p "Press Enter when you have updated the config..."

# Step 5: Run inference with visualization
echo ""
print_info "Running inference with visualization capture..."
echo ""
echo "Command to run:"
echo "  python src/inference_with_visualization.py --device 0 --exp_config A_exps/sample_viz.yml"
echo ""
read -p "Press Enter to run inference, or Ctrl+C to exit..."

python src/inference_with_visualization.py --device 0 --exp_config A_exps/sample_viz.yml

print_success "Inference complete!"
echo ""

# Step 6: Generate additional visualizations
print_info "Generating additional visualizations..."
echo ""

# Find the first attention file
ATTENTION_FILE=$(find F_visualizations -name "attention_qid*.pkl" -type f | head -n 1)

if [ -z "$ATTENTION_FILE" ]; then
    echo "Warning: No attention files found in F_visualizations/"
else
    echo "Processing: $ATTENTION_FILE"
    python src/visualization/visualize_attention.py \
        --attention_file "$ATTENTION_FILE" \
        --output_dir F_visualizations
    
    print_success "Additional visualizations generated!"
fi

echo ""
echo "============================================================"
echo "Visualization Complete!"
echo "============================================================"
echo ""
echo "Results saved to:"
echo "  - Answers: E_answers/"
echo "  - Attention data: F_visualizations/attention_qid*.pkl"
echo "  - Visualizations: F_grafi/"
echo ""
echo "To generate more visualizations for a specific question:"
echo "  python src/visualization/visualize_attention.py \\"
echo "    --attention_file F_visualizations/attention_qid1.pkl \\"
echo "    --layer 15 --head 10"
echo ""
print_success "Quick start complete! Check VISUALIZATION_GUIDE.md for more details."
