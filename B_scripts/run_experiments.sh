#!/bin/bash
# Script to run VAR assumption validation experiments

# Default values
EXPERIMENT="both"
MODEL_PATH=""
DATASET_PATH="C_datasets/refcoco"
DATASET_SPLIT="val"
OUTPUT_DIR="E_experiments"
DEVICE="0"
TAU="20.0"

# Help message
show_help() {
    cat << EOF
Usage: ${0##*/} [OPTIONS]

Run VAR assumption validation experiments.

OPTIONS:
    -e, --experiment TYPE       Which experiment to run (harmful_recycling, logit_ranking, both)
                                Default: both
    -m, --model PATH            Path to LLaVA model checkpoint (REQUIRED)
    -d, --dataset PATH          Path to RefCOCO dataset (Default: C_datasets/refcoco)
    -s, --split SPLIT           Dataset split (train, val, test) (Default: val)
    -o, --output DIR            Output directory (Default: E_experiments)
    --device DEVICE             CUDA device number (Default: 0)
    --tau TAU                   Sink token detection threshold (Default: 20.0)
    --num-search N              Number of samples to search (Default: 100)
    --num-intervention N        Number of intervention samples (Default: 5)
    --num-logit N               Number of samples for logit ranking (Default: 50)
    --layers LAYERS             Comma-separated layer indices (e.g., "10,20,30")
    -h, --help                  Show this help message

EXAMPLES:
    # Run both experiments with LLaVA-1.5-7B
    ${0##*/} --experiment both --model /path/to/llava-v1.5-7b

    # Run only harmful recycling experiment
    ${0##*/} -e harmful_recycling -m /path/to/llava-v1.5-7b --tau 25.0

    # Run logit ranking with specific layers
    ${0##*/} -e logit_ranking -m /path/to/llava-v1.5-7b --layers 10,20,30

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--experiment)
            EXPERIMENT="$2"
            shift 2
            ;;
        -m|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        -d|--dataset)
            DATASET_PATH="$2"
            shift 2
            ;;
        -s|--split)
            DATASET_SPLIT="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --tau)
            TAU="$2"
            shift 2
            ;;
        --num-search)
            NUM_SEARCH="$2"
            shift 2
            ;;
        --num-intervention)
            NUM_INTERVENTION="$2"
            shift 2
            ;;
        --num-logit)
            NUM_LOGIT="$2"
            shift 2
            ;;
        --layers)
            LAYERS="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$MODEL_PATH" ]; then
    echo "Error: Model path is required!"
    echo ""
    show_help
    exit 1
fi

# Build command
CMD="python src/run_experiments.py \
    --experiment $EXPERIMENT \
    --model_path $MODEL_PATH \
    --dataset_path $DATASET_PATH \
    --dataset_split $DATASET_SPLIT \
    --output_dir $OUTPUT_DIR \
    --device $DEVICE \
    --tau $TAU"

# Add optional arguments if provided
[ ! -z "$NUM_SEARCH" ] && CMD="$CMD --num_search_samples $NUM_SEARCH"
[ ! -z "$NUM_INTERVENTION" ] && CMD="$CMD --num_intervention_samples $NUM_INTERVENTION"
[ ! -z "$NUM_LOGIT" ] && CMD="$CMD --num_logit_samples $NUM_LOGIT"
[ ! -z "$LAYERS" ] && CMD="$CMD --layers $LAYERS"

# Print command
echo "Running experiments with command:"
echo "$CMD"
echo ""

# Execute
eval $CMD
