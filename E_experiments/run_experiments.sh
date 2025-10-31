#!/bin/bash
# Launcher script for VAR Critique Experiments

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}VAR Critique Experiments Launcher${NC}"
echo -e "${GREEN}================================${NC}"
echo ""

# Check if experiment type is provided
if [ $# -lt 1 ]; then
    echo -e "${YELLOW}Usage: $0 <experiment_type> [options]${NC}"
    echo ""
    echo "Experiment types:"
    echo "  1 | harmful    - Run Harmful Recycling Intervention Experiment"
    echo "  2 | ranking    - Run Logit Ranking Failure Experiment"
    echo "  both           - Run both experiments sequentially"
    echo ""
    echo "Example:"
    echo "  $0 1 --device 0 --max_samples 100"
    echo "  $0 harmful --device 0 --num_intervention_samples 5"
    echo "  $0 both --device 0"
    exit 1
fi

EXPERIMENT_TYPE=$1
shift  # Remove first argument, rest are passed to Python scripts

# Set default config files
CONFIG_HARMFUL="A_exps/exp1_harmful_recycling.yml"
CONFIG_RANKING="A_exps/exp2_logit_ranking.yml"

run_harmful_recycling() {
    echo -e "${GREEN}Running Experiment 1: Harmful Recycling Intervention${NC}"
    echo -e "${YELLOW}This experiment will:${NC}"
    echo "  1. Search for samples with 'relevant sink tokens'"
    echo "  2. Run three-mode intervention experiments"
    echo "  3. Compare baseline vs VAR vs ideal-VAR performance"
    echo ""
    
    python E_experiments/g_harmful_recycling_intervention.py \
        --exp_config $CONFIG_HARMFUL \
        "$@"
    
    echo -e "${GREEN}✓ Experiment 1 completed${NC}"
    echo ""
}

run_logit_ranking() {
    echo -e "${GREEN}Running Experiment 2: Logit Ranking Failure${NC}"
    echo -e "${YELLOW}This experiment will:${NC}"
    echo "  1. Compute consensus logits using ICHs"
    echo "  2. Rank visual tokens by relevance"
    echo "  3. Calculate Top-K contamination rate"
    echo ""
    
    python E_experiments/g_logit_ranking_failure.py \
        --exp_config $CONFIG_RANKING \
        "$@"
    
    echo -e "${GREEN}✓ Experiment 2 completed${NC}"
    echo ""
}

case $EXPERIMENT_TYPE in
    1|harmful)
        run_harmful_recycling "$@"
        ;;
    2|ranking)
        run_logit_ranking "$@"
        ;;
    both)
        run_harmful_recycling "$@"
        echo -e "${YELLOW}---${NC}"
        run_logit_ranking "$@"
        ;;
    *)
        echo -e "${RED}Error: Unknown experiment type '$EXPERIMENT_TYPE'${NC}"
        echo "Valid types: 1, harmful, 2, ranking, both"
        exit 1
        ;;
esac

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}All experiments completed!${NC}"
echo -e "${GREEN}Results saved in F_experiment_results/${NC}"
echo -e "${GREEN}================================${NC}"
