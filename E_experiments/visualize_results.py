"""
Visualization script for experiment results.

This script generates plots and tables to visualize the results from both experiments.

Usage:
    python visualize_results.py --exp1_results F_experiment_results/harmful_recycling_intervention/12345/
    python visualize_results.py --exp2_results F_experiment_results/logit_ranking_failure/12345/
    python visualize_results.py --exp1_results ... --exp2_results ...  # Both
"""

import argparse
import json
import os
import os.path as osp
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.backends.backend_pdf import PdfPages
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Visualization features will be limited.")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Warning: numpy not installed. Some statistics will be limited.")


def visualize_harmful_recycling(results_dir):
    """Visualize results from Experiment 1: Harmful Recycling Intervention"""
    
    print(f"\n{'='*80}")
    print("Visualizing Experiment 1: Harmful Recycling Intervention")
    print(f"{'='*80}\n")
    
    # Load results
    results_file = osp.join(results_dir, "results.json")
    summary_file = osp.join(results_dir, "summary.json")
    
    if not osp.exists(results_file):
        print(f"Error: Results file not found at {results_file}")
        return
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    print(f"Loaded {len(results)} intervention experiment results")
    print(f"Samples with relevant sinks: {summary['samples_with_relevant_sinks']}")
    print()
    
    # Generate text report
    output_dir = osp.join(results_dir, "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    
    report_file = osp.join(output_dir, "text_report.txt")
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Experiment 1: Harmful Recycling Intervention - Detailed Report\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total samples examined: {summary['total_samples_examined']}\n")
        f.write(f"Samples with relevant sinks found: {summary['samples_with_relevant_sinks']}\n")
        f.write(f"Intervention experiments run: {summary['intervention_experiments_run']}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("Detailed Results per Sample\n")
        f.write("-"*80 + "\n\n")
        
        for i, result in enumerate(results):
            f.write(f"Sample {i+1} (QID: {result['qid']})\n")
            f.write(f"  Image: {result['image']}\n")
            f.write(f"  Query: {result['query']}\n")
            f.write(f"  Num relevant sink tokens: {result['num_relevant_sinks']}\n")
            f.write(f"  Relevant sink indices: {result['relevant_sink_tokens']}\n\n")
            
            f.write(f"  Mode A (Baseline):\n    {result['response_mode_A_baseline']}\n\n")
            f.write(f"  Mode B (VAR - Expected Harm):\n    {result['response_mode_B_var']}\n\n")
            f.write(f"  Mode C (Ideal VAR):\n    {result['response_mode_C_ideal_var']}\n\n")
            f.write("-"*80 + "\n\n")
    
    print(f"✓ Generated text report: {report_file}")
    
    # Generate HTML report with better formatting
    html_file = osp.join(output_dir, "report.html")
    with open(html_file, 'w') as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Harmful Recycling Intervention - Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; }
        .summary { background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }
        .sample { border: 1px solid #ddd; margin: 20px 0; padding: 20px; border-radius: 5px; }
        .mode { margin: 15px 0; padding: 10px; border-left: 4px solid #3498db; background: #f8f9fa; }
        .mode-a { border-left-color: #2ecc71; }
        .mode-b { border-left-color: #e74c3c; }
        .mode-c { border-left-color: #3498db; }
        .label { font-weight: bold; color: #555; }
        .response { margin-top: 5px; color: #333; }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
        .stat-box { background: #3498db; color: white; padding: 15px; border-radius: 5px; text-align: center; }
        .stat-value { font-size: 32px; font-weight: bold; }
        .stat-label { font-size: 14px; margin-top: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 Experiment 1: Harmful Recycling Intervention</h1>
        
        <div class="summary">
            <h2>Summary</h2>
            <div class="stats">
                <div class="stat-box">
                    <div class="stat-value">""" + str(summary['total_samples_examined']) + """</div>
                    <div class="stat-label">Samples Examined</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">""" + str(summary['samples_with_relevant_sinks']) + """</div>
                    <div class="stat-label">Relevant Sinks Found</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">""" + str(summary['intervention_experiments_run']) + """</div>
                    <div class="stat-label">Interventions Run</div>
                </div>
            </div>
        </div>
        
        <h2>Detailed Results</h2>
""")
        
        for i, result in enumerate(results):
            f.write(f"""
        <div class="sample">
            <h3>Sample {i+1} - QID: {result['qid']}</h3>
            <p><span class="label">Image:</span> {result['image']}</p>
            <p><span class="label">Query:</span> {result['query']}</p>
            <p><span class="label">Relevant Sink Tokens:</span> {result['num_relevant_sinks']} tokens at indices {result['relevant_sink_tokens']}</p>
            
            <div class="mode mode-a">
                <div class="label">Mode A (Baseline - No VAR)</div>
                <div class="response">{result['response_mode_A_baseline']}</div>
            </div>
            
            <div class="mode mode-b">
                <div class="label">Mode B (VAR - Expected to Harm)</div>
                <div class="response">{result['response_mode_B_var']}</div>
            </div>
            
            <div class="mode mode-c">
                <div class="label">Mode C (Ideal VAR - Exclude Relevant Sinks)</div>
                <div class="response">{result['response_mode_C_ideal_var']}</div>
            </div>
        </div>
""")
        
        f.write("""
    </div>
</body>
</html>
""")
    
    print(f"✓ Generated HTML report: {html_file}")
    print(f"\nOpen {html_file} in a browser to view formatted results.")


def visualize_logit_ranking(results_dir):
    """Visualize results from Experiment 2: Logit Ranking Failure"""
    
    print(f"\n{'='*80}")
    print("Visualizing Experiment 2: Logit Ranking Failure")
    print(f"{'='*80}\n")
    
    # Load results
    results_file = osp.join(results_dir, "results.json")
    summary_file = osp.join(results_dir, "summary.json")
    failures_file = osp.join(results_dir, "qualitative_failures.json")
    
    if not osp.exists(results_file):
        print(f"Error: Results file not found at {results_file}")
        return
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    if osp.exists(failures_file):
        with open(failures_file, 'r') as f:
            failures = json.load(f)
    else:
        failures = []
    
    stats = summary['aggregate_statistics']
    
    print(f"Analyzed {summary['samples_analyzed']} samples")
    print(f"Mean contamination rate: {stats['mean_contamination_rate']:.1%}")
    print(f"Top-1 irrelevant rate: {stats['top1_irrelevant_rate']:.1%}")
    print()
    
    # Generate visualizations directory
    output_dir = osp.join(results_dir, "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate histogram if matplotlib available
    if HAS_MATPLOTLIB and HAS_NUMPY:
        contamination_rates = [r['contamination_rate'] for r in results]
        
        plt.figure(figsize=(10, 6))
        plt.hist(contamination_rates, bins=20, edgecolor='black', alpha=0.7)
        plt.xlabel('Contamination Rate', fontsize=12)
        plt.ylabel('Number of Samples', fontsize=12)
        plt.title('Distribution of Top-K Contamination Rates', fontsize=14)
        plt.axvline(stats['mean_contamination_rate'], color='red', linestyle='--', 
                   label=f"Mean: {stats['mean_contamination_rate']:.1%}")
        plt.axvline(0.5, color='green', linestyle='--', label='50% threshold')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        hist_file = osp.join(output_dir, "contamination_histogram.png")
        plt.savefig(hist_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Generated histogram: {hist_file}")
    
    # Generate text report
    report_file = osp.join(output_dir, "text_report.txt")
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Experiment 2: Logit Ranking Failure - Detailed Report\n")
        f.write("="*80 + "\n\n")
        
        f.write("AGGREGATE STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Samples analyzed: {summary['samples_analyzed']}\n")
        f.write(f"Top-K: {summary['top_k']}\n")
        f.write(f"Mean contamination rate: {stats['mean_contamination_rate']:.2%}\n")
        f.write(f"Median contamination rate: {stats['median_contamination_rate']:.2%}\n")
        f.write(f"Std contamination rate: {stats['std_contamination_rate']:.2%}\n")
        f.write(f"Top-1 irrelevant rate: {stats['top1_irrelevant_rate']:.2%}\n")
        f.write(f"Top-1 irrelevant count: {stats['top1_irrelevant_count']}/{summary['samples_analyzed']}\n\n")
        
        f.write("QUALITATIVE FAILURES (Top-1 is Irrelevant)\n")
        f.write("-"*80 + "\n")
        f.write(f"Found {len(failures)} cases where Top-1 token was irrelevant\n\n")
        
        for i, failure in enumerate(failures[:20]):  # Show first 20
            f.write(f"Failure {i+1} (QID: {failure['qid']})\n")
            f.write(f"  Image: {failure['image']}\n")
            f.write(f"  Query: {failure['query']}\n")
            f.write(f"  Contamination rate: {failure['contamination_rate']:.1%}\n")
            f.write(f"  Best relevant token rank: {failure['ranking_stats']['best_relevant_rank']}\n")
            f.write(f"  Top-1 token index: {failure['ranking_stats']['top_1_index']} (IRRELEVANT)\n")
            f.write("\n")
    
    print(f"✓ Generated text report: {report_file}")
    
    # Generate HTML report
    html_file = osp.join(output_dir, "report.html")
    with open(html_file, 'w') as f:
        f.write(f"""
<!DOCTYPE html>
<html>
<head>
    <title>Logit Ranking Failure - Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #e74c3c; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .summary {{ background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
        .stat-box {{ background: #e74c3c; color: white; padding: 15px; border-radius: 5px; text-align: center; }}
        .stat-value {{ font-size: 32px; font-weight: bold; }}
        .stat-label {{ font-size: 14px; margin-top: 5px; }}
        .failure {{ border: 1px solid #ddd; margin: 20px 0; padding: 20px; border-radius: 5px; background: #fff5f5; }}
        .label {{ font-weight: bold; color: #555; }}
        .warning {{ color: #e74c3c; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 Experiment 2: Logit Ranking Failure</h1>
        
        <div class="summary">
            <h2>Aggregate Statistics</h2>
            <div class="stats">
                <div class="stat-box">
                    <div class="stat-value">{stats['mean_contamination_rate']:.1%}</div>
                    <div class="stat-label">Mean Contamination</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{stats['top1_irrelevant_rate']:.1%}</div>
                    <div class="stat-label">Top-1 Irrelevant Rate</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{summary['samples_analyzed']}</div>
                    <div class="stat-label">Samples Analyzed</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{len(failures)}</div>
                    <div class="stat-label">Qualitative Failures</div>
                </div>
            </div>
        </div>
        
        <h2>Key Finding</h2>
        <p style="font-size: 18px; padding: 15px; background: #fff3cd; border-left: 4px solid #ffc107;">
            <strong>{stats['mean_contamination_rate']:.1%}</strong> of Top-{summary['top_k']} tokens ranked by Q·K^T logits
            are actually <strong class="warning">irrelevant</strong> (outside ground-truth bounding box).
        </p>
        
        <h2>Notable Failures (Top-1 is Irrelevant)</h2>
""")
        
        for i, failure in enumerate(failures[:10]):  # Show first 10
            f.write(f"""
        <div class="failure">
            <h3>Failure {i+1} - QID: {failure['qid']}</h3>
            <p><span class="label">Image:</span> {failure['image']}</p>
            <p><span class="label">Query:</span> {failure['query']}</p>
            <p><span class="label">Contamination Rate:</span> {failure['contamination_rate']:.1%}</p>
            <p><span class="warning">⚠ Top-1 token (index {failure['ranking_stats']['top_1_index']}) is IRRELEVANT</span></p>
            <p><span class="label">Best relevant token rank:</span> #{failure['ranking_stats']['best_relevant_rank'] + 1}</p>
        </div>
""")
        
        f.write("""
    </div>
</body>
</html>
""")
    
    print(f"✓ Generated HTML report: {html_file}")
    print(f"\nOpen {html_file} in a browser to view formatted results.")


def main():
    parser = argparse.ArgumentParser(description="Visualize experiment results")
    parser.add_argument('--exp1_results', type=str,
                       help='Path to Experiment 1 results directory')
    parser.add_argument('--exp2_results', type=str,
                       help='Path to Experiment 2 results directory')
    
    args = parser.parse_args()
    
    if not args.exp1_results and not args.exp2_results:
        print("Error: Please provide at least one results directory")
        print("  --exp1_results for Harmful Recycling Intervention")
        print("  --exp2_results for Logit Ranking Failure")
        return
    
    if args.exp1_results:
        visualize_harmful_recycling(args.exp1_results)
    
    if args.exp2_results:
        visualize_logit_ranking(args.exp2_results)
    
    print(f"\n{'='*80}")
    print("Visualization complete!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
