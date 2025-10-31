"""
Helper script to prepare RefCOCO data for the experiments.

This script helps convert RefCOCO annotations to the required JSONL format
with bounding box information.

Usage:
    python prepare_refcoco_data.py \
        --refcoco_dir /path/to/refcoco \
        --output_dir /path/to/output \
        --split testA

Output format (JSONL):
    {
        "qid": 12345,
        "image": "COCO_train2014_000000123456.jpg",
        "text": "the woman in the red dress",
        "bbox": [x, y, width, height],
        "label": "person"
    }
"""

import argparse
import json
import os
import os.path as osp
from collections import defaultdict


def prepare_refcoco_format(refcoco_dir, output_dir, split='testA'):
    """
    Convert RefCOCO annotations to experiment-ready format.
    
    This is a template function. Actual implementation depends on your
    RefCOCO data format. Common formats include:
    - Original RefCOCO JSON format
    - COCO-style annotations
    - Pickle format
    
    Args:
        refcoco_dir: Directory containing RefCOCO data
        output_dir: Directory to save output JSONL
        split: Data split (testA, testB, val, etc.)
    """
    
    print(f"Preparing RefCOCO {split} data...")
    print(f"Input directory: {refcoco_dir}")
    print(f"Output directory: {output_dir}")
    
    # Example structure - adapt to your data format
    # This assumes you have RefCOCO in standard format
    
    # Load RefCOCO annotations (format depends on source)
    # Common sources:
    # 1. Original RefCOCO: uses pickle files
    # 2. Hugging Face datasets
    # 3. Custom JSON format
    
    # Placeholder data structure - replace with actual data loading
    examples = []
    
    # Example: If you have annotations in JSON format
    annotation_file = osp.join(refcoco_dir, f"{split}_annotations.json")
    
    if osp.exists(annotation_file):
        print(f"Loading annotations from {annotation_file}")
        with open(annotation_file, 'r') as f:
            data = json.load(f)
            
        # Convert to required format
        # This is highly dependent on your annotation format
        for idx, item in enumerate(data):
            example = {
                'qid': idx,
                'image': item.get('image_file', ''),
                'text': item.get('caption', '') or item.get('phrase', ''),
                'bbox': item.get('bbox', [0, 0, 0, 0]),  # [x, y, w, h]
                'label': item.get('category', '') or 'object'
            }
            examples.append(example)
    else:
        print(f"Warning: Annotation file not found at {annotation_file}")
        print("\nYou need to adapt this script to your RefCOCO data format.")
        print("Common formats:")
        print("  1. Pickle format (.pkl): Use pickle.load()")
        print("  2. JSON format: Already supported above")
        print("  3. Hugging Face datasets: Use datasets.load_dataset('refcoco')")
        print("\nExample for Hugging Face datasets:")
        print("  from datasets import load_dataset")
        print("  dataset = load_dataset('HuggingFaceM4/RefCOCO', split='testA')")
        print("\nExample minimal data structure:")
        print("  {")
        print('    "qid": 0,')
        print('    "image": "COCO_train2014_000000001234.jpg",')
        print('    "text": "the woman in red",')
        print('    "bbox": [100, 150, 200, 300],  # [x, y, width, height]')
        print('    "label": "person"')
        print("  }")
        return False
    
    # Save to JSONL
    os.makedirs(output_dir, exist_ok=True)
    output_file = osp.join(output_dir, f"{split}-questions.jsonl")
    
    with open(output_file, 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"\n✓ Saved {len(examples)} examples to {output_file}")
    return True


def create_sample_data(output_dir):
    """
    Create a small sample dataset for testing.
    This creates dummy data that follows the correct format.
    """
    print("Creating sample test data...")
    
    # Create sample examples
    samples = [
        {
            'qid': 0,
            'image': 'sample_image_001.jpg',
            'text': 'the red car on the left',
            'bbox': [50, 100, 200, 150],  # [x, y, width, height]
            'label': 'car'
        },
        {
            'qid': 1,
            'image': 'sample_image_002.jpg',
            'text': 'the woman wearing glasses',
            'bbox': [150, 80, 180, 250],
            'label': 'person'
        },
        {
            'qid': 2,
            'image': 'sample_image_003.jpg',
            'text': 'the book on the table',
            'bbox': [200, 300, 100, 80],
            'label': 'book'
        }
    ]
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = osp.join(output_dir, "sample-questions.jsonl")
    
    with open(output_file, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"✓ Created sample data at {output_file}")
    print(f"  Contains {len(samples)} sample entries")
    print("\nNote: You need to provide actual images in the image directory")
    print("  and replace this with real RefCOCO data for actual experiments.")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Prepare RefCOCO data for VAR critique experiments"
    )
    parser.add_argument(
        '--refcoco_dir',
        type=str,
        help='Directory containing RefCOCO data'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save processed JSONL files'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='testA',
        choices=['testA', 'testB', 'val', 'train'],
        help='RefCOCO split to process'
    )
    parser.add_argument(
        '--create_sample',
        action='store_true',
        help='Create sample test data instead of processing real data'
    )
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_data(args.output_dir)
    else:
        if not args.refcoco_dir:
            print("Error: --refcoco_dir is required when not using --create_sample")
            return
        
        success = prepare_refcoco_format(
            args.refcoco_dir,
            args.output_dir,
            args.split
        )
        
        if not success:
            print("\n" + "="*80)
            print("IMPORTANT: You need to adapt this script to your data format!")
            print("="*80)
            print("\nFor testing, you can create sample data with:")
            print(f"  python {__file__} --output_dir {args.output_dir} --create_sample")


if __name__ == '__main__':
    main()
