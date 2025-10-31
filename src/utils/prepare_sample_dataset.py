#!/usr/bin/env python3
"""
Sample Dataset Preparation for Visual Attention Sink Experiments

This script creates a small sample dataset for testing the visualization pipeline.
It generates synthetic questions in the POPE format.
"""

import json
import os
import argparse
from pathlib import Path


def create_sample_questions(num_questions=5, output_dir="D_datasets/SAMPLE"):
    """
    Create sample questions in POPE format for testing.
    
    Args:
        num_questions: Number of sample questions to generate
        output_dir: Directory to save the questions
    """
    questions_dir = os.path.join(output_dir, "Questions")
    os.makedirs(questions_dir, exist_ok=True)
    
    # Sample questions for testing
    sample_questions = [
        "Is there a cat in the image?",
        "Can you see a dog?",
        "Is there a person in the image?",
        "Is there a car in the image?",
        "Can you see a tree?",
        "Is there a building in the image?",
        "Is there a chair?",
        "Can you see a table?",
        "Is there a bird in the image?",
        "Is there a computer?",
    ]
    
    # Sample image filenames (you would need to provide actual images)
    sample_images = [
        "sample_image_1.jpg",
        "sample_image_2.jpg",
        "sample_image_3.jpg",
        "sample_image_4.jpg",
        "sample_image_5.jpg",
    ]
    
    questions_data = []
    
    for i in range(min(num_questions, len(sample_questions))):
        question = {
            "question_id": i + 1,
            "image": sample_images[i % len(sample_images)],
            "text": sample_questions[i],
            "label": "yes" if i % 2 == 0 else "no"  # Alternate yes/no for testing
        }
        questions_data.append(question)
    
    # Save to JSONL file
    output_file = os.path.join(questions_dir, "sample-questions.jsonl")
    with open(output_file, 'w') as f:
        for question in questions_data:
            f.write(json.dumps(question) + '\n')
    
    print(f"✓ Created {len(questions_data)} sample questions")
    print(f"✓ Saved to {output_file}")
    
    # Create placeholder for images directory
    images_dir = os.path.join(output_dir, "Images")
    os.makedirs(images_dir, exist_ok=True)
    
    print(f"\n📝 Next steps:")
    print(f"1. Add your test images to: {images_dir}/")
    print(f"   Required images: {', '.join(set([q['image'] for q in questions_data]))}")
    print(f"2. Update the experiment config to use this dataset:")
    print(f"   path_question_dir: {questions_dir}")
    print(f"   path_image_dir: {images_dir}")
    
    return output_file


def create_sample_config(dataset_dir="D_datasets/SAMPLE", output_file="A_exps/sample_viz.yml"):
    """
    Create a sample experiment configuration file.
    
    Args:
        dataset_dir: Path to the sample dataset
        output_file: Where to save the config file
    """
    config = f"""name_exp: sample_visualization
name_daset: SAMPLE
name_category: sample

# Paths - Update path_model and path_image_dir with your actual paths
path_image_dir: {dataset_dir}/Images
path_question_dir: {dataset_dir}/Questions
path_model: liuhaotian/llava-v1.5-7b  # Or path to local model

conv_mode: vicuna_v1

# Visual Attention Sink Logic Parameters
logic: 1
tau: 20
rho: 0.5
p: 0.6
summ: 0.2

# Generation Parameters
max_new_tokens: 128
except_last_layer: 1

# Visualization Parameters
capture_attention: true
max_viz_questions: 5  # Process all sample questions
"""
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(config)
    
    print(f"\n✓ Created sample config at {output_file}")
    print(f"\n🚀 To run the visualization experiment:")
    print(f"   python src/inference_with_visualization.py --device 0 --exp_config {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Create sample dataset for visual attention sink experiments"
    )
    parser.add_argument(
        "--num_questions",
        type=int,
        default=5,
        help="Number of sample questions to generate (default: 5)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="D_datasets/SAMPLE",
        help="Directory to save the sample dataset (default: D_datasets/SAMPLE)"
    )
    parser.add_argument(
        "--create_config",
        action="store_true",
        help="Also create a sample experiment config file"
    )
    
    args = parser.parse_args()
    
    print("Creating sample dataset for Visual Attention Sink experiments...\n")
    
    # Create sample questions
    questions_file = create_sample_questions(args.num_questions, args.output_dir)
    
    # Optionally create config
    if args.create_config:
        create_sample_config(args.output_dir, "A_exps/sample_viz.yml")
    
    print("\n✅ Sample dataset preparation complete!")


if __name__ == "__main__":
    main()
