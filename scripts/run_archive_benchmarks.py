import os
import subprocess
import argparse
import sys

# Predefined models based on models.txt and run_all_benchmarks.py
MODELS = [
    # Encoder-based
    "bert-base-uncased",
    "mental/mental-bert-base-uncased",
    "mental/mental-roberta-base",
    # "Akashpaul123/bert-suicide-detection",
    
    # Decoder-based
    "gpt2",
    # "OpenMentalHealth/Mental-GPT2",
    
    # Encoder-Decoder-based
    "facebook/bart-base",
    # "Tianlin668/MentalBART",
    # "t5-base",
]

DATASETS = [
    {
        "name": "mental_health_balanced",
        "path": "data/raw/mental_health_balanced.csv",
        "text_col": "text",
        "label_col": "condition"
    },
    {
        "name": "dreaddit",
        "path": "data/raw/dreaddit_StressAnalysis.csv",
        "text_col": "text",
        "label_col": "label"
    },
    {
        "name": "reddit_500",
        "path": "data/raw/500_anonymized_Reddit_users_posts_labels - 500_anonymized_Reddit_users_posts_labels.csv",
        "text_col": "Post",
        "label_col": "Label"
    }
]

def run_benchmarks(quick=False):
    max_samples = 100 if quick else None
    python_exe = sys.executable
    
    print(f"Starting Archive Benchmarks... (Mode: {'Quick' if quick else 'Full'})")
    
    for dataset in DATASETS:
        print(f"\n" + "#"*60)
        print(f"Dataset: {dataset['name']}")
        print(f"#"*60)
        
        for model in MODELS:
            print(f"\n" + "="*50)
            print(f"Processing model: {model} on {dataset['name']}")
            print("="*50)
            
            output_dir = f"./results/results_archive/{dataset['name']}/{model.replace('/', '_')}"
            os.makedirs(output_dir, exist_ok=True)
            
            # Skip if already done
            if os.path.exists(os.path.join(output_dir, "results.json")):
                print(f"Results for {model} already exist. Skipping...")
                continue

            cmd = [
                python_exe, "scripts/train_benchmark.py",
                "--model_name", model,
                "--dataset_path", dataset['path'],
                "--output_dir", output_dir,
                "--text_col", dataset['text_col'],
                "--label_col", dataset['label_col']
            ]
            
            if max_samples:
                cmd.extend(["--max_samples", str(max_samples)])
            
            try:
                subprocess.run(cmd, check=True)
            except Exception as e:
                print(f"Error benchmarking {model} on {dataset['name']}: {e}")
                continue

    # Final aggregation
    print(f"\n" + "#"*60)
    print("Benchmark complete. Aggregating results...")
    subprocess.run([python_exe, "scripts/aggregate_results.py"])

    # Zero-shot comparison tasks on student cases
    print(f"\n" + "#"*60)
    print("Step 1: Baseline Zero-shot Prediction (Single Model)")
    print(f"#"*60)
    
    # We use mental-bert-base-uncased trained on reddit_500 as it has matching labels
    target_model = "mental/mental-bert-base-uncased".replace('/', '_')
    model_path = f"./results/results_archive/reddit_500/{target_model}"
    
    if os.path.exists(model_path):
        subprocess.run([
            python_exe, "scripts/zero_shot_student_cases.py",
            "--model_path", model_path,
            "--csv_path", "data/raw/student_assignment_10_cases.csv"
        ])
    else:
        print(f"Warning: Target model for zero-shot task ({model_path}) not found. "
              "Please ensure mental/mental-bert-base-uncased on reddit_500 is in the benchmark list.")

    print(f"\n" + "#"*60)
    print("Step 2: Comparison of Zero-shot vs Few-shot (All reddit_500 models)")
    print(f"#"*60)
    
    if os.path.exists("./results/results_archive/reddit_500"):
        subprocess.run([
            python_exe, "scripts/compare_zero_shot_measures.py",
            "--models_dir", "./results/results_archive/reddit_500",
            "--csv_path", "data/raw/student_assignment_10_cases.csv"
        ])
    else:
        print("Warning: results/results_archive/reddit_500 not found. Skipping Step 2.")

    print(f"\n" + "#"*60)
    print("Step 3: Enhanced Zero-shot with BART-NLI")
    print(f"#"*60)
    subprocess.run([
        python_exe, "scripts/zero_shot_prompt_enhanced.py",
        "--csv_path", "data/raw/student_assignment_10_cases.csv"
    ])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run with only 100 samples for testing.")
    args = parser.parse_args()
    
    run_benchmarks(quick=args.quick)
