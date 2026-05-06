import os
import subprocess
import argparse

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
    "t5-base",
]

def run_benchmark(quick=False):
    max_samples = 100 if quick else None
    
    print(f"Starting benchmark... (Mode: {'Quick' if quick else 'Full'})")
    
    for model in MODELS:
        print(f"\n" + "="*50)
        print(f"Processing model: {model}")
        print("="*50)
        
        output_dir = f"./results/benchmarks/{model.replace('/', '_')}"
        os.makedirs(output_dir, exist_ok=True)
        
        cmd = [
            "python", "scripts/train_benchmark.py",
            "--model_name", model,
            "--output_dir", output_dir
        ]
        
        if max_samples:
            cmd.extend(["--max_samples", str(max_samples)])
        
        try:
            # Using run instead of Popen for simplicity and waiting
            subprocess.run(cmd, check=True)
        except Exception as e:
            print(f"Error benchmarking {model}: {e}")
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run with only 100 samples for testing.")
    args = parser.parse_args()
    
    run_benchmark(quick=args.quick)
