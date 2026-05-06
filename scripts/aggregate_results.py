import os
import json
import pandas as pd

def aggregate_results(results_dir="./results/results_archive"):
    all_results = []
    
    if not os.path.exists(results_dir):
        print(f"Results directory {results_dir} not found.")
        return

    for dataset_name in os.listdir(results_dir):
        dataset_path = os.path.join(results_dir, dataset_name)
        if not os.path.isdir(dataset_path):
            continue
            
        for model_name in os.listdir(dataset_path):
            model_path = os.path.join(dataset_path, model_name)
            if not os.path.isdir(model_path):
                continue
                
            results_file = os.path.join(model_path, "results.json")
            if os.path.exists(results_file):
                with open(results_file, "r") as f:
                    try:
                        data = json.load(f)
                        # Add metadata
                        row = {
                            "Dataset": dataset_name,
                            "Model": model_name.replace("_", "/"),
                            "Accuracy": data.get("eval_accuracy", data.get("accuracy")),
                            "F1-Macro": data.get("eval_f1", data.get("f1")),
                            "Precision": data.get("eval_precision", data.get("precision")),
                            "Recall": data.get("eval_recall", data.get("recall")),
                            "Epochs": data.get("epoch")
                        }
                        
                        # Add emissions if available
                        emissions_file = os.path.join(model_path, "emissions.json")
                        if os.path.exists(emissions_file):
                            with open(emissions_file, "r") as ef:
                                emissions_data = json.load(ef)
                                row["CO2_kg"] = emissions_data.get("emissions_kg")
                        
                        all_results.append(row)
                    except Exception as e:
                        print(f"Error reading {results_file}: {e}")

    if all_results:
        df = pd.DataFrame(all_results)
        # Sort by Dataset and F1 Score
        df = df.sort_values(["Dataset", "F1-Macro"], ascending=[True, False])
        
        # Save to CSV
        output_csv = "results/benchmarks/benchmark_summary.csv"
        df.to_csv(output_csv, index=False)
        print(f"Summary saved to {output_csv}")
        
        # Print Markdown Table
        print("\n### Benchmark Summary\n")
        print(df.to_markdown(index=False))
    else:
        print("No results found to aggregate.")

if __name__ == "__main__":
    aggregate_results()
