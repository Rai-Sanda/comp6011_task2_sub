import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_benchmarks(csv_file="results/benchmarks/benchmark_summary.csv"):
    if not os.path.exists(csv_file):
        print(f"File {csv_file} not found. Please run the benchmarks first.")
        return

    # Load results
    df = pd.read_csv(csv_file)
    
    # Set plot style
    sns.set_theme(style="whitegrid")
    
    # Create a plot for each dataset
    datasets = df['Dataset'].unique()
    
    for dataset in datasets:
        plt.figure(figsize=(12, 6))
        data = df[df['Dataset'] == dataset].sort_values("F1-Macro", ascending=False)
        
        ax = sns.barplot(x="F1-Macro", y="Model", data=data, palette="viridis")
        
        plt.title(f"Model Performance on {dataset} (F1-Macro Score)", fontsize=15)
        plt.xlabel("F1-Macro Score", fontsize=12)
        plt.ylabel("Model Name", fontsize=12)
        plt.xlim(0, 1.0) # F1 score is between 0 and 1
        
        # Add labels on bars
        for p in ax.patches:
            width = p.get_width()
            plt.text(width + 0.01, p.get_y() + p.get_height()/2, f'{width:.3f}', ha="left", va="center")
            
        plt.tight_layout()
        output_plot = f"results/figures/results_plot_{dataset}.png"
        plt.savefig(output_plot)
        print(f"Plot saved to {output_plot}")

if __name__ == "__main__":
    plot_benchmarks()
