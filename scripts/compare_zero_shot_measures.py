import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import os
import argparse
import json

# 代表的な事例（Few-shot用）
FEW_SHOT_EXAMPLES = {
    "Attempt": "I tried to kill my self once and failed badly. I am now in a hospital and I feel so empty.",
    "Behavior": "Last night I was sitting on the ledge of the bridge. I have the rope ready in my room.",
    "Ideation": "I have no future, no present. I just want to disappear. Everything feels like a waste of energy.",
    "Indicator": "Everything is too much. I keep making dark jokes and I feel like I am failing at everything.",
    "Supportive": "The weather is lovely today. I've been working on my garden and it feels satisfying."
}

def predict_mode(model, tokenizer, text, mode, classes):
    # 'No Measures' mode: Simple inference
    # 'With Measures' mode: Append label definitions or examples (conceptual few-shot)
    input_text = text
    if mode == "with_measures":
        # クラスごとの定義と例をヒントとして与える
        context = "Class Definitions and Examples:\n"
        for label, example in FEW_SHOT_EXAMPLES.items():
            context += f"- {label}: {example[:100]}...\n"
        input_text = context + "\nTarget Text to Classify:\n" + text

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=-1).item()
        
    return classes[pred_idx], probs[0][pred_idx].item()

def run_comparison(models_dir="./results/results_archive/reddit_500", csv_path="data/raw/student_assignment_10_cases.csv"):
    df = pd.read_csv(csv_path)
    model_names = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    
    all_results = {}

    for m_name in model_names:
        model_path = os.path.join(models_dir, m_name)
        classes_path = os.path.join(model_path, "classes.npy")
        if not os.path.exists(classes_path): continue
        
        print(f"\nEvaluating Model: {m_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        classes = np.load(classes_path, allow_pickle=True)
        
        model_results = {"no_measures": [], "with_measures": []}
        
        for mode in ["no_measures", "with_measures"]:
            correct = 0
            for idx, row in df.iterrows():
                pred, conf = predict_mode(model, tokenizer, row['Paraphrased Dialogue'], mode, classes)
                
                # Label normalization for accuracy check
                t = str(row['Risk Level']).lower()
                p = str(pred).lower()
                if (t == "safe" and p == "supportive") or (t == p):
                    correct += 1
                
                model_results[mode].append(pred)
            
            acc = correct / len(df)
            model_results[f"{mode}_accuracy"] = acc
            print(f"  Mode: {mode} -> Accuracy: {acc:.2f}")
            
        all_results[m_name] = model_results

    # 保存とサマリー出力
    summary_data = []
    for m_name, res in all_results.items():
        summary_data.append({
            "Model": m_name,
            "Baseline Acc": res["no_measures_accuracy"],
            "Few-shot Acc": res["with_measures_accuracy"],
            "Improvement": res["with_measures_accuracy"] - res["no_measures_accuracy"]
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\n### Comparison Summary (Baseline vs Few-shot Measures)")
    print(summary_df.to_markdown(index=False))
    
    with open("results/zero_shot_comparison_results.json", "w") as f:
        json.dump(all_results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", type=str, default="./results/results_archive/reddit_500", help="Directory containing trained models")
    parser.add_argument("--csv_path", type=str, default="data/raw/student_assignment_10_cases.csv", help="Path to the student cases CSV")
    args = parser.parse_args()
    
    run_comparison(models_dir=args.models_dir, csv_path=args.csv_path)
