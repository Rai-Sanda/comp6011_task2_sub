import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import os
import argparse
import json

def zero_shot_predict(model_path, csv_path, text_col='Paraphrased Dialogue', label_col='Risk Level'):
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Load classes
    classes = np.load(os.path.join(model_path, 'classes.npy'), allow_pickle=True)
    print(f"Model classes: {classes}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} cases from {csv_path}")
    
    results = []
    
    model.eval()
    with torch.no_grad():
        for idx, row in df.iterrows():
            text = row[text_col]
            true_label = row[label_col]
            
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_idx = torch.argmax(probs, dim=-1).item()
            pred_label = classes[pred_idx]
            
            results.append({
                "Case ID": int(row["Case ID"]),
                "text_snippet": text[:100] + "...",
                "true_label": str(true_label),
                "predicted_label": str(pred_label),
                "confidence": float(probs[0][pred_idx].item())
            })
            
    # Save results
    output_file = os.path.join(os.path.dirname(model_path), "student_cases_predictions.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Zero-shot predictions saved to {output_file}")
    
    # Calculate simple accuracy if labels match after normalization
    correct = 0
    for r in results:
        # Normalize labels for comparison (case-insensitive, etc.)
        t = str(r["true_label"]).lower()
        p = str(r["predicted_label"]).lower()
        if t == "safe" and p == "supportive": # Specific mapping
            correct += 1
        elif t == p:
            correct += 1
            
    print(f"Zero-shot Accuracy: {correct/len(results):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model directory")
    parser.add_argument("--csv_path", type=str, default="data/raw/student_assignment_10_cases.csv")
    args = parser.parse_args()
    
    zero_shot_predict(args.model_path, args.csv_path)
