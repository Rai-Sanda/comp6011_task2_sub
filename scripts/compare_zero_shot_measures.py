import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import os
import argparse
import json

# 臨床的な定義と複数の具体例（Few-shot / Prompting用）
# 例を2つに増やすことで、モデルが各カテゴリのバリエーションをより良く理解できるようにします。
FEW_SHOT_EXAMPLES = {
    "Attempt": {
        "definition": "Actual suicide attempt or clear, immediate intent to take one's life.",
        "examples": [
            "I tried to end it all last night by taking a bottle of pills. I'm in the ER now and I still wish it had worked.",
            "I survived a jumping attempt last week. My body is broken but my mind is still in that same dark place. I don't know why I was saved."
        ]
    },
    "Behavior": {
        "definition": "Specific suicidal behaviors, preparations, or planning.",
        "examples": [
            "I've been looking at maps to find a high enough bridge. I've already written letters to my family explaining why.",
            "I bought a gun today. I've been practicing how to use it so I don't mess up when the time comes. It's hidden in the garage."
        ]
    },
    "Ideation": {
        "definition": "Persistent thoughts of suicide, wanting to die, or feeling that life is not worth living, but without a specific immediate plan or action.",
        "examples": [
            "I wake up every day wishing I didn't. The world would be so much better off without me. I just want to disappear forever.",
            "What is the point of living when every day is just a repeat of the same misery? I think about death constantly, it seems so peaceful."
        ]
    },
    "Indicator": {
        "definition": "Signs of deep emotional distress, hopelessness, or mental health struggles.",
        "examples": [
            "Everything is too much. I'm failing at my job, my family is disappointed, and I feel like a total failure. I can't see any way out.",
            "I haven't left my bed in three days. I haven't showered or eaten. I feel like a hollow shell of a person, just waiting for the end."
        ]
    },
    "Supportive": {
        "definition": "General conversation, positive emotions, or daily activities showing no signs of mental health risk.",
        "examples": [
            "I had a wonderful time gardening today. The roses are finally in bloom and it's so peaceful to just sit and watch them.",
            "I'm looking forward to the weekend. I'm going to the movies with some friends and then grabbing dinner at that new Italian place."
        ]
    }
}

def predict_mode(model, tokenizer, text, mode, classes):
    input_text = text
    if mode == "with_measures":
        # クラスごとの定義と複数の例を構造化して提供
        prompt = "Instruction: Classify the following 'Target Text' into exactly one of the categories below.\n\n"
        prompt += "Categories and Definitions:\n"
        for label, data in FEW_SHOT_EXAMPLES.items():
            prompt += f"- {label}: {data['definition']}\n"
            for i, example in enumerate(data['examples']):
                prompt += f"  Example {i+1}: \"{example}\"\n"
        
        prompt += "\nTarget Text: " + text + "\n\nCategory:"
        input_text = prompt

    # トークン数の上限（512）を超えないように注意
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=-1).item()
        
    return classes[pred_idx], probs[0][pred_idx].item()

def run_comparison(models_dir="./results_archive/reddit_500", csv_path="source_code/student_assignment_10_cases.csv"):
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if not os.path.exists(models_dir):
        print(f"Error: Models directory not found at {models_dir}")
        return
        
    model_names = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    
    all_results = {}

    for m_name in model_names:
        model_path = os.path.join(models_dir, m_name)
        classes_path = os.path.join(model_path, "classes.npy")
        if not os.path.exists(classes_path):
            print(f"Skipping {m_name}: classes.npy not found.")
            continue
        
        print(f"\nEvaluating Model: {m_name}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            classes = np.load(classes_path, allow_pickle=True)
            model.eval()
        except Exception as e:
            print(f"Error loading model {m_name}: {e}")
            continue
        
        model_results = {"no_measures": [], "with_measures": []}
        
        for mode in ["no_measures", "with_measures"]:
            correct = 0
            predictions = []
            for idx, row in df.iterrows():
                pred, conf = predict_mode(model, tokenizer, row['Paraphrased Dialogue'], mode, classes)
                predictions.append(pred)
                
                # ラベルの正規化（評価用）
                t = str(row['Risk Level']).lower().strip()
                p = str(pred).lower().strip()
                
                # safe (student) -> supportive (model) のマッピングを考慮
                if (t == "safe" and p == "supportive") or (t == p):
                    correct += 1
            
            acc = correct / len(df)
            model_results[mode] = predictions
            model_results[f"{mode}_accuracy"] = acc
            print(f"  Mode: {mode.replace('_', ' ').capitalize()} -> Accuracy: {acc:.2f}")
            
        all_results[m_name] = model_results

    # サマリーの出力
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
    try:
        print(summary_df.to_markdown(index=False))
    except:
        print(summary_df)
    
    # 詳細結果をJSONで保存
    output_file = "zero_shot_comparison_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\nDetailed results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", type=str, default="./results_archive/reddit_500", help="Directory containing trained models")
    parser.add_argument("--csv_path", type=str, default="source_code/student_assignment_10_cases.csv", help="Path to the student cases CSV")
    args = parser.parse_args()
    
    run_comparison(models_dir=args.models_dir, csv_path=args.csv_path)
