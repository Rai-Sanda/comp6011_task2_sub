import pandas as pd
import torch
from transformers import pipeline
import os
import argparse
import json

def zero_shot_prompt_enhanced(csv_path, output_dir="./results/zero_shot_enhanced"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. プロンプト設計の更なる改良: 極めてシンプルにし、ラベル内に例を含める
    # NLIモデル（BART）は、複雑な指示よりも「何についてのテキストか」という単純なマッチングに強いです。
    label_definitions = {
        "Attempt": "suicide attempt (e.g., 'I just took the pills')",
        "Behavior": "suicidal behavior or preparation (e.g., 'I bought a rope', 'I wrote a note')",
        "Ideation": "suicidal thoughts or wishing to die (e.g., 'I want to end it', 'I wish I was dead')",
        "Indicator": "emotional distress or hopelessness (e.g., 'I am a failure', 'I feel so empty')",
        "Safe": "safe and normal conversation (e.g., 'I like gardening', 'I am planning a trip')"
    }
    
    candidate_labels = list(label_definitions.values())
    # ラベル定義から元のラベル名への逆引きマップ
    reverse_map = {v: k for k, v in label_definitions.items()}

    print("Initializing Zero-Shot Classification pipeline (NLI-based)...")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0 if torch.cuda.is_available() else -1)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} cases from {csv_path}")

    results = []
    
    for idx, row in df.iterrows():
        text = row['Paraphrased Dialogue']
        true_label = row['Risk Level']
        
        # 2. 推論の実行
        # テンプレートを極めてシンプルに
        output = classifier(
            text, 
            candidate_labels, 
            hypothesis_template="This text is about {}."
        )
        
        # 最もスコアの高い定義文を取得し、元のラベル名に戻す
        top_definition = output['labels'][0]
        pred_label = reverse_map[top_definition]
        confidence = output['scores'][0]
        
        results.append({
            "Case ID": int(row["Case ID"]),
            "true_label": true_label,
            "predicted_label": pred_label,
            "confidence": f"{confidence:.4f}",
            "matching_definition": top_definition,
            "text_snippet": text[:100] + "..."
        })
        print(f"Case {row['Case ID']}: True={true_label}, Pred={pred_label} (Conf: {confidence:.2f})")

    # 結果の保存
    output_file = os.path.join(output_dir, "prompt_enhanced_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    # 精度の計算
    correct = 0
    for r in results:
        if r["true_label"].lower() == r["predicted_label"].lower():
            correct += 1
            
    accuracy = correct / len(results)
    summary = {
        "total_cases": len(results),
        "correct_predictions": correct,
        "accuracy": accuracy
    }
    
    with open(os.path.join(output_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"\nEnhanced Zero-shot Accuracy: {accuracy:.2f}")
    print(f"Detailed results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="data/raw/student_assignment_10_cases.csv")
    args = parser.parse_args()
    
    zero_shot_prompt_enhanced(args.csv_path)
