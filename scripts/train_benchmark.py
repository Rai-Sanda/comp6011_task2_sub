import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import os
import argparse

def compute_metrics(pred):
    labels = pred.label_ids
    # Handling cases where predictions are returned as a tuple (logits, hidden_states, etc.)
    if isinstance(pred.predictions, tuple):
        preds = pred.predictions[0].argmax(-1)
    else:
        preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

import json
import ast
from codecarbon import EmissionsTracker

def train_model(model_name, dataset_path, output_dir, max_samples=None, text_col='text', label_col='condition'):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Carbon tracking
    tracker = EmissionsTracker(output_dir=output_dir, project_name=f"{model_name}_{os.path.basename(dataset_path)}")
    tracker.start()
    
    # Load dataset
    df = pd.read_csv(dataset_path)
    
    # Drop rows with missing text or labels
    df = df.dropna(subset=[text_col, label_col])
    
    if max_samples:
        df = df.sample(n=min(max_samples, len(df)), random_state=42)
    
    # Preprocess text if it looks like a list
    def preprocess_text(text):
        if isinstance(text, str) and text.startswith('[') and text.endswith(']'):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, list):
                    return " ".join(parsed)
            except:
                pass
        return text

    df[text_col] = df[text_col].apply(preprocess_text)
    
    # Encode labels
    le = LabelEncoder()
    df['label'] = le.fit_transform(df[label_col])
    num_labels = len(le.classes_)
    
    # Split dataset
    stratify_labels = df['label'].tolist() if df['label'].value_counts().min() > 1 else None
    
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df[text_col].tolist(), df['label'].tolist(), test_size=0.2, random_state=42, stratify=stratify_labels
    )
    
    # Recalculate stratify for val/test split
    val_stratify = test_labels if pd.Series(test_labels).value_counts().min() > 1 else None
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        test_texts, test_labels, test_size=0.5, random_state=42, stratify=val_stratify
    )
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Handle padding token for Decoder-only models (like GPT-2)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set padding side for Decoder-only models
    if any(m in model_name.lower() for m in ["gpt2", "llama", "mistral"]):
        tokenizer.padding_side = "left"
        
    def tokenize_function(texts):
        # We don't pad here, but use DataCollatorWithPadding for efficiency
        return tokenizer(texts, truncation=True, max_length=512)
    
    class MentalDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    print(f"Tokenizing dataset... (Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)})")
    train_encodings = tokenize_function(train_texts)
    val_encodings = tokenize_function(val_texts)
    test_encodings = tokenize_function(test_texts)
    
    train_dataset = MentalDataset(train_encodings, train_labels)
    val_dataset = MentalDataset(val_encodings, val_labels)
    test_dataset = MentalDataset(test_encodings, test_labels)
    
    # Model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    
    # Fix for GPT-2 padding
    if model.config.model_type == "gpt2":
        model.config.pad_token_id = model.config.eos_token_id

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(),
        report_to="none" # Avoid wandb etc. if not configured
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )
    
    trainer.train()
    
    # Save the model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Final evaluation on test set
    results = trainer.evaluate(test_dataset)
    # Convert numpy types to native python types for JSON serialization
    results = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in results.items()}
    results = {k: int(v) if isinstance(v, (np.int32, np.int64)) else v for k, v in results.items()}
    print(f"Test Results for {model_name}: {results}")
    
    # Save results to JSON
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f)
    
    # Qualitative analysis: Save some examples
    num_examples = 5
    examples = []
    # Get predictions for the first few test samples
    test_subset_texts = test_texts[:num_examples]
    test_subset_labels = test_labels[:num_examples]
    
    predictions = trainer.predict(test_dataset)
    if isinstance(predictions.predictions, tuple):
        preds_indices = predictions.predictions[0][:num_examples].argmax(-1)
    else:
        preds_indices = predictions.predictions[:num_examples].argmax(-1)
    
    for i in range(len(test_subset_texts)):
        true_label = le.inverse_transform([test_subset_labels[i]])[0]
        pred_label = le.inverse_transform([preds_indices[i]])[0]
        
        # Convert numpy types to native python types
        if hasattr(true_label, 'item'): true_label = true_label.item()
        if hasattr(pred_label, 'item'): pred_label = pred_label.item()
        
        examples.append({
            "text": test_subset_texts[i],
            "true_label": true_label,
            "predicted_label": pred_label
        })
    
    with open(os.path.join(output_dir, 'qualitative_examples.json'), 'w') as f:
        json.dump(examples, f, indent=4)
        
    # Save label encoder classes
    np.save(os.path.join(output_dir, 'classes.npy'), le.classes_)

    # Stop carbon tracking
    emissions = tracker.stop()
    print(f"Estimated CO2 emissions: {emissions} kg")
    # Save emissions info
    with open(os.path.join(output_dir, 'emissions.json'), 'w') as f:
        json.dump({"emissions_kg": emissions}, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="data/raw/mental_health_balanced.csv")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--label_col", type=str, default="condition")
    args = parser.parse_args()
    
    train_model(args.model_name, args.dataset_path, args.output_dir, args.max_samples, args.text_col, args.label_col)
