import pandas as pd
import torch
import numpy as np
import os
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from sklearn.preprocessing import LabelEncoder

def train_mapped_model():
    # Load mapped data
    df = pd.read_csv('data/processed/mental_health_mapped.csv')
    
    # Encode labels
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['mapped_condition'])
    np.save('results/results_archive/mental_health_mapped/classes.npy', le.classes_)
    
    # Split
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
    )
    
    # Model
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(le.classes_))
    
    # Datasets
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, texts, labels, tokenizer):
            self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
            self.labels = labels
        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item
        def __len__(self):
            return len(self.labels)
            
    train_dataset = Dataset(train_texts, train_labels, tokenizer)
    test_dataset = Dataset(test_texts, test_labels, tokenizer)
    
    training_args = TrainingArguments(
        output_dir='./results/results_archive/mental_health_mapped/checkpoints',
        num_train_epochs=1,
        per_device_train_batch_size=8,
        logging_steps=100,
        save_strategy="no"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )
    
    trainer.train()
    
    # Save model
    model.save_pretrained('./results/results_archive/mental_health_mapped')
    tokenizer.save_pretrained('./results/results_archive/mental_health_mapped')
    print("Model saved to ./results/results_archive/mental_health_mapped")

if __name__ == "__main__":
    os.makedirs('./results/results_archive/mental_health_mapped', exist_ok=True)
    train_mapped_model()
