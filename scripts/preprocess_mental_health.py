import pandas as pd
import ast

def map_condition(condition):
    if condition == 'Suicidal':
        # Simple heuristic mapping for demonstration; could be refined.
        # This is a placeholder; one might need manual inspection.
        return 'Ideation' 
    elif condition == 'Normal':
        return 'Supportive'
    else:
        return 'Indicator'

def preprocess_and_map():
    df = pd.read_csv('data/raw/mental_health_balanced.csv')
    df['mapped_condition'] = df['condition'].apply(map_condition)
    
    print("Class mapping distribution:")
    print(df['mapped_condition'].value_counts())
    
    # Save the mapped dataset
    df[['text', 'mapped_condition']].to_csv('data/processed/mental_health_mapped.csv', index=False)
    print("Mapped dataset saved to data/processed/mental_health_mapped.csv")

if __name__ == "__main__":
    preprocess_and_map()
