import pandas as pd
import numpy as np
import os

dataset_dir = os.path.dirname(__file__)

for file in ['train.csv', 'test.csv', 'val.csv']:
    filepath = os.path.join(dataset_dir, file)
    df = pd.read_csv(filepath)
    label_0 = df[df['label'] == 0]
    label_1 = df[df['label'] == 1]
    min_count = min(len(label_0), len(label_1))
    label_0_sampled = label_0.sample(n=min_count, random_state=42)
    balanced_df = pd.concat([label_0_sampled, label_1])
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    output_filepath = os.path.join(dataset_dir, f'balanced_{file}')
    balanced_df.to_csv(output_filepath, index=False)
    print(f'{file}: Balanced to {len(balanced_df)} samples ({min_count} each label)')