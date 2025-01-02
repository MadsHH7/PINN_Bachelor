import pandas as pd
from sympy import cos, sin, pi
import numpy as np

key = "001"
pct_data = 5 / 100 # Split into 5% Data

df = pd.read_csv(f'bend_data_mvel{key}.csv')

train_df = df.sample(frac = pct_data, random_state=42)
validation_df = df.drop(train_df.index)

train_df.to_csv(f'bend_data_mvel{key}_train.csv', index=False)
validation_df.to_csv(f'bend_data_mvel{key}_validation.csv', index=False)