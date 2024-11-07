import pandas as pd
from sympy import cos, sin, pi

# Load data
df = pd.read_csv('/zhome/e1/d/168534/Desktop/Bachelor_PINN/PINN_Bachelor/Data/U0pt1_Laminar.csv')

theta = 1.323541349
# x = x*cos(theta) - y*sin(theta)
# y = x*sin(theta) + y*cos(theta)

df['X_rot (m)'] = df.apply(lambda row: row['X (m)'] * cos(theta) - row['Y (m)'] * sin(theta), axis=1)

df.to_csv('U0pt1_Laminar_rot.csv', index = False)