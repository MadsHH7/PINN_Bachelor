import pandas as pd
from sympy import cos, sin, pi
import numpy as np

def rotate(df, keys, rotation_matrix):
    if rotation_matrix is not None:
        df[keys] = np.asarray([np.dot(rotation_matrix, entry) for _, entry in df[keys].iterrows()])
    return df

def translate(df, keys, translation):
    if translation is not None:
        df[keys] = df[keys] - translation
    return df

key = "1"
pct_data = 0.001

df = pd.read_csv(f'U0pt{key}_Laminar.csv')

keys_pts = ["X (m)", "Y (m)", "Z (m)"]
keys_vel = ["Velocity[i] (m/s)", "Velocity[j] (m/s)", "Velocity[k] (m/s)"]


theta = 1.323541349
angle = (pi / 2) + theta

rot_matrix = (
        [float(cos(angle)), float(-sin(angle)), 0],
        [float(sin(angle)), float(cos(angle)), 0],
        [0, 0, 1]
    )

translation= ([
        -0.05,
        -0.2,
        0
    ])

df = rotate(df, keys_pts, rot_matrix)
df = rotate(df, keys_vel, rot_matrix)
df = translate(df, keys_pts, translation)

df = df.sample(frac = pct_data, random_state=42)

df.to_csv(f'U0pt{key}_Laminar_rot.csv', index=False)