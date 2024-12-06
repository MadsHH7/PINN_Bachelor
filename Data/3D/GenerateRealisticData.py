import pandas as pd
from sympy import cos, sin, pi
import numpy as np
from pipe_bend_parameterized_geometry import PipeBend

def rotate(df, keys, rotation_matrix):
    if rotation_matrix is not None:
        df[keys] = np.asarray([np.dot(rotation_matrix, entry) for _, entry in df[keys].iterrows()])
    return df

def translate(df, keys, translation):
    if translation is not None:
        df[keys] = df[keys] - translation
    return df

key = "1"
pct_data = 5 / 100

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
        -0.,
        0
    ])

df = rotate(df, keys_pts, rot_matrix)
df = rotate(df, keys_vel, rot_matrix)
df = translate(df, keys_pts, translation)

bend_angle = (1.323541349,1.323541349) # I did not add any dimension to the bend
radius_pipe = (0.1,0.1)
radius_bend = (0.2,0.2)
inlet_pipe_length = (0.2,0.2)
outlet_pipe_length = (1,1)

radius = radius_pipe[0]


Pipe = PipeBend(bend_angle_range= bend_angle,
                radius_pipe_range=radius_pipe,
                radius_bend_range=radius_bend,
                inlet_pipe_length_range=inlet_pipe_length,
                outlet_pipe_length_range=outlet_pipe_length)

# bend_inlet_c = Pipe.bend_planes_centers[0]
bend_inlet_c = (radius_bend[0]*cos(0.25*bend_angle[0]), radius_bend[0]*sin(0.25*bend_angle[0]))
# bend_outlet_c = Pipe.bend_planes_centers[-1]
bend_outlet_c = (radius_bend[0]*cos(1.0*bend_angle[0]), radius_bend[0]*sin(1.0*bend_angle[0]))
print(bend_inlet_c)
print(bend_outlet_c)
bend_inlet_index = []
bend_outlet_index = []

print("Geometry was created")
for index, row in df.iterrows():
    if ((row[keys_pts[0]] - bend_inlet_c[0])**2 + (row[keys_pts[1]] - bend_inlet_c[1])**2 + row[keys_pts[2]]**2 <= radius**2):
        bend_inlet_index.append(index)

print("Inlet index was calulated")

for index, row in df.iterrows():
    if ((row[keys_pts[0]] - bend_outlet_c[0])**2 + (row[keys_pts[1]] - bend_outlet_c[1])**2 + row[keys_pts[2]]**2 <= radius**2):
        bend_outlet_index.append(index)

print("Outlet index was calculated")

bend_inlet = df.iloc[bend_inlet_index]
bend_outlet = df.iloc[bend_outlet_index]



# train_df = df.sample(frac = pct_data, random_state=42)

validation_df = df.drop(bend_inlet.index)
validation_df = validation_df.drop(bend_outlet.index)

bend_inlet.to_csv(f'U0pt{key}_Bend_Inlet.csv', index=False)
bend_outlet.to_csv(f'U0pt{key}_Bend_Outlet.csv', index=False)
validation_df.to_csv(f'U0pt{key}_Laminar_validation_BEND.csv', index=False)

# /zhome/e3/5/167986/Desktop/PINN/bin/python /zhome/e3/5/167986/Desktop/PINN_Bachelor/Data/GenerateRealisticData.py