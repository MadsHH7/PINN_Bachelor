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

df = pd.read_csv(f'U0pt{key}_Laminar_Rotated.csv')

keys_pts = ["X (m)", "Y (m)", "Z (m)"]
keys_vel = ["Velocity[i] (m/s)", "Velocity[j] (m/s)", "Velocity[k] (m/s)"]


bend_angle = (1.323541349,1.323541349) # I did not add any dimension to the bend
radius_pipe = (0.1,0.1)
radius_bend = (0.2,0.2)
inlet_pipe_length = (0.1,0.1)
outlet_pipe_length = (1,1)

radius = radius_pipe[0]


Pipe = PipeBend(bend_angle_range= bend_angle,
                radius_pipe_range=radius_pipe,
                radius_bend_range=radius_bend,
                inlet_pipe_length_range=inlet_pipe_length,
                outlet_pipe_length_range=outlet_pipe_length)

bend_inlet_index = []
bend_outlet_index = []
inlet_mid_index = []
outlet_mid_index = []

# bend_planes_index = []

n_inlet = (0,1,0)
n_outlet = (outlet_pipe_length[0] * cos(bend_angle[0] + pi / 2), outlet_pipe_length[0] * sin(bend_angle[0] + pi / 2),0)
# print("length of n_outlet: ", (n_outlet[0]**2+n_outlet[1]**2+n_outlet[2]**2)**0.5)
print(n_outlet)
print("Geometry was created")
# It seems like there are no points on the plane, which means that planes are probably not viabel.


# The center of the inlet and outlet
inlet_c = (radius_bend[0], -inlet_pipe_length[0],0)
outlet_c =  (-outlet_pipe_length[0]*sin(bend_angle[0]) + radius_bend[0]*cos(bend_angle[0]), outlet_pipe_length[0]*cos(bend_angle[0]) + radius_bend[0]*sin(bend_angle[0]),0)

# We use the inlet and outlet centers and pipe directions to calculate the location of the bend inlet and outlet.
b_i_c = (inlet_c[0]+inlet_pipe_length[0]*n_inlet[0],
         inlet_c[1]+inlet_pipe_length[0]*n_inlet[1],
          inlet_c[2]+inlet_pipe_length[0]*n_inlet[2])
b_o_c = (outlet_c[0]-outlet_pipe_length[0]*n_outlet[0],
         outlet_c[1]-outlet_pipe_length[0]*n_outlet[1],
         outlet_c[2]-outlet_pipe_length[0]*n_outlet[2])

m_i_c = (inlet_c[0]+0.5*inlet_pipe_length[0]*n_inlet[0],
         inlet_c[1]+0.5*inlet_pipe_length[0]*n_inlet[1],
          inlet_c[2]+0.5*inlet_pipe_length[0]*n_inlet[2])

m_o_c = (outlet_c[0]-0.5*outlet_pipe_length[0]*n_outlet[0],
         outlet_c[1]-0.5*outlet_pipe_length[0]*n_outlet[1],
         outlet_c[2]-0.5*outlet_pipe_length[0]*n_outlet[2])

print("Bend Inlet Center: ", b_i_c)
print("Bend Outlet Cetner: ", b_o_c)
count = 0
Pipe.inlet
# We iterate through all elements elements in the validation data, and check whether any are between the planes.
for index, row in df.iterrows(): # The middle of the inlet pipe


    # The bend inlet is defined to be at y=0 so this is easy to check.
    eq1 = row[keys_pts[1]] <= -0.05 + 0.005
    eq2 =row[keys_pts[1]] >= -0.05 -0.005
    
    if eq1 and eq2: # note n[0]=n[2]= 0 so we have only 1 component.

        inlet_mid_index.append(index)


for index, row in df.iterrows(): # The bend inlet


    # The bend inlet is defined to be at y=0 so this is easy to check.
    eq1 = row[keys_pts[1]] <= 0.005
    eq2 =row[keys_pts[1]] >= -0.005
    
    if eq1 and eq2: # note n[0]=n[2]= 0 so we have only 1 component.

        bend_inlet_index.append(index)

for index, row in df.iterrows(): # The outlet of the bend


    # We use the equation for 2 planes each shifted slightly from the true bend outlet
    eq1 = (n_outlet[0]* (row[keys_pts[0]]- b_o_c[0])
        + n_outlet[1] * (row[keys_pts[1]]-b_o_c[1])
        + n_outlet[2] * (row[keys_pts[2]] - b_o_c[2])) <= 0.005 # This value is choosen through trial and error.
    
    eq2 = (n_outlet[0]* (row[keys_pts[0]]- b_o_c[0])
        + n_outlet[1] * (row[keys_pts[1]]-b_o_c[1])
        + n_outlet[2] * (row[keys_pts[2]]-b_o_c[2])) >= -0.005
    
    if eq1 and eq2:
        bend_outlet_index.append(index)


for index, row in df.iterrows(): # The middle of the outlet pipe.


    # We use the equation for 2 planes each shifted slightly from the true bend outlet
    eq1 = (n_outlet[0]* (row[keys_pts[0]]- m_o_c[0])
        + n_outlet[1] * (row[keys_pts[1]]-m_o_c[1])
        + n_outlet[2] * (row[keys_pts[2]] - m_o_c[2])) <= 0.005 # This value is choosen through trial and error.
    
    eq2 = (n_outlet[0]* (row[keys_pts[0]]- m_o_c[0])
        + n_outlet[1] * (row[keys_pts[1]]-m_o_c[1])
        + n_outlet[2] * (row[keys_pts[2]]-m_o_c[2])) >= -0.005
    
    if eq1 and eq2:
        outlet_mid_index.append(index)


print("Index was calculated")
print("Points in inlet and outlet planes: ", len(bend_inlet_index)+len(bend_outlet_index))


print("Points in bend inlet plane: ", len(bend_inlet_index))
print("Points in bend outlet plane: ", len(bend_outlet_index))
print("Points in mid inlet plane: ", len(inlet_mid_index))
print("Points in mid outlet plane: ", len(outlet_mid_index))

length = df.shape[0]
inlet_velocity = 0.1
in_vel = [inlet_velocity] * length

# df2 = df.assign(inlet_vel = in_vel)
# print(df2.head())





sample_points = 125

bend_inlet = df.iloc[bend_inlet_index]
bend_outlet = df.iloc[bend_outlet_index]
inlet = df.iloc[inlet_mid_index]
outlet = df.iloc[outlet_mid_index]

bend_inlet = bend_inlet.sample(n=sample_points,random_state=42)
bend_outlet = bend_outlet.sample(n=sample_points,random_state=42)
inlet = inlet.sample(n=sample_points,random_state=42)
outlet = outlet.sample(n=sample_points,random_state=42)

sensor_data = pd.concat([bend_inlet,bend_outlet,inlet,outlet])


validation_df = df.drop(sensor_data.index)



sensor_data.to_csv(f'U0pt{key}_NotUni_0.5K_Train.csv', index=False)
validation_df.to_csv(f'U0pt{key}_NotUni_0.5K_Validation.csv', index=False)

# /zhome/e3/5/167986/Desktop/PINN/bin/python /zhome/e3/5/167986/Desktop/PINN_Bachelor/Data/3D/GenerateRealisticData.py