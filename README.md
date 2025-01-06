# Simulating flow states in curved pipes using physics-informed neural networks

This GitHub contains all the code used to generate the results of our thesis. It is sectioned into data and the scripts used to simulate our results. 

In the data folder you can find the raw data and the scripts used to sample our data. In the Data/Laplace_2D, we included a folder of MATLAB scripts created by Allan P. Engsig-Karup, Associate Professor, DTU Compute, which generate the validation data used for our 2D Laplace problem.
The data in the Navier_Stokes folders contain data supplied by FORCE Technology. The data specify the inlet velocity of the dataset. 001 = 0.01 and U0pt1 = 0.1

The training_scripts contain all the scripts we ran in order to generate the final results, in the Navier_Stokes folders we have included a helper function "PINN_Helper", which was created and shared by Andreas Hallbäck, FORCE Technology.
Likewise is pipe_bend_parameterized_geometry is used to create the geometry of our 3D pipe created and shared by Andreas Hallbäck, FORCE Technology.
