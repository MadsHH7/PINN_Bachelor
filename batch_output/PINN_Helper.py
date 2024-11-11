import os
import sys
from typing import Callable, Dict, List, Union

import numpy as np
import pandas as pd
import vtk
from modulus.sym.domain.inferencer import PointwiseInferencer
from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.geometry.primitives_2d import Channel2D, Line, Rectangle
from modulus.sym.utils.io.csv_rw import dict_to_csv
from sympy import Max, sqrt

# from pinn.shared.plane_plotter import ValidatorPlanePlotter


def sample_geometry(
    geometry,
    nr_interior_points,
    nr_boundary_points,
    data_path,
    interior_file_name,
    boundary_file_name,
    extra_input_keys,
    extra_input_values,
    parameterization,
    criteria_boundary=None,
    criteria_interior=None,

):
    interior = geometry.sample_interior(
        nr_points=nr_interior_points,
        parameterization=parameterization,
        criteria=criteria_interior,
    )
    dict_to_csv(interior, os.path.join(data_path, f"{interior_file_name}.csv"))
    interior_df=pd.read_csv(os.path.join(data_path, f"{interior_file_name}.csv"))
    interior_df[extra_input_keys] = extra_input_values

    boundary = geometry.sample_boundary(
        nr_points=nr_boundary_points,
        parameterization=parameterization,
        criteria=criteria_boundary,
    )
    dict_to_csv(boundary, os.path.join(data_path, f"{boundary_file_name}.csv"))
    boundary_df=pd.read_csv(os.path.join(data_path, f"{boundary_file_name}.csv"))
    boundary_df[extra_input_keys] = extra_input_values

    return interior_df, boundary_df

def get_train_data(
    number_of_samples=2000000,
    dataframe_path="/bechh/dataframes/01_Water_Water.csv",
    seed=1,
    input_keys=["x", "y", "z"],
    output_keys=["u", "v", "w"],
    output_keys_translation=["velocityi", "velocityj", "velocityk"],
):
    df = pd.read_csv(dataframe_path)

    # Use dropna() to remove rows with NaN values.
    df_no_nan = df.dropna()

    simulation_invar_random, simulation_outvar_random = get_random_data_points(
        df_no_nan,
        number_of_samples,
        seed,
        input_keys,
        output_keys,
        output_keys_translation,
    )

    simulation_outvar_random["continuity"] = np.zeros_like(simulation_outvar_random["u"])
    simulation_outvar_random["momentum_x"] = np.zeros_like(simulation_outvar_random["u"])
    simulation_outvar_random["momentum_y"] = np.zeros_like(simulation_outvar_random["u"])
    simulation_outvar_random["momentum_z"] = np.zeros_like(simulation_outvar_random["u"])
    simulation_outvar_random["p"] = np.zeros_like(simulation_outvar_random["u"])

    return simulation_invar_random, simulation_outvar_random



def get_branch_input(
    branch_data_path,
    trunk_input_length,
    desired_branch_input_keys=None,
    original_branch_input_keys=None,
):
    """Get data from the paths to use for a DeepONet.
    
    The branch net input must have shape (N, M) and the trunk net input must have shape (N, 1) where N must be equal to the trunk input size."""

    df_branch = pd.read_csv(branch_data_path)
    
    if desired_branch_input_keys is not None:
        df_branch = df_key_rename(df_branch, original_branch_input_keys, desired_branch_input_keys)

    branch_dict = df_branch.to_dict(orient="list")

    branch_input_dict = {
        key: np.tile(np.asarray([v for v in value])[np.newaxis], (trunk_input_length, 1)) for key, value in branch_dict.items() if key in desired_branch_input_keys
    }

    return branch_input_dict


def get_deeponet_data(
    branch_data_path,
    data_path,
    desired_branch_input_keys=None,
    original_branch_input_keys=None,
    desired_trunk_input_keys=None,
    original_trunk_input_keys=None,
    desired_output_keys=None,
    original_output_keys=None,
):
    """Get data from the paths to use for a DeepONet.
    
    The branch net input must have shape (N, M) and the trunk net input must have shape (N, 1) where N must be equal to the trunk input size."""

    df_branch = pd.read_csv(branch_data_path)
    df_data = pd.read_csv(data_path)
    
    if desired_branch_input_keys is not None:
        df_branch = df_key_rename(df_branch, original_branch_input_keys, desired_branch_input_keys)

    if desired_trunk_input_keys is not None:
        df_data = df_key_rename(df_data, original_trunk_input_keys, desired_trunk_input_keys)
    
    if desired_output_keys is not None:
        df_data = df_key_rename(df_data, original_output_keys, desired_output_keys)

    l = len(df_data)

    trunk_input_dict, output_dict = split_invar_outvar(
        df_data,
        desired_trunk_input_keys,
        desired_output_keys,
    )

    branch_dict = df_branch.to_dict(orient="list")

    branch_input_dict = {
        key: np.tile(np.asarray([v for v in value])[np.newaxis], (l, 1)) for key, value in branch_dict.items() if key in desired_branch_input_keys
    }

    return trunk_input_dict, branch_input_dict, output_dict, l


def create_line(p0, p1, t=(0, 0), scale=1, n=1):
    """Creates a line from p0 to p1 translated by t and scaled by scale. Returns the line and its length."""

    l = np.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)
    
    if p0[0] != p1[0]:
        a = np.arctan((p1[1] - p0[1]) / (p1[0] - p0[0]))
    else:
        a = -np.pi / 2
    line = Line((0, 0), (0, l), normal=n)
    line = line.rotate(angle=-np.pi / 2 + a)
    line = line.translate((t[0] + p0[0], t[1] + p0[1]))
    line = line.scale(scale)

    l = l * scale

    # Normal, but not needed atm. 
    # N = Line((0,0), (0, 1), normal=1)
    # if n==1:
    #     N = N.rotate(angle=-np.pi + a)
    # elif n==-1:
    #     N = N.rotate(angle=a)
    return line, l


def get_data(
    df_path="/data/pipe_bend/simulations_csv/vel_mag_first_sim_inlet_transformed.csv",
    desired_input_keys=None,
    original_input_keys=None,
    desired_output_keys=None,
    original_output_keys=None,
    input_scaling=None,
    velocity_scaling=None,
    translation=None,
    rotation_matrix=None,
):
    """Get data from the df_path to use for training the PINN. 
    
    Returns the length of the data frame and two dictionaries consisting of the input keys and output keys with their values from the data. 
    The data can be translated, rotated and scaled to match the geometric setup you desire."""
    
    df = pd.read_csv(df_path)

    if desired_output_keys is not None:
        df = df_key_rename(df, original_output_keys, desired_output_keys)

    if desired_input_keys is not None:
        df = df_key_rename(df, original_input_keys, desired_input_keys)

    # Use dropna() to remove rows with NaN values.
    df = df.dropna()

    df = translate(df, ["x", "y", "z"], translation)
    df = rotate(df, ["x", "y", "z"], rotation_matrix)
    df = scale(df, ["x", "y", "z"], input_scaling)
    df = scale_velocity(df, desired_output_keys, velocity_scaling)

    input_dict, output_dict = split_invar_outvar(
        df,
        desired_input_keys,
        desired_output_keys,
    )

    return input_dict, output_dict, len(df)


def add_validator_rorbuk(
    nodes,
    domain,
    df_path,
    input_keys=["x", "y", "z"],
    output_keys=["vel_mag"],
    output_keys_plot_naming=None,
    output_keys_translation=["velocity_magnitude"],
    batch_size=100,
    input_scaling=None,
    velocity_scaling=None,
    translation=None,
    rotation_matrix=None,
    plot_name="inlet",
):
    return add_validator_rorbuk_main(
        nodes=nodes,
        domain=domain,
        df_path=df_path,
        input_keys=input_keys,
        output_keys=output_keys,
        output_keys_plot_naming=output_keys_plot_naming,
        output_keys_translation=output_keys_translation,
        batch_size=batch_size,
        input_scaling=input_scaling,
        velocity_scaling=velocity_scaling,
        translation=translation,
        rotation_matrix=rotation_matrix,
        plot_name=plot_name,
        use_parabola=False,
        center=None,
        normal=None,
        radius=None,
        max_vel=None,
    )


def add_validator_rorbuk_parabola(
    nodes,
    domain,
    df_path,
    center,
    normal,
    radius,
    max_vel,
    input_keys=["x", "y", "z"],
    output_keys=["vel_mag"],
    output_keys_plot_naming=None,
    output_keys_translation=["velocity_magnitude"],
    batch_size=100,
    input_scaling=None,
    velocity_scaling=None,
    translation=None,
    rotation_matrix=None,
    plot_name="inlet_val_parabola",
):
    return add_validator_rorbuk_main(
        nodes=nodes,
        domain=domain,
        df_path=df_path,
        input_keys=input_keys,
        output_keys=output_keys,
        output_keys_plot_naming=output_keys_plot_naming,
        output_keys_translation=output_keys_translation,
        batch_size=batch_size,
        input_scaling=input_scaling,
        velocity_scaling=velocity_scaling,
        translation=translation,
        rotation_matrix=rotation_matrix,
        plot_name=plot_name,
        use_parabola=True,
        center=center,
        normal=normal,
        radius=radius,
        max_vel=max_vel,
    )


# def add_validator_rorbuk_main(
#     nodes,
#     domain,
#     df_path,
#     input_keys=["x", "y", "z"],
#     output_keys=["vel_mag"],
#     output_keys_plot_naming=None,
#     output_keys_translation=["velocity_magnitude"],
#     batch_size=100,
#     input_scaling=None,
#     velocity_scaling=None,
#     translation=None,
#     rotation_matrix=None,
#     plot_name="inlet",
#     use_parabola=False,
#     center=None,
#     normal=None,
#     radius=None,
#     max_vel=None,
# ):
#     """Add validator."""

#     # Load dataframe with pandas.
#     df = pd.read_csv(df_path)

#     # Rename keys in dictionary.
#     df = df_key_rename(df, output_keys_translation, output_keys)

#     # Use dropna() to remove rows with NaN values.
#     df = df.dropna()

#     # Transform the data to fit your geometric setup.
#     df = translate(df, input_keys, translation)
#     df = rotate(df, input_keys, rotation_matrix)
#     df = scale(df, input_keys, input_scaling)
#     df = scale_velocity(df, output_keys, velocity_scaling)

#     # If the inlet velocity is parabolic:
#     if use_parabola:
#         df_dict = df.to_dict(orient="list")
#         simulation_invar_plane = {key: [[v] for v in value] for key, value in df_dict.items() if key in input_keys}
#         u, v, w = circular_parabola_np(
#             df["x"].to_numpy(),
#             df["y"].to_numpy(),
#             df["z"].to_numpy(),
#             center=(center[0], center[1], 0),
#             normal=normal,
#             radius=radius,
#             max_vel=max_vel,
#         )
#         vel_mag = np.sqrt((u**2 + v**2 + w**2).astype(float))
#         simulation_outvar_plane = {"vel_mag": [[v] for v in vel_mag]}
#     else:
#         simulation_invar_plane, simulation_outvar_plane = split_invar_outvar(
#             df=df,
#             input_keys=input_keys,
#             output_keys=output_keys,
#         )

#     simulation_plane_validator = PointwiseValidator(
#         nodes=nodes,
#         invar=simulation_invar_plane,
#         true_outvar=simulation_outvar_plane,
#         batch_size=batch_size,
#         plotter=ValidatorPlanePlotter(
#             invar_keys=["x", "z"],
#             outvar_keys=output_keys,
#             figsize=(3 * 5, 4),
#             output_keys_plot_naming=output_keys_plot_naming,
#         ),
#     )
#     domain.add_validator(simulation_plane_validator, plot_name)


# def add_validator(
#     nodes,
#     domain,
#     number_of_samples=10000,
#     dataframe_path="/bechh/dataframes/01_Water_Water.csv",
#     seed=1,
#     input_keys=["x", "y", "z"],
#     output_keys=["u", "v", "w"],
#     output_keys_translation=["velocityi", "velocityj", "velocityk"],
#     dataframe_key=None,
#     batch_size=100,
#     translation=[0, 0, 0],
# ):
#     """Add validator."""
#     df = pd.read_csv(dataframe_path)

#     df = df_key_rename(df, output_keys_translation, output_keys)

#     if dataframe_key is not None:
#         df = df[(df["block_name"] == dataframe_key)]

#     # Use dropna() to remove rows with NaN values.
#     df = df.dropna()

#     df = translate(df, input_keys, translation)

#     simulation_invar_random, simulation_outvar_random = get_random_data_points(
#         df,
#         number_of_samples,
#         seed,
#         input_keys,
#         output_keys,
#         output_keys_translation,
#     )

#     simulation_random_validator = PointwiseValidator(
#         nodes=nodes,
#         invar=simulation_invar_random,
#         true_outvar=simulation_outvar_random,
#         batch_size=batch_size,
#     )
#     domain.add_validator(simulation_random_validator, "random_points")

#     simulation_invar_y_plane, simulation_outvar_y_plane = get_plane_data_points(
#         df,
#         inputs_criterion={
#             "y": {"point_range": np.asarray([0.100460, 0.101]) - translation[1]},
#         },
#         outputs_criterion={
#             "velocitymagnitude": 1.5,
#         },
#     )

#     simulation_y_plane_validator = PointwiseValidator(
#         nodes=nodes,
#         invar=simulation_invar_y_plane,
#         true_outvar=simulation_outvar_y_plane,
#         batch_size=batch_size,
#         plotter=ValidatorPlanePlotter(),
#     )
#     domain.add_validator(simulation_y_plane_validator, "y_plane")


def get_random_data_points(
    df,
    number_of_samples,
    seed=1,
    input_keys=["x", "y", "z"],
    output_keys=["u", "v", "w"],
):
    df_samples = df.sample(number_of_samples, random_state=seed)

    return split_invar_outvar(
        df_samples,
        input_keys,
        output_keys,
    )


def get_plane_data_points(
    df,
    inputs_criterion={
        "y": {
            "point_range": [0.100460, 0.101],
        },
    },
    outputs_criterion={
        "velocitymagnitude": 1.2,
    },
    input_keys=["x", "y", "z"],
    output_keys=["u", "v", "w"],
):
    df_criterion = None
    for key, value in inputs_criterion.items():
        df_criterion = boolean_df_criterion_between(df, key, value["point_range"], df_criterion)

    for key, value in outputs_criterion.items():
        df_criterion = boolean_df_criterion(df, key, value, df_criterion)

    df_plane = df[df_criterion]

    return split_invar_outvar(
        df=df_plane,
        input_keys=input_keys,
        output_keys=output_keys,
    )


def split_invar_outvar(
    df,
    input_keys=["x", "y", "z"],
    output_keys=["u", "v", "w"],
):
    samples_dict = df.to_dict(orient="list")

    simulation_invar = {key: [[v] for v in value] for key, value in samples_dict.items() if key in input_keys}
    simulation_outvar = {key: [[v] for v in value] for key, value in samples_dict.items() if key in output_keys}

    return simulation_invar, simulation_outvar


def boolean_df_criterion(df, key, value, expression=None):
    tmp_expression = df[key] < value
    if expression is None:
        expression = tmp_expression
    else:
        expression &= tmp_expression
    return expression


def boolean_df_criterion_between(df, key, value, expression=None):
    tmp_expression = df[key].between(value[0], value[1])
    if expression is None:
        expression = tmp_expression
    else:
        expression &= tmp_expression
    return expression


def get_block_names(case):
    block_names = [case.GetMetaData(i).Get(vtk.vtkCompositeDataSet.NAME()) for i in range(case.GetNumberOfBlocks())]
    for i, _ in enumerate(block_names):
        block_names[i] = block_names[i].replace(" ", "")
        block_names[i] = block_names[i].replace("[", "_")
        block_names[i] = block_names[i].replace("]", "_")
        block_names[i] = block_names[i].replace("/", "_")
        block_names[i] = block_names[i].lower()
    return block_names


def normalize_mesh(mesh, center, scale=1):
    # Normalize meshes.
    mesh = mesh.translate([-c for c in center])
    mesh = mesh.scale(scale)
    return mesh


def normalize_invar(invar, center, scale, dims=3):
    # Normalize invars.
    invar["x"] -= center[0]
    invar["y"] -= center[1]
    invar["z"] -= center[2]
    invar["x"] *= scale
    invar["y"] *= scale
    invar["z"] *= scale
    if "area" in invar.keys():
        invar["area"] *= scale**dims
    return invar


def circular_parabola(x, y, z, center, normal, radius, max_vel):
    # Helper function to calculate inlet velocity profile. Use parabolic profile.
    centered_x = x - center[0]
    centered_y = y - center[1]
    centered_z = z - center[2]
    distance = sqrt(centered_x**2 + centered_y**2 + centered_z**2)
    parabola = max_vel * Max((1 - (distance / radius) ** 2), 0)
    return normal[0] * parabola, normal[1] * parabola, normal[2] * parabola


def circular_parabola_np(x, y, z, center, normal, radius, max_vel):
    # Helper function to calculate inlet velocity profile. Use parabolic profile.
    centered_x = x - center[0]
    centered_y = y - center[1]
    centered_z = z - center[2]
    distance = np.sqrt(centered_x**2 + centered_y**2 + centered_z**2)
    parabola = max_vel * np.maximum((1 - (distance / radius) ** 2), 0)
    return normal[0] * parabola, normal[1] * parabola, normal[2] * parabola


def add_inferencer(
    nodes,
    domain,
    plot_name,
    df_path=None,
    df=None,
    input_keys=["x", "y", "z"],
    output_keys=["u", "v", "w"],
    batch_size=100,
    input_scaling=None,
    translation=None,
    rotation_matrix=None,
):
    """Add inferencer."""
    if df_path is not None:
        df = pd.read_csv(df_path)

    # Use dropna() to remove rows with NaN values.
    df = df.dropna()

    df = translate(df, ["x", "y", "z"], translation)
    df = rotate(df, ["x", "y", "z"], rotation_matrix)
    df = scale(df, ["x", "y", "z"], input_scaling)

    samples_dict = df.to_dict(orient="list")
    simulation_invar = {key: [[v] for v in value] for key, value in samples_dict.items() if key in input_keys}

    grid_inference = PointwiseInferencer(
        nodes=nodes,
        invar=simulation_invar,
        output_names=output_keys,
        batch_size=batch_size,
    )
    domain.add_inferencer(grid_inference, plot_name)


def translate(df, keys, translation):
    if translation is not None:
        df[keys] = df[keys] - translation
    return df


def rotate(df, keys, rotation_matrix):
    if rotation_matrix is not None:
        df[keys] = np.asarray([np.dot(rotation_matrix, entry) for _, entry in df[keys].iterrows()])
    return df


def scale(df, keys, scaling):
    if scaling is not None:
        df[keys] = df[keys] * scaling
    return df


def scale_velocity(df, keys, scaling):
    for key in ["u", "v", "w", "vel_mag"]:
        if key in keys:
            df = scale(df, key, scaling)
    return df


def df_key_rename(df, keys_from, keys_to):
    if keys_from is not None:
        for to_key, from_key in zip(keys_to, keys_from):
            df[to_key] = df[from_key]
    return df
