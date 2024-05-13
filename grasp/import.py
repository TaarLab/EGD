import os
import asyncio
import hashlib

import numpy as np
import networkx as nx
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from src.egd_2dof import fit, FitParameters, HyperParameters, extract_grasp_points
from utils import get_pcn_from_cameras, cleanup_shapes, get_obj_files, add_obj_file_to_sim


async def main():
    client = RemoteAPIClient()
    sim = initialize_simulation(client)
    # obj_directory = r"C:\Users\ARB\Desktop\meshes\3dnet"
    # obj_directory = r"C:\Users\ARB\Desktop\meshes\adversarial"
    obj_directory = r"C:\Users\ARB\Desktop\meshes\kit"
    dataset_directory = r"C:\Users\ARB\Desktop\dataset"
    obj_files = get_obj_files(obj_directory)
    load_modes = [0, 1]

    for obj_file in obj_files:
        for load_mode in load_modes:
            fid = hashlib.md5(obj_file.encode()).hexdigest()[:6]

            point_cloud_file = os.path.join(dataset_directory, f'{fid}{load_mode}_point_cloud.npy')
            grasps_file = os.path.join(dataset_directory, f'{fid}{load_mode}_grasps.npy')
            neg_grasps_file = os.path.join(dataset_directory, f'{fid}{load_mode}_neg_grasps.npy')

            if os.path.exists(point_cloud_file) and os.path.exists(grasps_file) and os.path.exists(neg_grasps_file):
                print(f"Skipping {obj_file} as npy files already exist.")
                continue

            shape_handles = add_obj_file_to_sim(fid, obj_file, obj_directory, sim, load_mode)

            if shape_handles is None:
                continue

            point_cloud = get_point_cloud(sim)

            np.random.seed(42)
            np.random.shuffle(point_cloud)

            graph = nx.Graph()
            grasp_errors = list()

            await fit(FitParameters(
                graph=graph,
                points=point_cloud,
                grasp_errors=grasp_errors,
                hyper_parameters=HyperParameters(
                    steps=200,
                    sample_size=10,
                    min_error=-20.0,
                    acceptable_error=-0.15,
                    min_exploration_error=-20.0,
                    max_exploitation_error=-0.05,
                    max_gripper_opening=0.085,
                    min_gripper_opening=0.01,
                    current_gripper_z_angle=0,
                )
            ))

            cleanup_shapes(sim, shape_handles)

            grasp_points = extract_grasp_points(graph)  # (22, 2, 3), (Grasp, Points, CoordinationXYZ)
            if grasp_points.shape[0] < 3:
                print(f"Skipping {obj_file} cause too few grasp points.")
                continue
            else:
                print(f"{obj_file}: {grasp_points.shape[0]} grasp points")
            indices_of_grasp_points = find_indices_of_grasp_points(point_cloud, grasp_points)
            np.save(grasps_file, indices_of_grasp_points)
            np.save(point_cloud_file, point_cloud)  # (7049, 6), 7049 Point with xyz and normal vector

            grasp_errors.sort(key=lambda x: x[0])
            min_error = -4
            min_error_grasps = [[grasp[1], grasp[2]] for grasp in grasp_errors if grasp[0] == min_error]
            neg_grasps = np.array(min_error_grasps)
            np.random.shuffle(neg_grasps)
            indices_of_neg_grasp_points = find_indices_of_grasp_points(point_cloud, neg_grasps[:1024])
            np.save(neg_grasps_file, indices_of_neg_grasp_points)


def find_indices_of_grasp_points(point_cloud: np.ndarray, grasp_points: np.ndarray):
    indices = []
    for grasp in grasp_points:
        grasp_indices = []
        for point in grasp:
            idx = np.where((point_cloud[:, :3] == point).all(axis=1))[0]
            if idx.size > 0:
                grasp_indices.append(idx[0])

        if len(grasp_indices) != 2:
            continue
        indices.append(grasp_indices)
    return np.array(indices)


def initialize_simulation(client):
    sim = client.require('sim')
    sim.setStepping(False)
    return sim


def get_point_cloud(sim):
    cameras = [
        '/Vision_sensor_1',
        '/Vision_sensor_2',
        '/Vision_sensor_3',
        '/Vision_sensor_4',
    ]
    return get_pcn_from_cameras(sim, cameras)


if __name__ == "__main__":
    asyncio.run(main())
