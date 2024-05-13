import asyncio
import time

import numpy as np
import networkx as nx
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from utils import get_pcn_from_cameras
from src.egd_2dof import FitParameters, fit, HyperParameters, extract_grasp_points


async def get_point_cloud(sim):
    cameras = [
        '/Vision_sensor_1',
        '/Vision_sensor_2',
        '/Vision_sensor_3',
        '/Vision_sensor_4',
    ]

    return get_pcn_from_cameras(sim, cameras)


async def main():
    client = RemoteAPIClient()

    sim = client.require('sim')
    sim.setStepping(False)

    start_time = time.time()
    point_cloud = await get_point_cloud(sim)
    print(f"Capture point cloud took {time.time() - start_time} seconds")

    np.random.seed(42)
    np.random.shuffle(point_cloud)

    results = []
    for steps in [10, 50]:
        for sample_size in [20, 40]:
            graph = nx.Graph()

            actual_start_time = time.time()

            await fit(FitParameters(
                graph=graph,
                points=point_cloud,
                hyper_parameters=HyperParameters(
                    steps=steps,
                    sample_size=sample_size,
                    min_error=-20.0,
                    acceptable_error=-0.2,
                    min_exploration_error=-20.0,
                    max_exploitation_error=-0.05,
                    max_gripper_opening=0.085,
                    min_gripper_opening=0.01,
                    current_gripper_z_angle=0
                )
            ))

            algorithm_time = time.time() - actual_start_time
            print(f"Algorithm total time: {algorithm_time} seconds")
            print('Algorithm Stopped!')

            grasp_points = extract_grasp_points(graph)
            print(f"Total Grasp: {len(grasp_points)}")

            results.append({
                'Point Cloud Size': point_cloud.shape[0],
                'Sample Size': sample_size,
                'Steps': steps,
                'Total Grasps': len(grasp_points),
                'Time': algorithm_time,
                'Max Error': max(list(map(lambda x: x[2]['error'], graph.edges(data=True))))
            })

    objectName = 'XXXXX'
    print(f"{'Object':<20}{'Point Cloud Size':<20}{'Steps':<10}{'Sample Size':<15}{'Total Grasps':<15}{'Time':<10}{'Max Error':<10}")
    for result in results:
        print(
            f"{objectName:<20}{result['Point Cloud Size']:<20}{result['Steps']:<10}{result['Sample Size']:<15}{result['Total Grasps']:<15}{result['Time']:<10.2f}{result['Max Error']:<10.4f}")


if __name__ == '__main__':
    asyncio.run(main())