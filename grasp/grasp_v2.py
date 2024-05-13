import math
import asyncio
import hashlib

import numpy as np
import networkx as nx
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from src.egd_2dof import fit, FitParameters, HyperParameters, extract_grasp_points, check_collision
from utils import get_pcn_from_cameras, cleanup_shapes, get_obj_files, add_obj_file_to_sim, setGripperData, \
    moveToPose_viaIK, moveToConfig_viaFK, vector_to_quaternion, wait_until_object_stops


async def main():
    client = RemoteAPIClient()
    sim, simIK = initialize_simulation(client)
    obj_directory = r"C:\Users\ARB\Desktop\meshes\3dnet"
    obj_files = get_obj_files(obj_directory)

    # Initialize robot joint values
    simJoints = [sim.getObject('/UR5/joint', {'index': i}) for i in range(6)]
    simTip = sim.getObject('/UR5/ikTip')
    simTarget = sim.getObject('/UR5/ikTarget')
    modelBase = sim.getObject('/UR5')
    gripperHandle = sim.getObject('/UR5/RG2')

    ikEnv = simIK.createEnvironment()

    # Prepare the IK group
    ikGroup = simIK.createGroup(ikEnv)
    simIK.addElementFromScene(ikEnv, ikGroup, modelBase, simTip, simTarget, simIK.constraint_pose)

    # Define FK movement data
    vel, accel, jerk = 180, 40, 80
    maxVel, maxAccel, maxJerk = [vel * math.pi / 180 for _ in range(6)], [accel * math.pi / 180 for _ in range(6)], [
        jerk * math.pi / 180 for _ in range(6)]

    # Define IK movement data
    ikMaxVel, ikMaxAccel, ikMaxJerk = [0.4, 0.4, 0.4, 1.8], [0.8, 0.8, 0.8, 0.9], [0.6, 0.6, 0.6, 0.8]

    pickConfig = [1.29158355133319, 0.3106908723883177, 0.0, 3.5940672100997455, 1.5705821039338792, 4.433069596719524]

    data = {
        'ikEnv': ikEnv,
        'ikGroup': ikGroup,
        'tip': simTip,
        'target': simTarget,
        'joints': simJoints
    }

    for obj_file in obj_files:
        fid = hashlib.md5(obj_file.encode()).hexdigest()[:5]

        sim.stopSimulation()

        shape_handles = []
        skip_file = False
        while True:
            try:
                shape_handles = add_obj_file_to_sim(fid, obj_file, obj_directory, sim)

                if shape_handles is None:
                    skip_file = True
                    break

                point_cloud = get_point_cloud(sim)
                break
            except:
                cleanup_shapes(sim, shape_handles)

        if skip_file:
            continue

        np.random.seed(42)
        np.random.shuffle(point_cloud)

        graph = nx.Graph()
        grasp_errors = list()

        await fit(FitParameters(
            graph=graph,
            points=point_cloud,
            grasp_errors=grasp_errors,
            hyper_parameters=HyperParameters(
                steps=50,
                sample_size=20,
                min_error=-20.0,
                acceptable_error=-0.25,
                min_exploration_error=-20.0,
                max_exploitation_error=-0.05,
                max_gripper_opening=0.085,
                min_gripper_opening=0.01,
                current_gripper_z_angle=0,
            )
        ))

        grasp_points = extract_grasp_points(graph)

        grasp_poses = []
        for i, grasp_point in enumerate(grasp_points):
            angle, point = check_collision(point_cloud[:, :3], grasp_point[0], grasp_point[1], np.array([0, 0, 0.4]),
                                           0.1, 0.05)
            if angle is not None:
                grasp_poses.append(
                    np.concatenate([point, vector_to_quaternion(angle, grasp_point[0] - grasp_point[1])])
                )

        if len(grasp_poses) > 0:
            grasp_pose = grasp_poses[0]

            client = RemoteAPIClient()

            sim = client.require('sim')
            simIK = client.require('simIK')
            sim.startSimulation()

            wait_until_object_stops(sim, shape_handles[0])

            starting_height = sim.getObjectPose(shape_handles[0], sim.handle_world)[2]

            setGripperData(sim, gripperHandle, True)

            moveToPose_viaIK(sim, simIK, ikMaxVel, ikMaxAccel, ikMaxJerk, list(grasp_pose), data)

            setGripperData(sim, gripperHandle, False)

            # NO MAGIC
            moveToPose_viaIK(sim, simIK, ikMaxVel, ikMaxAccel, ikMaxJerk, list(grasp_pose), data)

            await asyncio.sleep(0.5)

            moveToConfig_viaFK(sim, maxVel, maxAccel, maxJerk, pickConfig, data)

            ending_height = sim.getObjectPose(shape_handles[0], sim.handle_world)[2]

            is_successful = ending_height - starting_height > 0.2
        else:
            is_successful = False

        sim.stopSimulation()

        print(f"[{i}] Is Grasp Successful: " + str(is_successful))

        cleanup_shapes(sim, shape_handles)


def initialize_simulation(client):
    sim = client.require('sim')
    sim.setStepping(False)
    simIK = client.require('simIK')
    return sim, simIK


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
