import math
import asyncio
import time

import numpy as np
import pyvista as pv
import networkx as nx
from scipy.spatial.transform import Rotation
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from plotter import PlotterThread
from utils import get_pcn_from_cameras, setGripperData, moveToPose_viaIK, moveToConfig_viaFK
from src.egd_2dof import FitParameters, fit, HyperParameters, extract_grasp_points, check_collision

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
    simIK = client.require('simIK')

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

    pv.set_plot_theme('paraview')

    setGripperData(sim, gripperHandle, True)

    plotter_thread = PlotterThread()
    plotter_thread.start()

    plotters = {}

    plotter_thread.add_command(lambda: create_plotter(plotters))
    plotter_thread.set_interval(lambda: update(plotters))

    # input()

    i = 0
    env_handler = sim.getObject('/env')
    while True:
        object_handler = sim.getObjectChild(env_handler, i)
        if object_handler == -1:
            break
        prop = sim.getModelProperty(object_handler)
        prop |= sim.modelproperty_not_dynamic
        prop |= sim.modelproperty_not_respondable
        prop |= sim.modelproperty_not_visible
        sim.setModelProperty(object_handler, prop)
        i += 1

    i = 0
    while True:
        object_handler = sim.getObjectChild(env_handler, i)
        if object_handler == -1:
            break

        sim.setObjectInt32Param(object_handler, sim.objintparam_visibility_layer, 1)
        prop = sim.getModelProperty(object_handler)
        prop &= ~sim.modelproperty_not_dynamic
        prop &= ~sim.modelproperty_not_respondable
        prop &= ~sim.modelproperty_not_visible
        sim.setModelProperty(object_handler, prop)
        object_identifier = '/' + sim.getObjectName(object_handler)

        await evaluate(object_identifier, sim, gripperHandle, ikMaxVel, ikMaxAccel, ikMaxJerk,
                       maxVel, maxAccel, maxJerk, data, pickConfig, plotters, plotter_thread)

        prop |= sim.modelproperty_not_dynamic
        prop |= sim.modelproperty_not_respondable
        prop |= sim.modelproperty_not_visible
        sim.setModelProperty(object_handler, prop)
        i += 1

    sim.stopSimulation()


def create_plotter(plotters):
    plotter = pv.Plotter(shape=(2, 1), window_size=(600, 1000))
    plotter.ren_win.SetPosition(1350, 0)
    plotter.subplot(1, 0)
    plotter.enable_parallel_projection()
    plotter.view_xy()
    plotter.show_axes()
    plotter.subplot(0, 0)
    plotter.enable_parallel_projection()
    plotter.view_xy()
    plotter.show_axes()
    plotter.link_views()
    plotter.show(interactive_update=True)

    plotters['plotter'] = plotter


def update(plotters):
    for plotter in plotters.values():
        plotter.update()

async def get_grasp_poses(sim , plotters, plotter_thread):
    start_time = time.time()
    point_cloud = await get_point_cloud(sim)
    print(f"Capture point cloud took {time.time() - start_time} seconds")

    ps = pv.PointSet(point_cloud[:, :3])
    plotter_thread.add_command(lambda: plotters['plotter'].subplot(1, 0))
    plotter_thread.add_command(lambda: plotters['plotter'].add_points(np.array([[0, 0, 0]]), point_size=1, color=[191, 34, 139], render_points_as_spheres=True))
    plotter_thread.add_command(lambda: plotters['plotter'].add_points(ps, color=[191, 34, 139], render_points_as_spheres=True))
    plotter_thread.add_command(lambda: plotters['plotter'].subplot(0, 0))
    plotter_thread.add_command(lambda: plotters['plotter'].add_points(np.array([[0, 0, 0]]), point_size=1, color=[191, 34, 139], render_points_as_spheres=True))
    plotter_thread.add_command(lambda: plotters['plotter'].add_points(ps, color=[191, 34, 139], render_points_as_spheres=True))

    np.random.seed(42)
    np.random.shuffle(point_cloud)

    graph = nx.Graph()

    actual_start_time = time.time()

    await fit(FitParameters(
        graph=graph,
        points=point_cloud,
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
        ),
        on_iterate=on_iterate,
        on_step_end=on_step_end,
        on_step_start=on_step_start,
        on_exploitation_error=on_exploitation_error
    ))

    algorithm_time = time.time() - actual_start_time
    print(f"Algorithm total time: {algorithm_time} seconds")

    grasp_points = extract_grasp_points(graph)
    print(f"Total Grasp: {len(grasp_points)}")

    grasp_poses = []
    arrows = []
    for i, grasp_point in enumerate(grasp_points):
        angle, point = check_collision(point_cloud[:, :3], grasp_point[0], grasp_point[1], np.array([0, 0, 0.4]), 0.1, 0.05)
        if angle is not None:
            grasp_poses.append(np.concatenate([point, vector_to_quaternion(angle, grasp_point[0] - grasp_point[1])]))

            arrow = pv.Arrow(scale=0.06)
            rotation_matrix = Rotation.from_quat(grasp_poses[-1][3:]).as_matrix()
            arrow.points = (arrow.points.dot(rotation_matrix.T) + point)
            arrows.append(arrow)

    plotter_thread.add_command(lambda: plotters['plotter'].add_composite(pv.MultiBlock(arrows), color=(0, 255, 0)))

    plotter_thread.add_command(lambda: plotters['plotter'].subplot(1, 0))
    graph_points = []
    list(map(
        lambda edge: (graph_points.append(graph.nodes[edge[0]]['pos']), graph_points.append(graph.nodes[edge[1]]['pos'])),
        list(graph.edges)
    ))
    plotter_thread.add_command(lambda: plotters['plotter'].add_mesh(pv.PolyData(graph_points), point_size=15,
                     render_points_as_spheres=True, name='data',
                     color=[37, 207, 43]))

    lines = list(map(
        lambda edge: pv.Line(graph.nodes[edge[0]]['pos'], graph.nodes[edge[1]]['pos']),
        list(graph.edges)
    ))
    plotter_thread.add_command(lambda: plotters['plotter'].add_composite(pv.MultiBlock(lines), color=[37, 109, 161], line_width=10, name='edges'))

    def rotate():
        orbit = plotters['plotter'].generate_orbital_path(
            factor=2.0, n_points=256, shift=0.0, viewup=[0, -0.5, 1]
        )
        plotters['plotter'].orbit_on_path(orbit, viewup=[0, -0.25, 1], step=0.02)
        plotters['plotter'].orbit_on_path(orbit, viewup=[0, -0.25, 1], step=0.02)

    plotter_thread.add_command(lambda: rotate())

    max_error = max(list(map(lambda x: x[2]['error'], graph.edges(data=True))))

    return grasp_poses, algorithm_time, max_error


async def evaluate(object_identifier, sim, gripper_handle, ikMaxVel, ikMaxAccel, ikMaxJerk,
                   maxVel, maxAccel, maxJerk, data, pickConfig, plotters, plotter_thread):

    grasp_poses, algorithm_time, max_error = await get_grasp_poses(sim, plotters, plotter_thread)

    total_attempts = len(grasp_poses)
    successful_attempts = 0
    top_five_successful_attempts = 0

    for i, grasp_pose in enumerate(grasp_poses):
        tries_count = 1
        successful_tries = 0
        for s in range(tries_count):
            if successful_tries > tries_count / 2:
                break
            if s - successful_tries > tries_count / 2:
                break

            client = RemoteAPIClient()

            sim = client.require('sim')
            simIK = client.require('simIK')
            sim.startSimulation()

            starting_height = sim.getObjectPose(sim.getObject(object_identifier), sim.handle_world)[2]

            setGripperData(sim, gripper_handle, True)

            moveToPose_viaIK(sim, simIK, ikMaxVel, ikMaxAccel, ikMaxJerk, list(grasp_pose), data)

            setGripperData(sim, gripper_handle, False)

            # NO MAGIC
            moveToPose_viaIK(sim, simIK, ikMaxVel, ikMaxAccel, ikMaxJerk, list(grasp_pose), data)

            await asyncio.sleep(0.5)

            moveToConfig_viaFK(sim, maxVel, maxAccel, maxJerk, pickConfig, data)

            ending_height = sim.getObjectPose(sim.getObject(object_identifier), sim.handle_world)[2]

            is_successful = ending_height - starting_height > 0.2
            if is_successful:
                successful_tries += 1

            sim.stopSimulation()

            await asyncio.sleep(0.5)
        is_successful = successful_tries > tries_count / 2
        if is_successful:
            successful_attempts += 1
            if i < 5:
                top_five_successful_attempts += 1

        print(f"[{i}] Is Grasp Successful: " + str(is_successful))

    success_rate = successful_attempts / total_attempts * 100
    top_five_success_rate = top_five_successful_attempts / 5 * 100
    print(f"Object Identifier: {object_identifier}")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Total Attempts: {total_attempts}")
    print(f"Successful Attempts: {successful_attempts}")
    print(
        f"{'Object':<20}{'Total Grasps':<15}{'Time':<10}{'Max Error':<10}{'Success Rate':<20}{'Top 5 Success Rate':<25}")
    print(
        f"{object_identifier[1:]:<20}{total_attempts:<15}{algorithm_time:<10.2f}{max_error:<10.4f}{success_rate:<20}{top_five_success_rate:<25}")


if __name__ == '__main__':
    asyncio.run(main())
