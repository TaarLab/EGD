import time
import asyncio
import numpy as np
import pyvista as pv
import networkx as nx
from asyncio import sleep

from src.egd_2dof import fit, FitParameters, HyperParameters, extract_grasp_points, check_collision
from src.util import EXPLORATION_TABLE, EXPLOITATION_TABLE, enable_point_picking, select_elements

# Options
RECORDING = False
INITIAL_SLEEP = False
# INITIAL_SLEEP = False or RECORDING
DISABLE_GRAPHIC = False
SLEEP_ON_ITERATION = False
DISABLE_GRAPHIC_FOR_ITERATION = True
# SLEEP_ON_ITERATION = False or RECORDING
LOGGING_ERRORS = False
VISUALIZATION_TABLE_MODE = EXPLORATION_TABLE

start_time = time.time()
pv.set_plot_theme('paraview')
plotter = pv.Plotter()


async def test():
    global plotter

    if INITIAL_SLEEP:
        await sleep(6)

    if not DISABLE_GRAPHIC:
        if RECORDING:
            asyncio.create_task(record_plot())
        asyncio.create_task(update_plot())

    if DISABLE_GRAPHIC:
        from unittest.mock import MagicMock
        plotter = MagicMock()

    plotter.enable_parallel_projection()
    plotter.view_xy()
    plotter.show_axes()
    plotter.show(interactive_update=True)

    points = np.load('samples/elephant.npy')

    # Sort points based on their height
    # sorted_indices = np.argsort(np.abs(points[:, 2]) * -1)
    # points = points[sorted_indices]

    np.random.seed(42)
    np.random.shuffle(points)

    # Select specific init node for debugging purpose
    # random_points_indices = [10763, 28210]
    # result = calculate_exploitation_error(
    #     points[random_points_indices][0],
    #     points[random_points_indices][1],
    #     -4)

    graph = nx.Graph()

    if not RECORDING:
        enable_point_picking(plotter, points)

    plotter.add_points(np.array([[0, 0, 0]]), point_size=25, color=[0, 0, 191], render_points_as_spheres=True)
    ps = pv.PointSet(points[:, :3])
    plotter.add_points(ps, color=[191, 34, 139], render_points_as_spheres=True)
    plotter.add_ruler(
        pointa=[ps.bounds[0] - 0.1, ps.bounds[3], 0.0],
        pointb=[ps.bounds[0] - 0.1, ps.bounds[2], 0.0],
        flip_range=True,
        title="" if RECORDING else "Y Distance",
    )

    actual_start_time = time.time()

    await fit(FitParameters(
        graph=graph,
        points=points,
        hyper_parameters=HyperParameters(
            steps=50,
            sample_size=20,
            min_error=-20.0,
            acceptable_error=-0.2,
            min_exploration_error=-20.0,
            max_exploitation_error=-0.05,
            max_gripper_opening=0.085,
            min_gripper_opening=0.01,
            current_gripper_z_angle=0
        ),
        on_iterate=on_iterate,
        on_step_end=on_step_end,
        on_step_start=on_step_start,
        on_exploitation_error=on_exploitation_error
    ))

    print(f"Algorithm total time: {time.time() - actual_start_time} seconds")
    print('Algorithm Stopped!')

    grasp_points = extract_grasp_points(graph)

    print(f"Total Grasp: {len(grasp_points)}")

    for i, grasp_points in enumerate(grasp_points):
        angle, point = check_collision(points[:, :3], grasp_points[0], grasp_points[1], np.array([0, 0, 0.4]), 0.1, 0.05)
        print(point, angle)
        plotter.add_points(np.array([point]), point_size=25, color=[255, 0, 0], render_points_as_spheres=True)
        if angle is not None:
            plotter.add_arrows(point, point + angle, color="red", mag=0.1)

    # labels = cluster_points(grasp_points)
    #
    # pd = pv.PolyData(grasp_points[:, :3])
    # pd.point_data['cluster'] = labels
    # plotter.add_mesh(pd, point_size=35, scalars='cluster', cmap='jet',
    #                  render_points_as_spheres=True, name='data')

    while True:
        await sleep(1)


def visualize_graph(params: FitParameters):
    if not DISABLE_GRAPHIC:
        graph_points = np.array(list(nx.get_node_attributes(params.graph, 'pos').values()))
        plotter.add_mesh(pv.PolyData(graph_points), point_size=25,
                         render_points_as_spheres=True, name='data',
                         color=[37, 207, 43])
        plotter.add_arrows(
            graph_points,
            params.points[np.array(list(nx.get_node_attributes(params.graph, 'point_cloud_idx').values())), 3:],
            color="black", mag=0.01
        )

        lines = list(map(
            lambda edge: pv.Line(params.graph.nodes[edge[0]]['pos'], params.graph.nodes[edge[1]]['pos']),
            params.graph.edges
        ))
        plotter.add_composite(pv.MultiBlock(lines), color=[37, 109, 161], line_width=10, name='edges')


def visualize_error_table(params: FitParameters, exploration_error_table: np.ndarray, current_iteration: int):
    if not DISABLE_GRAPHIC and (not DISABLE_GRAPHIC_FOR_ITERATION or current_iteration % params.hyper_parameters.sample_size == 0):
        error_table_points = select_elements(
            params.points,
            (current_iteration - params.hyper_parameters.sample_size + 1) % len(params.points),
            params.hyper_parameters.sample_size
        )[:, :3]
        pd = pv.PolyData(error_table_points)
        if VISUALIZATION_TABLE_MODE == EXPLORATION_TABLE:
            pd.point_data['max_errors'] = exploration_error_table
        elif VISUALIZATION_TABLE_MODE == EXPLOITATION_TABLE:
            pass
            # TODO visualize the exploitation table
            # temp_error_table = np.ones((i + 1, len(graph.nodes))) * MIN_ERROR
            # for node in nodes_data:
            #     if not graph.nodes[node]['is_temp']:
            #         error = calculate_error(currentPoint, graph.nodes[node]['data'])
            #         temp_error_table[i][graph.nodes[node]['error_idx']] = error
            # pd.point_data['max_errors'] = [max(values) for values in temp_error_table[:i + 1]]
        plotter.add_mesh(pd, scalars='max_errors', cmap='hot',
                         point_size=10, render_points_as_spheres=True,
                         name='error', scalar_bar_args={'title': 'Max Errors'},
                         show_scalar_bar=False
                         )


def print_error_status(acceptable_error: float, error: float, target_point_idx: int, best_point_idx: int):
    if not LOGGING_ERRORS:
        return
    if error <= acceptable_error:
        print("Rejected Error:", error, f"({target_point_idx},"
                                        f" {best_point_idx})")
    else:
        print("Accepted Error:", error, f"({target_point_idx},"
                                        f" {best_point_idx})")


async def on_step_start(_: FitParameters):
    print('*' * 30)


async def on_step_end(params: FitParameters):
    global start_time
    print(f"Step took {time.time() - start_time} seconds to run")
    start_time = time.time()

    visualize_graph(params)

    if SLEEP_ON_ITERATION:
        await sleep(2)


async def on_iterate(params: FitParameters, exploration_error_table: np.ndarray, current_iteration: int):
    visualize_error_table(params, exploration_error_table, current_iteration)


async def on_exploitation_error(params: FitParameters, error: float, target_point_idx: int, best_point_idx: int):
    print_error_status(params.hyper_parameters.acceptable_error, error, target_point_idx, best_point_idx)


async def update_plot():
    global plotter

    while True:
        plotter.update()
        await sleep(0.1)


async def record_plot():
    pass
    # TODO record the steps as gif
    # plotter.open_gif("test.gif")
    #
    # await sleep(5)
    #
    # print('recording started!')
    #
    # while current_iteration < sample_size * step - 1:
    #     print('recording iteration: ' + str(current_iteration))
    #     plotter.write_frame()
    #     await sleep(0.1)
    #
    # print('recording finished!')
    # plotter.close()

if __name__ == '__main__':
    asyncio.run(test())
