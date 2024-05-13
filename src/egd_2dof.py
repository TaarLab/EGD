import numpy as np
import networkx as nx
from dataclasses import dataclass
from .util import find_best_point_to_explore, translate_to_origin, \
    cross_product, rotation_matrix_to_align_with_target, rotation_matrix_to_align_vectors
from typing import Callable, List, Tuple, Coroutine, Any, Optional
from .error_function import calculate_exploitation_error, calculate_exploration_error, GripperParameters


@dataclass
class HyperParameters:
    steps: int
    sample_size: int
    min_error: float
    acceptable_error: float
    min_exploration_error: float
    max_exploitation_error: float
    # Gripper parameter
    max_gripper_opening: float
    min_gripper_opening: float
    current_gripper_z_angle: float


@dataclass
class FitParameters:
    graph: nx.Graph
    points: np.ndarray
    hyper_parameters: HyperParameters
    error_table: Optional[List[Tuple[int, float]]] = None
    on_step_end: Optional[Callable[['FitParameters'], Coroutine[Any, Any, None]]] = None
    on_step_start: Optional[Callable[['FitParameters'], Coroutine[Any, Any, None]]] = None
    grasp_errors: Optional[List[Tuple[float, Any, Any]]] = None
    on_iterate: Optional[Callable[['FitParameters', np.ndarray, int], Coroutine[Any, Any, None]]] = None
    on_exploitation_error: Optional[Callable[['FitParameters', float, int, int], Coroutine[Any, Any, None]]] = None


def check_collision(point_cloud, point1: np.ndarray, point2: np.ndarray, gripper_position: np.ndarray,
                    GW: float, GH: float):
    midpoint = np.add(point1, point2) / 2
    translated_cloud = translate_to_origin(point_cloud, midpoint)
    normal_plane = cross_product(np.subtract(point1, gripper_position), np.subtract(point2, gripper_position))
    normal_plane = normal_plane / np.linalg.norm(normal_plane)
    normal_plane = rotation_matrix_to_align_vectors(point1, point2, -90).apply(normal_plane)
    transformation_matrix = rotation_matrix_to_align_with_target(normal_plane, [0, 0, 1])
    transformed_cloud = transformation_matrix.apply(translated_cloud)

    grasp_angle = normal_plane
    transformed_angle = transformation_matrix.apply(grasp_angle)

    # Disable collision check for now
    return grasp_angle * -1, midpoint

    collision_above, collision_below = False, False

    # Ground
    if midpoint[2] - GH < 0:
        collision_below = True

    for point in transformed_cloud:
        if (-GW / 2 <= point[0] <= GW / 2) and (-GW / 2 <= point[1] <= GW / 2):
            if point[2] > GH * np.sign(transformed_angle[2]):
                collision_above = True
            elif point[2] < -GH * np.sign(transformed_angle[2]):
                collision_below = True
        if collision_above and collision_below:
            break

    if collision_above and collision_below:
        rotation_matrix = rotation_matrix_to_align_vectors(point1, point2, -90)
        grasp_angle = rotation_matrix.apply(grasp_angle)
        transformed_angle = transformation_matrix.apply(grasp_angle)
        transformed_cloud = rotation_matrix.apply(transformed_cloud)
        collision_above, collision_below = False, False
        for point in transformed_cloud:
            if (-GW / 2 <= point[0] <= GW / 2) and (-GW / 2 <= point[1] <= GW / 2):
                if point[2] > GH * np.sign(transformed_angle[2]):
                    collision_above = True
                elif point[2] < -GH * np.sign(transformed_angle[2]):
                    collision_below = True
            if collision_above and collision_below:
                break
        if collision_above and collision_below:
            return None, midpoint  # No valid grasp found
        elif collision_above:
            return grasp_angle * -1, midpoint
        else:
            return grasp_angle, midpoint
    elif collision_above:
        return grasp_angle, midpoint
    else:
        return grasp_angle * -1, midpoint


def extract_grasp_points(graph):
    return np.array(list(
        map(
            lambda x: [x[0], x[1]],
            sorted(
                map(
                    lambda edge: [graph.nodes[edge[0]]['pos'], graph.nodes[edge[1]]['pos'], edge[2]['error']],
                    graph.edges(data=True)
                ),
                key=lambda x: -x[2]
            )
        )
    ))


async def fit(params: FitParameters):
    graph = params.graph
    points = params.points
    hp = params.hyper_parameters

    graph_key = 0

    graph.add_node(
        graph_key,
        pos=points[0][0:3],
        norm=points[0][3:6],
        data=points[0],
        point_cloud_idx=0,
        error_idx=len(graph.nodes),
        is_temp=False
    )

    error_table: List[Tuple[int, float]] = list([]) if params.error_table is None else params.error_table
    for i in range(len(graph.nodes)):
        error_table.append((i, hp.min_error))
    exploration_error_table: np.ndarray[float] = np.zeros(hp.sample_size, dtype=float)

    iteration = 0
    for exploration_error_table_idx, current_point in enumerate(points):

        if iteration > hp.steps * hp.sample_size:
            break

        for node in graph.nodes:
            current_node = graph.nodes[node]

            # Calculate error for grasp
            if not current_node['is_temp']:
                current_exploitation_error = error_table[current_node['error_idx']][1]
                if current_exploitation_error < hp.max_exploitation_error:
                    error = calculate_exploitation_error(
                        current_point, current_node['data'],
                        hp.acceptable_error,
                        GripperParameters(
                            max_gripper_opening=params.hyper_parameters.max_gripper_opening,
                            min_gripper_opening=params.hyper_parameters.min_gripper_opening,
                            current_gripper_z_angle=params.hyper_parameters.current_gripper_z_angle
                        )
                    )
                    if params.grasp_errors is not None:
                        params.grasp_errors.append((error, current_point[:3], current_node['data'][:3]))
                    if current_exploitation_error < error:
                        error_table[current_node['error_idx']] = (exploration_error_table_idx, error)

            # Calculate error for best point to explore
            current_exploration_error = exploration_error_table[exploration_error_table_idx % hp.sample_size]
            if current_exploration_error > hp.min_exploration_error:
                exploration_error = calculate_exploration_error(
                    current_point,
                    current_node['data'],
                    GripperParameters(
                        max_gripper_opening=params.hyper_parameters.max_gripper_opening,
                        min_gripper_opening=params.hyper_parameters.min_gripper_opening,
                        current_gripper_z_angle=params.hyper_parameters.current_gripper_z_angle
                    )
                )
                exploration_error_table[
                    exploration_error_table_idx % hp.sample_size
                    ] = min(exploration_error, current_exploration_error)

        if params.on_iterate is not None:
            await params.on_iterate(params, exploration_error_table, iteration)

        if iteration % hp.sample_size == 0 and iteration != 0:
            last_graph_node_count = len(graph.nodes)

            if params.on_step_start is not None:
                await params.on_step_start(params)

            # Exploitation
            error_table_idx = 0
            error_table_size = len(error_table)
            while error_table_idx < len(error_table):
                graph_node = next((k for k, v in nx.get_node_attributes(graph, 'error_idx').items()
                                   if v == error_table_idx), None)

                if graph.nodes[graph_node]['is_temp']:
                    error_table_idx += 1
                    continue

                best_point_for_grasp_idx, error = error_table[error_table_idx]
                best_point_for_grasp = points[best_point_for_grasp_idx]

                if params.on_exploitation_error is not None:
                    await params.on_exploitation_error(
                        params,
                        error,
                        graph.nodes[graph_node]['point_cloud_idx'],
                        best_point_for_grasp_idx
                    )

                if error < hp.acceptable_error:
                    error_table_idx += 1
                    continue

                existing_node_for_point_cloud = next(
                    (k for k, v in nx.get_node_attributes(graph, 'point_cloud_idx').items()
                     if v == best_point_for_grasp_idx), None
                )

                for u, v in list(graph.edges(graph_node)):
                    if graph.nodes[v]['is_temp'] and (existing_node_for_point_cloud is None
                                                      or v != existing_node_for_point_cloud):
                        graph.remove_edge(u, v)
                        error_idx = graph.nodes[v]['error_idx']
                        graph.remove_node(v)
                        error_table.pop(error_idx)

                        for node in graph.nodes:
                            if graph.nodes[node]['error_idx'] > error_idx:
                                graph.add_node(node, error_idx=graph.nodes[node]['error_idx'] - 1)

                        if error_idx < error_table_idx:
                            error_table_idx -= 1

                if existing_node_for_point_cloud is None:
                    graph_key += 1
                    graph.add_node(
                        graph_key,
                        pos=best_point_for_grasp[:3],
                        norm=best_point_for_grasp[3:],
                        data=best_point_for_grasp,
                        point_cloud_idx=best_point_for_grasp_idx,
                        error_idx=error_table_size + len(graph.nodes) - last_graph_node_count,
                        is_temp=True
                    )
                    graph.add_edge(graph_node, graph_key, error=error)

                error_table_idx += 1

            if hp.sample_size == len(points):
                for node in graph.nodes():
                    graph.nodes[node]['is_temp'] = True

            # Exploration
            best_point_to_explore_idx, error = find_best_point_to_explore(exploration_error_table)
            best_point_to_explore_idx += (iteration - hp.sample_size + 1)
            best_point_to_explore_idx %= len(points)
            best_point_to_explore = points[best_point_to_explore_idx]

            if best_point_to_explore_idx not in nx.get_node_attributes(graph, 'point_cloud_idx').values():
                graph_key += 1
                graph.add_node(graph_key, pos=best_point_to_explore[:3], norm=best_point_to_explore[3:],
                               data=best_point_to_explore, point_cloud_idx=best_point_to_explore_idx,
                               error_idx=error_table_size + len(graph.nodes) - last_graph_node_count, is_temp=False)

            for _ in range(error_table_size - len(error_table) + len(graph.nodes) - last_graph_node_count):
                error_table.append((0, hp.min_error))
            exploration_error_table: np.ndarray[float] = np.zeros(hp.sample_size, dtype=float)

            if params.on_step_end is not None:
                await params.on_step_end(params)

        iteration += 1
