import numpy as np
import open3d as o3d
import networkx as nx
from scipy.spatial import distance

EPS_B = 0.05
EPS_N = 0.0005
BETA = 0.5
MAX_AGE = 25
LAMBDA = 50

cube = np.load('glass.npy')
points = cube[:, 0:3]
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.paint_uniform_color([1,0,0])
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
# o3d.visualization.draw_geometries([pcd], point_show_normal=True)

def find_2_closest_point(currentPoint, graph):
    nodes_pos = nx.get_node_attributes(graph, 'pos')
    distances = {node: distance.euclidean(currentPoint, pos) for node, pos in nodes_pos.items()}
    return sorted(distances, key=distances.get)[:2]

def purge_graph(graph):
    edges = nx.get_edge_attributes(graph, 'age')
    affected_node = []
    
    for nodes, age in edges.items():
        if age > MAX_AGE:
            graph.remove_edge(*nodes)
            affected_node.append(nodes[0])
            affected_node.append(nodes[1])
            
    for node in affected_node:
        if len(graph.edges(node)) == 0:
            graph.remove_node(node)
            print("[!] Node Removed")

def purge_graph(graph):
    edges = nx.get_edge_attributes(graph, 'age')
    affected_nodes = []

    # Removing old edges and appending affected nodes
    old_edges = [nodes for nodes, age in edges.items() if age > MAX_AGE]
    graph.remove_edges_from(old_edges)
    affected_nodes.extend(old_edges)

    # Flatten affected_nodes list
    affected_nodes = [node for edge in affected_nodes for node in edge]

    # Removing nodes without edges
    nodes_to_remove = [node for node in affected_nodes if not graph.edges(node)]
    graph.remove_nodes_from(nodes_to_remove)
    
# Initialize graph
graph = nx.Graph()

C = 0

# 1. Randomly choose 2 initial point
random_points_indices = np.random.choice(len(points), size=2, replace=False)


for idx in random_points_indices:
    C += 1
    graph.add_node(C, pos=np.array(points[idx]), error=0)

graph.add_edge(C, C - 1, age=0)

iteration = 0
while iteration < 10000:
    for currentPoint in points:
        winner, runner_on = find_2_closest_point(currentPoint, graph)

        for u, v, attributes in graph.edges(winner, data=True):
            graph.add_edge(u, v, age=attributes['age']+1)

        graph.add_edge(winner, runner_on, age=0)

        graph.nodes[winner]['error'] += distance.euclidean(currentPoint, graph.nodes[winner]['pos']) * EPS_B
        graph.nodes[winner]['pos'] += (currentPoint - graph.nodes[winner]['pos']) * EPS_B

        for neighbor in graph.neighbors(winner):
            graph.nodes[neighbor]['pos'] += (currentPoint - graph.nodes[neighbor]['pos']) * EPS_N

        purge_graph(graph)

        if iteration % LAMBDA == 0:
            nodes_error = nx.get_node_attributes(graph, 'error')
            node_with_highest_error = max(nodes_error, key=nodes_error.get)

            node_with_highest_error_neighbors = [{'idx': node, **graph.nodes[node]} for node in graph.neighbors(node_with_highest_error)]
            node_with_highest_error_neighbor = max(node_with_highest_error_neighbors, key=lambda node: node['error'])['idx']

            graph.remove_edge(node_with_highest_error, node_with_highest_error_neighbor)
            C += 1
            graph.add_node(C, 
                           pos=(graph.nodes[node_with_highest_error]['pos'] + graph.nodes[node_with_highest_error_neighbor]['pos']) / 2, 
                           error=graph.nodes[node_with_highest_error]['error'] * BETA
                           )
            graph.add_edge(C, node_with_highest_error, age=0)
            graph.add_edge(C, node_with_highest_error_neighbor, age=0)

            graph.nodes[node_with_highest_error]['error'] *= BETA
            graph.nodes[node_with_highest_error_neighbor]['error'] *= BETA

        iteration += 1

    
graph_points = np.array(list(nx.get_node_attributes(graph, 'pos').values()))

graph_pcd = o3d.geometry.PointCloud()
graph_pcd.points = o3d.utility.Vector3dVector(graph_points)
graph_pcd.paint_uniform_color([0,1,0])
graph_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
# o3d.io.write_point_cloud("gng_glass.stl", graph_pcd)
o3d.visualization.draw_geometries([graph_pcd], point_show_normal=True)
# o3d.visualization.draw_geometries([pcd], point_show_normal=True)