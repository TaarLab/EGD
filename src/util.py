import numpy as np
import pyvista as pv
from pyvista import ColorLike
from sklearn.cluster import DBSCAN
from .error_function import plane_distance
from scipy.spatial.transform import Rotation

EXPLORATION_TABLE = 1
EXPLOITATION_TABLE = 2


def add_sphere(plotter: pv.Plotter, position: np.array, color: ColorLike, point_size: float, name: str):
    plotter.add_mesh(pv.Sphere(center=position, radius=point_size), name=name, color=color)


def point_picking_callback(mesh, pid, plotter, _):
    print(pid)
    point = mesh.points[pid]
    dargs = dict(name='labels', font_size=24)
    label = ['ID: {}'.format(pid)]
    plotter.add_point_labels(point, label, **dargs)


def enable_point_picking(plotter, points):
    plotter.enable_point_picking(callback=lambda mesh, pid: point_picking_callback(mesh, pid, plotter, points),
                                 show_message=True,
                                 picker='point', point_size=25,
                                 use_picker=True, show_point=True)


def find_best_point_to_explore(data):
    min_item = max(((index, value) for index, value in enumerate(data)), key=lambda x: x[1])
    return min_item[0] if min_item else None, min_item[1]


def select_elements(arr, _from, count):
    if _from + count > len(arr):
        result = np.concatenate((arr[_from:min(_from + count, len(arr))], arr[0:max(0, _from + count - len(arr))]))
    else:
        result = arr[_from:_from + count]
    return result


def compute_distances(points):
    n = len(points)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distances[i, j] = plane_distance(points[i][:3], points[j][3:], points[j][:3])
    return distances


def cluster_points(points, eps=0.01, min_samples=2):
    distances = compute_distances(points)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = dbscan.fit_predict(distances)
    return labels


def translate_to_origin(point_cloud, midpoint):
    """Translate point cloud such that midpoint is moved to the origin."""
    return point_cloud - midpoint


def compute_alignment_matrix(normal, target_normal):
    """Compute the rotation matrix to align 'normal' with 'target_normal' using scipy."""
    # Compute rotation vector
    v = np.cross(normal, target_normal)
    s = np.linalg.norm(v)
    if s != 0:  # To avoid division by zero
        v /= s

    # Compute angle between normal and target_normal
    theta = np.arccos(np.clip(np.dot(normal, target_normal), -1.0, 1.0))

    # Create rotation using scipy
    return Rotation.from_rotvec(theta * v)


def rotation_matrix_to_align_with_target(v, target):
    v = v / np.linalg.norm(v)

    k = target

    r = np.cross(v, k)
    r_norm = np.linalg.norm(r)

    if r_norm == 0:
        if np.dot(v, k) > 0:  # If v is the same as k
            return np.eye(3)
        else:  # v is in the opposite direction of k
            return -np.eye(3)

    r = r / r_norm

    theta = np.arccos(np.dot(v, k))

    # Compute rotation matrix using Rodrigues' formula
    return Rotation.from_rotvec(theta * r)

def cross_product(a, b):
    return np.cross(a, b)


def rotation_matrix_to_align_vectors(start, end, angle):
    axis = np.subtract(end, start)
    axis_normalized = axis / np.linalg.norm(axis)

    return Rotation.from_rotvec(axis_normalized * np.deg2rad(angle))