import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from src.util import cross_product


def create_point_cloud(depth_array, intrinsic_matrix):
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]

    height, width = depth_array.shape

    u = np.linspace(0, width - 1, width)
    v = np.linspace(0, height - 1, height)
    u, v = np.meshgrid(u, v)

    valid_depths = depth_array > 0
    z = depth_array[valid_depths]
    x = (u[valid_depths] - cx) * z / fx
    y = (v[valid_depths] - cy) * z / fy

    # r = rgb_array[:, :, 0][valid_depths]
    # g = rgb_array[:, :, 1][valid_depths]
    # b = rgb_array[:, :, 2][valid_depths]

    point_cloud = np.stack((x, y, z), axis=-1)

    return point_cloud


def get_intrinsic_matrix(resolution, view_angle):
    fx = fy = resolution[0] / (2 * math.tan(view_angle / 2))
    cx = resolution[0] / 2
    cy = resolution[1] / 2
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


def get_vision_sensor_intrinsic(sim, cam_handler):
    view_angle = sim.getObjectFloatParam(cam_handler, sim.visionfloatparam_perspective_angle)
    image, resolution = sim.getVisionSensorImg(cam_handler)
    return get_intrinsic_matrix(resolution, view_angle)


def read_point_cloud(sim, cam_handler, depth):
    point_cloud = create_point_cloud(depth, get_vision_sensor_intrinsic(sim, cam_handler))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
    # pcd.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:] / 255.0)
    return pcd


def get_pcd_from_cam(sim, cam_name, blank_cam_pcd):
    cam_handler = sim.getObject(cam_name)

    depth, resolution = sim.getVisionSensorDepth(cam_handler, 1)
    depth = np.frombuffer(depth, np.float32)
    # image, resolution = sim.getVisionSensorImg(cam_handler)

    # image = np.frombuffer(image, np.uint8)
    # image.resize([resolution[0], resolution[1], 3])
    depth.resize([resolution[0], resolution[1]])

    pcd = read_point_cloud(sim, cam_handler, depth)

    if blank_cam_pcd is not None:
        origin_handler = sim.getObject('/origin')

        pcd = remove_blank_from_pcd(pcd, blank_cam_pcd)

        origin_from_cam_pos = sim.getObjectPosition(origin_handler, cam_handler)
        origin_from_cam_pos[0] *= -1
        pcd = pcd.translate(np.array(origin_from_cam_pos) * -1)

        cam_ori_from_origin = sim.getObjectOrientation(cam_handler, origin_handler)
        cam_ori_from_origin = sim.alphaBetaGammaToYawPitchRoll(*cam_ori_from_origin)
        cam_ori = Rotation.from_euler('zyx', np.array(cam_ori_from_origin)).inv()
        pcd = pcd.rotate(cam_ori.as_matrix(), center=(0, 0, 0))

    return pcd


def remove_blank_from_pcd(pcd, blank_pcd):
    dist = np.asarray(pcd.compute_point_cloud_distance(blank_pcd))
    return pcd.select_by_index(np.where(dist > 0.0000000001)[0])


def pcd_as_np(pcd):
    point_cloud = np.asarray(pcd.points)
    point_cloud[:, 0] *= -1
    pcd.estimate_normals()
    normals = np.asarray(pcd.normals)
    point_cloud_with_normals = np.concatenate((point_cloud, normals), axis=1)
    return point_cloud_with_normals


def normal_correction(sim, camera_name, pcn):
    cam_handler = sim.getObject(camera_name)
    origin_handler = sim.getObject('/origin')
    cam_from_origin_pos = sim.getObjectPosition(cam_handler, origin_handler)

    pcn = np.array(pcn)

    vectors = pcn[:, :3] - cam_from_origin_pos

    vectors_norm = np.linalg.norm(vectors, axis=1)[:, np.newaxis]
    vectors_normalized = vectors / vectors_norm

    dot_products = np.sum(pcn[:, 3:] * vectors_normalized, axis=1)
    dot_products_negative = np.sum(pcn[:, 3:] * -vectors_normalized, axis=1)

    factors = np.where(dot_products > dot_products_negative, -1, 1)

    return pcn[:, 3:] * factors[:, np.newaxis]


def get_and_save_pcd(sim, camera_name, blank_cam_pcd):
    pcd_cam = get_pcd_from_cam(sim, camera_name, blank_cam_pcd)
    pcn_cam = pcd_as_np(pcd_cam)
    pcn_cam[:, 3:] = normal_correction(sim, camera_name, pcn_cam)
    return pcn_cam


def wait_until_object_stops(sim, object_handler):
    i = 0
    while i < 200:
        time.sleep(0.1)
        linear_velocity, angular_velocity = sim.getObjectVelocity(object_handler)
        if all(v < 0.0001 for v in linear_velocity + angular_velocity):
            break
        i += 1
    time.sleep(0.5)


def get_pcn_from_cameras(sim, cameras, env='/env'):
    env_handler = sim.getObject(env)

    object_handler = None
    while True:
        object_handler = sim.getObjectChild(env_handler, 0)
        if object_handler is None:
            time.sleep(1.0)
        else:
            break

    prop = sim.getModelProperty(object_handler)
    object_property = prop
    prop &= ~sim.modelproperty_not_dynamic
    prop &= ~sim.modelproperty_not_respondable
    prop |= sim.modelproperty_not_visible
    sim.setModelProperty(object_handler, prop)

    sim.startSimulation()

    blank_point_clouds = []

    def worker(camera_name):
        client = RemoteAPIClient()
        sim = client.getObject('sim')
        time.sleep(0.5)
        return get_pcd_from_cam(sim, camera_name, None)

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(worker, cameras))

    blank_point_clouds.extend(results)

    prop = sim.getModelProperty(object_handler)
    prop &= ~sim.modelproperty_not_dynamic
    prop &= ~sim.modelproperty_not_respondable
    prop &= ~sim.modelproperty_not_visible
    sim.setModelProperty(object_handler, prop)

    point_clouds = []

    def worker(camera_name, blank_point_cloud):
        client = RemoteAPIClient()
        sim = client.require('sim')
        wait_until_object_stops(sim, object_handler)
        return get_and_save_pcd(sim, camera_name, blank_point_cloud)

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(worker, cameras, blank_point_clouds))

    point_clouds.extend(results)

    sim.stopSimulation()

    sim.setModelProperty(object_handler, object_property)

    return np.concatenate(point_clouds, axis=0)


def load_obj(file_path, desired_height, max_width, load_mode):
    vertices = []
    faces = []
    min_coords = [float('inf')] * 3
    max_coords = [float('-inf')] * 3

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                _, x, y, z = line.split()
                x, y, z = float(x), float(y), float(z)
                if load_mode == 0:
                    coords = [x, y, -z]
                else:
                    coords = [-z, y, x]
                vertices.append(coords)
                min_coords = [min(min_coords[i], coords[i]) for i in range(3)]
                max_coords = [max(max_coords[i], coords[i]) for i in range(3)]
            elif line.startswith('f '):
                face = [int(idx.split('/')[0]) - 1 for idx in line.split()[1:]]
                faces.extend(face)

    if max_coords[2] - min_coords[2] == 0:
        return None

    scale_factor_height = desired_height / (max_coords[2] - min_coords[2])

    # Calculate width and scale factor based on max width
    current_width = max(abs(max_coords[1] - min_coords[1]), abs(max_coords[0] - min_coords[0]))
    scale_factor_width = max_width / current_width if (current_width * scale_factor_height) > max_width else 1000

    # Use the smaller scale factor to maintain proportions
    scale_factor = min(scale_factor_height, scale_factor_width)

    scaled_vertices = [[coord * scale_factor for coord in vertex] for vertex in vertices]
    scaled_min_max = [[min(vertex[i] for vertex in scaled_vertices), max(vertex[i] for vertex in scaled_vertices)] for i in range(3)]

    if any(scaled_max - scaled_min < 0.003 for scaled_min, scaled_max in scaled_min_max):
        return None

    flattened_scaled_vertices = [coord for vertex in scaled_vertices for coord in vertex]
    return flattened_scaled_vertices, faces


def get_obj_files(directory):
    return [f for f in os.listdir(directory) if f.endswith('.obj')]


def add_obj_file_to_sim(file_id, file_name, directory, sim, load_mode):
    file_path = os.path.join(directory, file_name)
    obj = load_obj(file_path, 0.1, 0.25, load_mode)

    if obj is None:
        return None

    vertices, indices = obj

    return create_simulation_shapes(sim, vertices, indices, file_id)


def create_simulation_shapes(sim, vertices, indices, file_id):
    env_handle = sim.getObjectHandle('env')
    shape_handles = []
    tries = 0
    while tries < 100:
        try:
            tries += 1
            h = sim.createShape(2, 20 * math.pi / 180, vertices, indices)

            min_z = min(vertices[2::3])
            current_pos = sim.getObjectPosition(h, -1)
            sim.setObjectPosition(h, [current_pos[0], current_pos[1], current_pos[2] - min_z + 0.02], -1)

            sim.setShapeColor(h, "", sim.colorcomponent_ambient, [0.5, 0.5, 0.5])
            sim.setObjectInt32Param(h, sim.shapeintparam_respondable, 1)
            sim.setObjectInt32Param(h, sim.shapeintparam_static, 0)
            sim.resetDynamicObject(h)

            sim.setObjectAlias(h, "object_" + str(file_id))
            sim.setObjectParent(h, env_handle, True)

            shape_handles.append(h)
            break
        except Exception as e:
            print(f"Failed to load: {file_id} due to {e}")
            cleanup_shapes(sim, shape_handles)
            time.sleep(0.1)
    return shape_handles


def run_simulation(sim, shape_handles):
    time.sleep(0.5)
    sim.startSimulation()
    time.sleep(2.0)
    sim.stopSimulation()
    time.sleep(0.5)
    cleanup_shapes(sim, shape_handles)


def cleanup_shapes(sim, shape_handles):
    try:
        for h in shape_handles:
            sim.removeObject(h)
    except Exception as e:
        print(f"Error during cleanup: {e}")


def setGripperData(sim, gripperHandle, open=True, velocity=None, force=None):
    if velocity is None:
        velocity = 0.5
    if force is None:
        force = 100
    if not open:
        velocity = -velocity

    dat = {'velocity': velocity, 'force': force}
    sim.writeCustomDataBlock(gripperHandle, 'activity', sim.packTable(dat))


def moveToPoseCallback(sim, simIK, q, velocity, accel, auxData):
    sim.setObjectPose(auxData['target'], sim.handle_world, q)
    simIK.handleGroup(auxData['ikEnv'], auxData['ikGroup'], {'syncWorlds': True})


def moveToPose_viaIK(sim, simIK, maxVelocity, maxAcceleration, maxJerk, targetQ, auxData):
    currentQ = sim.getObjectPose(auxData['tip'], sim.handle_world)
    callback = lambda q, velocity, accel, auxData: moveToPoseCallback(sim, simIK, q, velocity, accel, auxData)
    return sim.moveToPose(-1, currentQ, maxVelocity, maxAcceleration, maxJerk, targetQ, callback, auxData)


client = RemoteAPIClient()
sim = client.require('sim')


def moveToConfigCallback(config, velocity, accel, auxData):
    global sim
    for i, jh in enumerate(auxData['joints']):
        if sim.isDynamicallyEnabled(jh):
            sim.setJointTargetPosition(jh, config[i])
        else:
            sim.setJointPosition(jh, config[i])


def moveToConfig_viaFK(sim, maxVelocity, maxAcceleration, maxJerk, goalConfig, auxData):
    startConfig = [sim.getJointPosition(joint) for joint in auxData['joints']]
    sim.moveToConfig(-1, startConfig, None, None, maxVelocity,
                     maxAcceleration, maxJerk, goalConfig, None, moveToConfigCallback, auxData, None)


def vector_to_quaternion(v, perp_vector):
    v_norm = v / np.linalg.norm(v)

    reference_vector = np.array([-1, 0, 0])
    reference_vector = reference_vector / np.linalg.norm(reference_vector)

    if np.allclose(v_norm, -reference_vector):
        return np.array([0, 0, 0, -1])

    rotation_axis = cross_product(reference_vector, v_norm)

    theta = np.arccos(np.dot(reference_vector, v_norm))

    q1 = Rotation.from_rotvec(theta * rotation_axis)

    if perp_vector is not None:
        perp_vector /= np.linalg.norm(perp_vector)

        pitch_angle = np.arcsin(np.dot(perp_vector, reference_vector))

        q2 = Rotation.from_rotvec(pitch_angle * reference_vector)

        return (q1 * q2).as_quat()
    else:
        return q1.as_quat()
