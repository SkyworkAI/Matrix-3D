#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
from numbers import Number
from typing import *
import trimesh

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        
        if not os.path.exists(image_path) or "sky_mask" in image_path:
            print("skip =====", image_path)
            continue
        
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except Exception:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except Exception:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except Exception:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except Exception:
        pcd = None
    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readMultiScale(path, white_background,split, only_highres=False):
    cam_infos = []
    
    print("read split:", split)
    with open(os.path.join(path, 'metadata.json'), 'r') as fp:
        meta = json.load(fp)[split]
        
    meta = {k: np.array(meta[k]) for k in meta}
    
    # should now have ['pix2cam', 'cam2world', 'width', 'height'] in self.meta
    for idx, relative_path in enumerate(meta['file_path']):
        if only_highres and not relative_path.endswith("d0.png"):
            continue
        image_path = os.path.join(path, relative_path)
        image_name = Path(image_path).stem
        
        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = meta["cam2world"][idx]
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        image = Image.open(image_path)

        im_data = np.array(image.convert("RGBA"))

        bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

        norm_data = im_data / 255.0
        arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

        fovx = focal2fov(meta["focal"][idx], image.size[0])
        fovy = focal2fov(meta["focal"][idx], image.size[1])
        FovY = fovy 
        FovX = fovx

        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                        image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
    return cam_infos


def readMultiScaleNerfSyntheticInfo(path, white_background, eval, load_allres=False):
    print("Reading train from metadata.json")
    train_cam_infos = readMultiScale(path, white_background, "train", only_highres=(not load_allres))
    print("number of training images:", len(train_cam_infos))
    print("Reading test from metadata.json")
    test_cam_infos = readMultiScale(path, white_background, "test", only_highres=False)
    print("number of testing images:", len(test_cam_infos))
    if not eval:
        print("adding test cameras to training")
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except Exception:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info
# ok;
def readCamerasFromBlenderNPZ(path, transformsfile, white_background, extension=".png", prefix='mv_rgb'):
    cam_infos = []
    cam_path = os.path.join(path, transformsfile)
    cam_dict = np.load(cam_path)
    ex_cam_arr = cam_dict['arr_0'].reshape(-1, 4, 4)[:, :, :]
    with open(os.path.join(path, 'para.json')) as json_file:
        contents = json.load(json_file)
        focal_length = contents['focal_length_in_pixel']
    
    for idx in range(ex_cam_arr.shape[0]):
        cam_name = os.path.join(path, prefix, f'{idx:04d}' + extension)
        c2w = ex_cam_arr[idx]

        c2w[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)

        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        #R = (w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        image_path = os.path.join(path, cam_name)
        image_name = Path(cam_name).stem
        image = Image.open(image_path)
        fovy = focal2fov(focal_length, image.size[1])
        fovx = focal2fov(focal_length, image.size[0])
        FovY = fovy 
        FovX = fovx

        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                        image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def get_360_cameras_from_traincamera(train_cam_info, num_views_per_view=3):
    from scipy.spatial.transform import Rotation as Rot
    from scipy.spatial.transform import Slerp
    import math
    test_cam_info = []

    for i in range(len(train_cam_info) - 1):
        cam_info = train_cam_info[i]

        w2c0 = np.eye(4)
        w2c1 = np.eye(4)
        rot_0 = np.transpose(train_cam_info[i].R)
        rot_1 = np.transpose(train_cam_info[i + 1].R)
        FovY = train_cam_info[i].FovY
        FovX = train_cam_info[i].FovX
        image = train_cam_info[i].image
        image_path = train_cam_info[i].image_path
        width = train_cam_info[i].width
        height = train_cam_info[i].height

        quat_0 = Rot.from_matrix(rot_0).as_quat()
        quat_1 = Rot.from_matrix(rot_1).as_quat()
        trans_0 = train_cam_info[i].T
        trans_1 = train_cam_info[i + 1].T
        for j in np.linspace(0, 1, num_views_per_view):
            # ratio = np.sin(((j / num_views_per_view) - 0.5) * np.pi) * 0.5 + 0.5
            ratio = j
            key_times = [0, 1]
            slerp = Slerp(key_times, Rot.from_quat([quat_0, quat_1]))
            interp_quat = slerp(ratio)
            interp_rotation = interp_quat.as_matrix()

            interp_translation = (1 - j) * trans_0 + j * trans_1

            w2c = np.diag([1.0, 1.0, 1.0, 1.0])
            w2c = w2c.astype(np.float32)
            w2c[:3, :3] = interp_rotation
            w2c[:3, 3] = interp_translation

            R = np.transpose(w2c[:3, :3])
            T = w2c[:3, 3]
            image_name = train_cam_info[i].image_name + f'_{j}'
            idx = i * num_views_per_view + j
            insert_cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1])

            test_cam_info.append(insert_cam_info)
    
    return test_cam_info

# so if the data includes depth map, it reads it.
# otherwise use random initialization.

def sliding_window_1d(x: np.ndarray, window_size: int, stride: int, axis: int = -1):
    """
    Return x view of the input array with x sliding window of the given kernel size and stride.
    The sliding window is performed over the given axis, and the window dimension is append to the end of the output array's shape.

    Args:
        x (np.ndarray): input array with shape (..., axis_size, ...)
        kernel_size (int): size of the sliding window
        stride (int): stride of the sliding window
        axis (int): axis to perform sliding window over
    
    Returns:
        a_sliding (np.ndarray): view of the input array with shape (..., n_windows, ..., kernel_size), where n_windows = (axis_size - kernel_size + 1) // stride
    """
    assert x.shape[axis] >= window_size, f"kernel_size ({window_size}) is larger than axis_size ({x.shape[axis]})"
    axis = axis % x.ndim
    shape = (*x.shape[:axis], (x.shape[axis] - window_size + 1) // stride, *x.shape[axis + 1:], window_size)
    strides = (*x.strides[:axis], stride * x.strides[axis], *x.strides[axis + 1:], x.strides[axis])
    x_sliding = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return x_sliding


def sliding_window_nd(x: np.ndarray, window_size: Tuple[int,...], stride: Tuple[int,...], axis: Tuple[int,...]) -> np.ndarray:
    axis = [axis[i] % x.ndim for i in range(len(axis))]
    for i in range(len(axis)):
        x = sliding_window_1d(x, window_size[i], stride[i], axis[i])
    return x


def sliding_window_2d(x: np.ndarray, window_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]], axis: Tuple[int, int] = (-2, -1)) -> np.ndarray:
    if isinstance(window_size, int):
        window_size = (window_size, window_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    return sliding_window_nd(x, window_size, stride, axis)


def max_pool_1d(x: np.ndarray, kernel_size: int, stride: int, padding: int = 0, axis: int = -1):
    axis = axis % x.ndim
    if padding > 0:
        fill_value = np.nan if x.dtype.kind == 'f' else np.iinfo(x.dtype).min
        padding_arr = np.full((*x.shape[:axis], padding, *x.shape[axis + 1:]), fill_value=fill_value, dtype=x.dtype)
        x = np.concatenate([padding_arr, x, padding_arr], axis=axis)
    a_sliding = sliding_window_1d(x, kernel_size, stride, axis)
    max_pool = np.nanmax(a_sliding, axis=-1)
    return max_pool


def max_pool_nd(x: np.ndarray, kernel_size: Tuple[int,...], stride: Tuple[int,...], padding: Tuple[int,...], axis: Tuple[int,...]) -> np.ndarray:
    for i in range(len(axis)):
        x = max_pool_1d(x, kernel_size[i], stride[i], padding[i], axis[i])
    return x


def max_pool_2d(x: np.ndarray, kernel_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]], padding: Union[int, Tuple[int, int]], axis: Tuple[int, int] = (-2, -1)):
    if isinstance(kernel_size, Number):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, Number):
        stride = (stride, stride)
    if isinstance(padding, Number):
        padding = (padding, padding)
    axis = tuple(axis)
    return max_pool_nd(x, kernel_size, stride, padding, axis)

def depth_edge(depth: np.ndarray, atol: float = None, rtol: float = None, kernel_size: int = 3, mask: np.ndarray = None) -> np.ndarray:
    """
    Compute the edge mask from depth map. The edge is defined as the pixels whose neighbors have large difference in depth.
    
    Args:
        depth (np.ndarray): shape (..., height, width), linear depth map
        atol (float): absolute tolerance
        rtol (float): relative tolerance

    Returns:
        edge (np.ndarray): shape (..., height, width) of dtype torch.bool
    """
    if mask is None:
        diff = (max_pool_2d(depth, kernel_size, stride=1, padding=kernel_size // 2) + max_pool_2d(-depth, kernel_size, stride=1, padding=kernel_size // 2))
    else:
        diff = (max_pool_2d(np.where(mask, depth, -np.inf), kernel_size, stride=1, padding=kernel_size // 2) + max_pool_2d(np.where(mask, -depth, -np.inf), kernel_size, stride=1, padding=kernel_size // 2))

    edge = np.zeros_like(depth, dtype=bool)
    if atol is not None:
        edge |= diff > atol
    
    if rtol is not None:
        edge |= diff / depth > rtol
    return edge

def raw_depth_to_pointcloud(depthlist, K, Rt):
    B = len(depthlist)
    H, W = depthlist[0].shape[:2]
    xy_canvas = np.ones((H, W, 3), dtype=np.float32)
    xy_canvas[:,:,0] = np.linspace(0,W,num=W)[None,:]
    xy_canvas[:,:,1] = np.linspace(0,H,num=H)[:,None]
    world_coords_list = []
    for i in range(B):
        cur_depth_map = depthlist[i].astype(np.float32)[:,:,:1]
        
        cur_K = K[i]
        cur_Rt = Rt[i]
        cur_c2w = np.zeros((3,4), dtype=np.float32)
        cur_c2w[:,:3] = cur_Rt[:3,:3].transpose(1,0)
        cur_c2w[:,3:] = -cur_Rt[:3,:3].transpose(1,0)@cur_Rt[:3,3:]
        depth_vec = xy_canvas * cur_depth_map[:,:,:1]

        cur_K_inv = np.linalg.inv(cur_K).transpose(1,0)[None,None]

        world_coords = (depth_vec[...,None,:]@cur_K_inv)@cur_c2w[:,:3].transpose(1,0)[None,None] + cur_c2w[:,3][None,None,None]
        world_coords = world_coords[:,:,0,:]
        world_coords_list.append(world_coords)
        # ok then;
        # what else?

    return world_coords_list
# sample based on its area, and with color.
# even with color.
def generate_mesh_from_depth(depth0, K0, Rt0, threshold = 0.03, depth_threshold=201., rgb0=None):
    pts_wrd = raw_depth_to_pointcloud(depth0[None], K0[None], Rt0[None])[0]
    depth_edge_mask = ~depth_edge(depth0[:,:,0], rtol = threshold, kernel_size = 3, mask = None)
    depth_edge_mask = depth_edge_mask * (depth0[:,:,0] < depth_threshold)
    # get topology
    H, W = pts_wrd.shape[:2]
    pts_topology = np.zeros((H - 1, W - 1, 2, 3), dtype=np.int32)
    index_ = np.arange(H*W).astype(np.int32).reshape(H,W)
    pts_topology[:,:,0,0] = index_[:-1,:-1]
    pts_topology[:,:,0,2] = index_[:-1,1:]
    pts_topology[:,:,0,1] = index_[1:,:-1]

    pts_topology[:,:,1,0] = index_[:-1,1:]
    pts_topology[:,:,1,1] = index_[1:,:-1]
    pts_topology[:,:,1,2] = index_[1:,1:]

    d00 = depth_edge_mask[:-1,:-1]
    d01 = depth_edge_mask[:-1,1:]
    d10 = depth_edge_mask[1:,:-1]
    d11 = depth_edge_mask[1:,1:]
    
    mask0 = d00 * d10 * d01# ((d00 - d10)) * (np.abs(d00 - d01)<threshold) * (np.abs(d01 - d10)<threshold)
    mask1 = d11 * d01 * d10# (np.abs(d11 - d10)<threshold) * (np.abs(d11 - d01)<threshold) * (np.abs(d01 - d10)<threshold)
    
    pts_wrd = pts_wrd.reshape(-1,3)
    topo0 = pts_topology[:,:,0][mask0]
    topo1 = pts_topology[:,:,1][mask1]
    topo = np.concatenate([topo0, topo1], axis=0)
    # print("sdada, ",topo.shape, pts_topology.shape, mask0.shape, topo0.shape)
    if rgb0 is not None:
        rgb_wrd = rgb0.reshape(-1,3)
        mesh_trimesh = trimesh.Trimesh(vertices=pts_wrd, faces=topo, vertex_colors=rgb_wrd)
    else:
        mesh_trimesh = trimesh.Trimesh(vertices=pts_wrd, faces=topo)
    # mesh.export("../debug/sov.obj")
    
    return mesh_trimesh

def calculate_sampling_weight(mesh, Rt, weight_theta=1.25):
    face_area = mesh.area_faces
    vertex = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    face_vertices = vertex[faces]
    face_center = face_vertices.mean(axis=1)
    # print(face_center.shape, Rt.shape)
    fc_cam = face_center @ Rt[:3,:3].T + Rt[:3,3][None,:]
    fc_depth = fc_cam[:,2]
    # 保证最远处的面能有0.05的采样率
    min_depth, max_depth = fc_depth.min(), fc_depth.max()
    fc_depth = (fc_depth - min_depth)/(max_depth - min_depth)
    sampling_weight = np.exp(-fc_depth * weight_theta)
    return sampling_weight * face_area
    





def readBlenderNPZInfo(path, white_background, eval, extension=".png", interval=9, num_views_per_view=20, num_of_point_cloud=5000_000):
    print("Reading all camera params")
    all_cam_infos = readCamerasFromBlenderNPZ(path, "world_matrix.npz", white_background, extension)
    
    if eval:
        def get_idx_for_cam(cam_path):
            idx = int(cam_path.split('/')[-1][:-4])
            return idx
        train_cam_infos = [c for idx, c in enumerate(all_cam_infos) if get_idx_for_cam(c.image_path) % interval == 0]
        test_cam_infos = get_360_cameras_from_traincamera(all_cam_infos, num_views_per_view=num_views_per_view)
    else:
        train_cam_infos = all_cam_infos
        test_cam_infos = []
    print(f'train_cams={len(train_cam_infos)}, test_cams={len(test_cam_infos)}')

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    depth_exr_folder = os.path.join(path, "mv_depth")
    # HACK: add trigger here, later will add trigger somewhere else;
    use_mesh_sample = False
    if not os.path.exists(ply_path) and os.path.exists(depth_exr_folder):
        import torch, math
        from utils.graphics_utils import getWorld2View2
        def depths_to_points(world_view_transform, W, H, fov,fovY, depthmap):
            c2w = (world_view_transform.T).inverse()
            fx = W / (2 * math.tan(fov / 2.))
            fy = H / (2 * math.tan(fovY / 2.))
            #print(f"fovx fovy check: {fov} {fovY}")
            intrins = torch.tensor(
                [[fx, 0., W/2.],
                [0., fy, H/2.],
                [0., 0., 1.0]]
            ).float().cuda()
            grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float() + 0.5, torch.arange(H, device='cuda').float() + 0.5, indexing='xy')
            points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
            rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
            rays_o = c2w[:3,3]
            points = depthmap.reshape(-1, 1) * rays_d + rays_o
            return points
        # load ply from depth maps
        xyz = []
        rgbs = []

        for i in range(len(all_cam_infos)):
            cam_info = all_cam_infos[i]
            R = cam_info.R
            T = cam_info.T
            world2view = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).cuda()
            W = cam_info.width
            H = cam_info.height
            fov = cam_info.FovX
            fovy = cam_info.FovY
            depth_exr_path = cam_info.image_path.replace('mv_rgb','mv_depth').replace('png','exr')
            
            if os.path.exists(depth_exr_path):
                # print(depth_exr_path)
                depth_img = cv2.imread(depth_exr_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:,:,0]
                # 第一张图搞一个天空，后面就都不搞了。第一张图的天空部分，填上200的深度，后续都填300的深度。
                mask_ = depth_img < 500.

                # [512,512,3]
                depthmap = np.array(depth_img)
                depthmap[~mask_] = 500.
                depth_edge_mask = ~depth_edge(depthmap, atol=None, rtol=0.03, kernel_size=3, mask=None)
                mask_ = depth_edge_mask
                print(depthmap.shape, depth_edge_mask.shape, mask_.shape, i, len(all_cam_infos))
            depthmap = torch.from_numpy(depthmap).float().cuda()
            mask_cuda = torch.from_numpy(mask_).bool().cuda().reshape(-1) * torch.from_numpy(depth_edge_mask).bool().cuda().reshape(-1)
            if not use_mesh_sample:
                point = depths_to_points(world2view, W, H, fov,fovy, depthmap)[mask_cuda]
                
                img_arr = np.array(Image.open(cam_info.image_path)).astype(np.float32).reshape(-1, 3)[mask_.reshape(-1)]

                # m = 20_000
                # # m = 50_000
                # indices = np.random.choice(point.shape[0], m, replace=False)
                xyz.append(point.cpu().numpy())
                rgbs.append(img_arr)
            else:
                # ok...ok then.
                # 其实最重要的是颜色。
                print(f"mesh sample...{i}")
                fx = W / (2 * math.tan(fov / 2.))
                fy = H / (2 * math.tan(fovy / 2.))
                #print(f"fovx fovy check: {fov} {fovY}")
                K_ = np.array(
                    [[fx, 0., W/2.],
                    [0., fy, H/2.],
                    [0., 0., 1.0]]
                ,dtype=np.float32)

                w2c_ = ((world2view.T)).detach().cpu().numpy()

                depthmap_ = np.array(depth_img)
                sample_image = cv2.cvtColor(cv2.imread(cam_info.image_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
                cur_mesh = generate_mesh_from_depth(depthmap_[...,None], K_, w2c_, threshold = 0.03, rgb0 = sample_image)
                sampling_weight = calculate_sampling_weight(cur_mesh, w2c_)

                sample_pts,_,sample_color = trimesh.sample.sample_surface(cur_mesh, num_of_point_cloud//len(all_cam_infos)+1,face_weight=sampling_weight, sample_color=True)
                print(sample_pts.shape)
                xyz.append(sample_pts)
                rgbs.append(sample_color[:,:3].astype(np.float32))

        # 
        xyz = np.concatenate(xyz, axis=0)
        rgbs = np.concatenate(rgbs, axis=0)

        indices = np.random.choice(xyz.shape[0], min(num_of_point_cloud,xyz.shape[0]), replace=False)
        xyz = xyz[indices]
        rgbs = rgbs[indices]

        num_pts = xyz.shape[0]
        print(f"Generating point cloud from depth map (EXR) ({num_pts})...")
        pcd = BasicPointCloud(points=xyz, colors=rgbs, normals=np.zeros((num_pts, 3)))
        storePly(ply_path, xyz, rgbs)

    elif not os.path.exists(ply_path) and not os.path.exists(depth_exr_folder):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        # so i assume it is the problem.
        # We create random points inside the bounds of the synthetic Blender scenes
        # the finest of super earth
        # 它应该有什么样的性质呢？
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except Exception:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info
# 瞧一眼洗碗洗菜的家伙
# 虽然大概率是老爸来用辣
# 嘶。我突然意识到一个事情；似乎stereo matching也是从一个noise开始的，能不能用sd的这个手法做stereo matching呢？
sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Multi-scale": readMultiScaleNerfSyntheticInfo,
    "BlenderNPZ" : readBlenderNPZInfo,
}