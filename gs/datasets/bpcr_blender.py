import os
from typing import Any, Dict, List, Optional

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from pycolmap import SceneManager
from plyfile import PlyData, PlyElement
from PIL import Image
import torchvision
import json
import math

def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

from .normalize import (
    align_principle_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)


class Parser:
    """bpcr blender parser."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
    ):

        def read_from_json(path, extension='.png'):
            image_names = []
            image_paths = []
            camtoworlds = []

            with open(os.path.join(path, "transforms_train.json")) as json_file:
                contents = json.load(json_file)
                fovx = contents["camera_angle_x"]

                focal = 0.5 * 800 / np.tan(0.5 * fovx) 
                intrinsic = np.array([[focal, 0, 800 / 2], [0, focal, 800 / 2], [0, 0, 1]])

                frames = contents["frames"]
                for idx, frame in enumerate(frames):
                    cam_name = os.path.join(path, frame["file_path"] + extension)

                    # NeRF 'transform_matrix' is a camera-to-world transform
                    c2w = np.array(frame["transform_matrix"])
                    # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
                    c2w[:3, 1:3] *= -1

                    camtoworlds.append(c2w)
                    # get the world-to-camera transform and set R, T
                    # w2c = np.linalg.inv(c2w)

                    image_names.append(frame["file_path"].split('/')[-1] + extension)
                    image_paths.append(cam_name)

            with open(os.path.join(path, "transforms_test.json")) as json_file:
                contents = json.load(json_file)
                fovx = contents["camera_angle_x"]

                focal = 0.5 * 800 / np.tan(0.5 * fovx) 
                intrinsic = np.array([[focal, 0, 800 / 2], [0, focal, 800 / 2], [0, 0, 1]])

                frames = contents["frames"]
                for idx, frame in enumerate(frames):
                    cam_name = os.path.join(path, frame["file_path"] + extension)

                    # NeRF 'transform_matrix' is a camera-to-world transform
                    c2w = np.array(frame["transform_matrix"])
                    # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
                    c2w[:3, 1:3] *= -1

                    camtoworlds.append(c2w)
                    # get the world-to-camera transform and set R, T
                    # w2c = np.linalg.inv(c2w)

                    image_names.append(frame["file_path"].split('/')[-1] + extension)
                    image_paths.append(cam_name)

            return image_names, image_paths, camtoworlds, intrinsic


        self.data_dir = data_dir
        scene_id = data_dir.split('/')[-1]

        # 100 train, 200 test
        image_names, image_paths, camtoworlds, intrinsic = read_from_json(data_dir)



        num_seq = 300

        if factor != 1:
            print("Not support")
            assert False

        # TODO not sure whether needed
        # if normalize:
        #     print("Not support")
        #     assert False

        self.factor = factor
        self.normalize = normalize

        img_size = (800, 800) # (W, H)
        camera_ids = [1]*num_seq
        Ks_dict = dict()
        imsize_dict = dict()  # width, height

        Ks_dict[1] = intrinsic
        imsize_dict[1] = img_size
        camtoworlds = np.stack(camtoworlds, axis=0)
        
        #plyfile.PlyHeaderParseError: line 12: expected one of {}
        # ply_path = os.path.join("/home/renhaofan/dataset/pc/pointnerf", scene_id+'_pointnerf.ply')
        # print(ply_path)
        # plydata = PlyData.read(ply_path)
        # vertices = plydata['vertex']
        # points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
        # points_rgb = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T
        
        points = np.load(os.path.join("/home/renhaofan/dataset/pc/pointnerf", scene_id+'_xyz.npy'))
        points_rgb = np.load(os.path.join("/home/renhaofan/dataset/pc/pointnerf", scene_id+'_rgb.npy'))

        points_rgb *= 255
        points_rgb = points_rgb.astype(np.uint8)
        
        
        # NOTE:open3d not support python3.12 
        # every_k_points=4
        # o3d_pcd = o3d.io.read_point_cloud(ply_path)
        # if every_k_points > 1:
            # o3d_pcd = o3d.geometry.PointCloud.uniform_down_sample(o3d_pcd, every_k_points)
        # code below to generate xyz.npy and rgb.npy
        """
        pc_dir="/home/renhaofan/dataset/pc/pointnerf"
        scenes=["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]
        #scenes=["chair"]

        for scene in scenes:
            ply_path=os.path.join(pc_dir, scene+"_pointnerf.ply")
            # error
            #from plyfile import PlyData, PlyElement
            #plydata = PlyData.read(ply_path)
            pcd_load = o3d.io.read_point_cloud(ply_path)
            xyz_load = np.asarray(pcd_load.points)
            rgb_load = np.asarray(pcd_load.colors)
            np.save(os.path.join(pc_dir, scene+'_xyz.npy'), xyz_load)
            np.save(os.path.join(pc_dir, scene+'_rgb.npy'), rgb_load)
        """


        # Normalize the world space.
        if normalize:
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            points = transform_points(T1, points)

            T2 = align_principle_axes(points)
            camtoworlds = transform_cameras(T2, camtoworlds)
            points = transform_points(T2, points)

            transform = T2 @ T1
        else:
            transform = np.eye(4)
 
        self.image_names = image_names  # List[str], (num_images,)                                 #'_DSC8908.JPG'
        self.image_paths = image_paths  # List[str], (num_images,)                                 #'data/360_v2/treehill/images_4/_DSC8874.JPG'
        self.camtoworlds = camtoworlds  # np.ndarray, (num_images, 4, 4)                           # dtype float64
        self.camera_ids = camera_ids  # List[int], (num_images,)                                   # 全1
        self.Ks_dict = Ks_dict  # Dict of camera_id -> K                                           # {1: array..} dtype float64

#         Ks_dict[1]
# array([[1.05479268e+03, 0.00000000e+00, 6.33500000e+02],
#        [0.00000000e+00, 1.05140057e+03, 4.15750000e+02],
#        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

        # self.params_dict = params_dict  # Dict of camera_id -> params                              # {1: array([], dtype=float32)}
        self.imsize_dict = imsize_dict  # Dict of camera_id -> (width, height)                     # {1: (1267, 831)}
        self.points = points # np.ndarray, (num_points, 3)                                        # dtype 64
        # self.points_err = points_err  # np.ndarray, (num_points,)                                  # dtype32
        self.points_rgb = points_rgb  # np.ndarray, (num_points, 3)                                # uint8 0-255
        # self.point_indices = point_indices  # Dict[str, np.ndarray], image_name -> [M,]
        self.transform = transform  # np.ndarray, (4, 4)

        # size of the scene measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)

class Dataset:
    """A simple dataset class."""

    def __init__(
        self,
        parser: Parser,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
        white_background: bool = True
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        self.white_background = white_background

        train_num=100
        test_num=200

        if split == "train":
            self.indices = np.arange(train_num)
        else:
            self.indices = np.arange(train_num, test_num+train_num)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]

        if self.white_background:
            image = Image.open(self.parser.image_paths[index]).convert('RGBA')
            r, g, b, a = image.split()
            mask = np.array(a)
            white_background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            final_image = Image.alpha_composite(white_background, image)
            image = np.array(final_image.convert("RGB"))
        else:
            image = imageio.imread(self.parser.image_paths[index])[..., :3]
            image_mask = Image.open(self.parser.image_paths[index]).convert('RGBA')
            r, g, b, a = image_mask.split()
            mask = np.array(a)

        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()  # undistorted K
        camtoworlds = self.parser.camtoworlds[index]


        if self.patch_size is not None:
            # Random crop.
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            K[0, 2] -= x
            K[1, 2] -= y

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,  # the index of the image in the dataset
            "mask": torch.from_numpy(mask).float(),
        }

        return data


if __name__ == "__main__":
    import argparse
    import imageio.v2 as imageio
    import tqdm

    parser = argparse.ArgumentParser()
    # parser.add_argument("--data_dir", type=str, default="/home/renhaofan/dataset/dtu/dtu_114")
    parser.add_argument("--data_dir", type=str, default="/home/renhaofan/gs_example/examples/data/nerf_synthetic/drums")
    parser.add_argument("--factor", type=int, default=1)
    args = parser.parse_args()

    # Parse COLMAP data.
    parser = Parser(
        data_dir=args.data_dir, factor=args.factor, normalize=False, test_every=8
    )
    dataset = Dataset(parser, split="train")
    print(f"Dataset: {len(dataset)} images.")

    writer = imageio.get_writer("results/mask.mp4", fps=30)
    for data in tqdm.tqdm(dataset):
        image = data["mask"].numpy().astype(np.uint8)
        writer.append_data(image)
    writer.close()