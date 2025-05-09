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

def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths

from .normalize import (
    align_principle_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)



class Parser:
    """bpcr scannet parser."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
    ):
        
        def CameraRead(dir, line_num):
            camera_para = []
            f = open(dir)
            for i in range(line_num):
                line = f.readline()
                tmp = line.split()
                camera_para.append(tmp)
            camera_para = np.array(camera_para, dtype=np.float32)
            f.close()
            return camera_para
        
        # /home/renhaofan/dataset/scannet/0000
        self.data_dir = data_dir

        scene_id = data_dir.split('/')[-1]
        if scene_id == "0000":
            self.test_indices = np.array([10, 38, 67, 96, 125, 153, 182, 211, 240, 269])
        elif scene_id == "0043":
            self.test_indices = np.array([10, 18, 27, 36, 45, 54, 63, 72, 81, 90])
        elif scene_id == "0045":
            self.test_indices = np.array([10, 19, 29, 39, 49, 58, 68, 78, 88, 98])
        else:
            print(scene_id)
            import sys
            sys.exit(1)

        # if scene_id == "0000":
        #     test_idx = ["10.jpg", "38.jpg", "67.jpg", "96.jpg", "125.jpg", 
        #                 "153.jpg", "182.jpg", "211.jpg", "240.jpg", "269.jpg"]
        # elif scene_id == "0043":
        #     test_idx = ["10.jpg", "18.jpg", "27.jpg", "36.jpg", "45.jpg", 
        #                 "54.jpg", "63.jpg", "72.jpg", "81.jpg", "90.jpg"]
        # elif scene_id == "0045":
        #     test_idx = ["10.jpg", "19.jpg", "29.jpg", "39.jpg", "49.jpg", 
        #                 "58.jpg", "68.jpg", "78.jpg", "88.jpg", "98.jpg"]
        # else:
        #     assert False


        if factor != 1:
            print("Not support")
            assert False

        # TODO not sure whether needed
        # if normalize:
        #     print("Not support")
        #     assert False


        self.factor = factor
        self.normalize = normalize
        
        img_size = (1200, 960) # (W, H)
        scale1_c = (1296 - 1) / (img_size[0] - 1)
        scale2_c = (968 - 1) / (img_size[1] - 1)
        intrinsic_path = os.path.join(data_dir, 'intrinsic', 'intrinsic_color.txt')
        intrinsic = CameraRead(intrinsic_path, 4)[:3,:3]
        intrinsic[0:1, :] = intrinsic[0:1, :] / scale1_c
        intrinsic[1:2, :] = intrinsic[1:2, :] / scale2_c

        img_path = os.path.join(data_dir, 'color_select')
        pose_path = os.path.join(data_dir, 'pose_select')
        num_seq = len(os.listdir(pose_path))
        
        image_names = []
        image_paths = []
        camtoworlds = []
        
        camera_ids = [1]*num_seq
        Ks_dict = dict()
        params_dict = dict()
        imsize_dict = dict()  # width, height
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)

        Ks_dict[1] = intrinsic
        imsize_dict[1] = img_size

        for idx in range(num_seq):
            image_names.append(str(idx)+'.jpg')
            image_paths.append(os.path.join(img_path, str(idx)+'.jpg'))
            c2w = CameraRead(os.path.join(pose_path, str(idx) + '.txt'), 4)
            camtoworlds.append(c2w)
            # w2c = np.linalg.inv(c2w)

        camtoworlds = np.stack(camtoworlds, axis=0)

        ply_path = os.path.join(self.data_dir, scene_id[-2:]+'_npbgx4.ply')
        plydata = PlyData.read(ply_path)
        vertices = plydata['vertex']
        points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
        points_rgb = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T

        # open3d not support python3.12 
        # every_k_points=4
        # o3d_pcd = o3d.io.read_point_cloud(ply_path)
        # if every_k_points > 1:
            # o3d_pcd = o3d.geometry.PointCloud.uniform_down_sample(o3d_pcd, every_k_points)


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

        self.params_dict = params_dict  # Dict of camera_id -> params                              # {1: array([], dtype=float32)}
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
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        indices = np.arange(len(self.parser.image_names))
        
        if split == "train":
            self.indices = np.setdiff1d(indices, parser.test_indices)
            # self.indices = indices[indices % self.parser.test_every != 0]
        else:
            self.indices = parser.test_indices
            # self.indices = indices[indices % self.parser.test_every == 0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        image = imageio.imread(self.parser.image_paths[index])[..., :3]
        
        # img_size = (1200, 960) # (W, H)
        # convert (968, 1296, 3) ===> (img_size, 3) as bpcr
        image_pil = Image.fromarray(image).resize((1200, 960), Image.LANCZOS)
        image = np.asarray(image_pil).copy()

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
        }

        return data


if __name__ == "__main__":
    import argparse

    import imageio.v2 as imageio
    import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/renhaofan/dataset/scannet/0000")
    parser.add_argument("--factor", type=int, default=1)
    args = parser.parse_args()

    # Parse COLMAP data.
    parser = Parser(
        data_dir=args.data_dir, factor=args.factor, normalize=False, test_every=8
    )
    dataset = Dataset(parser, split="train")
    print(f"Dataset: {len(dataset)} images.")

    writer = imageio.get_writer("results/points.mp4", fps=30)
    for data in tqdm.tqdm(dataset):
        image = data["image"].numpy().astype(np.uint8)
        writer.append_data(image)
    writer.close()
