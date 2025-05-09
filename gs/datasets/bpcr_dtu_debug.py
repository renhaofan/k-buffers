import os
from typing import Any, Dict, List, Optional

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from pycolmap import SceneManager
from plyfile import PlyData, PlyElement
from PIL import Image

def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths

class Parser:
    """bpcr dtu parser."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
    ):

 
        def read_cam(path):
            data = []
            f = open(path)
            for i in range(10):
                line = f.readline()
                tmp = line.split()
                data.append(tmp)
            f.close()
            pose = np.array(data[1:5], dtype=np.float32)
            intrinsic = np.array(data[7:10], dtype=np.float32)
            return pose, intrinsic
        
        self.data_dir = data_dir

        scene_id = data_dir.split('/')[-1]
        
        # self.test_indices actually not used 
        # if scene_id == "dtu_110":
        #     self.test_indices = ['07', '12', '17', '22', '27', '32', '37', '42', '47']
        # elif scene_id == "dtu_114":
        #     self.test_indices = ['07', '12', '17', '22', '27', '32', '37', '42', '47']
        # elif scene_id == "dtu_118":
        #     self.test_indices = ['07', '12', '17', '22', '27', '32', '37', '42', '47']
        # else:
        #     print(scene_id)
        #     import sys
        #     sys.exit(1)



        num_seq = 64
        test_ = [7, 12, 17, 22, 27, 32, 37, 42, 47]
        self.train_indices = []
        self.test_indices = test_
        for i in range(num_seq):
            if i in test_:
                continue
            self.train_indices.append(i)
            # self.id_list.append(str(i).rjust(2,'0'))
        # else:
            # indices = test_
            # self.id_list = [str(i).rjust(2,'0') for i in test_]


        if factor != 1:
            print("Not support")
            assert False

        # TODO not sure whether needed
        # if normalize:
        #     print("Not support")
        #     assert False


        self.factor = factor
        self.normalize = normalize



        img_size = (1600, 1200) # (W, H)
        image_names = []
        image_paths = []
        mask_paths = []
        camtoworlds = []
        camera_ids = [1]*num_seq
        Ks_dict = dict()
        imsize_dict = dict()  # width, height

        for idx in range(num_seq):
            idx = str(idx).rjust(2, '0')
            image_names.append(f"0000{idx}.png")
            image_paths.append(os.path.join(data_dir, 'image', f"0000{idx}.png"))
            mask_paths.append(os.path.join(data_dir, 'mask', f"0{idx}.png"))

            cam_path = os.path.join(data_dir, 'cams_1', f"000000{idx}_cam.txt") 
            w2c, intrinsic = read_cam(cam_path)
            c2w = np.linalg.inv(w2c)
            camtoworlds.append(c2w)

        Ks_dict[1] = intrinsic
        imsize_dict[1] = img_size


        camtoworlds = np.stack(camtoworlds, axis=0)
        
        ply_path = os.path.join(self.data_dir, scene_id.replace('_', '')+'_npbgpp.ply')
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
        # if normalize:
        #     T1 = similarity_from_cameras(camtoworlds)
        #     camtoworlds = transform_cameras(T1, camtoworlds)
        #     points = transform_points(T1, points)

        #     T2 = align_principle_axes(points)
        #     camtoworlds = transform_cameras(T2, camtoworlds)
        #     points = transform_points(T2, points)

        #     transform = T2 @ T1
        # else:
        #     transform = np.eye(4)

        self.mask_paths = mask_paths
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
        # self.transform = transform  # np.ndarray, (4, 4)

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

        if split == "train":
            self.indices = np.asarray(parser.train_indices)
        else:
            self.indices = np.asarray(parser.test_indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        image = imageio.imread(self.parser.image_paths[index])[..., :3]
        mask = imageio.imread(self.parser.mask_paths[index])[..., :3]

        # mask[mask == 255] = 1
        # mask 背景的值是0，前景的值不是所有都是255，有244，8，2...
        # mask = mask / 255.0
        mask[mask != 0] = 1
        
        if self.white_background:
            image = image * mask + 255 * (1 - mask)
        else:
            image = image * mask

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
            "image_id": item,  # the index of the image in the dataset,
            "mask": torch.from_numpy(mask) # [H, W, 3] range in [0, 1]
        }

        return data


if __name__ == "__main__":
    import argparse

    import imageio.v2 as imageio
    import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/renhaofan/dataset/dtu/dtu_118")
    parser.add_argument("--factor", type=int, default=1)
    args = parser.parse_args()

    # Parse COLMAP data.
    parser = Parser(
        data_dir=args.data_dir, factor=args.factor, normalize=False, test_every=8
    )
    dataset = Dataset(parser, split="train", white_background=False)
    print(f"Dataset: {len(dataset)} images.")

    writer = imageio.get_writer("results/dtu118_black.mp4", fps=30)
    for data in tqdm.tqdm(dataset):
        image = data["image"].numpy().astype(np.uint8)
        writer.append_data(image)
    writer.close()
