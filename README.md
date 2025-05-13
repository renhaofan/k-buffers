# K-Buffers: A Plug-in Method for Enhancing Neural Fields with Multiple Buffers [IJCAI2025]

# Installation

## Install dependencies (NPF)

```
[PASS]
Ubuntu 20.04.1 LTS
NVIDIA GeForce RTX 3090
Python 3.8.18 (default, Sep 11 2023, 13:40:15)
[GCC 11.2.0] :: Anaconda, Inc. on linux
Pytorch version 1.13.1
Pytorch CUDA 11.7
Pytorch cudnn 8500


conda create -n bpcr_pro python=3.8 
conda activate bpcr_pro

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

python setup.py install

pip install Ninja
pip install ./shencoder
pip install ./gridencoder


pip install -i https://pypi.tuna.tsinghua.edu.cn/simple matplotlib
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple lpips
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple piqa
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorboard
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ConfigArgParse
pip install torchsummary

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python
conda install -c conda-forge openexr-python

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple open3d

# download vgg.pth automatially
# wget https://github.com/richzhang/PerceptualSimilarity/raw/master/lpips/weights/v0.1/vgg.pth /home/XXX/.cache/torch/hub/checkpoints/vgg.pth

# manually install vgg.pth if network unreachable
# wget https://gitee.com/renhaofan/nomachine/raw/master/vgg.pth
```



``` 
# Docker image [PASS]
# pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# setup.py 
# change `extra_compile_args = {"cxx": ["-std=c++14"]}`
# to `extra_compile_args = {"cxx": ["-std=c++17"]}`


pip install -i https://pypi.tuna.tsinghua.edu.cn/simple matplotlib
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple lpips
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple piqa
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorboard
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ConfigArgParse
pip install torchsummary

# if error meets
# OSError: libGL.so.1: cannot open shared object file: No such file or directory 
# apt-get install -y libglib2.0-0 libx11-6 libgl1-mesa-glx

apt-get install libglib2.0-0 
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python

apt install libx11-6
apt install libgl1-mesa-glx
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple open3d
conda install -c conda-forge openexr-python

```

``` 
[PASS]
Ubuntu 22.04.4 LTS
NVIDIA GeForce RTX 4090
Python 3.8.19 (default, Mar 20 2024, 19:58:24)
[GCC 11.2.0] :: Anaconda, Inc. on linux

Pytorch version 2.2.0
Pytorch CUDA 11.8
Pytorch cudnn 8700
```


## Install dependencies (3DGS)

```
[PASS]
Ubuntu 22.04.4 LTS
NVIDIA GeForce RTX 4090
Python 3.12.2 | packaged by conda-forge | (main, Feb 16 2024, 20:50:58) [GCC 12.3.0] on linux

Pytorch version 2.4.0
Pytorch CUDA 11.8
Pytorch cudnn 90100
gsplat_version: 1.2.0
```

## Data preparation
Follow the [RadianceMapping](https://github.com/seanywang0408/RadianceMapping) or just download from [baidudisk](https://pan.baidu.com/s/1TqoPuVavzd2TbDAkkRJLcA?pwd=pa1r)
```
├── data
    ├── nerf_synthetic
    ├── dtu
    |   ├── dtu_110
    │   │   │── cams_1
    │   │   │── image
    │   │   │── mask
    │   │   │── npbgpp.ply
    |   ├── dtu_114
    |   ├── dtu_118
    ├── scannet
    │   │   │──0000
    |   │   │   │──color_select
    |   │   │   │──pose_select
    |   │   │   |──intrinsic
    |   │   │   |──00.ply
    │   │   │──0043
    │   │   │──0045
    ├── pc
    |   ├── pointnerf
    │   │   │── chair_pointnerf.ply
    │   │   │── drums_pointnerf.ply  
```

## Training
```SHELL
# NPF_BASELINE: bpcr or frepcr
cd npf
bash dev_scripts/run_all.sh <GPU_ID> <NPF_BASELINE>

cd gs
bash dev_scripts/run_bpcr_all.sh <GPU_ID>
```


## Known BUGs
In certain case, running `main_fast_k.py` with `arg.version == 1` may trigger the CUDA error. This issue is caused by `gridencoder`, and the default version is currently set to 3.
```
Traceback (most recent call last):
  File "main_fast_k.py", line 188, in <module>
    loss.backward()
  File "/home/renhaofan/.conda/envs/bpcr_pro/lib/python3.8/site-packages/torch/_tensor.py", line 522, in backward
    torch.autograd.backward(
  File "/home/renhaofan/.conda/envs/bpcr_pro/lib/python3.8/site-packages/torch/autograd/__init__.py", line 266, in backward
    Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
RuntimeError: CUDA error: invalid configuration argument
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```

```
/opt/conda/conda-bld/pytorch_1704987290659/work/aten/src/ATen/native/cuda/IndexKernel.cu:92: operator(): block: [59829,0,0], thread: [120,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/opt/conda/conda-bld/pytorch_1704987290659/work/aten/src/ATen/native/cuda/IndexKernel.cu:92: operator(): block: [59829,0,0], thread: [121,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/opt/conda/conda-bld/pytorch_1704987290659/work/aten/src/ATen/native/cuda/IndexKernel.cu:92: operator(): block: [59829,0,0], thread: [122,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/opt/conda/conda-bld/pytorch_1704987290659/work/aten/src/ATen/native/cuda/IndexKernel.cu:92: operator(): block: [59829,0,0], thread: [123,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/opt/conda/conda-bld/pytorch_1704987290659/work/aten/src/ATen/native/cuda/IndexKernel.cu:92: operator(): block: [59829,0,0], thread: [124,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/opt/conda/conda-bld/pytorch_1704987290659/work/aten/src/ATen/native/cuda/IndexKernel.cu:92: operator(): block: [59829,0,0], thread: [125,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/opt/conda/conda-bld/pytorch_1704987290659/work/aten/src/ATen/native/cuda/IndexKernel.cu:92: operator(): block: [59829,0,0], thread: [126,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/opt/conda/conda-bld/pytorch_1704987290659/work/aten/src/ATen/native/cuda/IndexKernel.cu:92: operator(): block: [59829,0,0], thread: [127,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
Traceback (most recent call last):
  File "main_fast_k.py", line 160, in <module>
    output = renderer(zbuf, idbuf, ray, img_gt, mask_gt,
  File "/home/renhaofan/.conda/envs/bpcr_pro/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/renhaofan/.conda/envs/bpcr_pro/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/renhaofan/bpcr_kbuffer/model_k/renderer_k.py", line 169, in forward
    feature_map[unique_pixel_mask] = feature[inverse_indices, :]
RuntimeError: CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

```


The results in NeRF-Synthetic of 3DGS we reported in paper is from 3DGS original implementation. If you run `simple_trainer_bpcr_blender_BUG.py`  the results are full of strange floaters, which is lower PSNR than original implementation.
```
============PSNR===========
chair 35.88718032836914
drums 25.902305603027344
ficus 30.904375076293945
hotdog 37.30323028564453
lego 34.71310806274414
materials 29.690725326538086
mic 34.03371047973633
ship 30.37014389038086
--------------------------
Average: 32.350
```

## Updates

* 2025-05-09 Release the code.