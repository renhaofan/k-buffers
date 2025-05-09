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

## Known BUGs
In certain case, running `main_fast_k.py` with `arg.version == 1` may trigger the CUDA error. This issue is caused by `gridencoder`, and the default version is currently set to 4.
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

# simple_trainer_bpcr_blender_BUG.py
The results in NeRF-Synthetic of 3DGS we reported in paper is from 3DGS original implementation. If you use 

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