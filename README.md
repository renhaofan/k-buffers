# k-buffers
The code is available on the [alpha](https://github.com/renhaofan/k-buffers/tree/alpha) branch. There are still some redundant scripts and known bugs, but it can reproduce the results reported in the paper based on our tests.

| Dataset         | Method         | PSNR  | SSIM  | LPIPS |
|----------------|----------------|-------|-------|--------|
| **NeRF_Synthetic** | 3DGS (ours-1) | 33.62 | 0.969 | 0.019 |
|                | 3DGS (ours-2) | 33.63 | 0.969 | 0.019 |
|                | Difference     | 0.01  |       |       |
| **ScanNet**       | 3DGS (ours-1) | 25.55 | 0.785 | 0.406 |
|                | 3DGS (ours-2) | 25.37 | 0.784 | 0.405 |
|                | Difference     | -0.18 | -0.001| -0.001 |
| **DTU**           | 3DGS (ours-1) | 33.91 | 0.954 | 0.059 |
|                | 3DGS (ours-2) | 33.86 | 0.954 | 0.059 |
|                | Difference     | -0.05 |       |       |
| **360v2**         | 3DGS (ours-1) | 29.19 | 0.859 | 0.126 |
|                | 3DGS (ours-2) | 29.24 | 0.858 | 0.126 |
|                | Difference     | 0.05  | -0.001|        |

| Dataset           | Method         | PSNR  | SSIM  | LPIPS |
|------------------|----------------|-------|-------|--------|
| **Drums**         | BPCR (ours-1)  | 24.70 | 0.927 | 0.063 |
|                  | BPCR (ours-2)  | 24.70 | 0.927 | 0.063 |
|                  | Difference      |       |       |        |
|                  | FrePCR (ours-1)| 24.91 | 0.929 | 0.061 |
|                  | FrePCR (ours-2)| 24.90 | 0.930 | 0.059 |
|                  | Difference      | -0.01 | 0.001 | -0.002 |
| **ScanNet_0000**  | BPCR (ours-1)  | 24.58 | 0.742 | 0.415 |
|                  | BPCR (ours-2)  | 24.63 | 0.744 | 0.408 |
|                  | Difference      | 0.05  | 0.002 | -0.007 |
|                  | FrePCR (ours-1)| 24.32 | 0.742 | 0.417 |
|                  | FrePCR (ours-2)| 24.28 | 0.740 | 0.418 |
|                  | Difference      | -0.04 | -0.002| 0.001  |
