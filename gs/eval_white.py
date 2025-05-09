import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

output_dir="results/bpcr_blender_20240808011842/chair"
mask_path="data/nerf_synthetic/chair/val"


folder_path=os.path.join(output_dir, "renders")

device = "cuda"
metrics = {"psnr": [], "ssim": [], "lpips": []}
ssim_m = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
psnr_m = PeakSignalNoiseRatio(data_range=1.0).to(device)
lpips_m = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)


            
for filename in sorted(os.listdir(folder_path)):
    if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            # image = Image.open(self.parser.image_paths[index]).convert('RGBA')
            # white_background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            # final_image = Image.alpha_composite(white_background, image)
            # image = np.array(final_image.convert("RGB"))
        
        image_path = os.path.join(folder_path, filename)
        
        image = Image.open(image_path)
        im_data = np.array(image.convert("RGB"))

        image = Image.open().convert('RGBA')
        r, g, b, a = image.split()
        mask = np.array(a)

        height, width, _ = im_data.shape
        mid_point = width // 2
        
        # 竖直分割成两张图片
        gt = im_data[:, :mid_point]
        predict = im_data[:, mid_point:] * mask

        pixels = torch.from_numpy(gt).to(device).permute(2, 0, 1).float()
        colors = torch.from_numpy(predict).to(device).permute(2, 0, 1).float()
        pixels = pixels.unsqueeze(0) / 255.0 # [1, 3, H, W]
        colors = colors.unsqueeze(0) / 255.0  # [1, 3, H, W]

        metrics["psnr"].append(psnr_m(colors, pixels))
        metrics["ssim"].append(ssim_m(colors, pixels))
        metrics["lpips"].append(lpips_m(colors, pixels))

psnr = torch.stack(metrics["psnr"]).mean()
ssim = torch.stack(metrics["ssim"]).mean()
lpips = torch.stack(metrics["lpips"]).mean()
print(
    f"PSNR: {psnr.item():.3f}, SSIM: {ssim.item():.4f}, LPIPS: {lpips.item():.3f} "
)