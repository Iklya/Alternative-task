import cv2
import numpy as np
import torch
import piq
import sewar
import matplotlib.pyplot as plt

def load_images(image1_path, image2_path):
    """Функция, необходимая для обработки двух изображений"""
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    return img1, img2

def calculate_npcr(image1, image2):
    if image1.shape != image2.shape:
        return "Изображения должны быть одного размера"

    diff = image1 != image2 # Если они не одинаковые (иначе = 0)
    npcr = np.sum(diff) / diff.size * 100
    return npcr

def calculate_uaci(image1, image2):
    if image1.shape != image2.shape:
      return "Изображения должны быть одного размера"

    diff = np.abs(image1 - image2) / 255.0
    uaci = np.mean(diff) * 100
    return uaci

def sewar_funcs(image1, image2):
    return [("MSE", sewar.mse(image1, image2)),
            ("SSIM", sewar.ssim(image1, image2)[0]),
            ("MS-SSIM", sewar.msssim(image1, image2).real),
            ("RMSE", sewar.rmse(image1, image2)),
            ("PSNR", sewar.psnr(image1, image2)),
            ("PSNR-B", sewar.psnrb(image1, image2)),
            ("UQI", sewar.uqi(image1, image2)),
            ("SCC", sewar.scc(image1, image2)),
            ("RASE", sewar.rase(image1, image2)),
            ("SAM", sewar.sam(image1, image2)),
            ("VIFP", sewar.vifp(image1, image2)),
            ("ERGAS", sewar.ergas(image1, image2)),
            ("D_lambda", sewar.d_lambda(image1, image2))]

def piq_funcs(image1, image2):
    img1_tensor = torch.tensor(image1, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    img2_tensor = torch.tensor(image2, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    return [("IW-SSIM", piq.information_weighted_ssim(img1_tensor, img2_tensor).item()),
            ("DSS", piq.dss(img1_tensor, img2_tensor).item()),
            ("HaarPSI", piq.haarpsi(img1_tensor, img2_tensor).item()),
            ("MDSI", piq.mdsi(img1_tensor, img2_tensor).item()),
            ("SR-SIM", piq.srsim(img1_tensor, img2_tensor).item()),
            ("FSIM", piq.fsim(img1_tensor, img2_tensor, chromatic=False).item()),
            ("VSI", piq.vsi(img1_tensor, img2_tensor).item()),
            ("GMSD", piq.gmsd(img1_tensor, img2_tensor).item()),
            ("MS-GMSD", piq.multi_scale_gmsd(img1_tensor, img2_tensor).item())]

'''
# Пути изображений
image1_path = "tortoise.jpg"
image2_path = "tortoise_bad.jpg"
image1, image2 = load_images(image1_path, image2_path)

print("Metrics whose values are calculated using the sewar library:")
img1_np = np.array(image1)
img2_np = np.array(image2)

for name, res in sewar_funcs(img1_np, img2_np):
  print(f"{name}: {res}")


print("\nMetrics whose values are calculated using the piq library:")
img1_tensor = torch.tensor(image1, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
img2_tensor = torch.tensor(image2, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0

for name, res in piq_funcs(img1_np, img2_np):
    print(f"{name}: {res}")


npcr = calculate_npcr(image1, image2)
uaci = calculate_uaci(image1, image2)
print("\nAnother metrics values:")
print(f"NPCR: {npcr:.2f}%")
print(f"UACI: {uaci:.2f}%\n")
'''