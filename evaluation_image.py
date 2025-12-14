import cv2
import os
import matplotlib.pyplot as plt
import subprocess
import numpy as np
import time
import re
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torch
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure

def calculate_msssim(img1, img2):
    t1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    t2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)
    try:
        score = ms_ssim(t1, t2)
        return score.item()
    except Exception as e:
        return 0.0

# Setting
input_image = "lena.jpg"
# input_image = "aaron.jpg"
qualities = [10, 20, 30, 40, 50, 60, 70, 80, 90]
NUM_RUNS = 5

decoders = {
    "OpenCV": "INTERNAL",
    "Original": "./jpeg_decoder",
    "Add Interpolation": "./jpeg_decoder_interpolation",
    "Optimized": "./jpeg_decoder_optimized",
}

results = {
    name: {"psnr": [], "ssim": [], "msssim": [], "time": []} 
    for name in decoders.keys()
}
bpp_list = []

print(f"Starting RD Curve + Time Performance test (Average {NUM_RUNS} times)...")

if not os.path.exists(input_image):
    print(f"Error: Can't fint the input image {input_image}")
    exit()

img_gt = cv2.imread(input_image)
h, w, _ = img_gt.shape
total_pixels = h * w

print("Testing image...")
for q in qualities:
    filename = f"test_q{q}.jpg"
    cv2.imwrite(filename, img_gt, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    file_size_bits = os.path.getsize(filename) * 8
    bpp = file_size_bits / total_pixels
    bpp_list.append(bpp)

for name, exe_path in decoders.items():
    print(f"\nTesting decoder: {name}")
    
    for i, q in enumerate(qualities):
        filename = f"test_q{q}.jpg"
        
        if exe_path == "INTERNAL":
            total_time = 0.0
            decoded_img = None
            
            # Evaluate the decoding time
            for run_idx in range(NUM_RUNS):
                start_time = time.time()
                decoded_img = cv2.imread(filename)
                end_time = time.time()
                total_time += (end_time - start_time) * 1000

            avg_time = total_time / NUM_RUNS
            img_out = decoded_img
            
            if img_out is None:
                print("Error: OpenCV loading error")
                s_psnr, s_ssim, s_msssim = 0, 0, 0
            else:
                s_psnr = psnr(img_gt, img_out)
                s_ssim = ssim(img_gt, img_out, win_size=7, channel_axis=2)
                s_msssim = calculate_msssim(img_gt, img_out)

            results[name]["psnr"].append(s_psnr)
            results[name]["ssim"].append(s_ssim)
            results[name]["msssim"].append(s_msssim)
            results[name]["time"].append(avg_time)
            
            print(f"  Q={q} | Avg Time={avg_time:.2f}ms | PSNR={s_psnr:.2f} | SSIM={s_ssim:.2f} | MS-SSIM={s_msssim:.2f}")
            continue

        if not os.path.exists(exe_path):
            print(f"Error: Can not find execution file: {exe_path}")
            for k in results[name]: results[name][k].append(0)
            continue

        total_time = 0.0
        success = False
        
        for run_idx in range(NUM_RUNS):
            start_time_py = time.time()
            proc = subprocess.run([exe_path, filename], capture_output=True, text=True)
            end_time_py = time.time()
            
            match = re.search(r"Decoding Time:\s+([\d\.]+)\s+ms", proc.stdout)
            if match:
                duration_ms = float(match.group(1))
            else:
                duration_ms = (end_time_py - start_time_py) * 1000
            
            total_time += duration_ms
            
            expected_out = "out_interpolation.ppm"
            if "interpolation" not in exe_path and "filter" not in exe_path:
                 expected_out = "out.ppm"
            if "optimized" in exe_path:
                 expected_out = "out_optimized.ppm" 

            if run_idx == 0: 
                if os.path.exists(expected_out):
                    success = True
                elif os.path.exists("out.ppm"):
                    expected_out = "out.ppm"
                    success = True

        avg_time = total_time / NUM_RUNS
        
        if not success:
            print(f"Error: Can not decode for output ppm successfully")
            for k in results[name]: results[name][k].append(0)
            continue
            
        img_out = cv2.imread(expected_out)
        if img_out is None:
             print(f"Error: Can not load {expected_out}")
             for k in results[name]: results[name][k].append(0)
             continue
             
        img_out = img_out[:h, :w, :] 
        
        s_psnr = psnr(img_gt, img_out)
        s_ssim = ssim(img_gt, img_out, win_size=7, channel_axis=2)
        s_msssim = calculate_msssim(img_gt, img_out)
        
        results[name]["psnr"].append(s_psnr)
        results[name]["ssim"].append(s_ssim)
        results[name]["msssim"].append(s_msssim)
        results[name]["time"].append(avg_time)
        
        print(f"  Q={q} | Avg Time={avg_time:.2f}ms | PSNR={s_psnr:.2f} | SSIM={s_ssim:.2f} | MS-SSIM={s_msssim:.2f}")

print("\nPlotting Comparison Charts...")

styles = ['k--', 'x-', 's--', 'o-']
colors = ['black', 'gray', 'blue', 'red']

metrics_config = [
    ("PSNR", "psnr", "dB"),
    ("SSIM", "ssim", "Score"),
    ("MS-SSIM", "msssim", "Score"),
    ("Decoding Time", "time", "ms")
]

plt.figure(figsize=(14, 10))

for idx, (title, key, unit) in enumerate(metrics_config):
    plt.subplot(2, 2, idx + 1)
    
    for i, (name, data) in enumerate(results.items()):
        style = styles[i % len(styles)]
        color = colors[i % len(colors)]
        plt.plot(bpp_list, data[key], style, label=name, color=color, linewidth=2, markersize=6)
    
    plt.title(f'{title} Comparison')
    plt.xlabel('Bits Per Pixel (bpp)')
    plt.ylabel(f'{title} ({unit})')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

plt.tight_layout()
plt.savefig('performance_comparison_opencv_lena.png')
print("Image is stored by performance_comparison_opencv_lena.png")