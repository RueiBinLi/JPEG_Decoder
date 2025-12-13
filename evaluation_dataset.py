import cv2
import os
import matplotlib.pyplot as plt
import subprocess
import numpy as np
import glob
import time
import re
import argparse
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

DEFAULT_DATASET_DIR = "./CLIC_jpg" 
qualities = [10, 20, 30, 40, 50, 60, 70, 80, 90]

decoders = {
    "OpenCV (Standard)": "INTERNAL_OPENCV",
    "Original": "./jpeg_decoder",
    "Add Interpolation": "./jpeg_decoder_interpolation",
    "Optimized": "./jpeg_decoder_optimized"
}

metrics = ["psnr", "ssim", "msssim", "time"]
final_results = {name: {m: [] for m in metrics} for name in decoders}
avg_bpp_list = [] 

def run_dataset_benchmark(dataset_dir):
    image_paths = []
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    
    if not os.path.isdir(dataset_dir):
        print(f"Error: Can not find folder '{dataset_dir}'")
        return

    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(dataset_dir, ext)))
    
    if not image_paths:
        print(f"Error: Can find any images in '{dataset_dir}'")
        return

    print(f"Dataset path: {dataset_dir}")
    print(f"Find {len(image_paths)} images，Starting Speed & Quality Benchmark...")
    print("-" * 60)

    for q in qualities:
        print(f"Testing Quality = {q} ...")
        
        q_stats = {"total_bpp": 0.0, "count": 0}
        decoder_sums = {
            name: {"psnr": 0.0, "ssim": 0.0, "msssim": 0.0, "time": 0.0} 
            for name in decoders
        }

        for img_path in image_paths:
            img_gt = cv2.imread(img_path)
            if img_gt is None: continue
            
            h, w, _ = img_gt.shape
            
            temp_jpg = "temp_test.jpg"
            cv2.imwrite(temp_jpg, img_gt, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            
            file_bits = os.path.getsize(temp_jpg) * 8
            bpp = file_bits / (h * w)
            q_stats["total_bpp"] += bpp
            q_stats["count"] += 1

            for name, exe_path in decoders.items():
                img_out = None
                duration_ms = 0.0

                if exe_path == "INTERNAL_OPENCV":
                    start_t = time.time()
                    img_out = cv2.imread(temp_jpg)
                    end_t = time.time()
                    duration_ms = (end_t - start_t) * 1000.0
                
                else:
                    if os.path.exists(exe_path):
                        expected_out = "out.ppm" 
                        if "interpolation" in exe_path: 
                            expected_out = "out_interpolation.ppm"
                        elif "optimized" in exe_path or "filter" in exe_path:   
                            expected_out = "out_filter.ppm"
                        
                        if os.path.exists(expected_out):
                            os.remove(expected_out)

                        try:
                            start_py = time.time()
                            proc = subprocess.run([exe_path, temp_jpg], capture_output=True, text=True, check=True)
                            end_py = time.time()
                            
                            match = re.search(r"Decoding Time:\s+([\d\.]+)\s+ms", proc.stdout)
                            if match:
                                duration_ms = float(match.group(1))
                            else:
                                duration_ms = (end_py - start_py) * 1000.0

                        except subprocess.CalledProcessError:
                            print(f"Error: {name} execute failed")
                            continue

                        if os.path.exists(expected_out):
                            img_out = cv2.imread(expected_out)
                        else:
                            if os.path.exists("out.ppm"):
                                img_out = cv2.imread("out.ppm")
                            else:
                                pass 

                if img_out is not None:
                    if img_out.size == 0: continue

                    if img_out.shape != img_gt.shape:
                        h_gt, w_gt, _ = img_gt.shape
                        h_out, w_out, _ = img_out.shape
                        if h_out >= h_gt and w_out >= w_gt:
                            img_out = img_out[:h_gt, :w_gt, :]
                        else:
                            continue
                    
                    decoder_sums[name]["psnr"] += psnr(img_gt, img_out)
                    decoder_sums[name]["ssim"] += ssim(img_gt, img_out, win_size=7, channel_axis=2)
                    decoder_sums[name]["msssim"] += calculate_msssim(img_gt, img_out)
                    decoder_sums[name]["time"] += duration_ms

        if q_stats["count"] > 0:
            avg_bpp = q_stats["total_bpp"] / q_stats["count"]
            avg_bpp_list.append(avg_bpp)
            
            print(f"  -> BPP: {avg_bpp:.2f}")
            
            for name in decoders:
                count = q_stats["count"]
                final_results[name]["psnr"].append(decoder_sums[name]["psnr"] / count)
                final_results[name]["ssim"].append(decoder_sums[name]["ssim"] / count)
                final_results[name]["msssim"].append(decoder_sums[name]["msssim"] / count)
                avg_time = decoder_sums[name]["time"] / count
                final_results[name]["time"].append(avg_time)
                
                print(f"     [{name:20}] Time: {avg_time:.2f} ms | PSNR: {final_results[name]['psnr'][-1]:.2f}")

    print("\nDrawing Speed & Quality Charts...")
    
    styles = ['k--', 'x-', 's--', 'o-'] 
    colors = ['black', 'gray', 'blue', 'red'] 
    
    metrics_config = [
        ("PSNR", "psnr", "dB"),
        ("SSIM", "ssim", "Score"),
        ("MS-SSIM", "msssim", "Score"),
        ("Avg Decoding Time", "time", "ms")
    ]

    plt.figure(figsize=(14, 10))

    for idx, (title, key, unit) in enumerate(metrics_config):
        plt.subplot(2, 2, idx + 1)
        
        for i, (name, data) in enumerate(final_results.items()):
            style = styles[i % len(styles)]
            color = colors[i % len(colors)]
            lw = 2.5 if "Optimized" in name else 1.5
            plt.plot(avg_bpp_list, data[key], style, label=name, color=color, linewidth=lw, markersize=6)
        
        plt.title(f'{title}')
        plt.xlabel('Average BPP')
        plt.ylabel(f'{title} ({unit})')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

    plt.tight_layout()
    dataset_name = os.path.basename(os.path.normpath(dataset_dir))
    output_png = f"benchmark_{dataset_name}.png"
    plt.savefig(output_png)
    print(f"Image stored: {output_png}")
    
    if os.path.exists("temp_test.jpg"): os.remove("temp_test.jpg")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='JPEG Decoder Performance Benchmark')
    parser.add_argument('dataset', nargs='?', default=DEFAULT_DATASET_DIR, 
                        help=f'Path to the dataset directory (default: {DEFAULT_DATASET_DIR})')
    
    args = parser.parse_args()
    
    run_dataset_benchmark(args.dataset)