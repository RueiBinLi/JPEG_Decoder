# JPEG_Decoder

## Requirements
```bash
python==3.13.5
```
```bash
opencv-python
numpy==1.26.4
scikit-image
matplotlib
torch
torchmetrics
```

## Running Device
```bash
GPU: GTX 1650 super VRAM 4GB
CPU: AMD Ryzen R5 3600
Memory: 16GB
```

## Installation
1. Git Clone
```bash
git clone https://github.com/RueiBinLi/JPEG_Decoder.git
```
2. Install conda environment
```bash
conda create -n jpeg python=3.13.5 -y
conda activate jpeg
```
3. Install the required toolkit.
```bash
pip install -r requirements.txt
```
4. If you are using VSCode, compile the three c++ code.
```bash
g++ -O3 ./jpeg_decoder.cpp -o ./jpeg_decoder
```
```bash
g++ -O3 ./jpeg_decoder_bilinear.cpp -o ./jpeg_decoder_bilinear
```
```bash
g++ -O3 ./jpeg_decoder_optimized.cpp -o ./jpeg_decoder_optimized
```

## Execution
1. JPEG Decoder Original Version
```bash
./jpeg_decoder lena.jpg
```
2. JPEG Decoder Bilinear Interpolation Version
```bash
./jpeg_decoder_bilinear lena.jpg
```
3. JPEG Decoder Optimized Version
```bash
./jpeg_decoder_optimized lena.jpg
```

## Evaluation
1. Image
```bash
python evaluation_image.py
```
1. Dataset: Kodak
```bash
python evaluation_dataset.py kodak_jpg
```
1. Dataset: CLIC 
```bash
python evaluation_dataset.py CLIC_jpg
``` 

## Evaluation Result of Image
1. Lena

![lena](./lena.jpg)

2. Result

![lena_result](./performance_comparison_opencv_lena.png)

## Evaluation Result of Dataset
1. [Kodak](https://www.kaggle.com/datasets/sherylmehta/kodak-dataset)
   
![kodak_result](./dataset_benchmark_speed_quality_kodak.png)
   
2. [CLIC](https://www.kaggle.com/datasets/mustafaalkhafaji95/clic-dataset) High Resolution Dataset
   
![CLIC_result](./dataset_benchmark_speed_quality_CLIC.png)
