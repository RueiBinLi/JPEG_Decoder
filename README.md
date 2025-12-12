# JPEG_Decoder

## Installation
1. Install the required toolkit.
```bash
pip install -r requirements.txt
```
2. If you are using VSCode, compile the three c++ code.
```bash
g++ -O3 ./jpeg_decoder.cpp -o ./jpeg_decoder
```
```bash
g++ -O3 ./jpeg_decoder_bilinear.cpp -o ./jpeg_decoder_bilinear
```
```bash
g++ -O3 ./jpeg_decoder_optimized.cpp -o ./jpeg_decoder_optimized
```

## Testing Image
1. Image: Lena

![lena](./lena.jpg)

2. Result

![lena](./lena.jpg)

## Testing Dataset
1. [Kodak](https://www.kaggle.com/datasets/sherylmehta/kodak-dataset)
   
![kodak_result](./dataset_benchmark_speed_quality_kodak.png)
   
2. [CLIC](https://www.kaggle.com/datasets/mustafaalkhafaji95/clic-dataset) High Resolution Dataset
   
