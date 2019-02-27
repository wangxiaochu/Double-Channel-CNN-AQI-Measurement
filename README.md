# Double-Channel-CNN-AQI-Measurement
This is the demo of paper "Air Quality Measurement Based on Double-Channel Convolutional Neural Network Ensemble Learning".

## Requirements

Python 3.5 <br>
Tensorflow 1.10.0 <br>
OpenCV 3.4.1 <br>
Numpy 1.15.4 <br>

## Usage
Put the images you want to test into the "images" folder, run the inference.py,there will output the images' air quality grade and
air quality index. <br>

```
python3 inference.py
```
## Files
The images in "images" folder are sample images, the name is label information, just like 1_44.jpg, the number before "\_" like "1" 
is air quality grade, the  number after "\_" like "44" is air quality index. <br>

The files in "models" is the trained models of air quality grade and index, where the "model-classification.pb" is the grade model 
and the  "model-regression.pb" is the index model.
