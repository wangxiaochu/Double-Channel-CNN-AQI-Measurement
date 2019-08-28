# Double-Channel-CNN-AQI-Measurement
Thanks for your concern!<br>
This is the demo of paper "Air Quality Measurement Based on Double-Channel Convolutional Neural Network Ensemble Learning".<br>
In this project, we provide the test dataset and the trained models used in our paper, we also provide a simple demo test code, which will make it easier for you to verify the algorithm in our paper.

## Requirements

Python 3.5 <br>
Tensorflow 1.10.0 <br>
OpenCV 3.4.1 <br>
Numpy 1.15.4 <br>

## Usage
In order to try the models and algorithm performance, you just need to run the inference.py. You will see the testing result which is same as the result in our paper. <br>
```
python3 inference.py
```
The return information you could see on the console include the name of each image, the AQI prediction and the AQI groundtruth. Their format is as follows:<br>
```
20171122_090300.jpg Prediction: 1 Label: 1
```
Meanwhile, we will give the accuracy and the neighbor accuracy as in our paper.<br>
## Files
The "images" folder contains images of whole test dataset, The environment images are distributed in different subfolders according to its grade, and the subfolder name is its grade label.<br>

The "models" folder contains the trained models, include pre-trained models and packaged models. The subfolder named "model_OR" is the original model which without the weighted features fusion, while "model_FF" with this method.
