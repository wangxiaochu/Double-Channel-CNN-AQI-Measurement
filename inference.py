#-*-coding:utf-8-*-
import tensorflow as tf
import os
import cv2
import numpy as np

def load_model(path):
    if not os.path.exists(path):
        raise ValueError("'path_to_model' is not exist.")

    model_graph = tf.Graph()
    with model_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return model_graph

def load_image(im_path, input_size=64):
    
    im = cv2.imread(im_path)
    h,w = im.shape[:2]
    if h > w:
        ratio = float(h/w)
        w = 500
        h = int(500*ratio)
    else:
        ratio = float(w/h)
        h = 500
        w = int(500*ratio)
    this_im = cv2.resize(im, (w, h))

    newImages = []
    image1 = this_im[:int(this_im.shape[0]/2), :, :]
    image2 = this_im[int(this_im.shape[0]/2):, :, :]
    images = [image1, image2]
    for this_im in images:
        height = this_im.shape[0]
        width = this_im.shape[1]
        #random crop
        end_y = height - input_size
        end_x = width - input_size
        x_start = np.random.randint(0, end_x)
        y_start = np.random.randint(0, end_y)
        newImage = this_im[y_start:y_start + input_size, x_start:x_start + input_size, :]
        newImage = np.expand_dims(np.array(newImage, dtype=np.float32), 0)
        newImages.append(newImage)
    
    return newImages[0],newImages[1]

def inference(image_dir):
    model_graph1 = load_model('models/model-classification.pb')
    model_graph2 = load_model('models/model-regression.pb')

    inputs11 = model_graph1.get_tensor_by_name('input_image:0')
    inputs12 = model_graph1.get_tensor_by_name('input_image_1:0')
    inputs21 = model_graph2.get_tensor_by_name('input_image:0')
    inputs22 = model_graph2.get_tensor_by_name('input_image_1:0')
    prediction1 = model_graph1.get_tensor_by_name('prediction:0')
    prediction2 = model_graph2.get_tensor_by_name('fc8/BiasAdd:0')
    keep_prob1 = model_graph1.get_tensor_by_name('keep_prob:0')
    keep_prob2 = model_graph2.get_tensor_by_name('keep_prob:0')

    input_size = inputs11.get_shape()[1]

    sess1 = tf.Session(graph=model_graph1)
    sess2 = tf.Session(graph=model_graph2)
    for file in os.listdir(image_dir):
        image_path = os.path.join(image_dir, file)
        results1 = [0, 0, 0, 0, 0, 0]
        results2 = 0
        for i in range(20):
            image1, image2 = load_image(image_path, input_size)
            pred1 = sess1.run(prediction1, feed_dict={inputs11: image1, inputs12: image2, keep_prob1: 1.0})
            pred2 = sess2.run(prediction2, feed_dict={inputs21: image1, inputs22: image2, keep_prob2: 1.0})
            results1[np.argmax(pred1)] += 1
            results2 += np.squeeze(pred2)
        result1 = np.argmax(results1) + 1
        result2 = int(np.round(results2/20))
        print(file, 'Grade: ', result1, 'Index: ', result2)

if __name__ == '__main__':
        image_dir = 'images'
        inference(image_dir)