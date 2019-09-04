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

def load_image(image, i, input_size=64):
    h = image.shape[0]
    w = image.shape[1]
    image1 = image[:int(h*0.5), :, :]
    image2 = image[int(h*0.5):, :, :]
    if i < 4:
        w_start = int((w/4)*i)
        h1_start = int((image1.shape[0]/4)*0)
        h2_start = int((image2.shape[0]/4)*0)
        im1 = image1[h1_start:h1_start+input_size, w_start:w_start+input_size, :]
        im2 = image2[h2_start:h2_start+input_size, w_start:w_start+input_size, :]
    elif 4 <= i < 8:
        w_start = int((w/4)*(i-4))
        h1_start = int((image1.shape[0]/4)*1)
        h2_start = int((image2.shape[0]/4)*1)
        if image1.shape[0]/4 < 64:
            h1_start = int((image1.shape[0]/4)*1-(64-image1.shape[0]/4))
        if image2.shape[0]/4 < 64:
            h2_start = int((image2.shape[0]/4)*1-(64-image2.shape[0]/4))
        im1 = image1[h1_start:h1_start+input_size, w_start:w_start+input_size, :]
        im2 = image2[h2_start:h2_start+input_size, w_start:w_start+input_size, :]
    elif 8 <= i < 12:
        w_start = int((w/4)*(i-8))
        h1_start = int((image1.shape[0]/4)*2)
        h2_start = int((image2.shape[0]/4)*2)
        if image1.shape[0]/4 < 64:
            h1_start = int((image1.shape[0]/4)*2-(64-image1.shape[0]/4))
        if image2.shape[0]/4 < 64:
            h2_start = int((image2.shape[0]/4)*2-(64-image2.shape[0]/4))
        im1 = image1[h1_start:h1_start+input_size, w_start:w_start+input_size, :]
        im2 = image2[h2_start:h2_start+input_size, w_start:w_start+input_size, :]
    elif 12 <= i < 16:
        w_start = int((w/4)*(i-12))
        h1_start = int((image1.shape[0]/4)*3)
        h2_start = int((image2.shape[0]/4)*3)
        if image1.shape[0]/4 < 64:
            h1_start = int((image1.shape[0]/4)*2-(64-image1.shape[0]/4))
        if image2.shape[0]/4 < 64:
            h2_start = int((image2.shape[0]/4)*2-(64-image2.shape[0]/4))
        im1 = image1[h1_start:h1_start+input_size, w_start:w_start+input_size, :]
        im2 = image2[h2_start:h2_start+input_size, w_start:w_start+input_size, :]
    return im1, im2

def test_batch(image_path):
    image = cv2.imread(image_path)
    images1 = []
    images2 = []
    for i in range(16):
        image1, image2 = load_image(image, i)
        images1.append(image1)
        images2.append(image2)
    images1 = np.array(images1, dtype=np.float32)
    images2 = np.array(images2, dtype=np.float32)
    return images1, images2

def inference(image_dir):
    model_graph = load_model('models/model_OR/model_OR.pb')

    inputs1 = model_graph.get_tensor_by_name('input_image:0')
    inputs2 = model_graph.get_tensor_by_name('input_image_1:0')
    prediction = model_graph.get_tensor_by_name('prediction:0')
    keep_prob = model_graph.get_tensor_by_name('keep_prob:0')

    sess = tf.Session(graph=model_graph)
    N = 0
    acc = 0
    nb_acc = 0
    for root, dirs, files in os.walk(image_dir):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            for file in os.listdir(dir_path):
                image_path = os.path.join(dir_path, file)
                gt_label = int(dir)
                image1, image2 = test_batch(image_path)
                pred = sess.run(prediction, feed_dict={inputs1: image1, inputs2: image2, keep_prob: 1.0})
                result = np.argmax(np.bincount(np.argmax(pred, 1))) + 1
                print(file, 'Prediction: ', result, 'Label: ', gt_label)
                if result == gt_label:
                    acc += 1
                if result == gt_label or (result-1) == gt_label or (result + 1) == gt_label:
                    nb_acc += 1
                N += 1
    print('Accuracy: ', acc/N)
    print('Neighbor Accuracy: ', nb_acc/N)


if __name__ == '__main__':
    image_dir = 'images'
    inference(image_dir)
