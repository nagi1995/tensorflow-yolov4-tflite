# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 21:35:59 2021

@author: Nagesh
"""
"""
References

https://github.com/haroonshakeel/tensorflow-yolov4-tflite
https://www.youtube.com/watch?v=XTiM6HU0x9k&ab_channel=CutTheChaos
https://github.com/krishnaik06/Deployment-flask

"""


#%%

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from math import ceil
from time import time
from flask import Flask, request, render_template
import os
from werkzeug.utils import secure_filename


#%%

app = Flask(__name__)

pics_folder = os.path.join("static", "images")
app.config["UPLOAD_FOLDER"] = pics_folder

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
FLAGS = {}
FLAGS["tiny"] = True
FLAGS["model"] = "yolov4"
STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)

score = .25
iou = 0.5
input_size = 416
weights_path = "./checkpoints/yolov4tiny-custom_best-416-tflite-fp16.tflite"

interpreter = tf.lite.Interpreter(model_path=weights_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()



@app.route("/")
def home():
    return render_template("index.html")


def detect_object_util(image_path):
    start = time()
    original_image = cv2.imread(image_path)
    h, w, _ = original_image.shape
    h_new = ceil(h/input_size) * input_size
    w_new = ceil(w/input_size) * input_size
    
    resized_image = cv2.resize(original_image, (w_new, h_new), cv2.INTER_LINEAR)
    
    col_list = []
    for i in range(h_new//input_size):
        row_list = []
        for j in range(w_new//input_size):
            tiled_image_original = resized_image[i*input_size : (i+1)*input_size, j*input_size : (j+1)*input_size, :] 
            tiled_image = tiled_image_original / 255.
            tiled_image = np.asarray(tiled_image).astype(np.float32)
            
            interpreter.set_tensor(input_details[0]['index'], [tiled_image])
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))
            
            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=iou,
                score_threshold=score)
            pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
            row_list.append(utils.draw_bbox(tiled_image_original, pred_bbox))
        col_list.append(cv2.hconcat(row_list))
    scaled_image = cv2.vconcat(col_list)
    reconstructed_image = cv2.resize(scaled_image, (w, h), interpolation = cv2.INTER_AREA)
    end = time()
    session.close()
    #cv2.imshow("predictions", reconstructed_image)
    cv2.imwrite(os.path.join(app.config["UPLOAD_FOLDER"], str(int(end)) + ".jpg"), reconstructed_image)
    
    return end, start

def delete_file():
    for f in  os.listdir(app.config["UPLOAD_FOLDER"]):
        os.remove(os.path.join(app.config["UPLOAD_FOLDER"], f))


@app.route("/predict", methods = ["GET", "POST"])
def predict():
    delete_file()
    if request.method == "POST":
        file = request.files["myfile"]
        filename = secure_filename(file.filename)
        if filename.lower().endswith(".jpg") or filename.lower().endswith(".png") or filename.lower().endswith(".tif"):
            file.save(filename)
            end, start = detect_object_util(filename)
            prediction_text1 = "Time taken for detection is {}s".format(np.round(end-start, 3))
            return render_template("index.html", prediction_text1 = prediction_text1, img_path = os.path.join(app.config["UPLOAD_FOLDER"], str(int(end)) + ".jpg"))
        else:
            return render_template("index.html", prediction_text1 = "Please upload a file with '.jpg' or '.png' format", img_path = "")
    else:
        return render_template("index.html")

 
if __name__ == "__main__":
    app.run(debug = True)