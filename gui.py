# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 18:55:42 2021

@author: Nagesh
"""

#%%
"""

References: 
https://github.com/haroonshakeel/tensorflow-yolov4-tflite
https://data-flair.training/blogs/python-project-traffic-signs-recognition/
https://www.youtube.com/watch?v=5M_J_SRGR3k&ab_channel=EssentialEngineering
https://www.youtube.com/watch?v=3E_fK5hCUnI&ab_channel=Codemy.com


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
from tkinter import filedialog, Tk, Label, Button, RIGHT, LEFT, TOP, BOTTOM, HORIZONTAL, Scale, StringVar, OptionMenu
from PIL import ImageTk, Image

#%%

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
FLAGS = {}
FLAGS["tiny"] = True
FLAGS["model"] = "yolov4"
STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
iou = 0.5
input_size = 416

OPTIONS = ["tf", "tflite"]

#%%

top=Tk()
top.state("zoomed")
top.title('Object Detection')
top.configure(background = "#CDCDCD")
label = Label(top, 
              background = "#CDCDCD", 
              font = ("arial", 15, "bold"))



sign_image = Label(top)

def detect_lite(file_path):
    
    weights_path = "./checkpoints/yolov4tiny-custom_best-416-tflite-fp16.tflite"
    score = conf_score_scale.get()
    
    interpreter = tf.lite.Interpreter(model_path = weights_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    
    start = time()
    original_image = cv2.imread(file_path)
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
            
            interpreter.set_tensor(input_details[0]["index"], [tiled_image])
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]["index"]) for i in range(len(output_details))]
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
    reconstructed_image = cv2.cvtColor(reconstructed_image, cv2.COLOR_BGR2RGB)
    end = time()
    
    return reconstructed_image, end-start

def detect(file_path):
    
    
    score = conf_score_scale.get()
    weights_path = "./checkpoints/yolov4-tiny-custom_best-416"
    
    start = time()
    original_image = cv2.imread(file_path)
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
            
            saved_model_loaded = tf.saved_model.load(weights_path, tags=[tag_constants.SERVING])
            infer = saved_model_loaded.signatures["serving_default"]
            batch_data = tf.constant([tiled_image])
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]
            
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
    reconstructed_image = cv2.cvtColor(reconstructed_image, cv2.COLOR_BGR2RGB)
    end = time()
    return reconstructed_image, end-start
    
def save_image(image):
    cv2.imwrite("./gui_save/" + str(int(time())) + ".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    label.configure(text = "Image saved to 'gui_save' folder")

def detect_object(file_path):
    global label_packed
    dropdown_selector = clicked.get()
    
    if dropdown_selector == "tf":
        reconstructed_image, detection_time = detect(file_path)
    elif dropdown_selector == "tflite":
        reconstructed_image, detection_time = detect_lite(file_path)
    else:
        pass
    im = cv2.resize(reconstructed_image, (500, 500))
    im = ImageTk.PhotoImage(Image.fromarray(im))
    
    sign_image.configure(image = im)
    sign_image.image = im
    label.configure(text = "Time taken for detection is {}s".format(np.round(detection_time, 3)))
    save_button = Button(top, 
                         text = "Save Image", 
                         command = lambda: save_image(reconstructed_image), 
                         padx = 10, pady = 5)
    
    save_button.configure(background = "#364156", 
                          foreground = "white", 
                          font = ("arial", 10, "bold"))
    
    save_button.place(relx = 0.69, rely = 0.87)


def show_detect_button(file_path):
    classify_b = Button(top, 
                        text = "Detect Objects", 
                        command = lambda: detect_object(file_path), 
                        padx = 10, pady = 5)
    
    classify_b.configure(background = "#364156", 
                         foreground = "white", 
                         font = ("arial", 10, "bold"))
    
    classify_b.place(relx = 0.19, rely = 0.87)

def select_image():
    try:
        file_path = filedialog.askopenfilename()
        
        if ".jpg" in file_path or ".png" in file_path or ".JPG" in file_path or ".PNG" in file_path:
                
            uploaded = Image.open(file_path)
            # uploaded.thumbnail(((500), (500)))
            uploaded = uploaded.resize((500, 500))
            im = ImageTk.PhotoImage(uploaded)
            sign_image.configure(image = im)
            sign_image.image = im
            label.configure(text = "")
            show_detect_button(file_path)
        else:
            
            label.configure(text = "Please select images with '.jpg', '.png', '.JPG', '.PNG' extensions")
            
    except:
        pass



conf_score_scale = Scale(top, 
                         from_ = 0.01, 
                         to = 0.99, 
                         length = 130, 
                         resolution = 0.01, 
                         orient = HORIZONTAL)

conf_score_scale.set(0.25)
conf_score_label = Label(top, 
                         text = "Confidence Selector", 
                         bg = "#364156", 
                         fg = "white", 
                         font = ('arial',10,'bold'))

clicked = StringVar()
clicked.set("tflite")
dropdown = OptionMenu(top, clicked, *OPTIONS)
dropdown_label = Label(top, 
                       text = "drop-down selector", 
                       bg = "#364156", 
                       fg = "white", 
                       font = ('arial',10,'bold'))

upload = Button(top, 
                text = "Select an image", 
                command = select_image, 
                padx = 10, pady = 5)

upload.configure(background = '#364156', 
                 foreground='white', 
                 font = ('arial',10,'bold'))


conf_score_label.place(relx = 0.01, rely = 0.01)
conf_score_scale.place(relx = 0.01, rely = 0.045)
dropdown_label.place(relx = 0.01, rely = 0.15)
dropdown.place(relx = 0.025, rely = 0.19)

upload.pack(side = BOTTOM, pady = 50)
sign_image.pack(side = BOTTOM)
label.pack(side = RIGHT, expand = True)
top.mainloop()
