#----------------------------------------------
#--- Author         : Ahmet Ozlu
#--- Mail           : ahmetozlu93@gmail.com
#--- Date           : 27th January 2018
#----------------------------------------------

# Imports
import tensorflow as tf
import json

# Object detection imports
from utils import backbone
from api import object_counting_api
def recognizeImage(filename):
    #input_video = "https://www.mcdonalds.com.my/storage/foods/May2018/8Cw1if8XxZFkpuavHqp9.jpg"
    # By default I use an "SSD with Mobilenet" model here. See the detection model zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
    input_video = "."+str(filename)
    detection_graph, category_index = backbone.set_model('inference_graphGPU3', 'labelmap.pbtxt')

    is_color_recognition_enabled = 0

    result = object_counting_api.single_image_object_counting(str(input_video), detection_graph, category_index, is_color_recognition_enabled) # targeted objects counting

    mydict = {}
    result = result.replace("'","")
    result = result.replace(" ","")
    splitted = result.split(",")

    for v in splitted:
        aux = v.split(":")
        mydict[aux[0]] = aux[2]
        
    return mydict
