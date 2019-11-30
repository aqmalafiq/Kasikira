import tensorflow as tf
import cv2
import numpy as np
from utils import visualization_utils as vis_util
from datetime import datetime

# Variables
total_passed_vehicle = 0  # using it to count vehicles
class mamakDetector:
    def __init__(self, detection_graph):
        # tf.device('/device:GPU:0')
        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    
    def startSession(self,detection_graph):
        config = tf.ConfigProto(intra_op_parallelism_threads=8, 
                                inter_op_parallelism_threads=2, 
                                allow_soft_placement=True, 
                                device_count = {'CPU': 8})
        detection_graph.as_default()
        self.sess = tf.Session(graph=detection_graph,config=config)

    def endSession(self):
        self.sess.close()

    def detectStream(self,input_video,frameNumber,category_index,is_color_recognition_enabled):
        counting_mode = "..."
        if type(input_video) == np.ndarray:
            input_frame = cv2.imdecode(input_video, cv2.IMREAD_COLOR)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(input_frame, axis=0)

            # Actual detection.
            t = datetime.now()
            (boxes, scores, classes, num) = self.sess.run(
                [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                feed_dict={self.image_tensor: image_np_expanded})
            t2 = datetime.now()
            print(t2 - t)

            # insert information text to video frame
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Visualization of the results of a detection.        
            counter, csv_line, counting_mode = vis_util.visualize_boxes_and_labels_on_image_array(frameNumber,
                                                                                                    input_frame,
                                                                                                    1,
                                                                                                    is_color_recognition_enabled,
                                                                                                    np.squeeze(boxes),
                                                                                                    np.squeeze(classes).astype(np.int32),
                                                                                                    np.squeeze(scores),
                                                                                                    category_index,
                                                                                                    use_normalized_coordinates=True,
                                                                                                    line_thickness=4)
            if(len(counting_mode) == 0):
                cv2.putText(input_frame, "...", (10, 35), font, 0.8, (0,255,255),2,cv2.FONT_HERSHEY_SIMPLEX)                       
            else:
                cv2.putText(input_frame, counting_mode, (10, 35), font, 0.8, (0,255,255),2,cv2.FONT_HERSHEY_SIMPLEX)
            
            _,final_img = cv2.imencode(".jpeg", input_frame)
        return counting_mode,final_img
    
    def detectSingleImage(self, input_video, detection_graph, category_index, is_color_recognition_enabled):
        counting_mode = "..."         
        if "http" in input_video :
            img = cv2.VideoCapture(input_video)
            if(img.isOpened()) :
                ret,input_frame = img.read()
        else :
            input_frame = cv2.imread(input_video)
        
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(input_frame, axis=0)

        # Actual detection.
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})

        # insert information text to video frame
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Visualization of the results of a detection.        
        counter, csv_line, counting_mode = vis_util.visualize_boxes_and_labels_on_single_image_array(1,input_frame,
                                                                                            1,
                                                                                            is_color_recognition_enabled,
                                                                                            np.squeeze(boxes),
                                                                                            np.squeeze(classes).astype(np.int32),
                                                                                            np.squeeze(scores),
                                                                                            category_index,
                                                                                            use_normalized_coordinates=True,
                                                                                            line_thickness=4)
        if(len(counting_mode) == 0):
            cv2.putText(input_frame, "...", (10, 35), font, 0.8, (0,255,255),2,cv2.FONT_HERSHEY_SIMPLEX)                       
        else:
            cv2.putText(input_frame, counting_mode, (10, 35), font, 0.8, (0,255,255),2,cv2.FONT_HERSHEY_SIMPLEX)
        status = cv2.imwrite(str(input_video),input_frame)
        # _,final_img = cv2.imencode(".jpeg", input_frame)
        while status == False :
            cv2.waitKey(100)
        return counting_mode
