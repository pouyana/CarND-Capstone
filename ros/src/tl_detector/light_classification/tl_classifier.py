from styx_msgs.msg import TrafficLight
import numpy as np
import tensorflow as tf
import rospy
import threading as t
import cv2


class TLClassifier(object):

    def __init__(self):
        self.graph_path = rospy.get_param("tl_graph_file")
        self.class_labels = rospy.get_param("tl_classes")
        self.threshold = rospy.get_param("tl_threshold")

        self.counter = 0
        self.sess = None
        self.classes = None
        self.detection_graph = None
        self.state = TrafficLight.UNKNOWN
        if self.detection_graph is None:
            self.detection_graph = tf.Graph()

        if self.sess is None:
            with self.detection_graph.as_default():
                od_graph_def = tf.GraphDef()
                od_graph_def.ParseFromString(tf.gfile.GFile(self.graph_path, "rb").read())
                tf.import_graph_def(od_graph_def, name="")
            self.sess = tf.Session(graph=self.detection_graph)

            self.image_tensor = self.detection_graph.get_tensor_by_name("image_tensor:0")
            self.detection_box = self.detection_graph.get_tensor_by_name("detection_boxes:0")
            self.scores = self.detection_graph.get_tensor_by_name("detection_scores:0")
            self.classes = self.detection_graph.get_tensor_by_name("detection_classes:0")
            self.detection_num = self.detection_graph.get_tensor_by_name("num_detections:0")

    def get_prediction(self, image):
        """
        Returns the prediction for the given image

        :param image:
        :return:
        """
        image = self.process_image(image)
        image_np_expanded = np.expand_dims(image, axis=0)

        with self.detection_graph.as_default():
            pred_boxes, pred_scores, pred_classes = self.sess.run([self.detection_box, self.scores, self.classes],
                                                                      feed_dict={self.image_tensor: image_np_expanded})
            pred_boxes = pred_boxes.squeeze()
            pred_scores = pred_scores.squeeze()
            pred_classes = np.squeeze(pred_classes).astype(np.int32)

        for i, box in enumerate(pred_boxes):
            if pred_scores[i] > self.threshold:
                class_id = pred_classes[i]
                pred_score = pred_scores[i]
                print("Pred Score", pred_score, "Pred class", class_id)
                if class_id == 3:
                    self.state = TrafficLight.GREEN
                elif class_id == 1:
                    self.state = TrafficLight.RED
                elif class_id == 2:
                    self.state = TrafficLight.YELLOW
                else:
                    self.state = TrafficLight.UNKNOWN

    def process_image(self, image):
        image = cv2.resize(image, (300, 300))
        return image

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        self.get_prediction(image)
        return self.state