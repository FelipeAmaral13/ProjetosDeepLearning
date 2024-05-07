import cv2
import supervision as sv
from ultralytics import YOLO
import os

class ObjectDetector:
    def __init__(self):
        self.model = YOLO(os.path.join('model', 'yolov8s.pt'))

    def detect(self, image_path):
        image = cv2.imread(image_path)
        result = self.model(image)[0]
        detections = sv.Detections.from_ultralytics(result)

        return detections

    def visualize(self, image_path, detections, save_path=None):
        image = cv2.imread(image_path)
        bounding_box_annotator = sv.BoundingBoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        labels = [detect[5]['class_name'] for detect in detections]

        annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

        if save_path is not None:
            cv2.imwrite(save_path, annotated_image)
            print(f"Annotated image saved at {save_path}")
        else:
            sv.plot_image(annotated_image)
