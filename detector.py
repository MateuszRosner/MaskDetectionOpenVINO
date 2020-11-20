import numpy as np
import argparse
import imutils
import time
import cv2
import os

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream

from pyvino_utils.models.openvino_base.base_model import Base

COLOR = {"Green": (0, 255, 0), "Red": (0, 0, 255)}

class Detector():
    def __init__(self):
        pass

    def __del__(self):
        cv2.destroyAllWindows()
        self.vs.stop()

    def start_video_stream(self, source):
        self.vs = VideoStream(src=source).start()
        time.sleep(2.0)
        print("Video streaming established")

    def process_video(self):
        frame = self.vs.read()
        frame = imutils.resize(frame, width=300, height=300)

        return (frame)

class FaceDetection(Base):
    def __init__(self, model_name="models/face-detection-adas-0001", source_width=300, source_height=300, 
                 device="CPU", threshold=0.80, extensions=None, **kwargs):
        super().__init__(model_name, source_width, source_height, device, threshold, extensions, **kwargs)

    def preprocess_output(self, inference_results, image, show_bbox=False, **kwargs):
        results = {}
        if not (self._init_image_w and self._init_image_h):
            raise RuntimeError("Initial image width and height cannot be None.")
        if len(inference_results) == 1:
            inference_results = inference_results[0]

        bbox_coord = []
        for box in inference_results[0][0]:  # Output shape is 1x1xNx7
            conf = box[2]
            if conf >= self.threshold:
                xmin = int(box[3] * self._init_image_w)
                ymin = int(box[4] * self._init_image_h)
                xmax = int(box[5] * self._init_image_w)
                ymax = int(box[6] * self._init_image_h)
                bbox_coord.append((xmin, ymin, xmax, ymax))
                if show_bbox:
                    self.draw_output(image, xmin, ymin, xmax, ymax, **kwargs)

        results["image"] = image
        results["bbox_coord"] = bbox_coord
        return results

    @staticmethod
    def draw_output(image, xmin, ymin, xmax, ymax, label="Person", padding_size=(0.05, 0.25), scale=1, thickness=1, **kwargs):
        _label = None
        if kwargs.get("mask_detected"):
            _label = (
                (f"{label} Wearing Mask", COLOR["Green"])
                if float(kwargs.get("mask_detected")) < kwargs.get("threshold", 0.8)
                else (f"{label} No Mask!!!", COLOR["Red"])
            )

        label = _label if _label is not None else (label, COLOR["Green"])

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax,), color=label[1], thickness=thickness)

class MaskDetection(Base):
    def __init__(
        self,
        model_name = "models/face_mask",
        source_width=None,
        source_height=None,
        device="HDDL",
        threshold=0.80,
        extensions=None,
        **kwargs,
    ):
        super().__init__(
            model_name,
            source_width,
            source_height,
            device,
            threshold,
            extensions,
            **kwargs,
        )

    def preprocess_output(self, inference_results, image, show_bbox=False, **kwargs):
        results = {}
        results["flattened_predictions"] = np.vstack(inference_results).ravel()
        results["image"] = image
        return results

    def draw_output(
        self, image, inference_results, **kwargs,
    ):
        pass