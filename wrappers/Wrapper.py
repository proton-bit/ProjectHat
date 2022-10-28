import torch
import numpy as np
import json
import cv2

from yolov7.models.experimental import attempt_load
from yolov7.utils.general import non_max_suppression_kpt
from yolov7.utils.plots import output_to_keypoint, plot_skeleton_kpts

class PoseDetector:
    def __init__(
            self,
            weights_path : str,
            device : torch.device = torch.device("cpu"),
            image_size : int = 640,
            conf_thresh : float = 0.25,
            iou_thresh : float = 0.65
    ) -> None:
        self.model = attempt_load(weights_path)
        self.device = device
        self.image_size = image_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

    def prepare_image(self, image : np.ndarray) -> torch.tensor:
        # Resize image to the inference size
        orig_h, orig_w = image.shape[:2]
        image = cv2.resize(image, (self.image_size, self.image_size))

        # Transform image from numpy to torch format
        image_pt = torch.from_numpy(image).permute(2, 0, 1).to(self.device)
        image_pt = image_pt.float() / 255.0

        return orig_h, orig_w, image_pt

    def prepare_outputs(self, pred, orig_w, orig_h) -> np.ndarray:
        # NMS
        pred = non_max_suppression_kpt(
            pred, self.conf_thresh, self.iou_thresh,
            nc=self.model.yaml['nc'], nkpt=self.model.yaml['nkpt'], kpt_label=True
        )
        pred = output_to_keypoint(pred)[:, 7:]

        # Resize boxes to the original image size
        pred[:, 0::3] *= orig_w / self.image_size
        pred[:, 1::3] *= orig_h / self.image_size
        return pred

    def inference_model(self, image):
        with torch.no_grad():
            return self.model(image[None], augment=False)[0]

    def predict_keypoints(self, image : np.ndarray) -> np.ndarray:

        orig_h, orig_w, image_pt = self.prepare_image(image)

        pred = self.inference_model(image_pt)

        pred = self.prepare_outputs(pred)

        return pred



