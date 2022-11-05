import os
import cv2
import sys
from pathlib import Path
import torch
import numpy as np


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

sys.path.append("yolov5")
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.torch_utils import select_device, smart_inference_mode
from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import (
    Profile,
    check_img_size,
    non_max_suppression,
    scale_boxes
)

class HeadDetector:
    def __init__(
            self,
            weights: str = "weights/yolov5.pt",  # path to pretrained weights
            data: str = "weights/coco128.yaml",  # path to config
            device='',  # device to run yolo
            half=False,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
    ) -> None:
        self.device = select_device(device)
        self.model = DetectMultiBackend(
            weights,
            device=self.device,
            dnn=dnn,
            data=data,
            fp16=half,

        )

    @smart_inference_mode()
    def __call__(
                self,
                image,
                imgsz=(640, 640),  # inference size (height, width)
                conf_thres=0.5,  # confidence threshold
                iou_thres=0.45,  # NMS IOU threshold
                max_det=1000,  # maximum detections per image
                classes=None,  # filter by class: --class 0, or --class 0 2 3
                agnostic_nms=False,  # class-agnostic NMS
                augment=False,  # augmented inference
                visualize=False,  # visualize features
        ):

            stride, names, pt = self.model.stride, self.model.names, self.model.pt
            imgsz = check_img_size(imgsz, s=stride)  # check image size

            # Run inference
            self.model.warmup(imgsz=(1, 3, *imgsz))  # warmup
            seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

            im = letterbox(image, imgsz)[0]  # padded resize
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous

            with dt[0]:
                im = torch.from_numpy(im).to(self.model.device)
                im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                pred = self.model(im, augment=augment, visualize=visualize)

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            # Process predictions
            bboxes = list()
            confs= list()

            for i, det in enumerate(pred):  # per image
                seen += 1
                p, im0, frame = "frame", image.copy(), seen

                s = '%gx%g ' % im.shape[2:]  # print string
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):

                        coords = [int(a) for a in xyxy]
                        if conf > conf_thres:
                            bboxes.append(coords)
                            confs.append((conf))

                # Print time (inference-only)
                # LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
                return bboxes, confs

    def draw_bboxes(self, image, bboxes, confs, color = (200, 0, 255)):
        for coord, conf in sorted(list(zip(bboxes, confs)), key = lambda a: a[0][0], reverse=True):
            cv2.rectangle(
                image,
                (int(coord[0]), int(coord[1])),
                (int(coord[2]), int(coord[3])),
                color,
                10,
            )
            cv2.rectangle(
                image,
                (int(coord[0] - 5), int(coord[1] - 40)),
                (int(coord[0] + (len("face") + 7) * 16) , int(coord[1])),
                color,
                -1
            )

            cv2.putText(image,
                        f'face: {round(float(conf), 2)}',
                        (int(coord[0]),
                         int(coord[1] - 10)),
                        0, 1, (255, 255, 255), 2, cv2.LINE_AA)


