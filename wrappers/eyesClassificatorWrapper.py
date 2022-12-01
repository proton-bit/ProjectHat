import torchvision
import numpy as np
import torch
import json
import cv2

config = json.load(open('config.json'))

class EyesWrapper:
    def __init__(
            self,
            weights_path=config["eyes_model_weight_path"],
            confidence=config["eye_model_thresh"]
    ):
        self.image_height = 128
        self.image_width = 128
        self.confidence = confidence
        self.sigmoid = torch.nn.Sigmoid()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.init_model(weights_path)

    def init_model(self, weights_path):
        model = torchvision.models.mobilenet_v3_small()
        model.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=576, out_features=128, bias=True),
            torch.nn.Hardswish(),
            torch.nn.Dropout(p=0.2, inplace=True),
            torch.nn.Linear(in_features=128, out_features=1, bias=True)
        )
        model.load_state_dict(torch.load(weights_path, map_location=self.device))
        model.eval()
        return model

    def predict(self, image):
        input_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255
        input_image = cv2.merge([input_image]*3)
        input_image = cv2.resize(
            input_image,
            (self.image_width, self.image_height)
        )
        input_image = torch.tensor(
            np.expand_dims(input_image, axis=0)
        )
        input_image = input_image.permute([0, 3, 1, 2]).float().to(self.device)
        confidence = self.sigmoid(self.model(input_image)).item()
        cv2.imshow(str(confidence), image)
        cv2.waitKey()
        return confidence

    def __call__(self, image):
        return self.predict(image)
