import torchvision
import numpy as np
import torch
import json
import cv2

config = json.load(open('config.json'))

class EyesWrapper:
    def __init__(
            self,
            weights_path=config["blur_model_weight_path"],
            confidence=config["blur_model_thresh"]
    ):
        self.image_height = 128
        self.image_width = 128
        self.confidence = confidence
        self.softmax = torch.nn.Softmax(dim=1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.init_model(weights_path)
        self.classes = {'sharp': 0, 'defocused_blurred': 1, 'motion_blurred': 2}

    def init_model(self, weights_path):
        model = torchvision.models.mobilenet_v3_small()
        model.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=576, out_features=128, bias=True),
            torch.nn.Hardswish(),
            torch.nn.Dropout(p=0.2, inplace=True),
            torch.nn.Linear(in_features=128, out_features=3, bias=True)
        )
        model.load_state_dict(torch.load(weights_path, map_location=self.device))
        model.eval()
        return model

    def predict(self, image):
        input_image = cv2.resize(
            image / 255,
            (self.image_width, self.image_height)
        )
        input_image = torch.tensor(
            np.expand_dims(input_image, axis=0)
        )
        input_image = input_image.permute([0, 3, 1, 2]).float().to(self.device)
        outputs = self.model(input_image)
        normalized_outputs = torch.nn.functional.softmax(outputs, dim=1)
        print((normalized_outputs*10**4).int() / 10**4)
        # _, predicted_class_index = torch.max(outputs, 1)
        # predicted_class = list(self.classes.keys())[predicted_class_index.item()]
        # cv2.imshow(predicted_class, image)
        # cv2.waitKey()
        # return predicted_class

    def __call__(self, image):
        return self.predict(image)