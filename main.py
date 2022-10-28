import os
import cv2
import json
from wrappers import PoseDetector

config = json.load(open("config.json"))


poseDetector = PoseDetector(config["pose_weights"])

for image_file in os.listdir(config["test_directory"]):

    image_path = os.path.join(
        config["test_directory"],
        image_file
    )
    image = cv2.imread(image_path)

    pred = poseDetector.predict_keypoints(image)
    image = poseDetector.plot_skeleton(image, pred)
    poseDetector.write_annotations(
        image, pred
    )

    cv2.imshow("output", image)
    cv2.waitKey(0)